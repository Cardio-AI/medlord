""" Training functions for the different models. """
from collections import OrderedDict
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F


from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# ----------------------------------------------------------------------------------------------------------------------
# Controlnet
# ----------------------------------------------------------------------------------------------------------------------
def train_controlnet(
    controlnet: nn.Module,
    diffusion: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    #text_encoder,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    scale_factor: float = 1.0,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> float:
    scaler = GradScaler()
    raw_controlnet = controlnet.module if hasattr(controlnet, "module") else controlnet

    val_loss = eval_controlnet(
        controlnet=controlnet,
        diffusion=diffusion,
        stage1=stage1,
        scheduler=scheduler,
        #text_encoder=text_encoder,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        scale_factor=scale_factor,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_controlnet(
            controlnet=controlnet,
            diffusion=diffusion,
            stage1=stage1,
            scheduler=scheduler,
            #text_encoder=text_encoder,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=n_epochs,
            writer=writer_train,
            scaler=scaler,
            scale_factor=scale_factor,
        )
        print(f"Epoch: {epoch}")
        if lr_scheduler is not None:
            lr_scheduler.step()

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_controlnet(
                controlnet=controlnet,
                diffusion=diffusion,
                stage1=stage1,
                scheduler=scheduler,
                #text_encoder=text_encoder,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                scale_factor=scale_factor,
                controlnet_scale = 1.0,
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")


            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "controlnet": controlnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / ("checkpoint_" + str(epoch + 1) + ".pth")))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_controlnet.state_dict(), str(run_dir / "best_model.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_controlnet.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss
def ensure_integer_labels(tensor):
    """
    Convert tensor values to integers. This rounds to the nearest integer.
    """
    return tensor.to(torch.long)

def train_epoch_controlnet(
    controlnet: nn.Module,
    diffusion: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    writer: SummaryWriter,
    scaler: GradScaler,
    scale_factor: float = 1.0,
    probability_include: float = 0.5  # Probability of including each condition
) -> None:
    controlnet.train()
    torch.cuda.empty_cache()
    save_memory = False
    presaved = False

    probs = [0.7,0.7]
    
    for i, x in enumerate(loader):        
        images = x["image"].as_tensor().to(device)

        with torch.no_grad(): 
            conditions = [x[key].as_tensor().to(torch.float32) for key in 
                        ["ventricle_l","myocardium"]]

            # Ensure probs tensor is on the same device
            keep_masks = torch.rand(len(conditions), device=images.device) < torch.tensor(probs, device=images.device)

            for i, condition in enumerate(conditions):
                if not keep_masks[i]:  
                    conditions[i].zero_()

        cond = torch.cat(conditions, dim=1).to(device)

        del conditions
        torch.cuda.empty_cache()


        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            if presaved:
                images = images * scale_factor
                noise = torch.randn_like(images,device=device)
                noisy_e = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
            else:
                with torch.no_grad():
                    e = stage1(images)* scale_factor
                if save_memory:
                    del images
                    stage1.to("cpu")
                    torch.cuda.empty_cache()
                noise = torch.randn_like(e).to(device)
                noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            down_block_res_samples, mid_block_res_sample = controlnet(
                x=noisy_e, timesteps=timesteps, controlnet_cond=cond,conditioning_scale=1.0
            )
            if save_memory:
                del cond
                torch.cuda.empty_cache()
            noise_pred = diffusion(
                x=noisy_e,
                timesteps=timesteps,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                compute_encoder_grads=True
            )
            

            if scheduler.prediction_type == "v_prediction":
                if presaved:
                    target = scheduler.get_velocity(images, noise, timesteps)
                else:
                    target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise

            loss = F.smooth_l1_loss(noise_pred.float(), target.float())

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + i)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + i)

        #pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})
        #print("Epoch: ",epoch)
        #print("Loss: ",f"{losses['loss'].item():.5f}")
        if not presaved:
            if save_memory:
                stage1.to(device)
                torch.cuda.empty_cache()

        #print(torch.cuda.memory_summary(device))
        
        


@torch.no_grad()
def eval_controlnet(
    controlnet: nn.Module,
    diffusion: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    #text_encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    scale_factor: float = 1.0,
    controlnet_scale: float = 1.0,
) -> float:
    controlnet.eval()
    total_losses = OrderedDict()
    torch.cuda.empty_cache()
    presaved = False


    with torch.no_grad():

        for i, x in enumerate(loader):
            images = x["image"].as_tensor().to(device)
            shape = images.shape

            with torch.no_grad():   
                cond1 = x["ventricle_l"].as_tensor().to(torch.float32) 
                cond2 = x["myocardium"].as_tensor().to(torch.float32) 

                #cond = x["segments"].as_tensor().to(torch.bool).to(device)

            cond = torch.cat([cond1,cond2], dim=1).to(device)

            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
            with autocast(enabled=True):
                if presaved:
                    images = images * scale_factor
                    noise = torch.randn_like(images,device=device)
                    noisy_e = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                    
                else:
                
                    e = stage1(images)* scale_factor

                    noise = torch.randn_like(e).to(device)
                    noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

                    
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=noisy_e, timesteps=timesteps, controlnet_cond=cond,conditioning_scale=1.0
                ) 
            
                noise_pred = diffusion(
                    x=noisy_e,
                    timesteps=timesteps,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    compute_encoder_grads=True
                )

                if scheduler.prediction_type == "v_prediction":
                    if presaved:
                        target = scheduler.get_velocity(images, noise, timesteps)
                    else:
                        target = scheduler.get_velocity(e, noise, timesteps)
                elif scheduler.prediction_type == "epsilon":
                    target = noise
                loss = F.l1_loss(noise_pred.float(), target.float())

            loss = loss.mean()
            losses = OrderedDict(loss=loss)


            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * shape[0]

            writer.add_scalar(tag="val_loss", scalar_value=total_losses["loss"], global_step=step)



        for k in total_losses.keys():
            total_losses[k] /= len(loader.dataset)

        for k, v in total_losses.items():
            writer.add_scalar(f"{k}", v, step)
        torch.cuda.empty_cache()
        

    return total_losses["loss"]
