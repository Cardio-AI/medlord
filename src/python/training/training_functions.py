""" Training functions for the different models. """
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.spectral_loss import JukeboxLoss
from pynvml.smi import nvidia_smi
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from src.python.training.util import log_reconstructions
from src.python.functions.networks.schedulers import RFlowScheduler


import matplotlib.pyplot as plt



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")
def get_adv_weight(epoch, warmup_epochs=100, max_weight=1.0):

    if epoch < warmup_epochs:
        # Linear ramp-up: gradually increase adv_weight from 0 to max_weight
        return (epoch / warmup_epochs) * max_weight
    else:
        # Keep adv_weight constant after warm-up period
        return max_weight

# ----------------------------------------------------------------------------------------------------------------------
# AUTOENCODER
# ----------------------------------------------------------------------------------------------------------------------
def train_vqgan(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    g_scheduler: torch.optim.lr_scheduler._LRScheduler,  
    d_scheduler: torch.optim.lr_scheduler._LRScheduler,  
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    adv_weight: float,
    perceptual_weight: float,
    adv_start: int,
) -> float:
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_vqgan(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        adv_weight=adv_weight if start_epoch >= adv_start else 0.0,
        perceptual_weight=perceptual_weight,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    for epoch in range(start_epoch, n_epochs):
        adv_weight = get_adv_weight(epoch, warmup_epochs=adv_start, max_weight=0.005)
        train_epoch_vqgan(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            writer=writer_train,
            adv_weight=adv_weight if epoch >= adv_start else 0.0,
            perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )
        g_scheduler.step()
        d_scheduler.step()

        if (epoch + 1) % eval_freq == 0:
            
            model_ref = model.module if isinstance(model, torch.nn.DataParallel) else model

            if hasattr(model_ref, "quantizer") and hasattr(model_ref.quantizer, "quantizer"):
                unused_mask = model_ref.quantizer.quantizer.ema_cluster_size < model_ref.quantizer.quantizer.restart_threshold  
                num_unused = unused_mask.sum().item()
                print(f"Number of unused embeddings: {num_unused}")
                
                # Log to TensorBoard (Validation)
                writer_val.add_scalar("Codebook/Unused_Embeddings", num_unused, epoch)
            val_loss = eval_vqgan(
                model=model,
                discriminator=discriminator,
                perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                adv_weight=adv_weight,
                perceptual_weight=perceptual_weight,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
                "ema_cluster_size": model.quantizer.quantizer.ema_cluster_size,  # Add this
                "ema_w": model.quantizer.quantizer.ema_w,
            }
            torch.save(checkpoint, str(run_dir / f"checkpoint_epoch_{epoch + 1}.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
 
    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_vqgan(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    adv_weight: float,
    perceptual_weight: float,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
) -> None:
    model.train()
    discriminator.train()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    jukebox_loss = JukeboxLoss(spatial_dims=3)

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"].as_tensor().to(device)
        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            reconstruction, quantization_loss = model(images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            j_loss = jukebox_loss(reconstruction.float(), images.float())

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # ADDED ENTROPY LOSS TO ENFORCE DIVERSE CODEBOOK - ON EMA CLUSTERS FOR SMOOTH LEARNING

            cluster_size = model.quantizer.quantizer.ema_cluster_size 
            cluster_prob = cluster_size / (cluster_size.sum() + 1e-8)

            entropy = -torch.sum(cluster_prob * torch.log(cluster_prob + 1e-8))  # Avoid log(0)
            entropy_loss = -entropy

            entropy_weight = 0.001  # Start with low weight, increase as resolution of patches increases.

            loss = l1_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss + j_loss+ entropy_weight * entropy_loss
            
            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            quantization_loss = quantization_loss.mean()
            g_loss = generator_loss.mean()
            j_loss = j_loss.mean()
            entropy_loss = entropy_loss.mean()

            #loss = l1_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss + j_loss

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                quantization_loss=quantization_loss,
                g_loss=g_loss,
                j_loss=j_loss,
                entropy_loss=entropy_loss,

            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5) 
        scaler_g.step(optimizer_g)
        scaler_g.update()


        # DISCRIMINATOR
        if adv_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                d_loss = adv_weight * discriminator_loss
                d_loss = d_loss.mean()

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.5) 
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            discriminator_loss = torch.tensor([0.0]).to(device)

        losses["d_loss"] = discriminator_loss

        writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

    pbar.set_postfix(
        {
            "epoch": epoch,
            "loss": f"{losses['loss'].item():.6f}",
            "l1_loss": f"{losses['l1_loss'].item():.6f}",
            "p_loss": f"{losses['p_loss'].item():.6f}",
            "g_loss": f"{losses['g_loss'].item():.6f}",
            "d_loss": f"{losses['d_loss'].item():.6f}",
            "lr_g": f"{get_lr(optimizer_g):.6f}",
            "lr_d": f"{get_lr(optimizer_d):.6f}",
        },
    )


@torch.no_grad()
def eval_vqgan(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    adv_weight: float,
    perceptual_weight: float,
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    jukebox_loss = JukeboxLoss(spatial_dims=3)
    total_losses = OrderedDict()
    for x in loader:
        images = x["image"].as_tensor().to(device)
        with autocast(enabled=True):
            # GENERATOR
            reconstruction, quantization_loss = model(images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            j_loss = jukebox_loss(reconstruction.float(), images.float())

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # DISCRIMINATOR
            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            else:
                discriminator_loss = torch.tensor([0.0]).to(device)

            # ADDED ENTROPY LOSS TO ENFORCE DIVERSE CODEBOOK - ON EMA CLUSTERS FOR SMOOTH LEARNING
            cluster_size = model.quantizer.quantizer.ema_cluster_size 
            cluster_prob = cluster_size / (cluster_size.sum() + 1e-8)
            entropy = -torch.sum(cluster_prob * torch.log(cluster_prob + 1e-8))  # Avoid log(0)
            entropy_loss = -entropy

            entropy_weight = 0.001

            #loss = l1_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss + j_loss + perplexity_weight * perplexity_loss
            loss = l1_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss + j_loss + entropy_weight * entropy_loss
            
            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            quantization_loss = quantization_loss.mean()
            g_loss = generator_loss.mean()
            d_loss = discriminator_loss.mean()
            j_loss = j_loss.mean()
            entropy_loss = entropy_loss.mean()

            #loss = l1_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss + j_loss
            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                quantization_loss=quantization_loss,
                g_loss=g_loss,
                d_loss=d_loss,
                j_loss=j_loss,
                entropy_loss=entropy_loss
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    log_reconstructions(
        image=images,
        reconstruction=reconstruction,
        writer=writer,
        step=step,
    )

    return total_losses["l1_loss"]

# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Model
# ----------------------------------------------------------------------------------------------------------------------
def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
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
    raw_model = model.module if hasattr(model, "module") else model

    #log_dir = log_dir 
    #tracker = CarbonTracker(epochs=50,epochs_before_pred=10,monitor_epochs=-1,log_dir=log_dir,interpretable=True,verbose=2)

    val_loss = eval_ldm(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        sample=False,
        scale_factor=scale_factor,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    iteration = len(train_loader) * start_epoch
    for epoch in range(start_epoch, n_epochs):
        #tracker.epoch_start()
        iteration = train_epoch_ldm(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            scale_factor=scale_factor,
            start_iteration=iteration,
            checkpoint_interval=eval_freq,
            run_dir=run_dir,  # Save checkpoints directly from train_epoch_ldm
            best_loss=best_loss,
        )
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ldm(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                sample=False,
                scale_factor=scale_factor,
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / ("checkpoint.pth")))
            torch.save(raw_model.state_dict(), str(run_dir / ("checkpoint_" + str(epoch + 1) + ".pth")))

            

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))

        #tracker.epoch_end()
    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))
    #torch.save(checkpoint, str(run_dir / ("checkpoint.pth")))

    #tracker.stop()
    
    return val_loss


def train_epoch_ldm(
    model: torch.nn.Module,
    stage1: torch.nn.Module,
    scheduler: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler,
    scale_factor: float = 1.0,
    start_iteration: int = 0,
    checkpoint_interval: int = 10000,  # Save checkpoint every checkpoint_interval steps
    run_dir: Path = Path("checkpoints"),
    best_loss: float = float('inf')
) -> int:
    model.train()
    iteration = start_iteration  # Start counting from the provided start iteration
    # Check if latents are presaved 
    presaved_latents = True

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"].as_tensor().to(device)
        
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)

        if presaved_latents:
            with autocast(enabled=True):
                images = images * scale_factor
                noise = torch.randn_like(images).to(device)
                if isinstance(scheduler, RFlowScheduler):
                    timesteps = scheduler.sample_timesteps(images)
                    
                noisy_e = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                noise_pred = model(x=noisy_e, timesteps=timesteps)
                
                #del noisy_e
                #torch.cuda.empty_cache()
                if isinstance(scheduler, RFlowScheduler):
                    target = images - noise
                else:
                    if scheduler.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(images, noise, timesteps)
                    elif scheduler.prediction_type == "epsilon":
                        target = noise
                #del images, noise
                #torch.cuda.empty_cache()
                loss = F.smooth_l1_loss(noise_pred.float(), target.float()) #Huber loss
                #loss = F.l1_loss(noise_pred.float(), target.float())
        else:
            with autocast(enabled=True):
                with torch.no_grad():
                    e = stage1(images) * scale_factor

                #del images
                #stage1.to("cpu")
                #torch.cuda.empty_cache()

                noise = torch.randn_like(e).to(device)
                if isinstance(scheduler, RFlowScheduler):
                    timesteps = scheduler.sample_timesteps(e)
                noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
                noise_pred = model(x=noisy_e, timesteps=timesteps)

                #del noisy_e
                #torch.cuda.empty_cache()
                if isinstance(scheduler, RFlowScheduler):
                    target = images - noise
                else:
                    if scheduler.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(e, noise, timesteps)
                    elif scheduler.prediction_type == "epsilon":
                        target = noise
                #del e,noise
                #torch.cuda.empty_cache()
                #loss = F.l1_loss(noise_pred.float(), target.float())
                loss = F.smooth_l1_loss(noise_pred.float(), target.float()) #Huber loss
        losses = OrderedDict(loss=loss)
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), iteration)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), iteration)

        #pbar.set_postfix({"iteration": iteration, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})
        pbar.set_postfix({"epoch": epoch+1, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})
        
        iteration += 1  # Increment global iteration counter
        #stage1.to(device)
        #torch.cuda.empty_cache()

    return iteration

@torch.no_grad()
def eval_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    sample: bool = False,
    scale_factor: float = 1.0,
) -> float:
    model.eval()
    total_losses = OrderedDict()
    presaved_latents = True
    scale_factor = 1.0


    for x in loader:
        images = x["image"].as_tensor().to(device)
        shape = images.shape
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
        if presaved_latents:
            with autocast(enabled=True):
                images = images * scale_factor
                noise = torch.randn_like(images).to(device)
                if isinstance(scheduler, RFlowScheduler):
                    timesteps = scheduler.sample_timesteps(images)
                noisy_e = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                noise_pred = model(x=noisy_e, timesteps=timesteps)
                
                if isinstance(scheduler, RFlowScheduler):
                    target = images - noise
                else:
                    if scheduler.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(images, noise, timesteps)
                    elif scheduler.prediction_type == "epsilon":
                        target = noise
                #del images,noise
                #torch.cuda.empty_cache()
                loss = F.smooth_l1_loss(noise_pred.float(), target.float()) #Huber loss
                #loss = F.l1_loss(noise_pred.float(), target.float()) 
        else:
            with autocast(enabled=True):
                e = stage1(images) * scale_factor

                #del images
                #torch.cuda.empty_cache()
                noise = torch.randn_like(e).to(device)
                if isinstance(scheduler, RFlowScheduler):
                    timesteps = scheduler.sample_timesteps(images) 
                noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
                noise_pred = model(x=noisy_e, timesteps=timesteps)

                if isinstance(scheduler, RFlowScheduler):
                    target = images - noise
                else:
                    if scheduler.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(e, noise, timesteps)
                    elif scheduler.prediction_type == "epsilon":
                        target = noise
                #loss = F.l1_loss(noise_pred.float(), target.float()) 
                loss = F.smooth_l1_loss(noise_pred.float(), target.float()) #Huber loss
        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():

            total_losses[k] = total_losses.get(k, 0) + v.item() * shape[0]
        #del e, noise, noisy_e, noise_pred, target
        #torch.cuda.empty_cache()

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    return total_losses["loss"]

# ----------------------------------------------------------------------------------------------------------------------
# Autoregressive transformer
# ----------------------------------------------------------------------------------------------------------------------
def train_transformer(
    model: nn.Module,
    ordering,
    stage1: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    inferer,
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
) -> float:
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

   
    val_loss = eval_transformer(
        model=model,
        ordering=ordering,
        stage1=stage1,
        loader=val_loader,
        inferer=inferer,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_transformer(
            model=model,
            ordering=ordering,
            stage1=stage1,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
        )
        scheduler.step()

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_transformer(
                model=model,
                ordering=ordering,
                stage1=stage1,
                loader=val_loader,
                inferer=inferer,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "transformer": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }

            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, str(checkpoint_path))
            print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))


    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))


    return val_loss


def train_epoch_transformer(
    model: nn.Module,
    ordering,
    stage1: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler,
) -> None:
    model.train()
    presaved_latents = False

    ce_loss = CrossEntropyLoss()

    raw_model = model.module if hasattr(model, "module") else model
    raw_stage1 = stage1.module if hasattr(stage1, "module") else stage1

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            if presaved_latents:
                latent = images
            else:
                with torch.no_grad():
                    latent = stage1(images)

            latent = latent.reshape(latent.shape[0], -1)
            latent = latent[:, ordering.get_sequence_ordering()]

            target = latent.clone()
            latent = F.pad(latent, (1, 0), "constant", raw_stage1.model.num_embeddings)
            latent = latent[:, :-1]
            latent = latent.long()

            max_seq_len = raw_model.max_seq_len
            start = 0
            logits = model(x=latent)
            target = target[:, start : start + max_seq_len]

            logits = logits.transpose(1, 2)

            loss = ce_loss(logits, target)

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})


@torch.no_grad()
def eval_transformer(
    model: nn.Module,
    ordering,
    stage1: nn.Module,
    loader: torch.utils.data.DataLoader,
    inferer,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
) -> float:
    model.eval()
    total_losses = OrderedDict()
    presaved_latents = False

    ce_loss = CrossEntropyLoss()
    raw_model = model.module if hasattr(model, "module") else model
    raw_stage1 = stage1.module if hasattr(stage1, "module") else stage1

    for x in loader:
        images = x["image"].to(device)

        with autocast(enabled=True):
            if presaved_latents:
                latent = images
            else:
                with torch.no_grad():
                    latent = stage1(images)
            latent = latent.reshape(latent.shape[0], -1)
            latent = latent[:, ordering.get_sequence_ordering()]

            target = latent.clone()
            latent = F.pad(latent, (1, 0), "constant", raw_stage1.model.num_embeddings)
            latent = latent[:, :-1]
            latent = latent.long()
            print(latent.min(), latent.max())

            max_seq_len = raw_model.max_seq_len
            start = 0
            logits = model(x=latent)
            target = target[:, start : start + max_seq_len]

            logits = logits.transpose(1, 2)

            loss = ce_loss(logits, target)

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    if step % 50 == 0:
        vq_gan_model = raw_stage1.model
        transformer_model = raw_model
        sample = inferer.sample(
            starting_tokens=vq_gan_model.num_embeddings
            * torch.ones(1, 1).to(device),
            latent_spatial_dim=(28,28,16),
            vqvae_model=vq_gan_model,
            transformer_model=transformer_model,
            ordering=ordering,
            verbose=False,
        )
        print(sample.shape)
        fig = plt.figure()
        
        max_z = sample.shape[0]
        # select spaced 25%, 50%, 75% of the z dimension
        slices = [int(max_z * 0.25), int(max_z * 0.5), int(max_z * 0.75)]
        for i in range(len(slices)):
            plt.subplot(1, len(slices), i + 1)
            plt.imshow(sample[0, 0, slices[i], :, :].cpu(), cmap="gray")
        plt.show()


        writer.add_figure(
            tag="samples", figure=fig, global_step=step
        )

    return total_losses["loss"]
