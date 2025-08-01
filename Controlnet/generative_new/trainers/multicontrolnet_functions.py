""" Training functions for the different models. """
from collections import OrderedDict
from pathlib import Path


###Testing
import json
from generative.networks.nets import VQVAE, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
####

import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.losses.adversarial_loss import PatchAdversarialLoss
#from generative.inferers import ControlNetLatentDiffusionInferer
from src.inferers import ControlNetLatentDiffusionInferer, LatentDiffusionInferer
#from generative.inferers import LatentDiffusionInferer
#from pynvml.smi import nvidia_smi
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

'''
def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")'''




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
    torch.autograd.set_detect_anomaly(True)

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
            writer=writer_train,
            scaler=scaler,
            scale_factor=scale_factor,
        )

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
            #print_gpu_memory_report()

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


def train_epoch_controlnet(
    controlnet: nn.Module,
    diffusion: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler,
    scale_factor: float = 1.0,
    probability_include: float = 0.5  # Probability of including each condition
) -> None:
    controlnet.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        torch.cuda.empty_cache()
        images = x["image"].to(device)
        cond1 = x["artery"].to(device)#dtype=torch.float32
        cond2 = x["plaque"].to(device)
        cond3 = x["aorta"].to(device)
        cond4 = x["ventricle_l"].to(device)
        cond5 = x["ventricle_r"].to(device)
        cond6 = x["atrium_l"].to(device)
        cond7 = x["atrium_r"].to(device)
        cond8 = x["myocardium"].to(device)

        # Generate random masks for each condition
        mask_cond1 = torch.rand(1).to(device) < probability_include
        mask_cond2 = torch.rand(1).to(device) < probability_include
        mask_cond3 = torch.rand(1).to(device) < probability_include
        mask_cond4 = torch.rand(1).to(device) < probability_include
        mask_cond5 = torch.rand(1).to(device) < probability_include
        mask_cond6 = torch.rand(1).to(device) < probability_include
        mask_cond7 = torch.rand(1).to(device) < probability_include
        mask_cond8 = torch.rand(1).to(device) < probability_include

        # Apply masks to condition tensors
        cond1 = cond1 * mask_cond1
        cond2 = cond2 * mask_cond2
        cond3 = cond3 * mask_cond3
        cond4 = cond4 * mask_cond4
        cond5 = cond5 * mask_cond5
        cond6 = cond6 * mask_cond6
        cond7 = cond7 * mask_cond7
        cond8 = cond8 * mask_cond8


        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                e = stage1(images)
            
            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            # Concatenate the masked condition tensors
            cond = torch.cat([cond1, cond2, cond3, cond4, cond5,cond6,cond7,cond8], dim=1)
            #cond = torch.cat([cond1, cond2, cond3, cond4, cond5], dim=1)
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=noisy_e, timesteps=timesteps, controlnet_cond=cond
            )

            noise_pred = diffusion(
                x=noisy_e,
                timesteps=timesteps,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            )

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.l1_loss(noise_pred.float(), target.float())  # Using L1 loss instead of MSE

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})
        

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
    controlnet_latent_inferer = ControlNetLatentDiffusionInferer(scheduler)
    diffusion_inferer = LatentDiffusionInferer(scheduler,scale_factor=1.0)
    
    #controlnet_inferer = ControlNetLatentDiffusionInferer(scheduler)
    total_losses = OrderedDict()
    guidance_scale = 7.0
    #progress_bar = tqdm(enumerate(loader),total=len(loader),ncols=110,position=0,leave=True)
    torch.cuda.empty_cache()

    ###Testing
    '''
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8,
        out_channels=8,
        num_channels=(256, 512, 768),
        attention_levels=(False, True, True),
        num_res_blocks=2,
        num_head_channels=(0,512,768),
                #num_head_channels=256,
                #with_conditioning=True,
                #cross_attention_dim=512,
            )
    
    unet_checkpoint = torch.load("/mnt/sds/sd20i001/marvin/evaluation/VQGAN_Firas/3D_LDM_test2/checkpoint.pth")
    unet.load_state_dict(unet_checkpoint["model_state_dict"]) #change to model_state_dict for VQGAN
    unet = unet.to(device)
    unet.eval()'''

    #scheduler = DDPMScheduler(num_train_timesteps=1000,schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205)
    
    '''
    vqvae_checkpoint_path = Path("/mnt/sds/sd20i001/marvin/evaluation/VQGAN_Firas/vq_gan_equal_resize/checkpoint_800.pth")
    vqvae_config_path = vqvae_checkpoint_path.parent / "vqvae_config.json"
    with open(vqvae_config_path, "r") as f:
        vqvae_config = json.load(f)
    device = torch.device("cuda")
    vqvae = VQVAE(**vqvae_config)
    vqvae_checkpoint = torch.load(vqvae_checkpoint_path)
    vqvae.load_state_dict(vqvae_checkpoint["model_state_dict"]) #change to model_state_dict for VQGAN
    vqvae = vqvae.to(device)
    vqvae.eval()'''
    
    #scheduler.set_timesteps(num_inference_steps=1000)
    ####

    with torch.no_grad():
        for i, x in enumerate(loader):
            images = x["image"].to(device)
            #reports = x["report"].to(device)
            cond1 = x["artery"].to(device)#dtype=torch.float32
            cond2 = x["plaque"].to(device)
            cond3 = x["aorta"].to(device)
            cond4 = x["ventricle_l"].to(device)
            cond5 = x["ventricle_r"].to(device)
            cond6 = x["atrium_l"].to(device)
            cond7 = x["atrium_r"].to(device)
            cond8 = x["myocardium"].to(device)

            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

            with autocast(enabled=True):
                e = stage1(images)
                #e = stage1(images) * scale_factor
                #e = stage1().encode_stage_2_inputs(images)
                #prompt_embeds = text_encoder(reports.squeeze(1))
                #prompt_embeds = prompt_embeds[0]

                noise = torch.randn_like(e).to(device)
                '''
                noise_pred = controlnet_inferer(inputs = images, autoencoder_model=stage1,diffusion_model=diffusion,controlnet=controlnet,noise=e,timesteps=timesteps,
                                                cn_cond=cond)
                '''
                noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

                #cond = torch.cat([cond1, cond2, cond3, cond4, cond5], dim=1)
                cond = torch.cat([cond1, cond2, cond3, cond4, cond5,cond6,cond7,cond8], dim=1)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=noisy_e, timesteps=timesteps, controlnet_cond=cond
                ) #add context=prompt_embeds, for text embedding
            
                noise_pred = diffusion(
                    x=noisy_e,
                    timesteps=timesteps,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

                if scheduler.prediction_type == "v_prediction":
                    # Use v-prediction parameterization
                    target = scheduler.get_velocity(e, noise, timesteps)
                elif scheduler.prediction_type == "epsilon":
                    target = noise
                loss = F.l1_loss(noise_pred.float(), target.float())
                print(i)
            loss = loss.mean()
            losses = OrderedDict(loss=loss)
            print(i)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

            writer.add_scalar(tag="val_loss", scalar_value=total_losses["loss"], global_step=step)

            
            torch.cuda.empty_cache()

        for k in total_losses.keys():
            total_losses[k] /= len(loader.dataset)

        for k, v in total_losses.items():
            writer.add_scalar(f"{k}", v, step)

        torch.cuda.empty_cache()
        
    ###Removed for now:
    
    '''

    with torch.no_grad():
            noise = torch.randn((1, 8,48,48,32)).to(device) #change with other sizes
             #change here for DDIM
            stage1 = stage1.module if hasattr(stage1, "module") else stage1
            sample_uncond = diffusion_inferer.sample(input_noise=noise,autoencoder_model=stage1,diffusion_model=diffusion,scheduler=scheduler)[0,0].detach().cpu().numpy()
            fig_sample_uncond = plt.figure()
            sample_uncond_slice = sample_uncond[...,sample_uncond.shape[2] // 2]
            plt.imshow(sample_uncond_slice, cmap='gray')
            plt.title('Sample uncond slice')
            plt.axis('off')


    with torch.no_grad():
            #noise = torch.randn((1, 8,48 ,48, 32)).to(device)
            sample, intermediates = controlnet_latent_inferer.sample(
                    input_noise = noise,
                    autoencoder_model=stage1,
                    diffusion_model = diffusion,
                    controlnet = controlnet,
                    cn_cond = cond,
                    scheduler = scheduler,
                    save_intermediates=True,
                    intermediate_steps=100,
                    verbose=True,
            )

    print('Noise',noise.shape)
    print('Cond',cond.shape)
    print('Sample',sample.shape)
    print('Sample uncond',sample_uncond.shape)

    decoded_images = []

# Assuming intermediates is a list of images (tensors)
    for image in intermediates:
        with torch.no_grad():
            decoded_images.append(image)

        # Concatenate the images along the last dimension
    chain_slices = torch.cat([image[0, 0, :, :, image.shape[2] // 2].unsqueeze(0) for image in decoded_images], dim=-1).cpu().numpy()

    chain_slices = chain_slices.squeeze(-3)
        

    cond_slice = cond[0, 0, :, :, cond.shape[2]//2].detach().cpu().numpy()
    sample_slice = sample[0, 0, :, :, cond.shape[2]//2].detach().cpu().numpy()

    fig_cond = plt.figure()
    plt.imshow(cond_slice, cmap='gray')
    plt.title('Cond Slice')
    plt.axis('off')

    fig_sample = plt.figure()
    plt.imshow(sample_slice, cmap='gray')
    plt.title('Sample Slice')
    plt.axis('off')

    

    fig_chain = plt.figure()
    plt.imshow(chain_slices, cmap='gray')
    plt.title('Chain Slice')
    plt.axis('off')

        # Add the figures to TensorBoard
    writer.add_figure("Cond Slice", fig_cond, global_step=step)
    writer.add_figure("Sample Slice", fig_sample, global_step=step)
    writer.add_figure("Sample uncond slice", fig_sample_uncond,global_step=step)
    writer.add_figure("Decoded Images", fig_chain, global_step=step)

        # Close the matplotlib figures to free up resources
    plt.close(fig_cond)
    plt.close(fig_sample)

    '''

    return total_losses["loss"]
