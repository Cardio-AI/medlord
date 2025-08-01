""" Script to generate sample images from the diffusion model.

In the generation of the images, the script is using a DDIM scheduler.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.cuda.amp import autocast
from generative_new.networks.nets import VQVAE, DiffusionModelUNet, ControlNet
from generative_new.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.config import print_config
from omegaconf import OmegaConf

from tqdm import tqdm



import nibabel as nib
import os
import numpy as np


from dataloader_controlnet import get_dataloaders
from initialization import initialize_models_and_optim

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.")
    parser.add_argument("--vqvae_ckpt", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_ckpt", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--controlnet_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--reference_path", help="Path to reference image to get affine.")
    parser.add_argument("--stage1_config_file_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_config_file_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--controlnet_config_file_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--training_ids", help="Location of file with test ids.")
    parser.add_argument("--validation_ids", help="Location of file with test ids.")
    parser.add_argument("--controlnet_scale", type=float, default=1.0, help="")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="")
    parser.add_argument("--x_size", type=int, default=64, help="Latent space x size.")
    parser.add_argument("--y_size", type=int, default=64, help="Latent space y size.")
    parser.add_argument("--z_size", type=int, default=64, help="Latent space z size.")
    parser.add_argument("--num_workers", type=int, help="")
    parser.add_argument("--batch_size", type=int, help="")
    parser.add_argument("--seed", type=int,default=42, help="")
    parser.add_argument("--epoch", type=int, help="")
    parser.add_argument("--cache_data", type=int, default=0, help="")
    parser.add_argument("--scale_factor", type=float, help="Latent space y size.")
    parser.add_argument("--num_inference_steps", type=int, help="")

    args = parser.parse_args()
    return args


def main(args,controlnet_ckpt,epoch):
    print_config()
    #set_determinism(seed=42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / args.run_dir
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")


    print("Getting data...")
    train_loader, test_loader = get_dataloaders(
        train_ids_path="data/ids_full/controlnet_train.csv",
        val_ids_path="data/ids_full/controlnet_test.csv",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        luna=True 
    )

    print(f"Loading Models...")
    models = initialize_models_and_optim(args, resume=resume, run_dir=run_dir)

    stage1 = models["vqvae"]
    diffusion = models["diffusion"]
    controlnet = models["controlnet"]
    optimizer = models["optimizer"]
    lr_scheduler = models["lr_scheduler"]
    scheduler = models["scheduler"]
    start_epoch = models["start_epoch"]
    best_loss = models["best_loss"]
    device = models["device"]
    checkpoint = torch.load(controlnet_ckpt)

    scheduler.set_timesteps(args.num_inference_steps)


    num_crops = 1
    pad = False
    reference_path = "path/reference_image"
    reference_image = nib.load(reference_path)

    for iteration in range(num_crops):

        epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        #iteration_dir = os.path.join(epoch_dir, f"iteration_{iteration}")
        iteration_dir = epoch_dir
        os.makedirs(iteration_dir, exist_ok=True)

        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, x in pbar:
            torch.cuda.empty_cache()
            
            filename = os.path.basename(os.path.split(reference_path)[0])
            print(filename)
            with torch.no_grad():  
                cond1 = x["aorta"].as_tensor().to(torch.float32)
                cond2 = x["artery"].as_tensor().to(torch.float32)
                cond3 = x["plaque"].as_tensor().to(torch.float32)   # Keep these on CPU first
                cond4 = x["ventricle_l"].as_tensor().to(torch.float32) 
                cond5 = x["ventricle_r"].as_tensor().to(torch.float32) 
                cond6 = x["atrium_l"].as_tensor().to(torch.float32) 
                cond7 = x["atrium_r"].as_tensor().to(torch.float32) 
                cond8 = x["myocardium"].as_tensor().to(torch.float32) 
                cond9 = x["pulmonary"].as_tensor().to(torch.float32) 
            cond = torch.cat([cond1,cond2,cond3, cond4, cond5, cond6, cond7, cond8,cond9], dim=1).to(device)

            h,w,d = cond2.shape[-3], cond2.shape[-2], cond2.shape[-1]
            noise = torch.randn((1, 9, h//4,w//4,d//4)).to(device)

            with torch.no_grad():
                with autocast(enabled=True):
                    progress_bar = tqdm(scheduler.timesteps)
                    for t in progress_bar:
                        noise_input = noise

                        down_block_res_samples, mid_block_res_sample = controlnet(
                            x=noise_input,
                            timesteps=torch.Tensor((t,)).to(noise.device).long(),
                            controlnet_cond=cond,
                            conditioning_scale=1.0,
                        )

                        model_output = diffusion(
                            x=noise_input,
                            timesteps=torch.Tensor((t,)).to(noise.device).long(),
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        )

                        noise_pred = model_output

                        noise, _ = scheduler.step(noise_pred, t, noise)

            torch.cuda.empty_cache()
            with torch.no_grad():
                sample = stage1.decode_stage_2_outputs(noise / args.scale_factor)

            

            sample = np.clip(sample.cpu().numpy(),-1,1)
            sample = (((sample +1) * ((3000)/2)) -1000).astype(np.int16)
          

            mask_dir = os.path.join(iteration_dir,'masks')
            image_dir = os.path.join(iteration_dir,'images')
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"image_{filename}")
            mask1_path = os.path.join(mask_dir, f"aorta_{filename}")
            mask2_path = os.path.join(mask_dir, f"artery_{filename}")
            mask3_path = os.path.join(mask_dir, f"plaque_{filename}")
            mask4_path = os.path.join(mask_dir, f"ventricle_l_{filename}")
            mask5_path = os.path.join(mask_dir, f"ventricle_r_{filename}")
            mask6_path = os.path.join(mask_dir, f"atrium_l_{filename}")
            mask7_path = os.path.join(mask_dir, f"atrium_r_{filename}")
            mask8_path = os.path.join(mask_dir, f"myocardium_{filename}")
            mask9_path = os.path.join(mask_dir, f"pulmonary_{filename}")
  
            nib.save(nib.Nifti1Image(sample.squeeze(), affine=reference_image.affine, header=reference_image.header), image_path)#.squeeze()
            print(f"File saved at: {image_path}")
            nib.save(nib.Nifti1Image(cond1.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask1_path)

            nib.save(nib.Nifti1Image(cond2.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask2_path)
            nib.save(nib.Nifti1Image(cond3.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask3_path)
            nib.save(nib.Nifti1Image(cond4.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask4_path)
            nib.save(nib.Nifti1Image(cond5.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask5_path)
            nib.save(nib.Nifti1Image(cond6.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask6_path)
            nib.save(nib.Nifti1Image(cond7.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask7_path)
            nib.save(nib.Nifti1Image(cond8.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask8_path)
            nib.save(nib.Nifti1Image(cond9.cpu().numpy().squeeze(), affine=reference_image.affine, header=reference_image.header), mask9_path)
            del sample
            del noise
            del cond
            torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args() 
    for epoch in range(10,1001,10):
        checkpoint_dir = f"/checkpoint_{epoch}.pth"
        main(args,checkpoint_dir,epoch) 
