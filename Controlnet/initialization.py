# initialization.py

import torch
from pathlib import Path
from omegaconf import OmegaConf
from torch import optim
import torch.nn as nn
from generative_new.networks.nets import VQVAE, DiffusionModelUNet, ControlNet
from generative_new.networks.schedulers import DDPMScheduler


class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model as a workaround for the DataParallel usage in the training loop."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        z = self.model.encode_stage_2_inputs(x)
        
        return z

def load_vqvae_model(config_path, checkpoint_path):
    print(f"Loading Stage 1 from {checkpoint_path}")
    config = OmegaConf.load(config_path)
    vqvae = VQVAE(**config["stage1"]["params"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vqvae.load_state_dict(checkpoint["state_dict"])
    vqvae = Stage1Wrapper(model=vqvae)
    vqvae.eval()
    print("Loaded VQ-VAE model with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    return vqvae, config["stage1"]["params"]["embedding_dim"]


def load_diffusion_model(checkpoint_path, ddpm_channels, spatial_dims=3):
    print(f"Loading Diffusion from {checkpoint_path}")
    diffusion = DiffusionModelUNet(
        spatial_dims=spatial_dims,
        in_channels=ddpm_channels,
        out_channels=ddpm_channels,
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        num_head_channels=(0, 0, 32, 32),
        use_flash_attention=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    diffusion.load_state_dict(checkpoint)
    diffusion.eval()
    return diffusion


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def initialize_models_and_optim(args, resume=False, run_dir=None):
    # Load Stage 1 (VQ-VAE)
    vqvae_model, ddpm_channels = load_vqvae_model(
        config_path="/mnt/sds-hd/sd20i001/marvin/cfgs/stage1/ldm_project/vqgan_ds4.yaml",
        checkpoint_path=Path(args.vqvae_ckpt),
    )


    # Load Diffusion Model
    diffusion_model = load_diffusion_model(
        checkpoint_path=Path(args.diffusion_ckpt),
        ddpm_channels=ddpm_channels,
        spatial_dims=args.spatial_dimension,
    )

    # Load ControlNet from config
    config = OmegaConf.load(args.config_file)
    controlnet = ControlNet(**config["controlnet"].get("params", {}))

    # Freeze diffusion model
    freeze_model(diffusion_model)

    # Scheduler
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", {}))

    # Device setup
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        vqvae_model = torch.nn.DataParallel(vqvae_model)
        diffusion_model = torch.nn.DataParallel(diffusion_model)
        controlnet = torch.nn.DataParallel(controlnet)

    vqvae_model = vqvae_model.to(device)
    diffusion_model = diffusion_model.to(device)
    controlnet = controlnet.to(device)

    # Optimizer and LR scheduler
    optimizer = optim.AdamW(controlnet.parameters(), lr=config["controlnet"]["base_lr"])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    # Resume logic
    best_loss = float("inf")
    start_epoch = 0
    if resume and run_dir and (run_dir / "checkpoint.pth").exists():
        print("Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        controlnet.load_state_dict(checkpoint["controlnet"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = float(config["controlnet"]["base_lr"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print("No checkpoint found or resume not enabled.")

    return {
        "vqvae": vqvae_model,
        "diffusion": diffusion_model,
        "controlnet": controlnet,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "scheduler": scheduler,
        "start_epoch": start_epoch,
        "best_loss": best_loss,
        "config": config,
        "device": device,
    }
