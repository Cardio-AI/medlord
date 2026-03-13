"""
Training script for ControlNet in the latent space of the trained LDM.
"""

import argparse
from pathlib import Path
import sys
sys.path.append("/home/marvin/repositories/medlord")

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
import os
from omegaconf import OmegaConf

from src.models.vqvae import VQVAE
from src.models.diffusion_unet import DiffusionModelUNet
from src.models.controlnet import ControlNet
from src.models.ddpmscheduler import DDPMScheduler

from src.training.controlnet_trainer import ControlNetTrainer
from src.data.dataloading import get_controlnet_dataloader


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_vqvae", type=str, required=False)

    parser.add_argument("--vqvae_ckpt", type=str, required=False)
    parser.add_argument("--ldm_ckpt", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--training_ids", type=str, required=True)
    parser.add_argument("--validation_ids", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # -----------------------
    # DDP setup
    # -----------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.set_float32_matmul_precision("high")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        is_main = rank == 0
    else:
        rank = 0
        is_main = True

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Reproducibility
    # -----------------------
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # -----------------------
    # Config 
    # -----------------------
    config = OmegaConf.load(args.config)

    run_dir = Path(args.output_dir) / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    writer_train = SummaryWriter(run_dir / "logs" / "train") if is_main else None
    writer_val = SummaryWriter(run_dir / "logs" / "val") if is_main else None

    # -----------------------
    # Data
    # -----------------------

    use_precomputed_latents = config.training.get("use_precomputed_latents", False)

    train_loader, val_loader = get_controlnet_dataloader(
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        condition_keys = list(config.controlnet.condition_keys),
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        roi_size=tuple(config.training.roi_size),
        use_precomputed_latents=use_precomputed_latents,
        preload_latents=config.training.cache, #enable to preload into memory
        use_persistent=config.training.persistent #if online encoding, it is recommended to store data in cache_dir and enable
    )

    # -----------------------
    # Stage 1 (VQ-VAE encoder)
    # -----------------------

    if not use_precomputed_latents:

        if is_main:
            print(f"Loading VQ-VAE from {args.vqvae_ckpt}")

        config_vqvae = OmegaConf.load(args.config_vqvae)

        stage1 = VQVAE(**config_vqvae.stage1.params)

        checkpoint = torch.load(args.vqvae_ckpt, map_location="cpu")
        stage1.load_state_dict(checkpoint["state_dict"], strict=False)

        stage1.eval()
        stage1.requires_grad_(False)
        stage1 = stage1.to(device)

    else:

        if is_main:
            print("Using precomputed latents — skipping VQ-VAE initialization.")

        stage1 = None

    # -----------------------
    # Diffusion model (frozen)
    # -----------------------

    diffusion = DiffusionModelUNet(**config.pretrained_model.params)

    if is_main:
        print(f"Loading pretrained LDM from {args.ldm_ckpt}")

    checkpoint = torch.load(args.ldm_ckpt, map_location="cpu")
    diffusion.load_state_dict(checkpoint["model"])

    diffusion.requires_grad_(False)
    diffusion.eval()
    diffusion = diffusion.to(device)

    # -----------------------
    # ControlNet
    # -----------------------

    controlnet = ControlNet(**config.controlnet.params).to(device)

    if world_size > 1:
        controlnet = DDP(
            controlnet,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # -----------------------
    # Diffusion scheduler
    # -----------------------

    scheduler = DDPMScheduler(**config.ldm.scheduler)

    # -----------------------
    # Optimizer + LR scheduler
    # -----------------------

    optimizer = optim.AdamW(controlnet.parameters(), lr=config.training.lr)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.training.lr_gamma,
    )

    # -----------------------
    # Resume checkpoint
    # -----------------------

    checkpoint_path = run_dir / "last_checkpoint.pth"

    start_epoch = 0
    best_loss = float("inf")
    checkpoint = None

    if checkpoint_path.exists():

        if is_main:
            print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        controlnet.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if checkpoint.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))

        if world_size > 1:
            dist.barrier()

    # -----------------------
    # Trainer
    # -----------------------

    trainer = ControlNetTrainer(
        controlnet=controlnet,
        diffusion=diffusion,
        stage1=stage1,
        scheduler=scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        config=config,
        writer_train=writer_train,
        writer_val=writer_val,
        is_main=is_main,
        start_epoch=start_epoch,
        best_loss=best_loss,
    )

    trainer.train()

    if is_main:
        writer_train.close()
        writer_val.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()