""" Training script for VQ-VAE GAN. Mostly adapted from: https://github.com/Warvito/monai-vqvae-diffusion """
import argparse
import warnings
from pathlib import Path

import sys
import os
sys.path.append('/media/marvin/D/Marvin/MedLoRD/medlord_journal')

import torch
import torch.optim as optim

from generative.losses.perceptual import PerceptualLoss

from src.python.functions.networks.nets import VQVAE #change to generative_new?
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator #change to generative_new?

from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions import train_vqgan
from util import get_dataloader

from torch.optim.lr_scheduler import ExponentialLR


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location of model to resume.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--adv_start", type=int, default=25, help="Epoch when the adversarial training starts.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

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

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    print("Getting data...")
    cache_dir = output_dir / "cached_data_aekl"
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
        model_type="autoencoder_luna",
    )

    print("Creating model...")
    config = OmegaConf.load(args.config_file)
    model = VQVAE(**config["stage1"]["params"])
    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"])
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)

    model = model.to(device)
    perceptual_loss = perceptual_loss.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    optimizer_g = optim.AdamW(model.parameters(), lr=config["stage1"]["base_lr"])
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=config["stage1"]["disc_lr"])
    g_scheduler = ExponentialLR(optimizer_g, gamma=0.9999)
    d_scheduler = ExponentialLR(optimizer_d, gamma=0.9999)

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    def remove_module_prefix(state_dict):
        """Remove 'module.' prefix from checkpoint keys if present."""
        return {k.replace("module.", ""): v for k, v in state_dict.items()}

    def add_module_prefix(state_dict):
        """Add 'module.' prefix to checkpoint keys if needed."""
        return {f"module.{k}" if not k.startswith("module.") else k: v for k, v in state_dict.items()}
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model_state_dict = checkpoint["state_dict"]
        if isinstance(model, torch.nn.DataParallel):
            model_state_dict = add_module_prefix(model_state_dict)
        else:
            model_state_dict = remove_module_prefix(model_state_dict)
        model.load_state_dict(model_state_dict)
        discriminator_state_dict = checkpoint["discriminator"]
        if isinstance(discriminator, torch.nn.DataParallel):
            discriminator_state_dict = add_module_prefix(discriminator_state_dict)
        else:
            discriminator_state_dict = remove_module_prefix(discriminator_state_dict)
        discriminator.load_state_dict(discriminator_state_dict)
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        for g in optimizer_g.param_groups:
            g['lr'] = float(config["stage1"]["base_lr"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        for d in optimizer_d.param_groups:
            d['lr'] = float(config["stage1"]["disc_lr"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        if "ema_cluster_size" in checkpoint and "ema_w" in checkpoint:
            model.quantizer.quantizer.ema_cluster_size = checkpoint["ema_cluster_size"]
            model.quantizer.quantizer.ema_w = checkpoint["ema_w"]
            print("EMA parameters successfully restored!")
        else:
            print("Warning: EMA parameters not found in checkpoint. This may affect codebook usage.")
    else:
        print(f"No checkpoint found.")

    # Train model
    print(f"Starting Training")
    val_loss = train_vqgan(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        g_scheduler=g_scheduler,
        d_scheduler=d_scheduler,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        adv_weight=config["stage1"]["adv_weight"],
        perceptual_weight=config["stage1"]["perceptual_weight"],
        adv_start=args.adv_start,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
