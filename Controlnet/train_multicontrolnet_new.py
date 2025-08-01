""" Training script for the controlnet model in the latent space of the pretraine AEKL model. """
import argparse
import warnings
from pathlib import Path

import pandas as pd

import torch

import torch.optim as optim

from generative_new.networks.nets import VQVAE, DiffusionModelUNet, ControlNet
from generative_new.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from src.trainers.multicontrolnet_functions import train_controlnet
from src.data.get_loader import get_training_data_loader

from sklearn.model_selection import train_test_split
import numpy as np

import scipy.ndimage
from dataloader_controlnet import get_dataloaders
from initialization import initialize_models_and_optim


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--vqvae_ckpt", help="Path readable by load_model.")
    parser.add_argument("--diffusion_ckpt", help="Path readable by load_model.")
    parser.add_argument("--spatial_dimension",type=int, help="Spatial dimension of the model")
    parser.add_argument("--scale_factor", type=float, help="Path readable by load_model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--cache_data", type=int, default=0, help="Cache data, True or False")
    parser.add_argument("--experiment", help="Mlflow experiment name.")

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
    train_loader, val_loader = get_dataloaders(
        train_ids_path="data/ids_full/controlnet_train.csv",
        val_ids_path="data/ids_full/controlnet_validation.csv",
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

    # Train model
    print(f"Starting Training")
    val_loss = train_controlnet(
        controlnet=controlnet,
        diffusion=diffusion,
        stage1=stage1,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        scale_factor=args.scale_factor,
        lr_scheduler = lr_scheduler,
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
