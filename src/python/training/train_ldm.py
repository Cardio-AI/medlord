""" Training script for the diffusion model in the latent space of the pretraine AEKL model. """
import argparse
import warnings
from pathlib import Path

import sys
import os
sys.path.append('/media/marvin/D/Marvin/MedLoRD/medlord_journal')

import ast

#import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from src.python.functions.networks.nets import DiffusionModelUNet, VQVAE
from src.python.functions.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from src.python.training.training_functions import train_ldm
from transformers import CLIPTextModel
from src.python.training.util import get_dataloader


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--cache_dir", type=str,help="Output directory.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--config_vqvae", help="Location of file with validation ids.")
    parser.add_argument("--vqvae_ckpt", help="Checkppoint of VQGAN.")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Path readable by load_model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=2, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--image_roi",
        default=None,
        help="Specify central ROI crop of inputs, as a tuple, with -1 to not crop a dimension.",
        type=ast.literal_eval,
    )

    args = parser.parse_args()
    return args


class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model as a workaround for the DataParallel usage in the training loop."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        z = self.model.encode_stage_2_inputs(x)
        
        return z

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor: #was named 'forward' before
        z = self.model.encode_stage_2_inputs(x)
        
        return z
    def decode_stage_2_outputs(self,x: torch.Tensor) -> torch.Tensor:

        z = self.model.decode_stage_2_outputs(x)

        return z
    

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
    cache_dir = output_dir / str(args.cache_dir) 
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        image_roi=args.image_roi,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
        model_type="presaved",
    )

    # Load Autoencoder to produce the latent representations
    print(f"Loading Stage 1 from {args.vqvae_ckpt}")
    vqvae_checkpoint_path = Path(args.vqvae_ckpt)
    config = OmegaConf.load(args.config_vqvae)
    stage1 = VQVAE(**config["stage1"]["params"])
    vqvae_checkpoint = torch.load(vqvae_checkpoint_path,map_location="cpu")
    stage1.load_state_dict(vqvae_checkpoint["state_dict"])

    stage1 = Stage1Wrapper(model=stage1)
    stage1.eval()
    print("Loaded vqvae model with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create the diffusion model
    print("Creating model...")
    config = OmegaConf.load(args.config_file)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        stage1 = torch.nn.DataParallel(stage1)
        diffusion = torch.nn.DataParallel(diffusion)
    
    stage1 = stage1.to(device)
    diffusion = diffusion.to(device)
   
    optimizer = optim.AdamW(diffusion.parameters(), lr=config["ldm"]["base_lr"]) 
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9999)

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0

    #Check if latents are presaved
    presaved_latents = True
    if presaved_latents:
        del stage1
        torch.cuda.empty_cache()
        stage1 = None

    if resume:
        print(f"Using checkpoint!")
        with torch.no_grad():  # Ensure no gradients are computed during checkpoint loading
            checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
            diffusion.load_state_dict(checkpoint["diffusion"])

            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
        
            del checkpoint
            torch.cuda.empty_cache()
    else:
        print(f"No checkpoint found.")
    
    # Train model

    print(f"Starting Training")
    val_loss = train_ldm(
        model=diffusion,
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
