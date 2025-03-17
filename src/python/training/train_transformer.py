""" Training script for the diffusion model in the latent space of the pretraine AEKL model. """
import argparse
import warnings
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from generative.networks.nets import DecoderOnlyTransformer, VQVAE
from generative.utils.enums import OrderingType
from generative.utils.ordering import Ordering
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions import train_transformer
from util import get_dataloader
from torch.optim.lr_scheduler import ExponentialLR
from generative.inferers import VQVAETransformerInferer


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--stage1_config", help="Path readable by load_model.")
    parser.add_argument("--stage1_ckpt", help="Path readable by load_model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")


    args = parser.parse_args()
    return args


class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model as a workaround for the DataParallel usage in the training loop."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.model.index_quantize(x)
        return e


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path("")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / args.run_dir
    if run_dir.exists() and (run_dir / "checkpoint_epoch_525.pth").exists():
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
    cache_dir = output_dir / "cached_data_diffusion"
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
        model_type="transformer",
    )


    print(f"Loading Stage 1 from {args.stage1_ckpt}")
    vqvae_checkpoint_path = Path(args.stage1_ckpt)
    vqvae_config = OmegaConf.load(args.stage1_config)
    stage1 = VQVAE(**vqvae_config["stage1"]["params"])
    vqvae_checkpoint = torch.load(vqvae_checkpoint_path,map_location="cpu")
    stage1.load_state_dict(vqvae_checkpoint["state_dict"]) #changed model_state_dict

    stage1 = Stage1Wrapper(model=stage1)
    stage1.eval()
    print("Loaded vqvae model with config:")
    for k, v in vqvae_config.items():
        print(f"  {k}: {v}")
    # Create the diffusion model
    print("Creating model...")
    config = OmegaConf.load(args.config_file)
    transformer = DecoderOnlyTransformer(**config["transformer"].get("params", dict()))

    ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=3, dimensions=(1,28,28,16))

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        stage1 = torch.nn.DataParallel(stage1)
        transformer = torch.nn.DataParallel(transformer)

    stage1 = stage1.to(device)
    transformer = transformer.to(device)

    optimizer = optim.AdamW(transformer.parameters(), lr=config["transformer"]["base_lr"])
    scheduler = ExponentialLR(optimizer, gamma=0.9999)

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint_epoch_525.pth"),map_location="cpu")
        '''if 'module.' in list(checkpoint["transformer"].keys())[0]:
            checkpoint["transformer"] = {k.replace('module.', ''): v for k, v in checkpoint["transformer"].items()}'''
        transformer.load_state_dict(checkpoint["transformer"])
        # Issue loading optimizer https://github.com/pytorch/pytorch/issues/2830
        transformer = transformer.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g['lr'] = float(config["transformer"]["base_lr"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print(f"No checkpoint found.")
    
    # Train model
    print(f"Starting Training")
  

    inferer = VQVAETransformerInferer()
    val_loss = train_transformer(
        model=transformer,
        ordering=ordering,
        stage1=stage1,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        inferer=inferer,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
    )



if __name__ == "__main__":
    args = parse_args()
    main(args)
