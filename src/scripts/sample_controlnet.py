"""
Conditional sampling from a trained ControlNet + LDM.
A CSV file provides the mask paths (one row per subject). The same spatial
transforms used during encoding are applied, then the masks are passed as
conditioning to the ControlNet at every denoising step. One output volume is
generated per CSV row.
"""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
from torch import amp
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
)

from src.models.vqvae import VQVAE
from src.models.diffusion_unet import DiffusionModelUNet
from src.models.controlnet import ControlNet
from src.models.ddimscheduler import DDIMScheduler
from src.models.ddpmscheduler import DDPMScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional sampling from a trained ControlNet + LDM")

    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage1_cfg", type=str, required=True)
    parser.add_argument("--diff_ckpt", type=str, required=True)
    parser.add_argument("--diff_cfg", type=str, required=True)
    parser.add_argument("--controlnet_ckpt", type=str, required=True)
    parser.add_argument("--controlnet_cfg", type=str, required=True)

    parser.add_argument("--csv", type=str, required=True,
                        help="CSV file with mask paths, one row per subject. "
                             "Columns must include all condition keys. "
                             "An optional 'image' column is used for output file naming.")
    parser.add_argument("--condition_keys", nargs="+", required=True,
                        help="Column names in the CSV corresponding to the mask conditions")

    parser.add_argument("--output_dir", type=str, default="samples_cond")
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--conditioning_scale", type=float, default=1.0,
                        help="ControlNet conditioning scale")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--latent_shape", type=int, nargs=3, required=True,
                        metavar=("D", "H", "W"), help="Latent spatial dimensions e.g. 128 128 64")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[512, 512, 256],
                        metavar=("H", "W", "D"), help="Spatial crop/pad size matching training")
    parser.add_argument("--reference_nii", type=str, default=None,
                        help="Reference .nii.gz to copy affine from")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_stage1(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = VQVAE(**cfg.model.params)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    skipped = [k for k in state_dict if k not in model_keys]
    if skipped:
        print(f"Warning: skipped {len(skipped)} stage1 keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    model.load_state_dict(filtered, strict=False)

    return model.to(device).eval().requires_grad_(False)


def load_diffusion(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = DiffusionModelUNet(**cfg.model.params)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("ema", ckpt))
    model.load_state_dict(state_dict)

    return model.to(device).eval().requires_grad_(False), cfg


def load_controlnet(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = ControlNet(**cfg.controlnet.params)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Prefer EMA shadow weights if available
    ema_state = ckpt.get("ema")
    if ema_state is not None:
        shadow = ema_state["shadow"]
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name])
        print("Loaded ControlNet EMA weights.")
    else:
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict)
        print("Loaded ControlNet model weights (no EMA found).")

    return model.to(device).eval().requires_grad_(False)


def load_masks(mask_paths_by_key, roi_size, device):
    """
    Load and preprocess NIfTI masks with the same spatial transforms used during encoding.
    mask_paths_by_key: dict mapping condition_key -> path string, or None if mask is missing.
    Missing masks are substituted with a zero tensor of the correct spatial shape.
    Returns a tensor of shape [1, n_masks, H, W, D].
    """
    present_keys = [k for k, p in mask_paths_by_key.items() if p is not None]
    missing_keys = [k for k, p in mask_paths_by_key.items() if p is None]

    if missing_keys:
        print(f"  Warning: missing masks for {missing_keys}, substituting zeros.")

    masks_out = {}

    if present_keys:
        transforms = Compose([
            LoadImaged(keys=present_keys),
            EnsureChannelFirstd(keys=present_keys),
            CenterSpatialCropd(keys=present_keys, roi_size=roi_size),
            SpatialPadd(keys=present_keys, spatial_size=roi_size),
            ToTensord(keys=present_keys),
        ])
        data = transforms({k: mask_paths_by_key[k] for k in present_keys})
        for k in present_keys:
            masks_out[k] = data[k]  # [1, H, W, D]

    # Infer spatial shape from a loaded mask, or fall back to roi_size
    spatial_shape = next(iter(masks_out.values())).shape[1:] if masks_out else tuple(roi_size)
    for k in missing_keys:
        masks_out[k] = torch.zeros(1, *spatial_shape)

    # Preserve original key order, cat along channel dim, add batch dim
    ordered = [masks_out[k] for k in mask_paths_by_key]
    cond = torch.cat(ordered, dim=0).unsqueeze(0)  # [1, n_masks, H, W, D]

    return cond.to(device)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load affine from reference image if provided
    affine = None
    if args.reference_nii is not None:
        affine = nib.load(args.reference_nii).affine
        print(f"Using affine from {args.reference_nii}")

    # Load models
    print("Loading VQ-GAN...")
    stage1 = load_stage1(args.stage1_cfg, args.stage1_ckpt, device)

    print("Loading diffusion model...")
    unet, diff_cfg = load_diffusion(args.diff_cfg, args.diff_ckpt, device)

    print("Loading ControlNet...")
    controlnet = load_controlnet(args.controlnet_cfg, args.controlnet_ckpt, device)

    # Scheduler
    if args.scheduler == "ddpm":
        scheduler = DDPMScheduler(**diff_cfg.scheduler)
    else:
        scheduler = DDIMScheduler(**diff_cfg.scheduler)
    scheduler.set_timesteps(args.timesteps)

    scale_factor = args.scale_factor
    latent_shape = tuple(args.latent_shape)
    in_channels = diff_cfg.model.params.in_channels

    df = pd.read_csv(args.csv)
    print(f"Found {len(df)} subject(s) in {args.csv}")
    print(f"Sampling with {args.scheduler.upper()} ({args.timesteps} steps)...")

    for idx, row in df.iterrows():
        mask_paths_by_key = {
            key: str(row[key]) if pd.notna(row[key]) else None
            for key in args.condition_keys
        }

        cond = load_masks(mask_paths_by_key, args.roi_size, device)

        x = torch.randn((1, in_channels, *latent_shape), device=device)

        with torch.no_grad(), amp.autocast(device_type=device.type):
            for t in scheduler.timesteps:
                t_batch = torch.tensor([t], device=device)

                down_res, mid_res = controlnet(
                    x=x,
                    timesteps=t_batch,
                    controlnet_cond=cond,
                    conditioning_scale=args.conditioning_scale,
                )

                noise_pred = unet(
                    x=x,
                    timesteps=t_batch,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                )

                x, _ = scheduler.step(noise_pred, t, x)

            x = x / scale_factor
            recon = stage1.decode_stage_2_outputs(x)

        # Convert to HU and save
        vol = np.clip(recon[0, 0].float().cpu().numpy(), -1.0, 1.0)
        hu = (((vol + 1.0) * (2000.0 / 2.0)) - 1000.0).astype(np.int16)

        stem = Path(row["image"]).stem if "image" in row else f"{idx:03d}"
        out_path = out_dir / f"{stem}_cond.nii.gz"
        nib.save(nib.Nifti1Image(hu, affine), out_path)
        print(f"  [{idx+1}/{len(df)}] Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
