"""
Unconditional sampling from a trained Latent Diffusion Model.
"""

import argparse
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.multiprocessing as mp
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
from torch import amp

from src.models.diffusion_unet import DiffusionModelUNet
from src.models.vqvae import VQVAE
from src.models.ddimscheduler import DDIMScheduler
from src.models.ddpmscheduler import DDPMScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained LDM")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage1_cfg", type=str, required=True)
    parser.add_argument("--diff_cfg", type=str, required=True)
    # Single checkpoint (original behaviour)
    parser.add_argument("--diff_ckpt", type=str, default=None,
                        help="Path to a single diffusion checkpoint")
    # Epoch-range mode: point to the run directory containing checkpoint_epoch_N.pth files
    parser.add_argument("--diff_run_dir", type=str, default=None,
                        help="Directory containing checkpoint_epoch_N.pth files")
    parser.add_argument("--epoch_start", type=int, default=None,
                        help="First epoch to sample (inclusive)")
    parser.add_argument("--epoch_end", type=int, default=None,
                        help="Last epoch to sample (inclusive)")
    parser.add_argument("--epoch_step", type=int, default=100,
                        help="Step between epochs (default: 100)")
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--reference_nii", type=str, default=None,
                        help="Reference .nii.gz to copy affine from")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--latent_shape", type=int, nargs=3, required=True,
                        metavar=("D", "H", "W"), help="Latent spatial dimensions e.g. 128 128 64")
    parser.add_argument("--unet_batch_size", type=int, default=1,
                        help="Number of latents to denoise in parallel before decoding one-by-one. "
                             "Increase if GPU utilisation during denoising is below ~90%%.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Validate: must specify either --diff_ckpt or --diff_run_dir + epoch range
    if args.diff_ckpt is None and args.diff_run_dir is None:
        parser.error("Specify either --diff_ckpt or --diff_run_dir with --epoch_start/--epoch_end")
    if args.diff_run_dir is not None and (args.epoch_start is None or args.epoch_end is None):
        parser.error("--diff_run_dir requires --epoch_start and --epoch_end")

    return args


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
    ema = ckpt.get("ema")
    if ema is not None:
        state_dict = ema["shadow"]
    else:
        state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)

    return model.to(device).eval().requires_grad_(False), cfg


def _run_diffusion(unet, stage1, scheduler, diff_cfg, indices, out_dir,
                   affine, scale_factor, latent_shape, device, gpu_id=None,
                   unet_batch_size=1):
    """
    Denoise in chunks of unet_batch_size (better GPU utilisation), then decode
    one-by-one with empty_cache() between to prevent the decoder's peak allocation
    from blocking the next UNet forward pass.
    """
    in_channels = diff_cfg.model.params.in_channels
    tag = f"[GPU {gpu_id}] " if gpu_id is not None else ""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process indices in chunks so we denoise unet_batch_size latents at once
    for chunk_start in range(0, len(indices), unet_batch_size):
        chunk = indices[chunk_start : chunk_start + unet_batch_size]
        batch = len(chunk)

        # ── Denoise entire chunk in one batched forward pass ──────────────────
        x = torch.randn((batch, in_channels, *latent_shape), device=device)
        with torch.no_grad(), amp.autocast(device_type=device.type):
            for t in scheduler.timesteps:
                noise_pred = unet(x=x, timesteps=torch.full((batch,), t, device=device, dtype=torch.long))
                x, _ = scheduler.step(noise_pred, t, x)
            x = x / scale_factor

        latents_cpu = x.float().cpu()
        del x
        torch.cuda.empty_cache()

        # ── Decode one-by-one, clearing cache after each ──────────────────────
        for b, i in enumerate(chunk):
            latent = latents_cpu[b : b + 1].to(device)
            with torch.no_grad(), amp.autocast(device_type=device.type):
                recon = stage1.decode_stage_2_outputs(latent)
            del latent

            vol = np.clip(recon[0, 0].float().cpu().numpy(), -1.0, 1.0)
            hu = (((vol + 1.0) * (2000.0 / 2.0)) - 1000.0).astype(np.int16)
            out_path = out_dir / f"sample_{i:03d}.nii.gz"
            nib.save(nib.Nifti1Image(hu, affine), out_path)
            print(f"    {tag}Saved {out_path}", flush=True)

            del recon
            torch.cuda.empty_cache()


def _persistent_worker(rank, gpu_id, stage1_cfg, stage1_ckpt, diff_cfg_path,
                        job_queue, done_queue, scale_factor, latent_shape, unet_batch_size):
    """
    Long-lived worker process. Loads stage1 once and keeps the UNet architecture
    resident, hot-swapping only the weights for each new epoch checkpoint.
    """
    # Must be set before CUDA initialises to avoid allocator fragmentation OOM
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    device = torch.device(f"cuda:{gpu_id}")

    stage1 = load_stage1(stage1_cfg, stage1_ckpt, device)

    # Build UNet architecture once — only weights are swapped per epoch
    diff_cfg = OmegaConf.load(diff_cfg_path)
    unet = DiffusionModelUNet(**diff_cfg.model.params).to(device).eval().requires_grad_(False)

    while True:
        job = job_queue.get()
        if job is None:  
            break

        ckpt_path, out_dir_str, indices, affine, scheduler_name, timesteps = job

        # Hot-swap weights only (no architecture rebuild, no stage1 reload)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ema = ckpt.get("ema")
        state_dict = ema["shadow"] if ema is not None else ckpt.get("model", ckpt)
        unet.load_state_dict(state_dict)

        if scheduler_name == "ddpm":
            scheduler = DDPMScheduler(**diff_cfg.scheduler)
        else:
            scheduler = DDIMScheduler(**diff_cfg.scheduler)
        scheduler.set_timesteps(timesteps)

        _run_diffusion(unet, stage1, scheduler, diff_cfg, indices, Path(out_dir_str),
                       affine, scale_factor, tuple(latent_shape), device, gpu_id=gpu_id,
                       unet_batch_size=unet_batch_size)

        done_queue.put(rank)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()
    use_multi_gpu = n_gpus > 1
    print(f"Detected {n_gpus} GPU(s) — {'multi-GPU persistent workers' if use_multi_gpu else 'single-GPU'}.")

    # Load affine from reference image if provided
    affine = None
    if args.reference_nii is not None:
        ref = nib.load(args.reference_nii)
        affine = ref.affine
        print(f"Using affine from {args.reference_nii}")

    # Build list of (epoch_label, ckpt_path, output_subdir) jobs
    if args.diff_run_dir is not None:
        run_dir = Path(args.diff_run_dir)
        jobs = []
        for epoch in range(args.epoch_start, args.epoch_end + 1, args.epoch_step):
            ckpt = run_dir / f"checkpoint_epoch_{epoch-1}.pth"
            if not ckpt.exists():
                print(f"Warning: checkpoint not found, skipping: {ckpt}")
                continue
            jobs.append((epoch, ckpt, Path(args.output_dir) / f"epoch_{epoch}"))
    else:
        jobs = [(None, Path(args.diff_ckpt), Path(args.output_dir))]

    # ── Multi-GPU path: persistent workers ────────────────────────────────────
    if use_multi_gpu:
        nprocs = min(n_gpus, args.n_samples)
        indices_per_rank = [list(range(i, args.n_samples, nprocs)) for i in range(nprocs)]
        # Only spawn as many workers as needed (in case n_samples < n_gpus)
        active_ranks = [r for r in range(nprocs) if indices_per_rank[r]]

        ctx = mp.get_context("spawn")
        job_queues = [ctx.Queue() for _ in range(nprocs)]
        done_queue = ctx.Queue()

        workers = []
        for rank in active_ranks:
            p = ctx.Process(
                target=_persistent_worker,
                args=(rank, rank, args.stage1_cfg, args.stage1_ckpt, args.diff_cfg,
                      job_queues[rank], done_queue, args.scale_factor, args.latent_shape,
                      args.unet_batch_size),
                daemon=True,
            )
            p.start()
            workers.append(p)

        print(f"Spawned {len(active_ranks)} persistent worker(s) (GPUs {active_ranks}).")

        for epoch, ckpt_path, out_dir in jobs:
            label = f"epoch {epoch}" if epoch is not None else ckpt_path.name
            print(f"\n[{label}] {ckpt_path}")

            for rank in active_ranks:
                job_queues[rank].put((
                    str(ckpt_path), str(out_dir), indices_per_rank[rank],
                    affine, args.scheduler, args.timesteps,
                ))

            # Wait for all workers to finish this epoch before moving to the next
            for _ in active_ranks:
                done_queue.get()

        # Shut down workers
        for rank in active_ranks:
            job_queues[rank].put(None)
        for p in workers:
            p.join()

    # ── Single-GPU path ────────────────────────────────────────────────────────
    else:
        print("Loading VQGAN...")
        stage1 = load_stage1(args.stage1_cfg, args.stage1_ckpt, device)

        # Build UNet architecture once; hot-swap weights per epoch
        diff_cfg_obj = OmegaConf.load(args.diff_cfg)
        unet = DiffusionModelUNet(**diff_cfg_obj.model.params).to(device).eval().requires_grad_(False)

        for epoch, ckpt_path, out_dir in jobs:
            label = f"epoch {epoch}" if epoch is not None else ckpt_path.name
            print(f"\n[{label}] {ckpt_path}")

            # Hot-swap weights only
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ema = ckpt.get("ema")
            state_dict = ema["shadow"] if ema is not None else ckpt.get("model", ckpt)
            unet.load_state_dict(state_dict)

            if args.scheduler == "ddpm":
                scheduler = DDPMScheduler(**diff_cfg_obj.scheduler)
            else:
                scheduler = DDIMScheduler(**diff_cfg_obj.scheduler)
            scheduler.set_timesteps(args.timesteps)

            indices = list(range(args.n_samples))
            print(f"  Sampling {args.n_samples} volumes ({args.scheduler.upper()}, {args.timesteps} steps)...")
            _run_diffusion(unet, stage1, scheduler, diff_cfg_obj, indices, out_dir,
                           affine, args.scale_factor, tuple(args.latent_shape), device,
                           unet_batch_size=args.unet_batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
