# MedLoRD: A Medical Low-Resource Diffusion Model for High-Resolution 3D CT Image Synthesis

This repository contains the code for the paper **"MedLoRD: A Medical Low-Resource Diffusion Model for High-Resolution 3D CT Image Synthesis"**: https://arxiv.org/abs/2503.13211

MedLoRD generates high-dimensional medical volumes with resolutions up to **512×512×256**, on GPUs with as little as **24GB VRAM**.

<div align="center">
  <table>
    <tr>
      <td><img src="figures/lung_uncond.gif" width="220"/></td>
      <td><img src="figures/pccta_uncond.gif" width="220"/></td>
      <td><img src="figures/luna_cond.gif" width="220"/></td>
      <td><img src="figures/pccta_cond.gif" width="220"/></td>
    </tr>
    <tr>
      <td align="center"><i>LUNA Unconditional</i></td>
      <td align="center"><i>PCCTA Unconditional</i></td>
      <td align="center"><i>LUNA Conditional</i></td>
      <td align="center"><i>PCCTA Conditional</i></td>
    </tr>
  </table>
</div>

## Abstract

Advancements in AI for medical imaging offer significant potential. However, their applications are constrained by the limited availability of data and the reluctance of medical centers to share it due to patient privacy concerns. Generative models present a promising solution by creating synthetic data as a substitute for real patient data. However, medical images are typically high-dimensional, and current state-of-the-art methods are often impractical for computational resource-constrained healthcare environments. These models rely on data sub-sampling, raising doubts about their feasibility and real-world applicability. Furthermore, many of these models are evaluated on quantitative metrics that alone can be misleading in assessing the image quality and clinical meaningfulness of the generated images. To address this, we introduce **MedLoRD**, a generative diffusion model designed for computational resource-constrained environments. MedLoRD is capable of generating high-dimensional medical volumes with resolutions up to **512×512×256**, utilizing GPUs with only **24GB VRAM**, which are commonly found in standard desktop workstations. MedLoRD is evaluated across multiple modalities, including **Coronary Computed Tomography Angiography** and **Lung Computed Tomography** datasets. Extensive evaluations through radiological evaluation, relative regional volume analysis, adherence to conditional masks, and downstream tasks show that MedLoRD generates high-fidelity images closely adhering to segmentation mask conditions, surpassing the capabilities of current state-of-the-art generative models for medical image synthesis in computational resource-constrained environments.

<div align="center">
  <img src="figures/medlord.png" width="400" />
</div>

## Pretrained Models

Pretrained checkpoints trained on the LUNA dataset are available on Hugging Face Hub. Note: models trained on the PCCTA dataset are not publicly available as that dataset is private.

| Model | Download |
|---|---|
| VQ-GAN | `hf download AICM-HD/medlord vqgan_luna_ds8.pth --local-dir checkpoints/` |
| LDM (unconditional) | `hf download AICM-HD/medlord ldm_luna_ds8.pth --local-dir checkpoints/` |

Or download all checkpoints at once:

```bash
hf download AICM-HD/medlord --local-dir checkpoints/
```

Once downloaded, pass the checkpoint paths directly to the sampling or training scripts (see [Sampling](#sampling-from-the-ldm) below).

## Requirements

- Python 3.10
- CUDA 12.6
- PyTorch 2.10

Install all dependencies using the provided Conda environment:

```bash
conda env create -f environment.yaml -n medlord
conda activate medlord
```

## Data Preparation

Training expects CSV files with a column named `image` containing absolute paths to NIfTI (`.nii` / `.nii.gz`) CT volumes. Images are automatically clipped to the HU range `[-1000, 1000]` and scaled to `[-1, 1]`.

```
image
/data/ct_001.nii.gz
/data/ct_002.nii.gz
...
```

Split into a training CSV and a validation CSV before starting.

---

## Training: Unconditional LDM

Training proceeds in two stages: first the VQ-GAN autoencoder, then the diffusion model in latent space.

### Stage 1 — Train VQ-GAN

The VQ-GAN learns to compress 3D CT volumes into a compact 8-dimensional latent space using a combination of L1, perceptual, adversarial (PatchGAN), and spectral (Jukebox) losses.

```bash
python src/scripts/train_vqgan.py \
    --config configs/stage1/vqgan_ds4_new.yaml \
    --training_ids ids/train.csv \
    --validation_ids ids/val.csv \
    --output_dir outputs/ \
    --run_name vqgan_v1 \
    --cache_dir cache/vqgan_v1
```

The best checkpoint is saved to `outputs/vqgan_v1/best_checkpoint.pth`.

**Config options:** `vqgan_ds4_new.yaml` (2× downsampling, recommended), `vqgan_ds4_small.yaml` (reduced memory), `vqgan_ds8.yaml` (3× downsampling).

---

### Stage 2 — Pre-encode Images to Latents (recommended)

Pre-encoding the dataset once avoids redundant VQ-VAE forward passes during LDM training and significantly speeds up training. Set `use_precomputed_latents: True` in your LDM config.

```bash
python src/scripts/encode_images.py \
    --csv ids/train.csv \
    --output_dir data/latents/ \
    --vqvae_ckpt outputs/vqgan_v1/best_checkpoint.pth \
    --config configs/stage1/vqgan_ds4_new.yaml \
    --batch_size 1 \
    --device cuda
```

Repeat for the validation set. Each run produces a `latents.csv` in the output directory, with a column `image` pointing to the pre-encoded `.pt` tensors. Pass these CSVs as `--training_ids` and `--validation_ids` when training the LDM.

Already-encoded files are automatically skipped on re-runs.

---

### Stage 2 — Train LDM

Trains a 3D diffusion U-Net in the VQ-GAN latent space using a cosine noise schedule with v-prediction.

```bash
python src/scripts/train_ldm.py \
    --config configs/diffusion/medlord_new.yaml \
    --vqvae_ckpt outputs/vqgan_v1/best_checkpoint.pth \
    --config_vqvae configs/stage1/vqgan_ds4_new.yaml \
    --training_ids data/latents/train/latents.csv \
    --validation_ids data/latents/val/latents.csv \
    --output_dir outputs/ \
    --run_name ldm_v1
```

The best EMA checkpoint is saved to `outputs/ldm_v1/best_checkpoint.pth`.

---

### Sampling from the LDM

```bash
python src/scripts/sample_ldm.py \
  --stage1_ckpt /path/to/vqgan.ckpt \
  --stage1_cfg configs/stage1/vqgan_ds4_new.yaml \
  --diff_ckpt /path/to/diffusion.ckpt \
  --diff_cfg configs/diffusion/medlord.yaml \
  --latent_shape 16 16 8 \
  --output_dir samples \
  --n_samples 4 \
  --timesteps 300 \
  --scheduler ddpm \
  --scale_factor 1.0
```

Outputs are saved as `.nii.gz` files with HU values. EMA weights are used automatically if available in the checkpoint.

---

## Training: Conditional LDM (ControlNet)

ControlNet extends the trained LDM with mask-guided generation. The diffusion U-Net weights are frozen; only the ControlNet adapter is trained.

### Stage 3a — Pre-encode Images + Masks

Prepare a CSV with one column per condition key in addition to `image`:

```
image,lung,lung_nodules,...
/data/ct_001.nii.gz,/masks/ct_001_lung.nii.gz,...
```

Then encode:

```bash
python src/scripts/encode_images_cond.py \
    --csv ids/train_cond.csv \
    --output_dir data/latents_cond/ \
    --vqvae_ckpt outputs/vqgan_v1/best_checkpoint.pth \
    --config configs/stage1/vqgan_ds4_new.yaml \
    --condition_keys lung lung_nodules \
    --device cuda
```

Produces `controlnet_latents.csv` with paths to the encoded image latent and all preprocessed mask tensors. Run for both train and validation sets.

---

### Stage 3b — Train ControlNet

```bash
python src/scripts/train_controlnet.py \
    --config configs/diffusion/controlnet_new.yaml \
    --ldm_ckpt outputs/ldm_v1/best_checkpoint.pth \
    --vqvae_ckpt outputs/vqgan_v1/best_checkpoint.pth \
    --config_vqvae configs/stage1/vqgan_ds4_new.yaml \
    --training_ids data/latents_cond/train/controlnet_latents.csv \
    --validation_ids data/latents_cond/val/controlnet_latents.csv \
    --output_dir outputs/ \
    --run_name controlnet_v1
```

Key config parameters in `controlnet_new.yaml`:
- `condition_keys` — list of mask column names matching the encode step
- `r` — channel scaling factor for the adapter (0.5 = half the U-Net width)
- `control_dropout` — probability of dropping the entire condition during training (improves robustness)
- `initial_stride` — set to 2 for global/coarse masks, 1 for finer conditioning

---

### Sampling from the ControlNet

Provide a CSV with one row per subject and one column per condition key. If the CSV contains an `image` column its filename stem is used to name the output (e.g. `ct_001_cond.nii.gz`), otherwise the row index is used.

```bash
python src/scripts/sample_controlnet.py \
    --stage1_ckpt checkpoints/vqgan_luna.pth \
    --stage1_cfg configs/stage1/vqgan_ds4_new.yaml \
    --diff_ckpt outputs/ldm_v1/best_checkpoint.pth \
    --diff_cfg configs/diffusion/medlord_new.yaml \
    --controlnet_ckpt outputs/controlnet_v1/best_model.pth \
    --controlnet_cfg configs/diffusion/controlnet_new.yaml \
    --csv ids/test_cond.csv \
    --condition_keys lung lung_nodules \
    --latent_shape 128 128 64 \
    --output_dir samples/
```

One `.nii.gz` volume is generated per CSV row. EMA weights are loaded automatically if present in the checkpoint. The `--conditioning_scale` argument (default `1.0`) controls the strength of the mask guidance at inference time without retraining.

---

## Distributed Training

All training scripts support multi-GPU training via PyTorch DDP. Use `torchrun`:

```bash
torchrun --nproc_per_node=2 src/scripts/train_vqgan.py \
    --config configs/stage1/vqgan_ds4_new.yaml \
    --training_ids ids/train.csv \
    --validation_ids ids/val.csv \
    --output_dir outputs/ \
    --run_name vqgan_v1 \
    --cache_dir cache/vqgan_v1
```

The same applies to `train_ldm.py` and `train_controlnet.py`.

---

## License

This code is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@InProceedings{10.1007/978-3-032-05573-6_1,
author="Seyfarth, Marvin
and Dar, Salman Ul Hassan
and Ayx, Isabelle
and Fink, Matthias Alexander
and Schoenberg, Stefan O.
and Kauczor, Hans-Ulrich
and Engelhardt, Sandy",
editor="Fernandez, Virginia
and Wiesner, David
and Zuo, Lianrui
and Casamitjana, Adri{\`a}
and Remedios, Samuel W.",
title="MedLoRD: A Medical Low-Resource Diffusion Model for High-Resolution 3D CT Image Synthesis",
booktitle="Simulation and Synthesis in Medical Imaging",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="1--12",
isbn="978-3-032-05573-6"
}
```
