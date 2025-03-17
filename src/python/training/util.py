"""Utility functions for training."""
from pathlib import Path
from typing import Tuple, Union, Dict

import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ThresholdIntensityd, ScaleIntensityd,
    Resized, ToTensord,MapTransform, Transform,CenterSpatialCropd,SpatialPadd,RandRotated,RandFlipd,RandSpatialCropd
)
from monai.data import PersistentDataset, Dataset, DataLoader, CacheDataset

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tensorboardX import SummaryWriter



# ----------------------------------------------------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------------------------------------------------
def get_datalist(
    ids_path: str,
    extended_report: bool = False,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep=",")

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": f"{row['image']}",

            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


#Not used for MICCAI paper, but might be used in the future
def next_multiple_of_128(depth, max_depth=256):
    """Returns the next multiple of 128 for depth, but does not exceed max_depth."""
    new_depth = int(np.ceil(depth / 128) * 128)
    return min(new_depth, max_depth)

class ResizeDepthToMultipleOf128:
    """Custom transform to resize only the depth dimension to the next multiple of 128, up to max_depth."""
    def __init__(self, keys, max_depth=256):
        self.keys = keys
        self.max_depth = max_depth

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            original_shape = img.shape  # (C, H, W, D) or (H, W, D)
            new_depth = next_multiple_of_128(original_shape[-1], self.max_depth)
            data[key] = Resized(keys=[key], spatial_size=(original_shape[1], original_shape[2], new_depth), mode="trilinear")(data)[key]
        return data


def get_dataloader(
    cache_dir: Union[str, Path],
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    num_workers: int = 8,
    model_type: str = "autoencoder",
    image_roi=None,
):
    # Define transformations

    if model_type == "autoencoder":
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ThresholdIntensityd(keys=["image"], threshold=2000, above=False, cval=2000),
                ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
                ScaleIntensityd(keys=["image"],minv=-1.0,maxv=1.0),
                RandRotated(keys=["image"],range_x=0.0872665,range_y=0.0872665,range_z=0.0872665,prob=0.2),
                RandFlipd(keys=["image"],spatial_axis=1,prob=0.5),
                RandSpatialCropd(keys=["image"],roi_size=image_roi,random_size=False)
            ]
        )
        val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=2000, above=False, cval=2000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"],minv=-1.0,maxv=1.0),
            RandSpatialCropd(keys=["image"],roi_size=image_roi,random_size=False),
            ]
        )

    # Use this if you encode images on the run, with no augmentation
    if model_type == "diffusion":
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ThresholdIntensityd(keys=["image"], threshold=2000, above=False, cval=2000),
                ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
                ScaleIntensityd(keys=["image"],minv=-1.0,maxv=1.0),
                SpatialPadd(keys=["image"],spatial_size=image_roi),
                CenterSpatialCropd(keys=["image"],roi_size=image_roi),  
                ToTensord(keys=["image"]),       
            ]
        )
        val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=2000, above=False, cval=2000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"],minv=-1.0,maxv=1.0),
            SpatialPadd(keys=["image"],spatial_size=image_roi),
            CenterSpatialCropd(keys=["image"],roi_size=image_roi), 
            ]
        )
    #Use this in a non restricted setting, with donwsample/upsample transform
    if model_type == "non_restricted":
        max_depth = 512 # define maximum depth
        val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=2000, above=False, cval=2000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            ResizeDepthToMultipleOf128(keys=["image"], max_depth=max_depth), 
            ToTensord(keys=["image"]),
        ])
        train_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=2000, above=False, cval=2000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            ResizeDepthToMultipleOf128(keys=["image"], max_depth=max_depth), 
            ToTensord(keys=["image"]),
        ])
    if model_type == "transformer":
        val_transforms = Compose(
        [   LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=2000, above=False, cval=2000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"],minv=-1.0,maxv=1.0),
            Resized(keys=["image"],spatial_size=(448,448,256)), #remove this, if not restricted by 24gb GPU
            ToTensord(keys=["image"]),
        ]
    )
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ThresholdIntensityd(keys=["image"], threshold=2000, above=False, cval=2000),
                ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
                ScaleIntensityd(keys=["image"],minv=-1.0,maxv=1.0), 
                Resized(keys=["image"],spatial_size=(448,448,256)),
                ToTensord(keys=["image"]),  
            ]
        )
    #Use this if latents are presaved on harddrive 
    if model_type == "presaved":
        val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            ToTensord(keys=["image"]),  
        ]
    )
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),       
            ]
        )

    train_dicts = get_datalist(ids_path=training_ids)
    #Preloading data into Cache for more training speed. Adjust cache_rate to smaller value, if running out of RAM.
    train_ds = CacheDataset(data=train_dicts,transform=train_transforms,cache_rate=1.0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dicts = get_datalist(ids_path=validation_ids)
    val_ds = CacheDataset(data=val_dicts,transform=val_transforms,cache_rate=0.1)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return train_loader, val_loader


# ----------------------------------------------------------------------------------------------------------------------
# LOGS
# ----------------------------------------------------------------------------------------------------------------------

def get_figure(
    img: torch.Tensor,
    recons: torch.Tensor,
    slice_axes: list = [0, 1, 2],  # List of axes to plot slices along
    slice_idx: int = 8  # Fixed slice index; can also make this a parameter if you want more control
):
    # Assuming img and recons are 5D tensors: (batch size, channels, x_size, y_size, z_size)

    num_axes = len(slice_axes)
    fig, axes = plt.subplots(2, num_axes, figsize=(6 * num_axes, 12), dpi=300)

    for i, axis in enumerate(slice_axes):
        if axis == 0:  # x-axis
            img_slice = np.clip(img[0, 0, slice_idx, :, :].cpu().numpy(), a_min=-1, a_max=1)
            recons_slice = np.clip(recons[0, 0, slice_idx, :, :].cpu().numpy(), a_min=-1, a_max=1)
        elif axis == 1:  # y-axis
            img_slice = np.clip(img[0, 0, :, slice_idx, :].cpu().numpy(), a_min=-1, a_max=1)
            recons_slice = np.clip(recons[0, 0, :, slice_idx, :].cpu().numpy(), a_min=-1, a_max=1)
        elif axis == 2:  # z-axis
            img_slice = np.clip(img[0, 0, :, :, slice_idx].cpu().numpy(), a_min=-1, a_max=1)
            recons_slice = np.clip(recons[0, 0, :, :, slice_idx].cpu().numpy(), a_min=-1, a_max=1)

        # Plot original image slice
        axes[0, i].imshow(img_slice, cmap="gray")
        axes[0, i].set_title(f"Original - Axis {axis}")
        axes[0, i].axis("off")

        # Plot reconstructed image slice
        axes[1, i].imshow(recons_slice, cmap="gray")
        axes[1, i].set_title(f"Reconstructed - Axis {axis}")
        axes[1, i].axis("off")

    # Remove white space around the plots
    plt.tight_layout(pad=0.0)

    return fig

def log_reconstructions(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    title: str = "RECONSTRUCTION",
) -> None:
    fig = get_figure(
        image,
        reconstruction,
        slice_axes=[0, 1, 2],  # Specify all axes to visualize
    )
    writer.add_figure(title, fig, step)

