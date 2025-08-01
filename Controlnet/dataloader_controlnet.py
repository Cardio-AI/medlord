import pandas as pd
import numpy as np
from pathlib import Path
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    CenterSpatialCropd,
    ToTensord,
)
from monai.utils import set_determinism


def get_datalist(ids_path: str):
    """Get data dicts for multi-label medical segmentation data."""
    df = pd.read_csv(ids_path, sep="\t")

    data_dicts = []
    for _, row in df.iterrows():
        data_dicts.append({
            "image": f"{row['Image']}",
            "aorta": f"{row['aorta.nii']}",
            "artery": f"{row['Artery']}",
            "plaque": f"{row['Plaque']}",
            "ventricle_l": f"{row['Ventricle_left']}",
            "ventricle_r": f"{row['Ventricle_right']}",
            "atrium_l": f"{row['Atrium_left']}",
            "atrium_r": f"{row['Atrium_right']}",
            "myocardium": f"{row['Myocardium']}",
            "pulmonary": f"{row['Pulmonary']}",
        })

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_datalist_luna(ids_path: str):
    """Get data dicts for LUNA dataset."""
    df = pd.read_csv(ids_path, sep=",")
    data_dicts = [{"image": row["image"], "segments": row["segments"]} for _, row in df.iterrows()]
    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def previous_multiple_of_32(size):
    return int(np.floor(size / 32) * 32)


class CenterCropToMultipleOf32:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            new_sizes = [previous_multiple_of_32(dim) for dim in img.shape[-3:]]
            cropper = CenterSpatialCropd(keys=[key], roi_size=new_sizes)
            data = cropper(data)
        return data


transforms_presaved_luna = Compose([
    LoadImaged(keys=["image", "segments"]),
    EnsureChannelFirstd(keys=["segments"]),
    Resized(keys=["segments"], spatial_size=(512, 512, 256), mode="trilinear"),
    ToTensord(keys=["image", "segments"]),
])


def get_dataloaders(
    train_ids_path: str,
    val_ids_path: str,
    batch_size: int,
    num_workers: int = 4,
    cache_rate_train: float = 1.0,
    cache_rate_val: float = 0.0,
    transform=transforms_presaved_luna,
    luna: bool = False,
):
    """Returns train and validation dataloaders."""
    set_determinism(seed=42)  # Optional, can make seed configurable

    get_list_fn = get_datalist_luna if luna else get_datalist

    train_dicts = get_list_fn(train_ids_path)
    val_dicts = get_list_fn(val_ids_path)

    print(f"Training set size: {len(train_dicts)}")
    print(f"Validation set size: {len(val_dicts)}")

    train_ds = CacheDataset(data=train_dicts, transform=transform, cache_rate=cache_rate_train)
    val_ds = CacheDataset(data=val_dicts, transform=transform, cache_rate=cache_rate_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )
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