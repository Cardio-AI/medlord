import pandas as pd
import torch.distributed as dist
from monai import transforms
from monai.data import CacheDataset, Dataset, ThreadDataLoader, partition_dataset
import os


def get_data_dicts(ids_path: str, test_frac, shuffle: bool = False, first_n=False):

    """Get data dicts for data loaders."""
    
    img_files = [os.path.join(ids_path, f) for f in os.listdir(ids_path) if f.endswith('.nii.gz')]

    num_test = 16
    num_train = len(img_files)- num_test
    train_dict = [{"image": fname} for fname in img_files[:num_train]]
    val_dict = [{"image": fname} for fname in img_files[num_train:]]
    return train_dict, val_dict


def get_training_data_loader(
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    only_val: bool = False,
    augmentation: bool = True,
    drop_last: bool = False,
    num_workers: int = 1,
    num_val_workers: int = 1,
    cache_data=False,
    first_n=None,
    is_grayscale=False,
    add_vflip=False,
    add_hflip=False,
    image_size=None,
    image_roi=None,
    spatial_dimension=3,
    test_frac = 0.1,
    keys = ['image'],
):

    val_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        transforms.RandSpatialCropd(keys=keys,roi_size=(256,256,256),random_size=False), #was 330x300x230 ,was 256,256,256
        #transforms.NormalizeIntensityd(keys=["image"]),
        #transforms.Resized(keys=["image"],spatial_size=(256,256,128)),
        transforms.ToTensord(keys=["image"]),
    ])

    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        transforms.RandSpatialCropd(keys=keys,roi_size=(256,256,256),random_size=False), #was 330x300x230
        #transforms.NormalizeIntensityd(keys=["image"]),
        transforms.RandFlipd(
                keys=["image"],
                spatial_axis=0,
                prob=0.5,
            ),
        transforms.RandAffined(
                keys=["image"],
                translate_range=(-2, 2),
                scale_range=(-0.05, 0.05),
                spatial_size=[256,256,256],
                prob=0.5,
            ),
        transforms.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
        transforms.RandAdjustContrastd(keys=["image"], gamma=(0.97, 1.03), prob=0.1),
        #transforms.Resized(keys=["image"],spatial_size=(256,256,128)),    
        transforms.ToTensord(keys=["image"]),
    ])
    

    # no augmentation for now
    if augmentation:
        train_transforms = train_transforms
    else:
        train_transforms = val_transforms

    train_dicts, val_dicts = get_data_dicts(validation_ids, test_frac,shuffle=False, first_n=first_n)

    if first_n:
        val_dicts = val_dicts[:first_n]

    if cache_data:
        val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(
            data=val_dicts,
            transform=val_transforms,
        )
    print(val_ds[0]["image"].shape)
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_val_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    if only_val:
        return val_loader


    if cache_data:
        train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )
    print('Numer of training samples: ',len(train_dicts))
    print('Number of validation samples: ', len(val_dicts))
    return train_loader, val_loader
