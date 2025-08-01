import pandas as pd
import torch.distributed as dist
from monai import transforms
from monai.data import CacheDataset, Dataset, ThreadDataLoader, partition_dataset


def get_datalist(
    ids_path: str,
    extended_report: bool = False,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": f"{row['image']}",
                "cond": f"{row['artery']}",
                
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_training_data_loader(
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    only_val: bool = False,
    augmentation: bool = True,
    drop_last: bool = False,
    num_workers: int = 8,
    num_val_workers: int = 3,
    cache_data=False,
    first_n=None,
    is_grayscale=False,
    add_vflip=False,
    add_hflip=False,
    image_size=None,
    image_roi=None,
    spatial_dimension=3,
    keys = ["image","cond"]
):

    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
            transforms.RandSpatialCropd(keys=keys,roi_size=(330,330,220),random_size=False),
            #transforms.RandSpatialCropd(keys=keys,roi_size=(64,64,64),random_size=False), #was 330x330x220
            #transforms.NormalizeIntensityd(keys=["image"]),
            transforms.Resized(keys=["image"],spatial_size=(192,192,128)),
            transforms.Resized(keys=["cond"],spatial_size=(48,48,32)),
            transforms.RandFlipd(
                    keys=keys,
                    spatial_axis=0,
                    prob=0.5,
                ),
            transforms.ToTensord(keys=keys),
        ]
    )
    val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=keys),
                transforms.EnsureChannelFirstd(keys=keys),
                transforms.ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
                transforms.RandSpatialCropd(keys=keys,roi_size=(330,330,220),random_size=False),
                transforms.Resized(keys=["image"],spatial_size=(192,192,128)),
                transforms.Resized(keys=["cond"],spatial_size=(48,48,32)),
                #transforms.RandSpatialCropd(keys=keys,roi_size=(64,64,64),random_size=False),
                transforms.ToTensord(keys=keys),
            ]
        )

    # no augmentation for now
    if augmentation:
        train_transforms = train_transforms
    else:
        train_transforms = val_transforms

    val_dicts = get_datalist(validation_ids)

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

    train_dicts = get_datalist(training_ids)

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

    return train_loader, val_loader

