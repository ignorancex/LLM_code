import numpy as np
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is for NCR
    label 2 is for ED
    label 4 is for ET
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                result = []
                result.append(np.logical_or(d[key] == 1, d[key] == 4))
                # merge labels 1, 2 and 4 to construct WT
                result.append(np.logical_or(np.logical_or(d[key] == 4, d[key] == 1), d[key] == 2))
                # label 4 is ET
                result.append(d[key] == 4)
                d[key] = np.stack(result, axis=0).astype(np.float32)
            else:
                if self.allow_missing_keys:
                    continue
        return d


def get_train_transforms(keys):
    return Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=keys),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            CenterSpatialCropd(keys=keys, roi_size=[128, 128, 128]),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys=keys, factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=1.0),
        ]
    ).set_random_state(1999)


def get_val_transforms(keys):
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=keys),
            Spacingd(
                keys=keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            CenterSpatialCropd(keys=keys, roi_size=[128, 128, 128]),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ]
    ).set_random_state(1999)


def get_train_seg_transforms():
    return Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    ).set_random_state(1999)

def get_val_seg_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    ).set_random_state(1999)
    

def get_train_pretrain_transforms_with_seg(keys):
    full = keys + ["seg"]
    mode = ["bilinear"] * len(keys) + ["nearest"]
    return Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=full),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=full),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=full,
                pixdim=(1.0, 1.0, 1.0),
                mode=mode,
            ),
            CenterSpatialCropd(keys=full, roi_size=[128, 128, 128]),
            RandFlipd(keys=full, prob=0.5, spatial_axis=0),
            RandFlipd(keys=full, prob=0.5, spatial_axis=1),
            RandFlipd(keys=full, prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys=keys, factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=1.0),
        ]
    ).set_random_state(1999)


def get_val_pretrain_transforms_with_seg(keys):
    full = keys + ["seg"]
    mode = ["bilinear"] * len(keys) + ["nearest"]
    return Compose(
        [
            LoadImaged(keys=full),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=full),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=full,
                pixdim=(1.0, 1.0, 1.0),
                mode=mode,
            ),
            CenterSpatialCropd(keys=full, roi_size=[128, 128, 128]),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ]
    ).set_random_state(1999)
