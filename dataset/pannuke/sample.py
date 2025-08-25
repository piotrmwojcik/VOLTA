# example_pannuke_usage.py
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- Albumentations transforms ----
# Whole-image transform (applied to "image" only)
from dataset.pannuke.dataset import PanNukeDataset

img_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Patch & its masks transform (same geometry applied to both patch and masks)
patch_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ---- required caches (because cache_patch=True asserts these exist) ----
shared_caches = {
    "cache_patch": {},
    "cache_segmentation": {},
    "cache_morphological": {},
}

# Directory layout expected under root_dir:
# root_dir/
#   Images/         (RGB .png/.jpg ...)
#   Labels/         (text files with an integer label per image)
#   Locations/      (text "x,y" per image; used to center the patch)
#   Map/            (per-image .npy instance-map for segmentation → converted to binary)
#   Patches/        (RGB patch images; names derived from Images via image_to_patch_name_convertor)
#   Segmentation/   (per-image .npy instance-map used to create a binary segmentation for the patch)
#   Embedding/      (optional, only if hovernet_enable=True)

root_dir = "/data/pwojcik/Fold 2/"  # <-- change me

ds = PanNukeDataset(
    root_dir=root_dir,
    transform=img_transform,          # Albumentations
    target_transform=None,            # (optional) e.g., label mapping
    patch_transform=patch_transform,  # Albumentations (applied to patch + masks)
    patch_size=256,                   # crop size around the specified location
    mask_ratio=0.25,                  # create a boolean ROI-mask around the cell center (relative size)
    cache_patch=True,
    hovernet_enable=False,            # set True only if you have Embedding/
    return_file_name=False,
    dataset_size=None,                # e.g., 128 to truncate for quick tests
    valid_labels=None,                # e.g., [0,1,2] to filter classes
    shared_dictionaries=shared_caches
)

loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

# ---- grab one batch and inspect shapes/types ----
batch = next(iter(loader))
(
    images,            # Tensor [B,3,H,W] after img_transform
    labels,            # LongTensor [B]
    patches,           # Tensor [B,3,h,w] after patch_transform (if patch_size is set)
    slide_ids,         # LongTensor or int per sample (ID parsed from patch filename)
    offsets,           # LongTensor [B,2] -> (patch_left, patch_top) in the original patch image
    masks,             # BoolTensor [B,h,w] or None (if patch_size not set)
    segmentations,     # BoolTensor [B,h,w] (binary from instance map) or None
    extra_features     # np.array / Tensor [B, 38] (zeros in this template)
) = batch

print("images:", images.shape, images.dtype)
print("labels:", labels.shape, labels.dtype, labels[:4])
print("patches:", None if patches is None else (patches.shape, patches.dtype))
print("slide_ids:", slide_ids[:4] if isinstance(slide_ids, torch.Tensor) else slide_ids)
print("offsets:", offsets.shape, offsets[:4])
print("masks:", None if masks is None else (masks.shape, masks.dtype))
print("segmentations:", None if segmentations is None else (segmentations.shape, segmentations.dtype))
print("extra_features:", type(extra_features), getattr(extra_features, "shape", None))


