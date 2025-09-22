#!/usr/bin/env python3
"""
Build instance- and type-labeled arrays from all *__type-mask.png files.

Input:  /data/pwojcik/For_Piotr/gloms_rect_from_png_new/*__type-mask.png
Outputs (same folder, same stem):
  - <stem>_instances.npy   : HxW int32 unique instance IDs (across all classes)
  - <stem>_types.npy       : HxW int32 class IDs (palette index 0..5)
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage

# Hardcoded directory
SRC_DIR = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_new")

def read_type_map(mask_path: Path) -> np.ndarray:
    """
    Read the palette PNG without converting to RGB.
    Returns HxW int32 class IDs corresponding to the palette indices:
        0: _background, 1: _empty, 2: opal_480, 3: opal_520, 4: opal_570, 5: opal_620
    """
    img = Image.open(mask_path)
    # Ensure we keep palette indices; mode "P" yields an array of indices directly.
    type_map = np.array(img, dtype=np.int32)
    return type_map

def build_instance_map(type_map: np.ndarray) -> np.ndarray:
    """Create an instance map where each connected component (within each class) gets a unique ID."""
    H, W = type_map.shape
    inst_map = np.zeros((H, W), dtype=np.int32)

    next_id = 0
    max_class = int(type_map.max()) if type_map.size else -1

    for cls in range(max_class + 1):
        binary = (type_map == cls)
        if not np.any(binary):
            continue
        labeled, n = ndimage.label(binary)  # 4-connectivity by default; use structure for 8 if needed
        if n == 0:
            continue
        # Assign unique IDs across all classes
        # labeled == i are pixels of the ith component (1..n)
        for i in range(1, n + 1):
            inst_map[labeled == i] = next_id
            next_id += 1

    return inst_map

def main():
    files = sorted(SRC_DIR.glob("old__*__q*__type-mask.png"))
    if not files:
        print(f"[WARN] No type-mask files found in {SRC_DIR}")
        return

    for f in files:
        # 1) Read type map (class IDs as stored in the palette PNG)
        type_map = read_type_map(f)

        # 2) Build instance map
        inst_map = build_instance_map(type_map)

        # 3) Save both as .npy next to the mask
        stem = f.with_suffix("")  # drop .png
        out_types = stem.with_name(stem.name.replace("__type-mask", "_types") + ".npy")
        out_insts = stem.with_name(stem.name.replace("__type-mask", "_instances") + ".npy")

        np.save(out_types, type_map.astype(np.int32, copy=False))
        np.save(out_insts, inst_map.astype(np.int32, copy=False))

        print(
            f"[OK] {f.name} → "
            f"{out_types.name} (shape {type_map.shape}, classes up to {type_map.max()}), "
            f"{out_insts.name} (shape {inst_map.shape}, max ID {inst_map.max()})"
        )

    print(f"[DONE] Processed {len(files)} file(s).")

if __name__ == "__main__":
    main()
