#!/usr/bin/env python3
"""
Build instance-labeled arrays from all *__type-mask.png files in the hardcoded directory.

Input:  /data/pwojcik/For_Piotr/gloms_rect_from_png_within/*type-mask.png
Output: Same name with suffix _instances.npy

Each output array has the same HxW shape as the mask, but pixel values are
unique instance IDs numbered consecutively from 0 upward across all classes.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage

# Hardcoded directory
SRC_DIR = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within")


def build_instance_map(mask_path: Path) -> np.ndarray:
    """Create an instance map where each connected component gets a unique ID."""
    mask = np.array(Image.open(mask_path)).astype(np.int32)
    H, W = mask.shape
    inst_map = np.zeros((H, W), dtype=np.int32)

    next_id = 0
    max_class = mask.max()

    for cls in range(max_class + 1):
        binary = (mask == cls)
        if not np.any(binary):
            continue
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            inst_map[labeled == i] = next_id
            next_id += 1

    return inst_map

def main():
    files = sorted(SRC_DIR.glob("old__*__type-mask_*__q*.png"))
    if not files:
        print(f"[WARN] No type-mask files found in {SRC_DIR}")
        return

    for f in files:
        inst_map = build_instance_map(f)
        out_path = f.with_name(f.stem + "_instances.npy")
        np.save(out_path, inst_map)
        print(f"[OK] Saved {out_path} with shape {inst_map.shape} and max ID {inst_map.max()}")

    print(f"[DONE] Processed {len(files)} file(s).")


if __name__ == "__main__":
    main()
