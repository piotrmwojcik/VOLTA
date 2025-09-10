#!/usr/bin/env python3
"""
Split every *__type-mask.png inside /data/pwojcik/For_Piotr/gloms_rect_from_png_within/
into four quadrant tiles (q00, q01, q10, q11) matching the original image splits.

Example:
  old__K_hNiere_S1_PAS__polygon_0115__type-mask.png
    -> old__K_hNiere_S1_PAS__mask_0115__q00.png
    -> old__K_hNiere_S1_PAS__mask_0115__q01.png
    -> old__K_hNiere_S1_PAS__mask_0115__q10.png
    -> old__K_hNiere_S1_PAS__mask_0115__q11.png
"""

import re
from pathlib import Path
from PIL import Image

# Hardcoded source directory
SRC_DIR = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within")

def split_quadrants(img: Image.Image):
    """Return dict of 4 crops split by image midlines."""
    W, H = img.size
    mx, my = W // 2, H // 2
    boxes = {
        "q00": (0,  0,  mx, my),   # top-left
        "q01": (mx, 0,  W,  my),   # top-right
        "q10": (0,  my, mx, H),    # bottom-left
        "q11": (mx, my, W,  H),    # bottom-right
    }
    return {k: img.crop(b) for k, b in boxes.items()}

def build_outname(src: Path, qk: str) -> str:
    """Build output file name based on the pattern."""
    name = src.name
    m = re.match(r"^(.*)__polygon_(\d+)__type-mask\.png$", name, re.IGNORECASE)
    if m:
        pre, num = m.groups()
        return f"{pre}__mask_{num}__{qk}.png"
    stem, ext = src.stem, src.suffix
    return f"{stem}__{qk}{ext}"

def process_one(src: Path):
    try:
        with Image.open(src) as im:
            crops = split_quadrants(im)
        for qk, crop in crops.items():
            outname = build_outname(src, qk)
            dest = src.parent / outname
            crop.save(dest, format="PNG", optimize=True)
        print(f"[OK] {src.name} -> 4 tiles")
        return 1
    except Exception as e:
        print(f"[WARN] Failed {src}: {e}")
        return 0

def main():
    files = list(SRC_DIR.glob("*__type-mask.png"))
    if not files:
        print(f"[WARN] No files found in {SRC_DIR}")
        return
    total = 0
    for f in sorted(files):
        total += process_one(f)
    print(f"[DONE] Processed {total} file(s).")

if __name__ == "__main__":
    main()
