#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from PIL import Image

def clamp(val, lo, hi): return max(lo, min(val, hi))

def main():
    ap = argparse.ArgumentParser(description="Crop cells from one image using its __bboxes.json")
    ap.add_argument("--image", required=True, help="Path to crop PNG (e.g., old__CASE__polygon_0007.png)")
    ap.add_argument("--bboxes", default=None, help="Path to JSON (defaults to image path with __bboxes.json)")
    ap.add_argument("--out-dir", default=None, help="Where to save crops (default: <image_dir>/cells)")
    ap.add_argument("--pad", type=int, default=0, help="Optional pixel padding around each bbox")
    ap.add_argument("--min-size", type=int, default=1, help="Skip boxes smaller than this (min w/h)")
    ap.add_argument("--max-cells", type=int, default=None, help="Optional limit on number of crops")
    args = ap.parse_args()

    img_path = Path(args.image)
    if args.bboxes is None:
        # derive JSON name from the image file (…__bboxes.json)
        bb_path = img_path.with_name(img_path.stem + "__bboxes.json")
    else:
        bb_path = Path(args.bboxes)

    out_dir = Path(args.out_dir) if args.out_dir else (img_path.parent / "cells")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image and JSON
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    data = json.loads(Path(bb_path).read_text())

    boxes = data.get("boxes", [])
    saved = 0
    for i, rec in enumerate(boxes):
        x, y, w, h = rec["bbox"]
        lab = str(rec.get("label", ""))
        lid = rec.get("label_id", -1)

        # apply padding and clamp to image bounds
        left   = clamp(int(x) - args.pad, 0, W)
        top    = clamp(int(y) - args.pad, 0, H)
        right  = clamp(int(x + w) + args.pad, 0, W)
        bottom = clamp(int(y + h) + args.pad, 0, H)

        cw, ch = right - left, bottom - top
        if cw < args.min_size or ch < args.min_size:
            continue

        cell = im.crop((left, top, right, bottom))

        # safe label for filename
        safe_lab = lab if lab else "empty"
        safe_lab = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe_lab)

        out_name = f"{img_path.stem}__cell_{i:04d}__{safe_lab}_{lid}_{left}-{top}-{right}-{bottom}.png"
        cell.save(out_dir / out_name)
        saved += 1

        if args.max_cells is not None and saved >= args.max_cells:
            break

    print(f"Done. Saved {saved} crops to {out_dir}")

if __name__ == "__main__":
    main()
