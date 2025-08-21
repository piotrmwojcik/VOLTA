#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path
from PIL import Image, ImageDraw

def clamp(val, lo, hi): return max(lo, min(val, hi))

def main():
    ap = argparse.ArgumentParser(description="Crop cells and save boxed full images using __bboxes.json; also copy the original overlay.")
    ap.add_argument("--image", required=True, help="Path to crop PNG (e.g., old__CASE__polygon_0007.png)")
    ap.add_argument("--bboxes", default=None, help="Path to JSON (defaults to image path with __bboxes.json)")
    ap.add_argument("--out-dir", default=None, help="Where to save outputs (default: <image_dir>/cells)")
    ap.add_argument("--pad", type=int, default=0, help="Optional pixel padding around each bbox")
    ap.add_argument("--min-size", type=int, default=1, help="Skip boxes smaller than this (min w/h)")
    ap.add_argument("--max-cells", type=int, default=None, help="Optional limit on number of cells to save")
    ap.add_argument("--box-width", type=int, default=3, help="Rectangle line width on boxed image")
    ap.add_argument("--no-overlay-copy", action="store_true", help="Disable copying the original overlay into out-dir")
    args = ap.parse_args()

    img_path = Path(args.image)
    if args.bboxes is None:
        bb_path = img_path.with_name(img_path.stem + "__bboxes.json")
    else:
        bb_path = Path(args.bboxes)

    out_dir = Path(args.out_dir) if args.out_dir else (img_path.parent / "cells")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- copy the original overlay into out_dir (once) ---
    if not args.no_overlay_copy:
        overlay_src = img_path.with_name(img_path.stem + "__overlay.png")
        if overlay_src.exists():
            overlay_dst = out_dir / overlay_src.name
            if not overlay_dst.exists():
                shutil.copy2(overlay_src, overlay_dst)

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

        # apply padding and clamp
        left   = clamp(int(x) - args.pad, 0, W)
        top    = clamp(int(y) - args.pad, 0, H)
        right  = clamp(int(x + w) + args.pad, 0, W)
        bottom = clamp(int(y + h) + args.pad, 0, H)

        cw, ch = right - left, bottom - top
        if cw < args.min_size or ch < args.min_size:
            continue

        # 1) cropped cell
        cell = im.crop((left, top, right, bottom))
        safe_lab = lab if lab else "empty"
        safe_lab = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe_lab)
        crop_name = f"{img_path.stem}__cell_{i:04d}__{safe_lab}_{lid}_{left}-{top}-{right}-{bottom}.png"
        cell.save(out_dir / crop_name)

        # 2) full image with rectangle around this cell
        boxed = im.copy()
        draw = ImageDraw.Draw(boxed)
        draw.rectangle([(left, top), (max(left, right-1), max(top, bottom-1))],
                       outline=(255, 0, 0), width=args.box_width)
        box_name = f"{img_path.stem}__cell_{i:04d}__{safe_lab}_{lid}_{left}-{top}-{right}-{bottom}__boxed.png"
        boxed.save(out_dir / box_name)

        saved += 1
        if args.max_cells is not None and saved >= args.max_cells:
            break

    print(f"Done. Saved {saved} cells (crops + boxed). Outputs in: {out_dir}")

if __name__ == "__main__":
    main()
