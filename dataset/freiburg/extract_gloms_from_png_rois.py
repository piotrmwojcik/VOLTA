#!/usr/bin/env python3
from pathlib import Path
import warnings

import numpy as np
import geopandas as gpd
from PIL import Image
from PIL.Image import DecompressionBombWarning
from shapely.geometry import box

# Trust your images (huge PNGs)
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", DecompressionBombWarning)

def clamp_xyxy_to_image(x0, y0, x1, y1, W, H):
    x0 = max(0.0, min(float(x0), W))
    y0 = max(0.0, min(float(y0), H))
    x1 = max(0.0, min(float(x1), W))
    y1 = max(0.0, min(float(y1), H))
    x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
    y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)
    x_min_i = int(np.floor(x_min))
    y_min_i = int(np.floor(y_min))
    x_max_i = int(np.ceil(x_max))
    y_max_i = int(np.ceil(y_max))
    x_min_i = max(0, min(x_min_i, W))
    y_min_i = max(0, min(y_min_i, H))
    x_max_i = max(0, min(x_max_i, W))
    y_max_i = max(0, min(y_max_i, H))
    w = max(0, x_max_i - x_min_i)
    h = max(0, y_max_i - y_min_i)
    return x_min_i, y_min_i, w, h

# ----------------- core (PNG) -----------------
def cut_rois_png(
    dataset_tag: str,
    roi_geojson_path: str,
    png_path: str,
    out_root: str,
):
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_stem = Path(png_path).stem
    prefix = f"{dataset_tag}__{png_stem}"

    # Load ROIs (pixel space)
    gdf = gpd.read_file(roi_geojson_path)

    # Load PNG
    img = Image.open(png_path).convert("RGB")
    W_img, H_img = img.size

    print(f"\n[{dataset_tag}] Processing {png_stem} ({W_img}x{H_img})")

    n_saved = 0
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # crop the bounding rectangle of ROI
        minx, miny, maxx, maxy = geom.bounds
        x, y, w, h = clamp_xyxy_to_image(minx, miny, maxx, maxy, W_img, H_img)
        if w == 0 or h == 0:
            print(f"  ROI {i}: window out of bounds; skipping.")
            continue

        crop = img.crop((x, y, x + w, y + h))

        # Save only the crop
        crop_png = out_dir / f"{prefix}__polygon_{i:04d}.png"
        crop.save(crop_png)
        n_saved += 1

    print(f"  Saved {n_saved} ROI crops.")

# ----------------- batch runner (PNG) -----------------
if __name__ == "__main__":
    out_root = Path("/data/pwojcik/For_Piotr/gloms_marianna")
    out_root.mkdir(parents=True, exist_ok=True)

    # (tag, roi_geojson_dir, images_dir)
    datasets_cfg = [
        ("new",
         Path("/data/pwojcik/marianna_intership/masks_geojson_40x"),
         Path("/data/pwojcik/marianna_intership/mirax_registration_png")),
    ]

    exts = [".png"]

    def find_image_for(stem: str, images_dir: Path):
        for ext in exts:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        for p in images_dir.rglob("*.png"):
            if p.is_file() and (p.stem == stem or p.name.startswith(stem)):
                return p
        return None

    pairs = []
    for tag, labels_dir, images_dir in datasets_cfg:
        roi_geojsons = sorted(labels_dir.glob("*.geojson"))
        for gj in roi_geojsons:
            stem = gj.stem
            print('!!! ', stem, gj.stem)
            img = find_image_for(stem, images_dir)
            if img is None:
                print(f"[SKIP {tag}] Missing PNG for {stem}")
                continue
            pairs.append((tag, gj, img))

    if not pairs:
        print("No matching (ROI, PNG) pairs found across datasets.")
        raise SystemExit(1)

    # Process cases (ROIs only)
    for tag, gj, img in pairs:
        try:
            cut_rois_png(
                dataset_tag=tag,
                roi_geojson_path=str(gj),
                png_path=str(img),
                out_root=str(out_root),
            )
        except Exception as e:
            print(f"[ERROR {tag}] {gj.stem}: {e}")
