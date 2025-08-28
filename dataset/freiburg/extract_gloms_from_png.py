#!/usr/bin/env python3
import json, re, hashlib, warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import warnings
from PIL import Image, ImageDraw
from PIL.Image import DecompressionBombWarning  # <-- correct place
from shapely.geometry import box
from shapely.affinity import translate

# You no longer need rasterio-open; we only keep Affine.identity for rasterize alternative
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from affine import Affine

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", DecompressionBombWarning)

# ----------------- helpers -----------------
def extract_annotation_name(value):
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except Exception:
        pass
    if isinstance(value, dict):
        name = value.get("name") or value.get("displayName") or value.get("label") or ""
        return str(name).strip()
    return str(value).strip()

def stable_rgb(label_str: str):
    h = hashlib.md5(label_str.encode("utf-8")).digest()
    return (h[0], h[1], h[2])

def pretty_label(lab: str):
    return lab if lab != "" else "<empty>"

def clamp_xyxy_to_image(x0, y0, x1, y1, W, H):
    x0 = max(0.0, min(float(x0), W))
    y0 = max(0.0, min(float(y0), H))
    x1 = max(0.0, min(float(x1), W))
    y1 = max(0.0, min(float(y1), H))
    # ensure ordering
    x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
    y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)
    # integer pixel box (inclusive-exclusive)
    x_min_i = int(np.floor(x_min))
    y_min_i = int(np.floor(y_min))
    x_max_i = int(np.ceil(x_max))
    y_max_i = int(np.ceil(y_max))
    # clamp again after rounding
    x_min_i = max(0, min(x_min_i, W))
    y_min_i = max(0, min(y_min_i, H))
    x_max_i = max(0, min(x_max_i, W))
    y_max_i = max(0, min(y_max_i, H))
    w = max(0, x_max_i - x_min_i)
    h = max(0, y_max_i - y_min_i)
    return x_min_i, y_min_i, w, h


# ----------------- core (PNG) -----------------
def cut_rectangles_png_with_overlay(
    dataset_tag,                # 'old' or 'new' (used in filename)
    case_stem,                  # stem of the case (kept for compatibility)
    roi_geojson_path,           # ROI polygons (glomeruli) in pixel coords
    cells_geojson_path,         # cells polygons (with 'classification') in pixel coords
    png_path,                   # PNG path
    out_root,                   # root output dir (single folder for all images)
    label_to_id,                # GLOBAL mapping (mutable dict)
    label_to_rgb,               # GLOBAL colors (mutable dict)
    overlay_alpha=128
):
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_stem = Path(png_path).stem
    prefix = f"{dataset_tag}__{png_stem}"

    # Load ROIs & cells (pixel space)
    gdf = gpd.read_file(roi_geojson_path)
    cells_gdf = gpd.read_file(cells_geojson_path)

    if "classification" not in cells_gdf.columns:
        print(f"[WARN] 'classification' column not found in {cells_geojson_path}; using empty class.")
        cells_gdf = cells_gdf.assign(_label="")
    else:
        cells_gdf = cells_gdf.assign(_label=cells_gdf["classification"].map(extract_annotation_name))

    # Load PNG
    img = Image.open(png_path).convert("RGB")
    W_img, H_img = img.size

    cells_sindex = cells_gdf.sindex if not cells_gdf.empty else None

    print(f"\n[{dataset_tag}] Processing {png_stem} ({W_img}x{H_img})")

    for i, row in gdf.iterrows():
        glom_geom = row.geometry
        if glom_geom is None or glom_geom.is_empty:
            continue

        minx, miny, maxx, maxy = glom_geom.bounds
        # compute crop window in pixel coords, clamped to image
        x, y, w, h = clamp_xyxy_to_image(minx, miny, maxx, maxy, W_img, H_img)
        if w == 0 or h == 0:
            print(f"  ROI {i}: window out of bounds; skipping.")
            continue

        crop = img.crop((x, y, x + w, y + h))  # PIL (w,h)
        W, H = w, h

        # Prepare raster target & bbox collector
        label_raster = np.zeros((H, W), dtype=np.int32)
        bbox_records = []

        # Compute shapes for rasterization within this ROI
        if cells_sindex is not None:
            # shift glomerulus into crop coord space, intersect with crop box
            glom_shifted = translate(glom_geom, xoff=-x, yoff=-y).intersection(box(0, 0, W, H))
            if not glom_shifted.is_empty:
                # quick spatial filter using original bounds (in full image pixel coords)
                cand_idx = list(cells_sindex.intersection(glom_geom.bounds))
                if cand_idx:
                    cand = cells_gdf.iloc[cand_idx]
                    cand = cand[cand.geometry.intersects(glom_geom)]
                    shapes = []
                    for g, lab in zip(cand.geometry, cand["_label"]):
                        if g is None or g.is_empty:
                            continue
                        # shift to crop space and clip to crop & glom
                        g2 = translate(g, xoff=-x, yoff=-y).intersection(box(0, 0, W, H))
                        gi = g2.intersection(glom_shifted)
                        if gi.is_empty:
                            continue
                        if lab not in label_to_id:
                            new_id = max(label_to_id.values(), default=0) + 1
                            label_to_id[lab] = new_id
                            label_to_rgb[lab] = stable_rgb(lab)
                        shapes.append((gi, label_to_id[lab]))
                        # bbox in crop pixel space
                        gx0, gy0, gx1, gy1 = gi.bounds
                        bx, by, bw, bh = clamp_xyxy_to_image(gx0, gy0, gx1, gy1, W, H)
                        if bw > 0 and bh > 0:
                            bbox_records.append({
                                "bbox": [bx, by, bw, bh],
                                "label": lab,
                                "label_id": label_to_id[lab],
                            })
                    if shapes:
                        label_raster = rasterize(
                            shapes=shapes,
                            out_shape=(H, W),
                            transform=Affine.identity(),  # pixel space
                            fill=0,
                            default_value=0,
                            dtype="int32",
                            merge_alg=MergeAlg.replace
                        )

        # ---- binary mask: cells white (255), background black (0) ----
        bin_mask = (label_raster > 0).astype(np.uint8) * 255  # (H,W), uint8
        mask_img = Image.fromarray(bin_mask, mode="L")

        # ---- colored overlay from label_raster ----
        overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        for lab, val in label_to_id.items():
            m = (label_raster == val)
            if not np.any(m):
                continue
            r, g, b = label_to_rgb[lab]
            overlay_rgba[m, 0] = r
            overlay_rgba[m, 1] = g
            overlay_rgba[m, 2] = b
            overlay_rgba[m, 3] = overlay_alpha

        rgba = crop.convert("RGBA")
        out_overlay = Image.fromarray(overlay_rgba, mode="RGBA")
        out_overlay = Image.alpha_composite(rgba, out_overlay)

        # Save outputs (single folder, unique names)
        crop_png    = out_dir / f"{prefix}__polygon_{i:04d}.png"
        overlay_png = out_dir / f"{prefix}__polygon_{i:04d}__overlay.png"
        mask_png    = out_dir / f"{prefix}__polygon_{i:04d}__mask.png"
        json_path   = out_dir / f"{prefix}__polygon_{i:04d}__bboxes.json"

        crop.save(crop_png)
        out_overlay.convert("RGB").save(overlay_png)
        mask_img.save(mask_png)

        with open(json_path, "w") as f:
            json.dump({
                "image": crop_png.name,
                "width": int(W),
                "height": int(H),
                "boxes": bbox_records
            }, f, indent=2)

        print(f"  Saved: {crop_png.name}, {overlay_png.name}, {mask_png.name}, {json_path.name}")


# ----------------- batch runner (PNG) -----------------
if __name__ == "__main__":
    # Output root (shared single folder)
    out_root = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png")
    out_root.mkdir(parents=True, exist_ok=True)

    # Dataset groups: (tag, labels_dir, cells_dir, images_dir)
    datasets_cfg = [
        ("old",
         Path("/data/pwojcik/For_Piotr/new_labels/ROIs_geojson"),
         Path("/data/pwojcik/For_Piotr/new_labels/cells_labels_geojson"),
         Path("/data/pwojcik/For_Piotr/new_images")),  # this folder now holds PNGs
        # add more tuples if needed
    ]

    # Allowed image extensions (PNG only now)
    exts = [".png"]

    def find_image_for(stem: str, images_dir: Path):
        # exact match first
        for ext in exts:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        # loose match (e.g., stem prefixes file)
        for p in images_dir.rglob("*.png"):
            if p.is_file() and (p.stem == stem or p.name.startswith(stem)):
                return p
        return None

    def find_cells_for(stem: str, cells_dir: Path):
        cand = cells_dir / f"{stem}.geojson"
        if cand.exists():
            return cand
        for p in cells_dir.glob("*.geojson"):
            if p.stem == stem or p.name.startswith(stem):
                return p
        return None

    # Build list of (dataset_tag, stem, roi_geojson, png, cells)
    pairs = []
    for tag, labels_dir, cells_dir, images_dir in datasets_cfg:
        roi_geojsons = sorted(labels_dir.glob("*.geojson"))
        for gj in roi_geojsons:
            stem = gj.stem
            img = find_image_for(stem, images_dir)
            cells = find_cells_for(stem, cells_dir)
            if img is None or cells is None:
                print(f"[SKIP {tag}] Missing PNG or cells for {stem}")
                continue
            pairs.append((tag, stem, gj, img, cells))

    if not pairs:
        print("No matching (ROI, PNG, cells) triplets found across datasets.")
        raise SystemExit(1)

    # ---------- FIRST PASS: collect ALL labels globally ----------
    global_labels = set()
    for tag, stem, gj, img, cells in pairs:
        cells_gdf = gpd.read_file(cells)
        if "classification" not in cells_gdf.columns:
            if not cells_gdf.empty:
                global_labels.add("")
            continue
        labs = cells_gdf["classification"].map(extract_annotation_name)
        for lab in labs:
            global_labels.add("" if lab is None else str(lab))

    global_labels = sorted(global_labels)
    label_to_id  = {lab: i + 1 for i, lab in enumerate(global_labels)}  # 0 = background
    label_to_rgb = {lab: stable_rgb(lab) for lab in global_labels}
    print("Global labels:", [pretty_label(l) for l in global_labels])

    # Save a SINGLE legend + JSON right now (in out_root)
    legend_png  = out_root / "legend_all_classes.png"
    legend_json = out_root / "label_map.json"

    try:
        sw, pad = 18, 8
        H_leg = pad*2 + (len(global_labels)+1)*sw
        W_leg = 640
        leg = Image.new("RGB", (W_leg, H_leg), (255, 255, 255))
        drw = ImageDraw.Draw(leg)
        y = pad
        drw.text((pad, y), "Legend (annotation name → color)", fill=(0,0,0))
        y += sw
        for lab in global_labels:
            color = label_to_rgb[lab]
            drw.rectangle([pad, y, pad+sw, y+sw], fill=color)
            drw.text((pad+sw+8, y+2), pretty_label(lab), fill=(0,0,0))
            y += sw
        leg.save(legend_png)
        with open(legend_json, "w") as f:
            json.dump(
                {
                    "labels": global_labels,
                    "label_to_id": label_to_id,
                    "label_to_rgb": {k: list(v) for k, v in label_to_rgb.items()},
                    "note": "0 = background; colors are deterministic per label"
                },
                f,
                indent=2
            )
        print(f"[LEGEND] Saved global legend: {legend_png}")
        print(f"[LEGEND] Saved mapping JSON: {legend_json}")
    except Exception as e:
        print(f"[WARN] Could not save global legend: {e}")

    # ---------- SECOND PASS: process all cases using GLOBAL mappings ----------
    mapping_len_before = len(label_to_id)
    for tag, stem, gj, img, cells in pairs:
        try:
            cut_rectangles_png_with_overlay(
                dataset_tag=tag,
                case_stem=stem,
                roi_geojson_path=str(gj),
                cells_geojson_path=str(cells),
                png_path=str(img),
                out_root=str(out_root),     # single folder
                label_to_id=label_to_id,
                label_to_rgb=label_to_rgb,
                overlay_alpha=128
            )
        except Exception as e:
            print(f"[ERROR {tag}] {stem}: {e}")

    # If new labels appeared mid-run, refresh legend to include them
    if len(label_to_id) != mapping_len_before:
        global_labels = sorted(label_to_id.keys())
        try:
            sw, pad = 18, 8
            H_leg = pad*2 + (len(global_labels)+1)*sw
            W_leg = 640
            leg = Image.new("RGB", (W_leg, H_leg), (255, 255, 255))
            drw = ImageDraw.Draw(leg)
            y = pad
            drw.text((pad, y), "Legend (annotation name → color)", fill=(0,0,0))
            y += sw
            for lab in global_labels:
                color = label_to_rgb[lab]
                drw.rectangle([pad, y, pad+sw, y+sw], fill=color)
                drw.text((pad+sw+8, y+2), pretty_label(lab), fill=(0,0,0))
                y += sw
            leg.save(legend_png)
            with open(legend_json, "w") as f:
                json.dump(
                    {
                        "labels": global_labels,
                        "label_to_id": label_to_id,
                        "label_to_rgb": {k: list(v) for k, v in label_to_rgb.items()},
                        "note": "0 = background; colors are deterministic per label"
                    },
                    f,
                    indent=2
                )
            print(f"[LEGEND] Updated global legend with newly seen labels.")
        except Exception as e:
            print(f"[WARN] Could not update legend: {e}")
