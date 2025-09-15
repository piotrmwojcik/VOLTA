#!/usr/bin/env python3
import json, hashlib, warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
from PIL import Image, ImageDraw
from PIL.Image import DecompressionBombWarning
from shapely.geometry import box
from shapely.affinity import translate

from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from affine import Affine

# Trust your images (huge PNGs)
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", DecompressionBombWarning)

IGNORE_LABELS = {"unclassified"}

CLASSES = [
    "_empty",       # 0
    "opal_480",     # 1
    "opal_520",     # 2
    "opal_570",     # 3
    "unclassified", # 4
    "opal_620",     # 5
]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}

def map_to_known_class(norm_lab: str) -> str:
    """Normalize → map unknown labels to 'unclassified' (except '_empty')."""
    if norm_lab in CLASSES:
        return norm_lab
    return "_empty" if norm_lab == "_empty" else "unclassified"

# ----------------- helpers -----------------
def extract_annotation_name(value):
    if value is None:
        return ""
    try:
        import numpy as _np
        if isinstance(value, float) and _np.isnan(value):
            return ""
    except Exception:
        pass
    if isinstance(value, dict):
        name = value.get("name") or value.get("displayName") or value.get("label") or ""
        return str(name).strip()
    return str(value).strip()

def normalize_label(lab: str) -> str:
    lab = (lab or "").strip().lower().replace(" ", "_")
    return lab if lab else "_empty"

def stable_rgb(label_str: str):
    # deterministic color from normalized label
    h = hashlib.md5(label_str.encode("utf-8")).digest()
    return (h[0], h[1], h[2])

def pretty_label(lab: str):
    return lab if lab != "" else "<empty>"

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
def cut_rectangles_png_with_overlay(
    dataset_tag,
    case_stem,
    roi_geojson_path,
    cells_geojson_path,
    png_path,
    out_root,
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
        x, y, w, h = clamp_xyxy_to_image(minx, miny, maxx, maxy, W_img, H_img)
        if w == 0 or h == 0:
            print(f"  ROI {i}: window out of bounds; skipping.")
            continue

        crop = img.crop((x, y, x + w, y + h))
        W, H = w, h

        # Binary raster (inside-ROI cells only)
        label_raster = np.zeros((H, W), dtype=np.uint8)
        bbox_records = []

        # For colored overlay without numeric ids, keep label->list_of_geoms in crop coords
        per_label_geoms = {}

        if cells_sindex is not None:
            # Shift glomerulus into crop coords, clip to crop box
            glom_shifted = translate(glom_geom, xoff=-x, yoff=-y).intersection(box(0, 0, W, H))
            if not glom_shifted.is_empty:
                # quick candidates by bbox; then STRICT containment (within)
                cand_idx = list(cells_sindex.intersection(glom_geom.bounds))
                if cand_idx:
                    cand = cells_gdf.iloc[cand_idx]
                    # strict containment in original (full-image) space
                    cand = cand[cand.geometry.centroid.within(glom_geom)]

                    for g, raw_lab in zip(cand.geometry, cand["_label"]):
                        if g is None or g.is_empty:
                            continue
                        # shift cell polygon to crop coords and clip to crop box
                        g2 = translate(g, xoff=-x, yoff=-y).intersection(box(0, 0, W, H))
                        if g2.is_empty or (not g2.within(glom_shifted)):
                            continue

                        lab = map_to_known_class(normalize_label(raw_lab))
                        if lab in IGNORE_LABELS:
                            continue  # <- skip unclassified entirely
                        per_label_geoms.setdefault(lab, []).append(g2)

                        # bbox in crop pixel space
                        gx0, gy0, gx1, gy1 = g2.bounds
                        bx, by, bw, bh = clamp_xyxy_to_image(gx0, gy0, gx1, gy1, W, H)
                        if bw > 0 and bh > 0:
                            bbox_records.append({
                                "bbox": [bx, by, bw, bh],
                                "label": lab,
                            })

        # ---- build binary mask (union of all per-label shapes) ----
        if per_label_geoms:
            shapes_with_values = [
                (geom, LABEL_TO_ID[label])
                for label, geoms in per_label_geoms.items()
                for geom in geoms
            ]
            type_raster = rasterize(
                shapes=shapes_with_values,
                out_shape=(H, W),
                transform=Affine.identity(),
                fill=0,  # background = "_empty"
                default_value=0,
                dtype="uint8",
                merge_alg=MergeAlg.replace
            )
        else:
            type_raster = np.zeros((H, W), dtype=np.uint8)

        # Optional: palette PNG for visualization (values remain class IDs 0..5)
        palette = [
            255, 255, 255,  # 0: _empty
            66, 135, 245,  # 1: opal_480
            40, 167, 69,  # 2: opal_520
            255, 193, 7,  # 3: opal_570
            111, 66, 193,  # 5: opal_620
        ]
        palette += [0, 0, 0] * (256 - len(CLASSES))  # pad

        type_mask_img = Image.fromarray(type_raster, mode="P")
        type_mask_img.putpalette(palette)

        # ---- colored overlay (per label, no numeric ids) ----
        overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        if per_label_geoms:
            for lab, geoms in per_label_geoms.items():
                tmp = rasterize(
                    shapes=[(g, 1) for g in geoms],
                    out_shape=(H, W),
                    transform=Affine.identity(),
                    fill=0,
                    default_value=0,
                    dtype="uint8",
                    merge_alg=MergeAlg.replace
                )
                m = tmp.astype(bool)
                if not np.any(m):
                    continue
                r, g_, b = stable_rgb(lab)
                overlay_rgba[m, 0] = r
                overlay_rgba[m, 1] = g_
                overlay_rgba[m, 2] = b
                overlay_rgba[m, 3] = overlay_alpha

        rgba_crop = crop.convert("RGBA")
        out_overlay = Image.alpha_composite(rgba_crop, Image.fromarray(overlay_rgba, mode="RGBA"))

        # Save outputs
        crop_png    = out_dir / f"{prefix}__polygon_{i:04d}.png"
        overlay_png = out_dir / f"{prefix}__polygon_{i:04d}__overlay.png"
        type_mask_png = out_dir / f"{prefix}__polygon_{i:04d}__type-mask.png"
        json_path   = out_dir / f"{prefix}__polygon_{i:04d}__bboxes.json"

        crop.save(crop_png)
        out_overlay.convert("RGB").save(overlay_png)
        type_mask_img.save(type_mask_png)

        with open(json_path, "w") as f:
            json.dump({
                "image": crop_png.name,
                "width": int(W),
                "height": int(H),
                "boxes": bbox_records
            }, f, indent=2)

        print(f"  Saved: {crop_png.name}, {overlay_png.name}, {type_mask_png.name}, {json_path.name}")


# ----------------- batch runner (PNG) -----------------
if __name__ == "__main__":
    out_root = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_new")
    out_root.mkdir(parents=True, exist_ok=True)

    datasets_cfg = [
        ("old",
         Path("/data/pwojcik/For_Piotr/new_labels/ROIs_geojson"),
         Path("/data/pwojcik/For_Piotr/new_labels/cells_labels_geojson"),
         Path("/data/pwojcik/For_Piotr/new_images")),
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

    def find_cells_for(stem: str, cells_dir: Path):
        cand = cells_dir / f"{stem}.geojson"
        if cand.exists():
            return cand
        for p in cells_dir.glob("*.geojson"):
            if p.stem == stem or p.name.startswith(stem):
                return p
        return None

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

    # FIRST PASS: collect global normalized labels (for legend only)
    global_labels = set()
    for tag, stem, gj, img, cells in pairs:
        cells_gdf = gpd.read_file(cells)
        if "classification" not in cells_gdf.columns:
            if not cells_gdf.empty:
                global_labels.add("_empty")
            continue
        for lab in cells_gdf["classification"].map(extract_annotation_name):
            norm = normalize_label("" if lab is None else str(lab))
            mapped = map_to_known_class(norm)
            if mapped in IGNORE_LABELS:
                continue  # <- do not show unclassified in legend
            global_labels.add(norm)

    global_labels = sorted(global_labels)
    print("Global labels:", [pretty_label(l) for l in global_labels])

    # Legend (normalized labels with deterministic colors)
    legend_png  = out_root / "legend_all_classes.png"
    legend_json = out_root / "label_map.json"
    try:
        sw, pad = 18, 8
        H_leg = pad*2 + (len(global_labels)+1)*sw
        W_leg = 640
        leg = Image.new("RGB", (W_leg, H_leg), (255, 255, 255))
        drw = ImageDraw.Draw(leg)
        y = pad
        drw.text((pad, y), "Legend (normalized label → color)", fill=(0,0,0))
        y += sw
        for lab in global_labels:
            color = stable_rgb(lab)
            drw.rectangle([pad, y, pad+sw, y+sw], fill=color)
            drw.text((pad+sw+8, y+2), pretty_label(lab), fill=(0,0,0))
            y += sw
        leg.save(legend_png)
        with open(legend_json, "w") as f:
            json.dump(
                {
                    "labels": global_labels,
                    "label_to_rgb": {lab: list(stable_rgb(lab)) for lab in global_labels},
                    "note": "colors are deterministic per normalized label"
                },
                f,
                indent=2
            )
        print(f"[LEGEND] Saved global legend: {legend_png}")
        print(f"[LEGEND] Saved mapping JSON: {legend_json}")
    except Exception as e:
        print(f"[WARN] Could not save global legend: {e}")

    # SECOND PASS: process cases
    for tag, stem, gj, img, cells in pairs:
        try:
            cut_rectangles_png_with_overlay(
                dataset_tag=tag,
                case_stem=stem,
                roi_geojson_path=str(gj),
                cells_geojson_path=str(cells),
                png_path=str(img),
                out_root=str(out_root),
                overlay_alpha=128
            )
        except Exception as e:
            print(f"[ERROR {tag}] {stem}: {e}")
