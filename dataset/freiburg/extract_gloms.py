#!/usr/bin/env python3
import json, re, hashlib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from affine import Affine
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import translate
from PIL import Image, ImageDraw, ImageFont


# ----------------- helpers -----------------
def extract_annotation_name(value):
    """
    Return the annotation name string from a 'classification' field.
    Accepts:
      - dict like {'name': 'Podocyte', ...}
      - plain string ('Podocyte')
      - None/NaN -> ''  (treated as a real class)
    """
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
    """Deterministic RGB color for a given label string (including empty)."""
    h = hashlib.md5(label_str.encode("utf-8")).digest()
    return (h[0], h[1], h[2])


def pretty_label(lab: str):
    return lab if lab != "" else "<empty>"


# ----------------- core function -----------------
def cut_rectangles_multich_with_overlay(
    geojson_path,                # ROI polygons (glomeruli)
    cells_geojson_path,          # cells polygons (with 'classification' column)
    ome_tiff_path,               # OME-TIFF path with GTIFF_DIR subdatasets
    output_dir,                  # where to write PNGs
    prefix=None,                 # filename prefix (defaults to TIFF stem)
    overlay_alpha=128,           # 0..255 transparency for class overlay
    label_to_id=None,            # GLOBAL mapping label->id (required for single legend)
    label_to_rgb=None            # GLOBAL mapping label->RGB (same across all images)
):
    assert label_to_id is not None and label_to_rgb is not None, "Pass global label mappings"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = Path(ome_tiff_path).stem

    # Load ROI & cells
    gdf = gpd.read_file(geojson_path)
    cells_gdf = gpd.read_file(cells_geojson_path)

    if "classification" not in cells_gdf.columns:
        raise RuntimeError(
            f"'classification' column not found in {cells_geojson_path}. "
            f"Columns: {list(cells_gdf.columns)}"
        )

    # Derive annotation name; keep empty names as their own class ('')
    cells_gdf = cells_gdf.assign(_label=cells_gdf["classification"].map(extract_annotation_name))

    # Open subdatasets (channels)
    with rasterio.open(ome_tiff_path) as src0:
        subdatasets = src0.subdatasets
        if not subdatasets:
            raise RuntimeError(f"No subdatasets found in {ome_tiff_path}. Is this an OME-TIFF?")
    def dir_index(s):
        m = re.search(r"GTIFF_DIR:(\d+):", s)
        return int(m.group(1)) if m else 0
    subdatasets = sorted(subdatasets, key=dir_index)
    datasets = [rasterio.open(sref) for sref in subdatasets]

    try:
        first = datasets[0]
        print(f"\nProcessing image: {ome_tiff_path}")
        print("  Channels (subdatasets):", len(datasets))
        print("  CRS (first sds):", first.crs)

        # GEO vs PIXEL mode
        geo_mode = (first.crs is not None) and (gdf.crs is not None)

        # Reproject labels in GEO mode
        if geo_mode:
            if gdf.crs != first.crs:
                gdf = gdf.to_crs(first.crs)
                print(f"  Reprojected ROI GeoJSON to {first.crs}")
            if cells_gdf.crs is not None and cells_gdf.crs != first.crs:
                cells_gdf = cells_gdf.to_crs(first.crs)
                print(f"  Reprojected Cells GeoJSON to {first.crs}")

        # Spatial index for cells
        cells_sindex = cells_gdf.sindex if not cells_gdf.empty else None

        for i, row in gdf.iterrows():
            glom_geom = row.geometry
            if glom_geom is None or glom_geom.is_empty:
                continue

            minx, miny, maxx, maxy = glom_geom.bounds
            cropped_list = []
            used_transform = None
            target_shape = None
            win_for_cells = None

            # Read same window from each channel
            for ds in datasets:
                if geo_mode:
                    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
                else:
                    col_off = int(max(0, np.floor(minx)))
                    row_off = int(max(0, np.floor(miny)))
                    width   = int(max(0, np.ceil(maxx - minx)))
                    height  = int(max(0, np.ceil(maxy - miny)))
                    win = Window(col_off=col_off, row_off=row_off, width=width, height=height)

                # clip & int
                win = win.intersection(Window(0, 0, ds.width, ds.height))
                win = Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
                if win.width == 0 or win.height == 0:
                    print(f"  ROI {i}: window out of bounds; skipping.")
                    cropped_list = []
                    break

                arr = ds.read(window=win)  # (bands, H, W) — usually (1, H, W)
                if used_transform is None:
                    # Avoid georeferencing warnings in pixel mode
                    used_transform = ds.window_transform(win) if geo_mode else Affine.identity()
                    target_shape = arr.shape[1:]
                    win_for_cells = win
                else:
                    if arr.shape[1:] != target_shape:
                        raise RuntimeError(f"  ROI {i}: crop size mismatch across subdatasets.")
                cropped_list.append(arr)

            if not cropped_list:
                continue

            stacked = np.concatenate(cropped_list, axis=0)  # (B, H, W)
            B, H, W = stacked.shape

            # Build RGB preview from first 3 channels (per-band min-max)
            if B >= 3:
                rgb = stacked[:3].astype(np.float32)
            else:
                rgb = np.vstack([stacked[:1]] * 3).astype(np.float32)
            for c in range(3):
                band = rgb[c]
                mn, mx = float(band.min()), float(band.max())
                rgb[c] = (band - mn) / (mx - mn) if mx > mn else 0.0
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
            rgb_img = Image.fromarray(rgb, mode="RGB")

            # Raster of class ids (0 bg), **inside glomerulus only**
            label_raster = np.zeros((H, W), dtype=np.int32)

            if cells_sindex is not None:
                if geo_mode:
                    # Candidate cells by bbox, keep those intersecting glomerulus
                    cand_idx = list(cells_sindex.intersection(glom_geom.bounds))
                    if cand_idx:
                        cand = cells_gdf.iloc[cand_idx]
                        cand = cand[cand.geometry.intersects(glom_geom)]
                        shapes = []
                        for g, lab in zip(cand.geometry, cand["_label"]):
                            if g is None or g.is_empty:
                                continue
                            gi = g.intersection(glom_geom)
                            if gi.is_empty:
                                continue
                            if lab not in label_to_id:
                                # Unknown label (not seen globally) -> add as new id on the fly
                                new_id = max(label_to_id.values(), default=0) + 1
                                label_to_id[lab] = new_id
                                label_to_rgb[lab] = stable_rgb(lab)
                            shapes.append((gi, label_to_id[lab]))
                        if shapes:
                            label_raster = rasterize(
                                shapes=shapes,
                                out_shape=(H, W),
                                transform=used_transform,
                                fill=0,
                                default_value=0,
                                dtype="int32",
                                merge_alg=MergeAlg.replace  # last wins on overlaps
                            )
                else:
                    # Pixel mode: translate polygons into crop-local px coords
                    col0, row0 = int(win_for_cells.col_off), int(win_for_cells.row_off)
                    glom_shifted = translate(glom_geom, xoff=-col0, yoff=-row0).intersection(box(0, 0, W, H))
                    if not glom_shifted.is_empty:
                        cand_idx = list(cells_sindex.intersection(glom_geom.bounds))
                        if cand_idx:
                            cand = cells_gdf.iloc[cand_idx]
                            cand = cand[cand.geometry.intersects(glom_geom)]
                            shapes = []
                            for g, lab in zip(cand.geometry, cand["_label"]):
                                if g is None or g.is_empty:
                                    continue
                                g2 = translate(g, xoff=-col0, yoff=-row0).intersection(box(0, 0, W, H))
                                gi = g2.intersection(glom_shifted)
                                if gi.is_empty:
                                    continue
                                if lab not in label_to_id:
                                    new_id = max(label_to_id.values(), default=0) + 1
                                    label_to_id[lab] = new_id
                                    label_to_rgb[lab] = stable_rgb(lab)
                                shapes.append((gi, label_to_id[lab]))
                            if shapes:
                                label_raster = rasterize(
                                    shapes=shapes,
                                    out_shape=(H, W),
                                    transform=Affine.identity(),
                                    fill=0,
                                    default_value=0,
                                    dtype="int32",
                                    merge_alg=MergeAlg.replace
                                )

            # Build colored overlay from label_raster (global colors)
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

            # Compose and save
            rgba = rgb_img.convert("RGBA")
            out_overlay = Image.alpha_composite(rgba, Image.fromarray(overlay_rgba, mode="RGBA"))

            crop_png    = out_dir / f"{prefix}_polygon_{i:04d}.png"
            overlay_png = out_dir / f"{prefix}_polygon_{i:04d}_overlay.png"
            rgb_img.save(crop_png)
            out_overlay.convert("RGB").save(overlay_png)
            print(f"  Saved: {crop_png} and {overlay_png}")

    finally:
        for ds in datasets:
            ds.close()


# ----------------- batch runner with GLOBAL legend -----------------
if __name__ == "__main__":
    # Paths
    labels_dir = Path("/data/pwojcik/For_Piotr/Labels/glom_labels")   # ROI polygons (one .geojson per case)
    cells_dir  = Path("/data/pwojcik/For_Piotr/Labels/cells_labels")   # cells polygons (with 'classification')
    images_dir = Path("/data/pwojcik/For_Piotr/Images")               # OME-TIFFs
    out_root   = Path("/data/pwojcik/For_Piotr/gloms_rect")           # output root
    out_root.mkdir(parents=True, exist_ok=True)

    # Allowed image extensions in order of preference
    exts = [".ome.tiff", ".ome.tif", ".tif", ".tiff"]

    def find_image_for(stem: str) -> Path:
        for ext in exts:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        for p in images_dir.rglob("*"):
            if p.is_file() and any(str(p.name).lower().endswith(ext) for ext in exts):
                if p.stem == stem or p.name.startswith(stem):
                    return p
        return None

    def find_cells_for(stem: str) -> Path:
        cand = cells_dir / f"{stem}.geojson"
        if cand.exists():
            return cand
        for p in cells_dir.glob("*.geojson"):
            if p.stem == stem or p.name.startswith(stem):
                return p
        return None

    # Build list of matching cases
    roi_geojsons = sorted(labels_dir.glob("*.geojson"))
    pairs = []
    for gj in roi_geojsons:
        stem = gj.stem
        img = find_image_for(stem)
        cells = find_cells_for(stem)
        if img is None or cells is None:
            print(f"[SKIP] Missing image or cells for {stem}")
            continue
        pairs.append((stem, gj, img, cells))
    if not pairs:
        print("No matching (ROI, image, cells) triplets found.")
        raise SystemExit(1)

    # ---------- FIRST PASS: collect ALL labels across all cases ----------
    global_labels = set()
    for stem, gj, img, cells in pairs:
        cells_gdf = gpd.read_file(cells)
        if "classification" not in cells_gdf.columns:
            print(f"[WARN] 'classification' column missing in {cells}; skipping label collection for this file.")
            continue
        labs = cells_gdf["classification"].map(extract_annotation_name)
        for lab in labs:
            global_labels.add("" if lab is None else str(lab))

    # Build GLOBAL mappings (sorted for stability)
    global_labels = sorted(global_labels)
    label_to_id  = {lab: i + 1 for i, lab in enumerate(global_labels)}  # 0 = background
    label_to_rgb = {lab: stable_rgb(lab) for lab in global_labels}
    print("Global labels:", [pretty_label(l) for l in global_labels])

    # Save a SINGLE legend for all images
    legend_png = out_root / "legend_all_classes.png"
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
    for stem, gj, img, cells in pairs:
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing: {stem} ===")
        print(f"ROI GeoJSON:   {gj}")
        print(f"Cells GeoJSON: {cells}")
        print(f"Image:         {img}")
        print(f"Output:        {out_dir}")

        try:
            cut_rectangles_multich_with_overlay(
                geojson_path=str(gj),
                cells_geojson_path=str(cells),
                ome_tiff_path=str(img),
                output_dir=str(out_dir),
                prefix=img.stem,                 # name crops after original image
                overlay_alpha=128,               # semi-transparent
                label_to_id=label_to_id,         # GLOBAL mapping
                label_to_rgb=label_to_rgb        # GLOBAL colors
            )
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
