import re, hashlib
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
from PIL import Image


def _pick_label_field(gdf, explicit=None):
    if explicit and explicit in gdf.columns:
        return explicit
    candidates = ["label", "type", "class", "cell_type", "category", "phenotype", "name", "Name"]
    for c in candidates:
        if c in gdf.columns:
            return c
    raise RuntimeError(f"No cell label column found. Available columns: {list(gdf.columns)}. "
                       f"Pass cell_label_field=...")

def _stable_rgb(label):
    # Deterministic color from label string
    h = hashlib.md5(str(label).encode("utf-8")).digest()
    return (h[0], h[1], h[2])  # RGB 0..255


def cut_rectangles_multich_with_overlay(
    geojson_path,                 # ROI polygons (glomeruli)
    cells_geojson_path,           # cell polygons (segmented cells) WITH labels
    ome_tiff_path,                # OME-TIFF with subdatasets
    output_dir,                   # where to save PNGs
    prefix=None,                  # filename prefix (defaults to TIFF stem)
    overlay_alpha=128,            # 0..255 transparency for the mask
    cell_label_field=None         # set explicit label column name if needed
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = Path(ome_tiff_path).stem

    # Load ROI (gloms) & cell polygons
    gdf = gpd.read_file(geojson_path)
    cells_gdf = gpd.read_file(cells_geojson_path)
    label_col = _pick_label_field(cells_gdf, cell_label_field)

    # Get & sort subdatasets (one per channel)
    with rasterio.open(ome_tiff_path) as src0:
        subdatasets = src0.subdatasets
        if not subdatasets:
            raise RuntimeError(f"No subdatasets found in {ome_tiff_path}.")
    def _dir_index(s):
        m = re.search(r"GTIFF_DIR:(\d+):", s)
        return int(m.group(1)) if m else 0
    subdatasets = sorted(subdatasets, key=_dir_index)

    datasets = [rasterio.open(sref) for sref in subdatasets]
    try:
        first = datasets[0]
        print(f"\nProcessing {ome_tiff_path}")
        print("  Subdatasets:", len(datasets))
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

        # Build a global (deterministic) mapping label -> id/color
        all_labels = sorted(map(str, cells_gdf[label_col].dropna().unique()))
        label_to_id = {lab: i+1 for i, lab in enumerate(all_labels)}  # 0 reserved for background
        label_to_rgb = {lab: _stable_rgb(lab) for lab in all_labels}
        print("  Label map:", label_to_id)

        for i, row in gdf.iterrows():
            glom_geom = row.geometry
            if glom_geom is None or glom_geom.is_empty:
                continue

            minx, miny, maxx, maxy = glom_geom.bounds
            cropped_list = []
            used_transform = None
            target_shape = None
            win_for_cells = None

            # Read same bbox window from every subdataset
            for ds in datasets:
                if geo_mode:
                    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
                else:
                    col_off = int(max(0, np.floor(minx)))
                    row_off = int(max(0, np.floor(miny)))
                    width   = int(max(0, np.ceil(maxx - minx)))
                    height  = int(max(0, np.ceil(maxy - miny)))
                    win = Window(col_off=col_off, row_off=row_off, width=width, height=height)

                win = win.intersection(Window(0, 0, ds.width, ds.height))
                win = Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
                if win.width == 0 or win.height == 0:
                    print(f"  ROI {i}: window out of bounds; skipping.")
                    cropped_list = []
                    break

                arr = ds.read(window=win)  # (bands, H, W)
                if used_transform is None:
                    used_transform = ds.window_transform(win)
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

            # ---- RGB preview from first 3 channels (per-band min-max) ----
            if B >= 3:
                rgb = stacked[:3].astype(np.float32)
            else:
                rgb = np.vstack([stacked[:1]] * 3).astype(np.float32)
            for c in range(3):
                band = rgb[c]
                mn, mx = float(band.min()), float(band.max())
                rgb[c] = (band - mn) / (mx - mn) if mx > mn else 0.0
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            rgb = np.transpose(rgb, (1, 2, 0))  # (H,W,3)
            rgb_img = Image.fromarray(rgb, mode="RGB")

            # ---- Label raster inside glomerulus only (0=bg, >0=class id) ----
            label_raster = np.zeros((H, W), dtype=np.int32)

            if cells_sindex is not None:
                if geo_mode:
                    cand_idx = list(cells_sindex.intersection(glom_geom.bounds))
                    if cand_idx:
                        cand = cells_gdf.iloc[cand_idx]
                        # Only cells intersecting glomerulus
                        cand = cand[cand.geometry.intersects(glom_geom)]
                        shapes = []
                        for g, lab in zip(cand.geometry, cand[label_col]):
                            if g is None or g.is_empty or lab is None:
                                continue
                            gi = g.intersection(glom_geom)
                            if gi.is_empty:
                                continue
                            val = label_to_id[str(lab)]
                            shapes.append((gi, val))
                        if shapes:
                            label_raster = rasterize(
                                shapes=shapes,
                                out_shape=(H, W),
                                transform=used_transform,
                                fill=0,
                                default_value=0,
                                dtype="int32",
                                merge_alg=MergeAlg.replace  # last wins if overlaps
                            )
                else:
                    # Pixel mode: translate to crop-local px, clip to window & glom
                    col0, row0 = int(win_for_cells.col_off), int(win_for_cells.row_off)
                    glom_shifted = translate(glom_geom, xoff=-col0, yoff=-row0).intersection(box(0, 0, W, H))
                    if not glom_shifted.is_empty:
                        cand_idx = list(cells_sindex.intersection(glom_geom.bounds))
                        if cand_idx:
                            cand = cells_gdf.iloc[cand_idx]
                            cand = cand[cand.geometry.intersects(glom_geom)]
                            shapes = []
                            for g, lab in zip(cand.geometry, cand[label_col]):
                                if g is None or g.is_empty or lab is None:
                                    continue
                                g2 = translate(g, xoff=-col0, yoff=-row0).intersection(box(0, 0, W, H))
                                gi = g2.intersection(glom_shifted)
                                if gi.is_empty:
                                    continue
                                val = label_to_id[str(lab)]
                                shapes.append((gi, val))
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

            # ---- Build colored overlay from label_raster ----
            overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            for lab, val in label_to_id.items():
                mask = (label_raster == val)
                if not np.any(mask):
                    continue
                r, g, b = label_to_rgb[lab]
                overlay_rgba[mask, 0] = r
                overlay_rgba[mask, 1] = g
                overlay_rgba[mask, 2] = b
                overlay_rgba[mask, 3] = overlay_alpha  # same alpha for all classes

            # Compose overlay on top of crop
            rgba = rgb_img.convert("RGBA")
            out_overlay = Image.alpha_composite(rgba, Image.fromarray(overlay_rgba, mode="RGBA"))

            # ---- Save PNGs ----
            crop_png = out_dir / f"{prefix}_polygon_{i:04d}.png"
            overlay_png = out_dir / f"{prefix}_polygon_{i:04d}_overlay.png"
            rgb_img.save(crop_png)
            out_overlay.convert("RGB").save(overlay_png)

            # (Optional) also save a legend image
            legend_png = out_dir / f"{prefix}_polygon_{i:04d}_legend.png"
            try:
                # tiny legend image
                swatch_h = 20
                pad = 6
                H_legend = swatch_h * (len(label_to_id) + 1) + pad * 2
                W_legend = 480
                legend = Image.new("RGB", (W_legend, H_legend), (255, 255, 255))
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(legend)
                y = pad
                draw.text((pad, y), "Legend (label → color)", fill=(0, 0, 0))
                y += swatch_h
                for lab in sorted(label_to_id.keys()):
                    color = label_to_rgb[lab]
                    draw.rectangle([pad, y, pad + swatch_h, y + swatch_h], fill=color)
                    draw.text((pad + swatch_h + 8, y + 2), str(lab), fill=(0, 0, 0))
                    y += swatch_h
                legend.save(legend_png)
            except Exception:
                pass

            print(f"  Saved: {crop_png} and {overlay_png}")

    finally:
        for ds in datasets:
            ds.close()


# ---- Batch runner ----
if __name__ == "__main__":
    # Folders
    labels_dir = Path("/data/pwojcik/For_Piotr/Labels/glom_labels")    # ROI polygons
    cells_dir  = Path("/data/pwojcik/For_Piotr/Labels/cells_labels")    # cell polygons (with labels)
    images_dir = Path("/data/pwojcik/For_Piotr/Images")
    out_root   = Path("/data/pwojcik/For_Piotr/gloms_rect")

    # Allowed image extensions (preference order)
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

    geojsons = sorted(labels_dir.glob("*.geojson"))
    if not geojsons:
        print(f"No .geojson files found in {labels_dir}")
        raise SystemExit(1)

    for gj in geojsons:
        stem = gj.stem
        img = find_image_for(stem)
        cells = find_cells_for(stem)

        if img is None:
            print(f"[SKIP] No matching image for {stem}")
            continue
        if cells is None:
            print(f"[SKIP] No matching cells GeoJSON for {stem}")
            continue

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
                prefix=img.stem,              # name crops after original image
                overlay_alpha=128,            # semi-transparent
                cell_label_field=None         # set to your column name if auto-detect fails
            )
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
