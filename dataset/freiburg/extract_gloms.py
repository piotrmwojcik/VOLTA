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

# ---------- NEW helpers ----------
def _pick_label_field(gdf, explicit=None):
    """
    Choose which column holds the cell labels.
    Supports common names + 'classification' used by QuPath.
    """
    if explicit and explicit in gdf.columns:
        return explicit
    candidates = [
        "classification", "label", "type", "class", "cell_type",
        "category", "phenotype", "name", "Name", "objectType"
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    raise RuntimeError(f"No cell label column found. Available columns: {list(gdf.columns)}. "
                       f"Pass cell_label_field=...")

def _extract_label(value):
    """
    Turn a label field value (which may be a dict like {'name': 'Podocyte'})
    into a plain string label. Returns None if missing/invalid.
    """
    if value is None:
        return None
    # handle NaN
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    # dict-like (common in QuPath: classification = {'name': 'X', ...})
    if isinstance(value, dict):
        for key in ("name", "displayName", "label"):
            if key in value and value[key] not in (None, ""):
                return str(value[key])
        # fallback to str of dict
        return str(value)
    # plain string / int / etc.
    s = str(value).strip()
    return s if s else None

def _stable_rgb(label):
    """Deterministic RGB from label string (stable across runs)."""
    h = hashlib.md5(str(label).encode("utf-8")).digest()
    return (h[0], h[1], h[2])

# ---------- REPLACE your function with this version ----------
def cut_rectangles_multich_with_overlay(
    geojson_path,                # ROI polygons (glomeruli)
    cells_geojson_path,          # cell polygons (segmented cells) WITH labels
    ome_tiff_path,               # OME-TIFF with subdatasets
    output_dir,                  # where to save PNGs
    prefix=None,                 # filename prefix (defaults to TIFF stem)
    overlay_alpha=128,           # 0..255 transparency for the mask
    cell_label_field=None        # set explicit label column name if needed
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = Path(ome_tiff_path).stem

    # Load ROI (gloms) & cell polygons
    gdf = gpd.read_file(geojson_path)
    cells_gdf = gpd.read_file(cells_geojson_path)

    # Pick label column & build mapping
    label_col = _pick_label_field(cells_gdf, cell_label_field)
    derived_labels = [ _extract_label(v) for v in cells_gdf[label_col] ]
    cells_gdf = cells_gdf.assign(_label=derived_labels)
    # keep only rows with a usable label
    cells_gdf = cells_gdf[cells_gdf["_label"].notna()]

    all_labels = sorted(map(str, cells_gdf["_label"].dropna().unique()))
    label_to_id = {lab: i+1 for i, lab in enumerate(all_labels)}  # 0 = background
    label_to_rgb = {lab: _stable_rgb(lab) for lab in all_labels}
    print("  Label map:", label_to_id)

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

        for i, row in gdf.iterrows():
            glom_geom = row.geometry
            if glom_geom is None or glom_geom.is_empty:
                continue

            minx, miny, maxx, maxy = glom_geom.bounds
            cropped_list, used_transform, target_shape, win_for_cells = [], None, None, None

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
                        cand = cand[cand.geometry.intersects(glom_geom)]
                        shapes = []
                        for g, lab in zip(cand.geometry, cand["_label"]):
                            if g is None or g.is_empty:
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
                                merge_alg=MergeAlg.replace
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
                            for g, lab in zip(cand.geometry, cand["_label"]):
                                if g is None or g.is_empty:
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
                overlay_rgba[mask, 3] = overlay_alpha

            # Compose overlay on top of crop
            rgba = rgb_img.convert("RGBA")
            out_overlay = Image.alpha_composite(rgba, Image.fromarray(overlay_rgba, mode="RGBA"))

            # ---- Save PNGs ----
            crop_png = out_dir / f"{prefix}_polygon_{i:04d}.png"
            overlay_png = out_dir / f"{prefix}_polygon_{i:04d}_overlay.png"
            rgb_img.save(crop_png)
            out_overlay.convert("RGB").save(overlay_png)
            print(f"  Saved: {crop_png} and {overlay_png}")

    finally:
        for ds in datasets:
            ds.close()
