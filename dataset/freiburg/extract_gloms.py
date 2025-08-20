import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import rasterize
from affine import Affine
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import translate
from PIL import Image


def cut_rectangles_multich_with_cells(
    geojson_path,                # polygons of ROIs to crop (e.g., glomeruli)
    cells_geojson_path,          # polygons for segmented cells
    ome_tiff_path,               # OME-TIFF (multi subdatasets)
    output_dir,                  # output folder
    prefix=None                  # filename prefix; defaults to TIFF stem
):
    """
    For each ROI polygon in `geojson_path`, take its bounding box, crop ALL subdatasets
    (channels) from the OME-TIFF, and also create a boolean cell mask selecting cells
    from `cells_geojson_path` that intersect the crop.

    Saves per ROI:
      - <prefix>_polygon_XXXX.tif   (multi-band GeoTIFF with all channels)
      - <prefix>_polygon_XXXX.png   (RGB quick-look from first 3 bands)
      - <prefix>_polygon_XXXX_mask.tif (single-band 0/1 mask, same size as crop)
      - <prefix>_polygon_XXXX_mask.png (quick-look mask)

    If the image has a CRS and the GeoJSONs have a CRS, polygons are reprojected to the image CRS.
    If the image has no CRS, polygon coordinates are assumed to be in pixel space (x=col, y=row).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = Path(ome_tiff_path).stem

    # Load ROI polygons (e.g., glomeruli) and cell polygons
    gdf = gpd.read_file(geojson_path)
    cells_gdf = gpd.read_file(cells_geojson_path)

    # Enumerate & sort subdatasets by GTIFF_DIR index
    with rasterio.open(ome_tiff_path) as src0:
        subdatasets = src0.subdatasets
        if not subdatasets:
            raise RuntimeError(f"No subdatasets found in {ome_tiff_path}. Is this an OME-TIFF?")
    def _dir_index(s):  # GTIFF_DIR:<n>:<path>
        m = re.search(r"GTIFF_DIR:(\d+):", s)
        return int(m.group(1)) if m else 0
    subdatasets = sorted(subdatasets, key=_dir_index)

    # Open all subdatasets
    datasets = [rasterio.open(sref) for sref in subdatasets]

    try:
        first = datasets[0]
        print(f"\nProcessing {ome_tiff_path}")
        print("  Subdatasets:", len(datasets))
        print("  CRS (first sds):", first.crs)
        print("  Sizes (w x h):", [(ds.width, ds.height) for ds in datasets])

        # GEO mode if both image & labels (at least ROIs) have CRS
        geo_mode = (first.crs is not None) and (gdf.crs is not None)

        # Reproject ROIs and Cells to raster CRS (geo mode)
        if geo_mode:
            if gdf.crs != first.crs:
                gdf = gdf.to_crs(first.crs)
                print(f"  Reprojected ROI GeoJSON to raster CRS {first.crs}")
            if cells_gdf.crs is not None and cells_gdf.crs != first.crs:
                cells_gdf = cells_gdf.to_crs(first.crs)
                print(f"  Reprojected Cells GeoJSON to raster CRS {first.crs}")
        else:
            # In pixel mode, we assume both ROI & Cells are in pixel coordinates.
            # If your cells GeoJSON is in a different coordinate system, convert beforehand.
            pass

        # Spatial index on cells for quick candidate lookup
        if not cells_gdf.empty:
            cells_sindex = cells_gdf.sindex
        else:
            cells_sindex = None

        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Bounding box of the ROI polygon
            minx, miny, maxx, maxy = geom.bounds

            cropped_list = []
            used_transform = None
            target_shape = None
            win_for_cells = None  # keep Window for pixel-mode cell translation

            for ds in datasets:
                # Build window either in GEO mode (bounds+transform) or PIXEL mode (bounds as pixels)
                if geo_mode:
                    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
                else:
                    # bounds are pixel coords: x->col, y->row (origin top-left)
                    col_off = int(max(0, np.floor(minx)))
                    row_off = int(max(0, np.floor(miny)))
                    width   = int(max(0, np.ceil(maxx - minx)))
                    height  = int(max(0, np.ceil(maxy - miny)))
                    win = Window(col_off=col_off, row_off=row_off, width=width, height=height)

                # Clip to dataset extent & enforce ints
                win = win.intersection(Window(0, 0, ds.width, ds.height))
                win = Window(
                    col_off=int(win.col_off), row_off=int(win.row_off),
                    width=int(win.width), height=int(win.height)
                )

                if win.width == 0 or win.height == 0:
                    print(f"  ROI {i}: window out of bounds for a subdataset, skipping ROI.")
                    cropped_list = []
                    break

                # Read rectangular window (no mask)
                out_img = ds.read(window=win)  # shape: (bands, h, w)

                if used_transform is None:
                    used_transform = ds.window_transform(win)
                    target_shape = out_img.shape[1:]  # (h, w)
                    win_for_cells = win
                else:
                    if out_img.shape[1:] != target_shape:
                        raise RuntimeError(
                            f"  Mismatch in crop sizes between subdatasets for ROI {i}: "
                            f"{out_img.shape[1:]} vs {target_shape}"
                        )

                cropped_list.append(out_img)

            if not cropped_list:
                continue

            # Stack ALL bands across subdatasets -> (B_total, H, W)
            stacked = np.concatenate(cropped_list, axis=0)
            B, H, W = stacked.shape
            print(f"  ROI {i}: stacked bands={B}, size={W}x{H}")

            # ---- Build cell mask for this crop ----
            mask_array = np.zeros((H, W), dtype=np.uint8)

            if cells_sindex is not None:
                if geo_mode:
                    # Use world-space bbox for candidate selection
                    crop_bbox_geom = box(minx, miny, maxx, maxy)
                    cand_idx = list(cells_sindex.intersection(crop_bbox_geom.bounds))
                    if cand_idx:
                        cand_cells = cells_gdf.iloc[cand_idx]
                        # Keep only those that actually intersect the bbox
                        cand_cells = cand_cells[cand_cells.geometry.intersects(crop_bbox_geom)]
                        # Rasterize directly with the crop transform
                        shapes = [(g.intersection(crop_bbox_geom), 1) for g in cand_cells.geometry if not g.is_empty]
                        if shapes:
                            mask_array = rasterize(
                                shapes=shapes,
                                out_shape=(H, W),
                                transform=used_transform,
                                fill=0,
                                default_value=1,
                                dtype='uint8'
                            )
                else:
                    # Pixel mode: translate cell geometries into crop-local pixel coords
                    col0, row0 = int(win_for_cells.col_off), int(win_for_cells.row_off)
                    crop_bbox_px = box(minx, miny, maxx, maxy)
                    cand_idx = list(cells_sindex.intersection(crop_bbox_px.bounds))
                    if cand_idx:
                        cand_cells = cells_gdf.iloc[cand_idx]
                        cand_cells = cand_cells[cand_cells.geometry.intersects(crop_bbox_px)]
                        # Shift by -col0, -row0 so the window origin becomes (0,0)
                        shifted = []
                        for g in cand_cells.geometry:
                            if g is None or g.is_empty:
                                continue
                            g2 = translate(g, xoff=-col0, yoff=-row0)
                            # Keep only part within the window (0..W, 0..H)
                            g2 = g2.intersection(box(0, 0, W, H))
                            if not g2.is_empty:
                                shifted.append((g2, 1))
                        if shifted:
                            mask_array = rasterize(
                                shapes=shifted,
                                out_shape=(H, W),
                                transform=Affine.identity(),  # pixel coords
                                fill=0,
                                default_value=1,
                                dtype='uint8'
                            )

            # ----- Save multi-band GeoTIFF -----
            profile = first.profile.copy()
            profile.update({
                "driver": "GTiff",
                "height": H,
                "width": W,
                "count": B,
                "transform": used_transform,
                "crs": first.crs,  # often None for OME; keep as-is
                # "compress": "deflate", "predictor": 2,
            })
            tif_path = out_dir / f"{prefix}_polygon_{i:04d}.tif"
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(stacked)

            # ----- Save RGB preview PNG (first 3 bands) -----
            png_path = out_dir / f"{prefix}_polygon_{i:04d}.png"
            if B >= 3:
                rgb = stacked[:3].astype(np.float32)
                for c in range(3):
                    band = rgb[c]
                    mn, mx = float(band.min()), float(band.max())
                    rgb[c] = (band - mn) / (mx - mn) if mx > mn else 0.0
                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                rgb = np.transpose(rgb, (1, 2, 0))
                Image.fromarray(rgb, mode="RGB").save(png_path)
            else:
                band = stacked[0].astype(np.float32)
                mn, mx = float(band.min()), float(band.max())
                gray = ((band - mn) / (mx - mn) if mx > mn else np.zeros_like(band))
                gray = (gray * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(gray, mode="L").save(png_path)

            # ----- Save mask (TIFF + PNG) -----
            mask_tif = out_dir / f"{prefix}_polygon_{i:04d}_mask.tif"
            mask_profile = profile.copy()
            mask_profile.update({"count": 1, "dtype": "uint8"})
            with rasterio.open(mask_tif, "w", **mask_profile) as mds:
                mds.write(mask_array[None, ...])  # (1, H, W)

            mask_png = out_dir / f"{prefix}_polygon_{i:04d}_mask.png"
            Image.fromarray((mask_array * 255).astype(np.uint8), mode="L").save(mask_png)

            print(f"  Saved: {tif_path}, {png_path}, {mask_tif}, {mask_png}")

    finally:
        for ds in datasets:
            ds.close()


# ---- Batch runner ----
if __name__ == "__main__":
    # Configure your folders
    labels_dir = Path("/data/pwojcik/For_Piotr/Labels/glom_labels")     # ROI polygons
    cells_dir  = Path("/data/pwojcik/For_Piotr/Labels/cells_labels")     # CELL polygons (one file per case)
    images_dir = Path("/data/pwojcik/For_Piotr/Images")
    out_root   = Path("/data/pwojcik/For_Piotr/gloms_rect")

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
        # Try exact match first
        cand = cells_dir / f"{stem}.geojson"
        if cand.exists():
            return cand
        # Loose match (e.g., suffixes)
        for p in cells_dir.glob("*.geojson"):
            if p.stem == stem or p.name.startswith(stem):
                return p
        return None

    geojsons = sorted(labels_dir.glob("*.geojson"))
    if not geojsons:
        print(f"No .geojson files found in {labels_dir}")
        raise SystemExit(1)

    for gj in geojsons:
        stem = gj.stem  # e.g., "A_hNiere_S3"
        img = find_image_for(stem)
        cells = find_cells_for(stem)
        if img is None:
            print(f"[SKIP] No matching image for {stem} in {images_dir}")
            continue
        if cells is None:
            print(f"[SKIP] No matching cells GeoJSON for {stem} in {cells_dir}")
            continue

        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing: {stem} ===")
        print(f"ROI GeoJSON:   {gj}")
        print(f"Cells GeoJSON: {cells}")
        print(f"Image:         {img}")
        print(f"Output:        {out_dir}")

        try:
            cut_rectangles_multich_with_cells(
                geojson_path=str(gj),
                cells_geojson_path=str(cells),
                ome_tiff_path=str(img),
                output_dir=str(out_dir),
                prefix=img.stem  # prefix outputs with the TIFF name
            )
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
