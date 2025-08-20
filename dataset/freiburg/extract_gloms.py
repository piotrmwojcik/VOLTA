import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
import geopandas as gpd
from PIL import Image


def cut_rectangles_multich(geojson_path, ome_tiff_path, output_dir, prefix=None):
    """
    For each polygon in the GeoJSON, take its bounding box (no mask),
    crop ALL subdatasets (channels) from the OME-TIFF to that window, and save:
      - <prefix>_polygon_XXXX.tif  (multi-band GeoTIFF with all channels)
      - <prefix>_polygon_XXXX.png  (RGB preview from first 3 bands, if present)

    If the image has a CRS and the GeoJSON has a CRS, polygons are reprojected to the image CRS.
    If the image has no CRS, polygon coordinates are assumed to be in pixel space (x=col, y=row).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = Path(ome_tiff_path).stem

    # Load polygons
    gdf = gpd.read_file(geojson_path)

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

        # GEO mode: reproject GeoJSON to raster CRS up-front (if both exist)
        geo_mode = (first.crs is not None) and (gdf.crs is not None)
        if geo_mode and gdf.crs != first.crs:
            gdf = gdf.to_crs(first.crs)
            print(f"  Reprojected GeoJSON to raster CRS {first.crs}")

        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Bounding box of the polygon
            minx, miny, maxx, maxy = geom.bounds

            cropped_list = []
            used_transform = None
            target_shape = None

            for ds in datasets:
                # Build window either in GEO mode (bounds+transform) or PIXEL mode (bounds as pixels)
                if geo_mode:
                    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
                else:
                    # Treat bounds as pixel coords: x->col, y->row (origin top-left)
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
                    print(f"  Polygon {i}: window out of bounds for a subdataset, skipping polygon.")
                    cropped_list = []
                    break

                # Read rectangular window (no mask)
                out_img = ds.read(window=win)  # shape: (bands, h, w)

                # Keep transform & check shape consistency across subdatasets
                if used_transform is None:
                    used_transform = ds.window_transform(win)
                    target_shape = out_img.shape[1:]  # (h, w)
                else:
                    if out_img.shape[1:] != target_shape:
                        raise RuntimeError(
                            f"  Mismatch in crop sizes between subdatasets for polygon {i}: "
                            f"{out_img.shape[1:]} vs {target_shape}"
                        )

                cropped_list.append(out_img)

            if not cropped_list:
                continue

            # Stack ALL bands across subdatasets -> (B_total, H, W)
            stacked = np.concatenate(cropped_list, axis=0)
            B, H, W = stacked.shape
            print(f"  Polygon {i}: stacked bands={B}, size={W}x{H}")

            # Save multi-band GeoTIFF
            profile = first.profile.copy()
            profile.update({
                "driver": "GTiff",
                "height": H,
                "width": W,
                "count": B,
                "transform": used_transform,
                "crs": first.crs,  # often None for OME; keep as-is
                # Optional compression:
                # "compress": "deflate", "predictor": 2
            })
            tif_path = out_dir / f"{prefix}_polygon_{i:04d}.tif"
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(stacked)

            # Save RGB preview PNG (first 3 bands, per-band min-max scaling)
            png_path = out_dir / f"{prefix}_polygon_{i:04d}.png"
            if B >= 3:
                rgb = stacked[:3].astype(np.float32)  # (3, H, W)
                for c in range(3):
                    band = rgb[c]
                    mn, mx = float(band.min()), float(band.max())
                    rgb[c] = (band - mn) / (mx - mn) if mx > mn else 0.0
                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                rgb = np.transpose(rgb, (1, 2, 0))  # -> (H, W, 3)
                Image.fromarray(rgb, mode="RGB").save(png_path)
            else:
                band = stacked[0].astype(np.float32)
                mn, mx = float(band.min()), float(band.max())
                gray = ((band - mn) / (mx - mn) if mx > mn else np.zeros_like(band))
                gray = (gray * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(gray, mode="L").save(png_path)

            print(f"  Saved: {tif_path} (all {B} bands) and {png_path} (preview)")

    finally:
        for ds in datasets:
            ds.close()


# ---- Batch runner ----
if __name__ == "__main__":
    # Configure your folders
    labels_dir = Path("/data/pwojcik/For_Piotr/Labels/glom_labels")
    images_dir = Path("/data/pwojcik/For_Piotr/Images")
    out_root   = Path("/data/pwojcik/For_Piotr/gloms_rect")

    # Allowed image extensions in order of preference
    exts = [".ome.tiff", ".ome.tif", ".tif", ".tiff"]

    def find_image_for(stem: str) -> Path:
        """Find first matching image by stem in images_dir with allowed extensions."""
        for ext in exts:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        # Fallback: loose match (handles names like stem + suffixes)
        for p in images_dir.rglob("*"):
            if p.is_file() and any(str(p.name).lower().endswith(ext) for ext in exts):
                # prefer exact stem match if possible
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
        if img is None:
            print(f"[SKIP] No matching image for {stem} in {images_dir}")
            continue

        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing: {stem} ===")
        print(f"GeoJSON: {gj}")
        print(f"Image:   {img}")
        print(f"Output:  {out_dir}")

        try:
            cut_rectangles_multich(
                geojson_path=str(gj),
                ome_tiff_path=str(img),
                output_dir=str(out_dir),
                prefix=img.stem  # prefix outputs with the TIFF name
            )
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
