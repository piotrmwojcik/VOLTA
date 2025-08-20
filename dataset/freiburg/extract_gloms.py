import os
import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
import geopandas as gpd
from shapely.geometry import mapping  # not strictly needed now, but fine to keep
from PIL import Image

def cut_rectangles_multich(geojson_path, ome_tiff_path, output_dir="output_rects"):
    """
    For each polygon in GeoJSON, take its bounding box (rectangular window),
    crop ALL subdatasets (channels) from the OME-TIFF to that window, and save:
      - polygon_XXXX.tif  (multi-band GeoTIFF with all channels)
      - polygon_XXXX.png  (RGB preview from first 3 channels, if present)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load polygons (expects polygon or multipolygon geometries in same CRS as the image)
    gdf = gpd.read_file(geojson_path)

    # Enumerate & sort subdatasets by GTIFF_DIR index
    with rasterio.open(ome_tiff_path) as src0:
        sds = src0.subdatasets
        if not sds:
            raise RuntimeError("No subdatasets found. Is this an OME-TIFF?")
    def _dir_index(s):  # GTIFF_DIR:<n>:<path>
        m = re.search(r"GTIFF_DIR:(\d+):", s)
        return int(m.group(1)) if m else 0
    sds = sorted(sds, key=_dir_index)

    # Open all subdatasets
    datasets = [rasterio.open(sref) for sref in sds]

    try:
        print("Subdatasets:", len(datasets))
        print("CRS (first sds):", datasets[0].crs)
        print("Sizes (w x h):", [(ds.width, ds.height) for ds in datasets])

        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # 1) Bounding box of the polygon
            minx, miny, maxx, maxy = geom.bounds

            cropped_list = []
            used_transform = None
            target_shape = None

            for ds in datasets:
                # 2) Convert bounds -> pixel window (clipped to raster extent)
                #    If there's no CRS/transform, this assumes geometries are already in pixel coords.
                win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)

                # Clip to dataset bounds and round to integers
                win = win.intersection(Window(0, 0, ds.width, ds.height))
                win = Window(
                    col_off=int(max(0, np.floor(win.col_off))),
                    row_off=int(max(0, np.floor(win.row_off))),
                    width=int(max(0, np.ceil(win.width))),
                    height=int(max(0, np.ceil(win.height))),
                )
                if win.width == 0 or win.height == 0:
                    print(f"Polygon {i}: window out of bounds, skipping.")
                    cropped_list = []
                    break

                # 3) Read the rectangular window (no mask)
                out_img = ds.read(window=win)  # shape: (bands, h, w)

                if used_transform is None:
                    used_transform = ds.window_transform(win)
                    target_shape = out_img.shape[1:]  # (h, w)
                else:
                    if out_img.shape[1:] != target_shape:
                        raise RuntimeError(
                            f"Mismatch in crop sizes between subdatasets for polygon {i}: "
                            f"{out_img.shape[1:]} vs {target_shape}"
                        )

                cropped_list.append(out_img)

            if not cropped_list:
                continue

            # 4) Stack ALL bands across subdatasets -> (B_total, H, W)
            stacked = np.concatenate(cropped_list, axis=0)
            B, H, W = stacked.shape
            print(f"Polygon {i}: stacked bands={B}, size={W}x{H}")

            # ----- Save multi-band GeoTIFF -----
            profile = datasets[0].profile.copy()
            profile.update({
                "driver": "GTiff",
                "height": H,
                "width": W,
                "count": B,
                "transform": used_transform,
                "crs": datasets[0].crs,  # often None for OME; keep as-is
                # Optional compression:
                # "compress": "deflate", "predictor": 2
            })
            tif_path = out_dir / f"polygon_{i:04d}.tif"
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(stacked)

            # ----- Save RGB preview PNG (first 3 bands, per-band min-max scaling) -----
            png_path = out_dir / f"polygon_{i:04d}.png"
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

            print(f"Saved: {tif_path} (all {B} bands) and {png_path} (preview)")

    finally:
        for ds in datasets:
            ds.close()

# --- Example call ---
cut_rectangles_multich(
    "/data/pwojcik/For_Piotr/Labels/glom_labels/A_hNiere_S3.geojson",
    "/data/pwojcik/For_Piotr/Images/A_hNiere_S3.ome.tiff",
    output_dir="/data/pwojcik/For_Piotr/gloms_rect"
)
