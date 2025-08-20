import os
import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from PIL import Image

def cut_polygons_multich(geojson_path, ome_tiff_path, output_dir="output_polygons"):
    """
    Crop ALL subdatasets (channels) from an OME-TIFF by polygons and save:
      - <name>.tif  (multi-band GeoTIFF with all channels)
      - <name>.png  (RGB preview from first 3 channels, if present)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load polygons (expects polygon or multipolygon geometries)
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

    # Open all subdatasets (lazy context manager)
    datasets = [rasterio.open(sref) for sref in sds]

    try:
        # Basic info
        print("Subdatasets:", len(datasets))
        print("Sizes (w x h) per subdataset:", [(ds.width, ds.height) for ds in datasets])
        print("CRS (from first sds):", datasets[0].crs)

        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Crop each subdataset to the polygon (returns (bands, h, w))
            cropped_list = []
            used_transform = None
            for ds in datasets:
                out_img, out_transform = mask(
                    ds, [mapping(geom)], crop=True, filled=True, nodata=0
                )
                if used_transform is None:
                    used_transform = out_transform
                    target_shape = out_img.shape[1:]  # (h, w)
                else:
                    # Safety: enforce identical sizes (OME subdatasets should match)
                    if out_img.shape[1:] != target_shape:
                        raise RuntimeError(
                            f"Mismatch in crop sizes between subdatasets: "
                            f"{out_img.shape[1:]} vs {target_shape}"
                        )
                cropped_list.append(out_img)  # may be (1,h,w) or (B,h,w)

            # Stack ALL bands across subdatasets -> (B_total, h, w)
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
                "crs": datasets[0].crs,   # often None for OME; keep as-is
                # You can add compression if desired:
                # "compress": "deflate", "predictor": 2
            })
            tif_path = out_dir / f"polygon_{i:04d}.tif"
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(stacked)

            # ----- Save RGB preview PNG (first 3 bands, scaled per-band) -----
            png_path = out_dir / f"polygon_{i:04d}.png"
            if B >= 3:
                rgb = stacked[:3].astype(np.float32)  # (3, H, W)
                # Per-band min-max to 0..255 (avoid div-by-zero)
                for c in range(3):
                    band = rgb[c]
                    mn, mx = float(band.min()), float(band.max())
                    if mx > mn:
                        band = (band - mn) / (mx - mn)
                    else:
                        band = np.zeros_like(band)
                    rgb[c] = band
                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                rgb = np.transpose(rgb, (1, 2, 0))  # -> (H, W, 3)
                Image.fromarray(rgb, mode="RGB").save(png_path)
            else:
                # Single-band preview: normalize to 8-bit grayscale
                band = stacked[0].astype(np.float32)
                mn, mx = float(band.min()), float(band.max())
                if mx > mn:
                    band = (band - mn) / (mx - mn)
                else:
                    band = np.zeros_like(band)
                gray = (band * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(gray, mode="L").save(png_path)

            print(f"Saved: {tif_path} (all {B} bands) and {png_path} (preview)")

    finally:
        for ds in datasets:
            ds.close()

# --- Example call ---
cut_polygons_multich(
    "/data/pwojcik/For_Piotr/Labels/glom_labels/A_hNiere_S3.geojson",
    "/data/pwojcik/For_Piotr/Images/A_hNiere_S3.ome.tiff",
    output_dir="/data/pwojcik/For_Piotr/gloms"
)
