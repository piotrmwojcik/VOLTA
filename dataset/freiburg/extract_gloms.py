import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from PIL import Image
import numpy as np


def cut_polygons(geojson_path, tiff_path, output_dir="output_polygons"):
    """
    Cut polygons from a GeoJSON over a GeoTIFF and save as PNGs.

    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file with polygons.
    tiff_path : str
        Path to the GeoTIFF raster.
    output_dir : str
        Directory to save PNG cutouts.
    """
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Load polygons
    gdf = gpd.read_file(geojson_path)

    # Open raster
    with rasterio.open(tiff_path) as src:
        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            # Crop raster to polygon
            out_image, out_transform = mask(src, [mapping(geom)], crop=True)
            out_image = out_image.transpose(1, 2, 0)  # (bands, h, w) -> (h, w, bands)

            # Remove nodata pixels (set transparent)
            mask_array = out_image.sum(axis=-1) == 0
            out_image = np.where(mask_array[..., None], 0, out_image)

            # Convert to uint8 (if needed)
            if out_image.dtype != np.uint8:
                out_image = ((out_image - out_image.min()) / (out_image.max() - out_image.min()) * 255).astype(np.uint8)

            # Save as PNG
            out_img_pil = Image.fromarray(out_image)
            save_path = os.path.join(output_dir, f"polygon_{i}.png")
            out_img_pil.save(save_path)

            print(f"Saved {save_path}")


cut_polygons("/data/pwojcik/For_Piotr/Labels/glom_labels/A_hNiere_S3.geojson",
             "/data/pwojcik/For_Piotr/Images/A_hNiere_S3.ome.tiff", output_dir="/data/pwojcik/For_Piotr/gloms")
