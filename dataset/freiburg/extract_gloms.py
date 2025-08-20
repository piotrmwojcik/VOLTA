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


def cut_rectangles_multich_with_overlay(
    geojson_path,                # ROI polygons
    cells_geojson_path,          # cell polygons
    ome_tiff_path,               # OME-TIFF with subdatasets
    output_dir,                  # where to save PNGs
    prefix=None,                 # filename prefix (defaults to TIFF stem)
    overlay_color=(255, 0, 0),   # mask overlay color (R,G,B)
    overlay_alpha=128            # 0..255 transparency of overlay
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = Path(ome_tiff_path).stem

    # Load ROI & cell polygons
    gdf = gpd.read_file(geojson_path)
    cells_gdf = gpd.read_file(cells_geojson_path)

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

        # Determine GEO vs PIXEL mode
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
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            minx, miny, maxx, maxy = geom.bounds
            cropped_list = []
            used_transform = None
            target_shape = None
            win_for_cells = None

            # Read the same bbox window from every subdataset
            for ds in datasets:
                if geo_mode:
                    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
                else:
                    # Treat ROI bounds as pixel coords (x=col, y=row)
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

                arr = ds.read(window=win)  # (bands, H, W) (usually 1, H, W)
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

            # Build an RGB preview from first 3 channels (per-band min-max)
            if B >= 3:
                rgb = stacked[:3].astype(np.float32)
            else:
                # If fewer than 3 channels, duplicate/extend to 3 for visualization
                rgb = np.vstack([stacked[:1]] * 3).astype(np.float32)  # (3,H,W)

            for c in range(3):
                band = rgb[c]
                mn, mx = float(band.min()), float(band.max())
                rgb[c] = (band - mn) / (mx - mn) if mx > mn else 0.0

            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)         # (3,H,W)
            rgb = np.transpose(rgb, (1, 2, 0))                        # (H,W,3)
            rgb_img = Image.fromarray(rgb, mode="RGB")

            # ---- Build boolean cell mask for this crop (H,W) ----
            mask_array = np.zeros((H, W), dtype=np.uint8)
            if cells_sindex is not None:
                if geo_mode:
                    crop_bbox_geom = box(minx, miny, maxx, maxy)
                    cand_idx = list(cells_sindex.intersection(crop_bbox_geom.bounds))
                    if cand_idx:
                        cand = cells_gdf.iloc[cand_idx]
                        cand = cand[cand.geometry.intersects(crop_bbox_geom)]
                        shapes = [(g.intersection(crop_bbox_geom), 1) for g in cand.geometry if g and not g.is_empty]
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
                    # Pixel mode: shift polygons by crop origin so (0,0) is top-left of crop
                    col0, row0 = int(win_for_cells.col_off), int(win_for_cells.row_off)
                    crop_bbox_px = box(minx, miny, maxx, maxy)
                    cand_idx = list(cells_sindex.intersection(crop_bbox_px.bounds))
                    if cand_idx:
                        cand = cells_gdf.iloc[cand_idx]
                        cand = cand[cand.geometry.intersects(crop_bbox_px)]
                        shifted = []
                        for g in cand.geometry:
                            if g is None or g.is_empty:
                                continue
                            g2 = translate(g, xoff=-col0, yoff=-row0).intersection(box(0, 0, W, H))
                            if not g2.is_empty:
                                shifted.append((g2, 1))
                        if shifted:
                            mask_array = rasterize(
                                shapes=shifted,
                                out_shape=(H, W),
                                transform=Affine.identity(),
                                fill=0,
                                default_value=1,
                                dtype='uint8'
                            )

            # ---- Save PNG of the crop ----
            crop_png = out_dir / f"{prefix}_polygon_{i:04d}.png"
            rgb_img.save(crop_png)

            # ---- Save overlay PNG (semi-transparent mask over crop) ----
            overlay_png = out_dir / f"{prefix}_polygon_{i:04d}_overlay.png"
            # Make an RGBA version of the crop
            rgba = rgb_img.convert("RGBA")
            # Create a solid color image with desired alpha, then use mask_array as the alpha mask multiplier
            overlay = Image.new("RGBA", (W, H), overlay_color + (0,))
            # Build alpha channel from mask_array (0 or 1) scaled by overlay_alpha
            alpha_channel = (mask_array.astype(np.uint8) * overlay_alpha)
            overlay.putalpha(Image.fromarray(alpha_channel, mode="L"))
            # Composite: crop under, colored mask over
            out_overlay = Image.alpha_composite(rgba, overlay)
            out_overlay.convert("RGB").save(overlay_png)

            print(f"  Saved: {crop_png} and {overlay_png}")

    finally:
        for ds in datasets:
            ds.close()


# ---- Batch runner ----
if __name__ == "__main__":
    # Folders
    labels_dir = Path("/data/pwojcik/For_Piotr/Labels/glom_labels")  # ROI polygons
    cells_dir  = Path("/data/pwojcik/For_Piotr/Labels/cell_labels")  # cell polygons
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
                prefix=img.stem,             # name crops after original image
                overlay_color=(255, 0, 0),   # red overlay
                overlay_alpha=128            # semi-transparent
            )
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
