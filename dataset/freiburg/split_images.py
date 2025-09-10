#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image

# ========= CONFIG =========
IN_DIR = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within")      # folder containing *_bboxes.json files
OUT_PATH = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within/train.json")
IMG_KEY_PREFIX = "/data/pwojcik/For_Piotr/gloms_rect_from_png_within"   # prefix for keys in the output

CLASSES = ["_empty", "opal_480", "opal_520", "opal_570", "unclassified", "opal_620"]
# ==========================

def split_and_remap_points(img: Image.Image, points_per_class):
    """Split image into 4 quadrants and remap points to each quadrant."""
    W, H = img.size
    mx, my = W // 2, H // 2

    # quadrant boxes
    boxes = {
        'q00': (0,    0,    mx,   my),   # top-left
        'q01': (mx,   0,    W,    my),   # top-right
        'q10': (0,    my,   mx,   H),    # bottom-left
        'q11': (mx,   my,   W,    H),    # bottom-right
    }

    crops = {k: img.crop(v) for k, v in boxes.items()}
    num_classes = len(points_per_class)
    quad_points = {k: [[] for _ in range(num_classes)] for k in boxes.keys()}

    def place_point(cx, cy, cls_idx):
        if 0 <= cx < mx and 0 <= cy < my:
            quad_points['q00'][cls_idx].append([cx, cy]); return
        if mx <= cx < W and 0 <= cy < my:
            quad_points['q01'][cls_idx].append([cx - mx, cy]); return
        if 0 <= cx < mx and my <= cy < H:
            quad_points['q10'][cls_idx].append([cx, cy - my]); return
        if mx <= cx < W and my <= cy < H:
            quad_points['q11'][cls_idx].append([cx - mx, cy - my]); return

    for cls_idx in range(num_classes):
        pts = points_per_class[cls_idx] if cls_idx < len(points_per_class) else []
        for p in pts:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                continue
            cx, cy = int(p[0]), int(p[1])
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            place_point(cx, cy, cls_idx)

    return crops, quad_points

def main():
    out_dict = {"classes": CLASSES}

    files = sorted(IN_DIR.glob("*__bboxes.json"))
    if not files:
        print(f"[WARN] No files found in {IN_DIR}")
        return

    for jf in files:
        with open(jf, "r") as f:
            data = json.load(f)

        img_name = data.get("image") or jf.stem.replace("__bboxes", "") + ".png"
        img_path = IN_DIR / img_name
        if not img_path.exists():
            print(f"[SKIP] Missing image: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Failed to open {img_path}: {e}")
            continue

        # ---------- build points per class (unchanged) ----------
        points_per_class = [[] for _ in CLASSES]
        for rec in data.get("boxes", []):
            bbox = rec.get("bbox") or []
            if len(bbox) != 4:
                continue
            bx, by, bw, bh = map(int, bbox)
            cx = bx + bw // 2
            cy = by + bh // 2
            lab = (rec.get("label") or "").strip().lower().replace(" ", "_")
            if lab not in CLASSES:
                continue
            cls_idx = CLASSES.index(lab)
            points_per_class[cls_idx].append([cx, cy])

        # ---------- split image into 4 crops (existing logic) ----------
        crops, quad_points = split_and_remap_points(img, points_per_class)

        # We'll reuse the exact same boxes to crop the mask, so compute them here too
        W, H = img.size
        mx, my = W // 2, H // 2
        boxes = {
            'q00': (0,    0,    mx,   my),   # top-left
            'q01': (mx,   0,    W,    my),   # top-right
            'q10': (0,    my,   mx,   H),    # bottom-left
            'q11': (mx,   my,   W,    H),    # bottom-right
        }

        stem = img_path.stem

        # ---------- NEW: locate and crop the corresponding type-mask ----------
        mask_name = f"{stem}__type-mask.png"
        mask_path = IN_DIR / mask_name
        mask = None
        if mask_path.exists():
            try:
                # Keep original mode (likely "P"); don't convert, to preserve class IDs and palette
                mask = Image.open(mask_path)
                if mask.size != (W, H):
                    print(f"[WARN] Mask size {mask.size} != image size {(W, H)} for {mask_path.name}")
            except Exception as e:
                print(f"[WARN] Failed to open mask {mask_path}: {e}")
                mask = None
        else:
            print(f"[WARN] No type-mask found for {stem} (expected {mask_path.name})")

        # ---------- save crops + (optionally) mask crops ----------
        for qk, crop_img in crops.items():
            out_img_name = f"{stem}__{qk}.png"
            out_img_path = IN_DIR / out_img_name
            crop_img.save(out_img_path)

            # If we have a mask, crop with the exact same box and save alongside
            if mask is not None:
                box = boxes[qk]
                mask_crop = mask.crop(box)
                out_mask_name = f"{stem}__{qk}__type-mask.png"
                out_mask_path = IN_DIR / out_mask_name
                # Save without conversion to keep palette + class IDs intact
                mask_crop.save(out_mask_path)

            out_key = f"{IMG_KEY_PREFIX}/{out_img_name}"
            out_dict[out_key] = quad_points[qk]

        print(f"[OK] {img_path} -> 4 crops" + (" + mask crops" if mask is not None else ""))

    with open(OUT_PATH, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"[DONE] Wrote {OUT_PATH} with {len(out_dict)-1} images.")


if __name__ == "__main__":
    main()
