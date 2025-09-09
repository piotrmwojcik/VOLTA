#!/usr/bin/env python3
import json, hashlib, warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import DecompressionBombWarning

# ---------- hardcoded inputs ----------
BOXES_JSON_PATH = Path("/Users/piotrwojcik/PycharmProjects/VOLTA/old__A_hNiere_S3.ome__polygon_0097__bboxes.json")
PNG_PATH        = Path("/Users/piotrwojcik/PycharmProjects/VOLTA/old__A_hNiere_S3.ome__polygon_0097.png")
# --------------------------------------

# allow very large images from trusted sources
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", DecompressionBombWarning)

def stable_rgb(s: str):
    h = hashlib.md5(s.encode("utf-8")).digest()
    return (h[0], h[1], h[2])

def clamp_box(x0, y0, x1, y1, W, H):
    x0 = max(0.0, min(float(x0), W))
    y0 = max(0.0, min(float(y0), H))
    x1 = max(0.0, min(float(x1), W))
    y1 = max(0.0, min(float(y1), H))
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    return int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))

def get_font(size_px: int) -> ImageFont.ImageFont:
    size_px = max(10, int(size_px))
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size_px)
    except Exception:
        return ImageFont.load_default()

def draw_centered_text(draw: ImageDraw.ImageDraw, xy_box, text: str, font: ImageFont.ImageFont):
    x0, y0, x1, y1 = xy_box
    try:
        tb = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    tx = x0 + max(0, (x1 - x0 - tw) // 2)
    ty = y0 + max(0, (y1 - y0 - th) // 2)
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255),
              stroke_width=2, stroke_fill=(0, 0, 0, 255))

def main():
    # load image
    img = Image.open(PNG_PATH).convert("RGB")
    W, H = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    # load our custom boxes JSON
    data = json.loads(BOXES_JSON_PATH.read_text())
    boxes_in = data.get("boxes", [])

    # optional sanity check against JSON-declared size
    dw, dh = int(data.get("width", W)), int(data.get("height", H))
    if (dw, dh) != (W, H):
        print(f"[WARN] JSON width/height ({dw}x{dh}) != image size ({W}x{H}). Using image size.")

    not_fit = 0
    not_fit_idx = []

    boxes_out = []
    for idx, b in enumerate(boxes_in, start=1):
        bbox = b.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = map(float, bbox)
        orig_x0, orig_y0, orig_x1, orig_y1 = x, y, x + w, y + h

        # check if the ORIGINAL box fully fits within the image
        fits = (
            0 <= orig_x0 < orig_x1 <= W and
            0 <= orig_y0 < orig_y1 <= H
        )
        if not fits:
            not_fit += 1
            not_fit_idx.append(idx)

        # draw clamped box
        x0, y0, x1, y1 = clamp_box(orig_x0, orig_y0, orig_x1, orig_y1, W, H)
        if x1 <= x0 or y1 <= y0:
            # completely outside after clamping; skip drawing
            continue

        label = str(b.get("label", "") or "")
        color = stable_rgb(label)
        draw.rectangle([x0, y0, x1, y1], fill=(*color, 0), outline=(*color, 255), width=2)

        box_w, box_h = (x1 - x0), (y1 - y0)
        font_size = max(12, min(40, int(min(box_w, box_h) * 0.5)))
        font = get_font(font_size)
        draw_centered_text(draw, (x0, y0, x1, y1), str(idx), font)

        boxes_out.append({"index": idx, "bbox": [x0, y0, x1 - x0, y1 - y0], "label": label})

    out_png = PNG_PATH.with_name(PNG_PATH.stem + "__boxes_numbered.png")
    overlay.save(out_png)

    out_json = out_png.with_suffix(".json")
    with open(out_json, "w") as f:
        json.dump({"image": out_png.name, "width": W, "height": H, "boxes": boxes_out}, f, indent=2)

    print(f"Saved: {out_png}  (boxes drawn: {len(boxes_out)})")
    print(f"Wrote: {out_json}")
    print(f"Boxes that do NOT fully fit inside the image: {not_fit} / {len(boxes_in)}")
    if not_fit_idx:
        print(f"  Indices (1-based) of non-fitting boxes: {not_fit_idx[:20]}{' …' if len(not_fit_idx) > 20 else ''}")

if __name__ == "__main__":
    main()
