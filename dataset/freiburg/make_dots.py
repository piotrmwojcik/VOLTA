#!/usr/bin/env python3
import json
import os.path
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ======== CONFIG ========
COMBINED_JSON = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within/train.json")   # path to your combined JSON
IMAGE_ROOT = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within/")                           # root to resolve image paths in JSON (if they are relative)
OUT_DIR = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within/dots_preview")                 # where to save dotted images
SAMPLE_N = None                                  # set to an int to only render a random sample, e.g. 50; or None for all
DOT_RADIUS = 4                                   # radius in pixels
DOT_ALPHA = 200                                  # 0..255
LEGEND = True                                    # draw legend
LEGEND_BG_ALPHA = 160                             # background alpha for legend box
# ========================

# The 6 classes (must match order in JSON["classes"])
CLASSES = ["_empty", "opal_480", "opal_520", "opal_570", "unclassified", "opal_620"]

# Nice distinct colors (R,G,B) mapped to each class
CLASS_COLORS = {
    "_empty":       (160, 160, 160),   # gray
    "opal_480":     ( 55, 126, 184),   # blue
    "opal_520":     ( 77, 175,  74),   # green
    "opal_570":     (228,  26,  28),   # red
    "unclassified": (152,  78, 163),   # purple
    "opal_620":     (255, 127,   0),   # orange
}

def draw_dot(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color_rgba):
    """Draw a filled circle centered at (cx,cy)."""
    x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r
    draw.ellipse([x0, y0, x1, y1], fill=color_rgba, outline=None, width=0)

def draw_legend(base_img: Image.Image, classes, colors):
    """Draw a small legend in the top-left corner."""
    # Try to load a default font; fall back to PIL's built-in
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 8
    swatch = 14
    spacing = 4
    # Compute legend size
    text_width = 0
    text_height = 0
    for c in classes:
        t = c
        w, h = (font.getlength(t), font.size) if font else (7*len(t), 12)
        text_width = max(text_width, int(w))
        text_height = max(text_height, int(h))
    W = int(pad*3 + swatch + text_width)
    H = int(pad*2 + len(classes)*(max(swatch, text_height) + spacing) - spacing)

    # Make overlay
    overlay = Image.new("RGBA", base_img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    # background box
    d.rectangle([0,0, W, H], fill=(0,0,0,LEGEND_BG_ALPHA))
    # rows
    y = pad
    for c in classes:
        rgb = colors[c]
        d.rectangle([pad, y, pad+swatch, y+swatch], fill=rgb+(255,))
        # text
        tx = pad*2 + swatch
        ty = y + (swatch - (font.size if font else 12))//2
        d.text((tx, ty), c, fill=(255,255,255,255), font=font)
        y += max(swatch, text_height) + spacing

    return Image.alpha_composite(base_img.convert("RGBA"), overlay)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(COMBINED_JSON, "r") as f:
        combo = json.load(f)

    # Validate classes
    json_classes = combo.get("classes", [])
    if json_classes != CLASSES:
        # We’ll continue, but warn if the order/content differs
        print("[WARN] JSON classes differ from expected list.")
        print("JSON:", json_classes)
        print("Expected:", CLASSES)

    # Collect image keys (all keys except 'classes')
    keys = [k for k in combo.keys() if k != "classes"]

    if SAMPLE_N is not None:
        keys = random.sample(keys, min(SAMPLE_N, len(keys)))

    for key in keys:
        rel_img_path = key  # e.g. "datasets/pannuke/Images/1_1047.png"
        points_per_class = combo[key]

        #img_path = (IMAGE_ROOT / rel_img_path).resolve()
        img_path = os.path.join(IMAGE_ROOT, rel_img_path)
        if not img_path.exists():
            print(f"[SKIP] Missing image: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Failed to open {img_path}: {e}")
            continue

        W, H = img.size
        # draw on transparent overlay to keep dots semi-transparent
        overlay = Image.new("RGBA", (W, H), (0,0,0,0))
        draw = ImageDraw.Draw(overlay)

        # Iterate classes in order
        for cls_idx, cls_name in enumerate(CLASSES):
            rgb = CLASS_COLORS[cls_name]
            rgba = rgb + (DOT_ALPHA,)
            pts = points_per_class[cls_idx] if cls_idx < len(points_per_class) else []
            for (cx, cy) in pts:
                # safety clamp
                if not (0 <= cx < W and 0 <= cy < H):
                    continue
                draw_dot(draw, int(cx), int(cy), DOT_RADIUS, rgba)

        dotted = Image.alpha_composite(img.convert("RGBA"), overlay)

        if LEGEND:
            dotted = draw_legend(dotted, CLASSES, CLASS_COLORS)

        out_path = OUT_DIR / (Path(rel_img_path).stem + "__dots.png")
        dotted.convert("RGB").save(out_path)
        print(f"[OK] {out_path}")

if __name__ == "__main__":
    main()
