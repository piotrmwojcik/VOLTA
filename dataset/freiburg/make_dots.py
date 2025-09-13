#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ========= CONFIG =========
COMBINED_JSON = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_new/train.json")   # path to your combined JSON                         # root to resolve image paths in JSON (if they are relative)
OUT_DIR = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_new/dots_preview")           # where to save dotted images
DOT_RADIUS = 4                                  # pixels
DOT_ALPHA = 200                                 # 0..255
DRAW_LEGEND = True
LEGEND_BG_ALPHA = 160
# ==========================

# Optional fixed colors for your 6 labels (fallback to hash-colors if a class isn't listed)
FIXED_COLORS = {
    "_empty":       (0, 0, 0),  # gray
    "opal_480":     ( 55, 126, 184),  # blue
    "opal_520":     ( 77, 175,  74),  # green
    "opal_570":     (228,  26,  28),  # red
    "opal_620":     (255, 127,   0),  # orange
}

def stable_rgb(name: str):
    """Deterministic color for a class name (fallback if not in FIXED_COLORS)."""
    import hashlib
    h = hashlib.md5(name.encode("utf-8")).digest()
    return (h[0], h[1], h[2])

def color_for(cls: str):
    return FIXED_COLORS.get(cls, stable_rgb(cls))

def draw_dot(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color_rgba):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color_rgba)

def draw_legend(img_rgba: Image.Image, classes):
    try:
        font = ImageFont.load_default()
        line_h = font.size
    except Exception:
        font = None
        line_h = 12
    pad = 8
    sw = 14
    gap = 4
    # compute legend size
    max_text_w = 0
    for c in classes:
        w = int(font.getlength(c)) if font else 7 * len(c)
        if w > max_text_w:
            max_text_w = w
    W = pad * 3 + sw + max_text_w
    H = pad * 2 + len(classes) * (max(sw, line_h) + gap) - gap

    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, 0, W, H], fill=(0, 0, 0, LEGEND_BG_ALPHA))
    y = pad
    for c in classes:
        rgb = color_for(c)
        d.rectangle([pad, y, pad + sw, y + sw], fill=rgb + (255,))
        tx = pad * 2 + sw
        ty = y + (max(sw, line_h) - line_h) // 2
        d.text((tx, ty), c, fill=(255, 255, 255, 255), font=font)
        y += max(sw, line_h) + gap
    return Image.alpha_composite(img_rgba, overlay)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(COMBINED_JSON, "r") as f:
        data = json.load(f)

    classes = data.get("classes", [])
    if not classes:
        print("[ERROR] 'classes' missing or empty in JSON.")
        return

    # all keys except 'classes' are image paths
    image_keys = [k for k in data.keys() if k != "classes"]
    if not image_keys:
        print("[WARN] No image entries found.")
        return

    for key in image_keys:
        rel_path = key  # use EXACTLY the key as the image path (no prefix/root join)
        img_path = Path(rel_path).expanduser()

        if not img_path.exists():
            print(f"[SKIP] Missing image: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Failed to open {img_path}: {e}")
            continue

        W, H = img.size
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # points_per_class should be list-of-lists aligned to 'classes'
        points_per_class = data[key]
        for cls_idx, cls_name in enumerate(classes):
            pts = points_per_class[cls_idx] if cls_idx < len(points_per_class) else []
            rgb = color_for(cls_name)
            rgba = rgb + (DOT_ALPHA,)
            for pt in pts:
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    continue
                cx, cy = int(pt[0]), int(pt[1])
                if 0 <= cx < W and 0 <= cy < H:
                    draw_dot(draw, cx, cy, DOT_RADIUS, rgba)

        dotted = Image.alpha_composite(img.convert("RGBA"), overlay)
        if DRAW_LEGEND:
            dotted = draw_legend(dotted, classes)

        out_name = f"{img_path.stem}__dots.png"
        out_path = OUT_DIR / out_name
        dotted.convert("RGB").save(out_path)
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
