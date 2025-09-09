#!/usr/bin/env python3
import json
from pathlib import Path

# ----- config -----
IN_DIR = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within")      # folder containing *_bboxes.json files
OUT_PATH = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within/train.json")
IMG_KEY_PREFIX = "/data/pwojcik/For_Piotr/gloms_rect_from_png_within"  # prefix for keys in the output

# classes we want
CLASSES = ["_empty", "opal_480", "opal_520", "opal_570", "unclassified", "opal_620"]

# normalized label -> class index
def norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

LABEL_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}


def bbox_center(bx, by, bw, bh, W, H):
    cx = int(round(bx + bw / 2.0))
    cy = int(round(by + bh / 2.0))
    # clip to image bounds
    cx = max(0, min(cx, max(0, W - 1)))
    cy = max(0, min(cy, max(0, H - 1)))
    return [cx, cy]

def main():
    out = {"classes": CLASSES}

    files = sorted(IN_DIR.glob("*__bboxes.json"))
    if not files:
        print(f"[WARN] No files found in {IN_DIR}")
    for jf in files:
        with open(jf, "r") as f:
            data = json.load(f)

        img_name = data.get("image") or jf.stem.replace("__bboxes", "") + ".png"
        W = int(data.get("width", 0))
        H = int(data.get("height", 0))
        boxes = data.get("boxes", [])

        # initialize per-class points
        points = [[] for _ in CLASSES]

        for rec in boxes:
            bbox = rec.get("bbox") or []
            if len(bbox) != 4:
                continue
            bx, by, bw, bh = map(int, bbox)
            lab = norm(rec.get("label", ""))

            if lab not in LABEL_TO_INDEX:
                # unknown label -> skip
                continue

            cls_idx = LABEL_TO_INDEX[lab]
            pt = bbox_center(bx, by, bw, bh, W, H)
            points[cls_idx].append(pt)

        key = f"{IMG_KEY_PREFIX}{img_name}"
        out[key] = points

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Wrote {OUT_PATH} with {len(out)-1} image entries.")

if __name__ == "__main__":
    main()
