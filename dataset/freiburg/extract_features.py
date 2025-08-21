#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from types import SimpleNamespace
from torchvision import models as tv
from torchvision import transforms as T
from torchvision.ops import roi_align


import torch
from torchvision import models as tv
import torch.nn as nn

class ResNet50Backbone(nn.Module):
    def __init__(self, base: tv.ResNet):
        super().__init__()
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x

def _get_student_state(ckpt):
    """Return the student's state_dict (flat), handling common DINO checkpoints."""
    if isinstance(ckpt, dict):
        if "student" in ckpt and isinstance(ckpt["student"], dict):
            sd = ckpt["student"].get("state_dict", ckpt["student"])
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    return sd

def _normalize_to_resnet_keys(sd):
    """
    Normalize keys to torchvision resnet style: conv1, bn1, layer1.*, ..., layer4.*, fc.
    Aggressively strips wrappers like: module., student., model., teacher., backbone., net., encoder_q., encoder_k., encoder.
    Drops head/proj/prototypes keys altogether.
    """
    strip_prefixes = (
        "module.", "student.", "model.", "teacher.",
        "backbone.", "net.", "encoder_q.", "encoder_k.", "encoder."
    )
    drop_if_starts = (
        "head", "prototypes", "projection", "dino_head",
        "cls_head", "mlp_head", "proj", "prototype", "queue", "fc_norm"
    )

    cleaned = {}
    for k, v in sd.items():
        k = k.strip()

        # repeatedly strip any of the known prefixes until none apply
        changed = True
        while changed:
            changed = False
            for p in strip_prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True

        # drop classifier / projection / head params
        if any(k.startswith(d) for d in drop_if_starts):
            continue

        cleaned[k] = v

    return cleaned

def load_dino_backbone_from_checkpoint(ckpt_path: str) -> ResNet50Backbone:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_sd = _get_student_state(ckpt)

    # quick ViT detection to avoid silent mismatch
    if any(k.startswith(("pos_embed", "patch_embed", "blocks", "encoder.blocks", "norm", "heads")) for k in raw_sd.keys()):
        raise RuntimeError("Checkpoint looks like ViT/DeiT weights. Use a ViT backbone variant, not ResNet.")

    # DEBUG: show a few raw keys
    raw_keys = list(raw_sd.keys())
    print("[debug] raw keys (first 10):", raw_keys[:10])

    bk = _normalize_to_resnet_keys(raw_sd)

    # DEBUG: show a few normalized keys
    norm_keys = list(bk.keys())
    print("[debug] normalized keys (first 10):", norm_keys[:10])

    # If somehow backbone.* still survived, strip it one more time (belt & suspenders)
    if any(k.startswith("backbone.") for k in bk.keys()):
        bk = { (k.split("backbone.", 1)[1] if k.startswith("backbone.") else k): v for k, v in bk.items() }

    base = tv.resnet50(pretrained=False)
    missing, unexpected = base.load_state_dict(bk, strict=False)
    print(f"[load] ResNet50 backbone <- checkpoint | missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print("  examples missing:", ", ".join(missing[:5]))
        print("  examples unexpected:", ", ".join(unexpected[:5]))

    return ResNet50Backbone(base)


# --------------------- Image / box utils ---------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

to_tensor = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def clamp(v, lo, hi): return max(lo, min(v, hi))

def centered_window(x, y, w, h, crop_size, W, H):
    """
    Given a bbox (x,y,w,h) on an image of size (W,H), return integer (left, top, right, bottom)
    for a square window of size crop_size centered at the bbox center. If it falls outside, we clamp.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    half = crop_size / 2.0
    left   = int(round(cx - half))
    top    = int(round(cy - half))
    right  = left + crop_size
    bottom = top + crop_size

    # Shift window to be inside image when possible
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > W:
        shift = right - W
        left -= shift
        right = W
        left = max(0, left)
    if bottom > H:
        shift = bottom - H
        top -= shift
        bottom = H
        top = max(0, top)

    # Final clamp
    left   = clamp(left, 0, max(0, W-1))
    top    = clamp(top, 0, max(0, H-1))
    right  = clamp(right, 1, W)
    bottom = clamp(bottom, 1, H)
    return left, top, right, bottom


def xywh_to_xyxy_in_crop(x, y, w, h, crop_left, crop_top, crop_W, crop_H):
    """
    Convert original image bbox to bbox **within the crop** coordinates.
    Returns clamped (x1, y1, x2, y2) in crop space.
    """
    x1 = x - crop_left
    y1 = y - crop_top
    x2 = x + w - crop_left
    y2 = y + h - crop_top
    x1 = clamp(x1, 0, crop_W)
    y1 = clamp(y1, 0, crop_H)
    x2 = clamp(x2, 0, crop_W)
    y2 = clamp(y2, 0, crop_H)
    # ensure min<=max
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser("Extract DINO ROI features per cell")
    ap.add_argument("--checkpoint", required=True, help="Path to DINO checkpoint (checkpoint.pth)")
    ap.add_argument("--data-dir", required=True, help="Folder containing crop PNGs and __bboxes.json files")
    ap.add_argument("--out-dir", required=True, help="Where to save features (.npz)")
    ap.add_argument("--crop-size", type=int, default=224, help="Square centered crop size fed to the model")
    ap.add_argument("--roi-size", type=int, default=1, help="ROI Align output size (1 gives a single feature vector)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 1) Load backbone
    backbone = load_dino_backbone_from_checkpoint(args.checkpoint).to(device).eval()

    # 2) List images
    data_dir = Path(args.data_dir)
    imgs = sorted([p for p in data_dir.glob("*.png") if "__overlay" not in p.name and "__cell_" not in p.name and "__boxed" not in p.name])

    if not imgs:
        print(f"[WARN] No PNGs found in {data_dir}")
        return

    with torch.no_grad():
        for img_path in imgs:
            bb_json = img_path.with_name(img_path.stem + "__bboxes.json")
            if not bb_json.exists():
                # silently skip images without boxes
                continue

            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            meta = json.loads(bb_json.read_text())
            boxes = meta.get("boxes", [])
            if not boxes:
                continue

            feats = []      # (N, C) if roi_size=1 else (N, C, roi_size, roi_size)
            rows = []       # metadata rows

            # We’ll batch a few crops for speed
            batch_imgs = []
            batch_rois = []   # each item will be a list[Tensor(K_i, 4)] with one ROI per image
            idx_map = []      # (img_idx, cell_idx) mapping

            # Decide stride/spatial_scale lazily after first forward
            spatial_scale = None
            feat_C = None

            for ci, rec in enumerate(boxes):
                x, y, w, h = rec["bbox"]
                lab = str(rec.get("label", ""))
                lid = int(rec.get("label_id", -1))

                # centered crop window on this cell
                l, t, r, b = centered_window(x, y, w, h, args.crop_size, W, H)
                crop = im.crop((l, t, r, b))   # size: (args.crop_size, args.crop_size) except at borders (we clamp)

                # If the crop is not exactly requested size due to borders, resize to target size
                if crop.size != (args.crop_size, args.crop_size):
                    crop = crop.resize((args.crop_size, args.crop_size), Image.BICUBIC)
                    # when we resize the whole crop, the ROI must be scaled accordingly
                    scale_x = args.crop_size / (r - l)
                    scale_y = args.crop_size / (b - t)
                    cx1, cy1, cx2, cy2 = xywh_to_xyxy_in_crop(x, y, w, h, l, t, r - l, b - t)
                    cx1, cy1 = cx1 * scale_x, cy1 * scale_y
                    cx2, cy2 = cx2 * scale_x, cy2 * scale_y
                else:
                    cx1, cy1, cx2, cy2 = xywh_to_xyxy_in_crop(x, y, w, h, l, t, args.crop_size, args.crop_size)

                # Prepare tensors
                img_t = to_tensor(crop)  # (3, H, W), normalized
                batch_imgs.append(img_t)
                batch_rois.append(torch.tensor([[cx1, cy1, cx2, cy2]], dtype=torch.float32))
                idx_map.append((ci, lab, lid, (l, t, r, b), (cx1, cy1, cx2, cy2)))

                # To keep memory in check, run in mini-batches of, say, 64 crops
                if len(batch_imgs) == 64 or ci == len(boxes) - 1:
                    inp = torch.stack(batch_imgs, dim=0).to(device)  # (B,3,H,W)
                    feat = backbone(inp)  # (B, C, h, w)
                    if spatial_scale is None:
                        # spatial_scale = feat_h / input_h = 1 / stride
                        h_feat = feat.shape[-2]
                        spatial_scale = h_feat / float(args.crop_size)
                        feat_C = feat.shape[1]

                    # roi_align expects list[Tensor(K_i, 4)] in image coords; one ROI per image
                    rois = [b.to(device) for b in batch_rois]
                    pooled = roi_align(
                        feat, rois,
                        output_size=(args.roi_size, args.roi_size),
                        spatial_scale=spatial_scale,
                        sampling_ratio=-1,
                        aligned=True
                    )  # shape: (sum K_i == B, C, roi, roi)

                    if args.roi_size == 1:
                        vec = pooled.view(pooled.size(0), feat_C)  # (B, C)
                        feats.append(vec.cpu().numpy())
                    else:
                        feats.append(pooled.cpu().numpy())         # (B, C, roi, roi)

                    # stash metadata rows
                    for (ci0, lab0, lid0, crop_xyxy, roi_xyxy) in idx_map:
                        rows.append({
                            "cell_index": int(ci0),
                            "label": lab0,
                            "label_id": int(lid0),
                            "crop_window_xyxy": list(map(int, crop_xyxy)),
                            "roi_in_crop_xyxy": [float(f"{v:.3f}") for v in roi_xyxy],
                        })

                    # reset batch
                    batch_imgs.clear()
                    batch_rois.clear()
                    idx_map.clear()

            if feats:
                feats = np.concatenate(feats, axis=0)  # (N, C) or (N, C, roi, roi)
                out_npz = out_dir / f"{img_path.stem}__roi_feats.npz"
                np.savez_compressed(
                    out_npz,
                    feats=feats,
                    rows=np.array(rows, dtype=object),
                    image=str(img_path.name),
                    crop_size=args.crop_size,
                    roi_size=args.roi_size
                )
                print(f"[OK] {img_path.name}: saved {feats.shape} to {out_npz}")
            else:
                print(f"[SKIP] {img_path.name}: no valid boxes")

if __name__ == "__main__":
    main()
