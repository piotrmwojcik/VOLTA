#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

# minimal, but nice to have if available
try:
    import pandas as pd
except Exception:
    pd = None

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def main():
    ap = argparse.ArgumentParser("Collate ROI features and make a t-SNE plot")
    ap.add_argument("--feats-dir", required=True, help="Folder with *__roi_feats.npz files")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (e.g., /path/to/out/roi)")
    ap.add_argument("--pca-dim", type=int, default=50, help="PCA dimension before t-SNE (0 to disable)")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--max-per-class", type=int, default=200, help="Max samples per class (for speed)")
    ap.add_argument("--normalize", type=int, default=1, help="L2-normalize features before PCA/TSNE (1/0)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    feats_dir = Path(args.feats_dir)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(feats_dir.glob("*__roi_feats.npz"))
    if not npz_files:
        raise SystemExit(f"No *__roi_feats.npz found in {feats_dir}")

    # ---- load & concatenate ----
    X_list, y_list, yid_list, src_img, cell_idx = [], [], [], [], []
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        X = data["feats"]                    # (N, C) or (N, C, r, r)
        rows = data["rows"]                  # array(dtype=object) of dicts
        if X.ndim == 4:                      # flatten spatial ROI if present
            N, C, r1, r2 = X.shape
            X = X.reshape(N, C * r1 * r2)
        else:
            N, C = X.shape

        for i in range(len(rows)):
            rec = rows[i].item() if hasattr(rows[i], "item") else rows[i]
            lab = str(rec.get("label", ""))
            lid = int(rec.get("label_id", -1))
            X_list.append(X[i])
            y_list.append(lab if lab else "<empty>")
            yid_list.append(lid)
            src_img.append(str(data.get("image", f.name)))
            cell_idx.append(int(rec.get("cell_index", i)))

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    yid = np.array(yid_list, dtype=int)
    src_img = np.array(src_img)
    cell_idx = np.array(cell_idx, dtype=int)

    # ---- per-class subsample (for speed/clarity) ----
    if args.max_per_class > 0:
        rng = np.random.default_rng(args.seed)
        keep_idx = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) > args.max_per_class:
                idx = rng.choice(idx, size=args.max_per_class, replace=False)
            keep_idx.append(idx)
        keep_idx = np.concatenate(keep_idx)
        X, y, yid, src_img, cell_idx = X[keep_idx], y[keep_idx], yid[keep_idx], src_img[keep_idx], cell_idx[keep_idx]

    # ---- optional L2-normalize ----
    if args.normalize:
        X = normalize(X, norm="l2", axis=1)

    # ---- optional PCA ----
    if args.pca_dim and args.pca_dim < X.shape[1]:
        pca = PCA(n_components=args.pca_dim, random_state=args.seed, svd_solver="auto")
        X_red = pca.fit_transform(X)
    else:
        X_red = X

    # ---- t-SNE ----
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        random_state=args.seed,
        learning_rate="auto",
        metric="euclidean",
        n_iter=1000,
        verbose=1,
    )
    Z = tsne.fit_transform(X_red)  # (N,2)

    # ---- save consolidated data ----
    np.savez_compressed(
        f"{out_prefix}_collated.npz",
        feats=X,
        labels=y,
        label_ids=yid,
        tsne=Z,
        src_image=src_img,
        cell_index=cell_idx,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        normalized=bool(args.normalize),
        seed=args.seed
    )

    if pd is not None:
        # also a CSV for easy inspection
        df_cols = {
            "x": Z[:, 0],
            "y": Z[:, 1],
            "label": y,
            "label_id": yid,
            "src_image": src_img,
            "cell_index": cell_idx,
        }
        # store features too (can be large; comment this if too big)
        for j in range(X.shape[1]):
            df_cols[f"f{j}"] = X[:, j]
        pd.DataFrame(df_cols).to_csv(f"{out_prefix}_collated.csv", index=False)

    # ---- plot ----
    plt.figure(figsize=(9, 7))

    classes = np.unique(y)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    idxs = np.array([cls_to_idx[c] for c in y], dtype=int)

    # choose a discrete colormap with enough distinct colors
    cmap_name = "tab10" if len(classes) <= 10 else ("tab20" if len(classes) <= 20 else "gist_ncar")
    cmap = plt.cm.get_cmap(cmap_name, len(classes))

    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=idxs, cmap=cmap, s=6, alpha=0.8)

    # build legend with colored proxies that match the scatter colors
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", markersize=6,
            markerfacecolor=cmap(i), markeredgecolor="none", label=cls
        )
        for i, cls in enumerate(classes)
    ]

    plt.legend(handles=handles, title="Classes",
               bbox_to_anchor=(1.05, 1.0), loc="upper left", borderaxespad=0.)
    plt.title("t-SNE of DINO ROI features")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_tsne.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
