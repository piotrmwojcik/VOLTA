#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

# optional (nice to have)
try:
    import pandas as pd
except Exception:
    pd = None

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def load_all_npz(feats_dir: Path):
    npz_files = sorted(feats_dir.glob("*__roi_feats.npz"))
    if not npz_files:
        raise SystemExit(f"No *__roi_feats.npz found in {feats_dir}")

    X_list, y_list, yid_list, src_img, cell_idx = [], [], [], [], []
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        X = data["feats"]                    # (N, C) or (N, C, r, r)
        rows = data["rows"]                  # array(dtype=object) of dicts
        if X.ndim == 4:
            N, C, r1, r2 = X.shape
            X = X.reshape(N, C * r1 * r2)
        else:
            N, C = X.shape

        for i in range(N):
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
    return X, y, yid, src_img, cell_idx


def print_class_stats(title: str, y: np.ndarray):
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    print(f"\n[stats] {title}:")
    for c, n in zip(classes[order], counts[order]):
        print(f"  {c:<30} : {n}")
    print(f"  {'TOTAL':<30} : {y.shape[0]}  |  classes: {len(classes)}")


def subsample(X, y, yid, src_img, cell_idx, max_per_class: int, max_total: int, seed: int):
    N0 = X.shape[0]
    rng = np.random.default_rng(seed)
    keep_idx = np.arange(N0)

    if max_per_class and max_per_class > 0:
        per_class_keep = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            per_class_keep.append(idx)
        keep_idx = np.concatenate(per_class_keep)

    if max_total and max_total > 0 and keep_idx.size > max_total:
        keep_idx = rng.choice(keep_idx, size=max_total, replace=False)

    X = X[keep_idx]; y = y[keep_idx]; yid = yid[keep_idx]
    src_img = src_img[keep_idx]; cell_idx = cell_idx[keep_idx]
    print(f"\n[tsne] using {X.shape[0]} samples (from {N0})")
    return X, y, yid, src_img, cell_idx


def choose_cmap(n_classes: int):
    if n_classes <= 10:  return plt.cm.get_cmap("tab10", n_classes)
    if n_classes <= 20:  return plt.cm.get_cmap("tab20", n_classes)
    return plt.cm.get_cmap("gist_ncar", n_classes)


def make_label_onehot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return one-hot matrix (N, K) for labels y over given classes order."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    idxs = np.array([cls_to_idx[c] for c in y], dtype=int)
    K = len(classes)
    M = np.zeros((y.shape[0], K), dtype=np.float32)
    M[np.arange(y.shape[0]), idxs] = 1.0
    return M


def main():
    ap = argparse.ArgumentParser("Collate ROI features and make a t-SNE plot (with optional label influence)")
    ap.add_argument("--feats-dir", required=True, help="Folder with *__roi_feats.npz files")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (e.g., /path/to/out/roi)")
    ap.add_argument("--normalize", type=int, default=1, help="L2-normalize features before PCA/TSNE (1/0)")
    ap.add_argument("--pca-dim", type=int, default=50, help="PCA dim before t-SNE (0 to disable)")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--max-per-class", type=int, default=0, help="Per-class cap; 0 = no cap")
    ap.add_argument("--max-total", type=int, default=0, help="Overall sample cap; 0 = no cap")
    ap.add_argument("--point-size", type=float, default=12.0, help="Marker size for t-SNE points")
    ap.add_argument("--label-weight", type=float, default=0.05,
                    help="Strength of label one-hot appended to features (0 disables)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    feats_dir = Path(args.feats_dir)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # ---- load all ----
    X, y, yid, src_img, cell_idx = load_all_npz(feats_dir)

    # ---- stats BEFORE subsampling ----
    print_class_stats("FULL set (before subsampling)", y)

    # ---- subsampling (optional) ----
    X, y, yid, src_img, cell_idx = subsample(
        X, y, yid, src_img, cell_idx,
        max_per_class=args.max_per_class,
        max_total=args.max_total,
        seed=args.seed
    )

    # ---- stats AFTER subsampling ----
    print_class_stats("t-SNE set (after subsampling)", y)

    # ---- optional L2-normalize ----
    if args.normalize:
        X = normalize(X, norm="l2", axis=1)

    # ---- inject small label signal (slight impact) ----
    if args.label_weight and args.label_weight > 0:
        classes = np.unique(y)
        L = make_label_onehot(y, classes)                      # (N, K)
        X = np.hstack([X, args.label_weight * L])              # concatenate with small weight
        print(f"[label] appended one-hot of size {L.shape[1]} with weight {args.label_weight}")

    # ---- optional PCA ----
    if args.pca_dim and args.pca_dim < X.shape[1]:
        pca = PCA(n_components=args.pca_dim, random_state=args.seed, svd_solver="auto")
        X_red = pca.fit_transform(X)
    else:
        X_red = X

    # ---- t-SNE (guard perplexity) ----
    max_perp = max(5.0, (X_red.shape[0] - 1) / 3.0)
    perp = min(args.perplexity, max_perp)
    if perp != args.perplexity:
        print(f"[tsne] lowering perplexity from {args.perplexity} to {perp:.1f} for {X_red.shape[0]} samples")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
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
        feats=X, labels=y, label_ids=yid, tsne=Z,
        src_image=src_img, cell_index=cell_idx,
        pca_dim=args.pca_dim, perplexity=float(perp),
        normalized=bool(args.normalize), seed=args.seed,
        label_weight=float(args.label_weight)
    )

    if pd is not None:
        # also CSV for inspection (comment out features if too large)
        df_cols = {
            "x": Z[:, 0], "y": Z[:, 1],
            "label": y, "label_id": yid,
            "src_image": src_img, "cell_index": cell_idx,
        }
        # (optional) attach features columns — can be large
        # for j in range(X.shape[1]):
        #     df_cols[f"f{j}"] = X[:, j]
        pd.DataFrame(df_cols).to_csv(f"{out_prefix}_collated.csv", index=False)

    # ---- plot ----
    plt.figure(figsize=(10, 8))
    classes = np.unique(y)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    idxs = np.array([cls_to_idx[c] for c in y], dtype=int)

    cmap = choose_cmap(len(classes))
    plt.scatter(Z[:, 0], Z[:, 1], c=idxs, cmap=cmap,
                s=args.point_size, alpha=0.85, linewidths=0)

    # legend with matching colors
    legend_ms = max(6, args.point_size * 0.6)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=legend_ms,
                   markerfacecolor=cmap(i), markeredgecolor="none", label=cls)
        for i, cls in enumerate(classes)
    ]
    plt.legend(handles=handles, title="Classes",
               bbox_to_anchor=(1.05, 1.0), loc="upper left", borderaxespad=0.)
    plt.title("t-SNE of DINO ROI features (with label hint)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_tsne.png", dpi=200)
    plt.close()

    # neat “saved” print
    lines = [f"Saved:",
             f"  - {out_prefix}_collated.npz"]
    if pd is not None:
        lines.append(f"  - {out_prefix}_collated.csv")
    lines.append(f"  - {out_prefix}_tsne.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
