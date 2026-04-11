#!/usr/bin/env python3
"""
Visualize DINO features saved by eval_knn.py (--dump_features).

Generates and saves to the features directory:
  - label_distribution.png  : class counts for train and test sets
  - pca.png                 : 2-D PCA of train and test features
  - pca3d.png               : 3-D PCA of train features
  - tsne.png                : 2-D t-SNE of train and test subsets
  - cosine_similarity.png   : cosine similarity between class centroids

Usage:
  python plot_features.py --features_dir OUTPUT_DINO_ITERATIVE/pass1_shape/features
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

CLASS_NAMES = ["Cats", "Dogs"]
COLORS = ["steelblue", "tomato"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(features_dir):
    train_feats  = torch.load(os.path.join(features_dir, "trainfeat.pth")).numpy()
    test_feats   = torch.load(os.path.join(features_dir, "testfeat.pth")).numpy()
    train_labels = torch.load(os.path.join(features_dir, "trainlabels.pth")).numpy()
    test_labels  = torch.load(os.path.join(features_dir, "testlabels.pth")).numpy()
    print(f"Train: {train_feats.shape},  labels: {np.unique(train_labels)}")
    print(f"Test:  {test_feats.shape},   labels: {np.unique(test_labels)}")
    return train_feats, test_feats, train_labels, test_labels


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_label_distribution(train_labels, test_labels, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, labels, title in zip(axes, [train_labels, test_labels], ["Train", "Test"]):
        counts = np.bincount(labels)
        ax.bar(CLASS_NAMES[:len(counts)], counts)
        ax.set_title(f"{title} label distribution")
        ax.set_ylabel("Count")
    plt.tight_layout()
    path = os.path.join(out_dir, "label_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_pca2d(train_feats, test_feats, train_labels, test_labels, out_dir):
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_feats)
    test_pca  = pca.transform(test_feats)
    print(f"PCA 2D explained variance ratio: {pca.explained_variance_ratio_}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, emb, labels, title in zip(
        axes,
        [train_pca, test_pca],
        [train_labels, test_labels],
        ["Train", "Test"],
    ):
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            mask = labels == cls_id
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=COLORS[cls_id], label=cls_name,
                       alpha=0.4, s=5, rasterized=True)
        ax.set_title(f"PCA — {title}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=3)
    plt.tight_layout()
    path = os.path.join(out_dir, "pca.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_pca3d(train_feats, train_labels, out_dir):
    pca3 = PCA(n_components=3)
    train_pca3 = pca3.fit_transform(train_feats)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = train_labels == cls_id
        ax.scatter(train_pca3[mask, 0], train_pca3[mask, 1], train_pca3[mask, 2],
                   c=COLORS[cls_id], label=cls_name, alpha=0.3, s=3, rasterized=True)
    ax.set_title("PCA 3D — Train")
    ax.legend(markerscale=3)
    plt.tight_layout()
    path = os.path.join(out_dir, "pca3d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_tsne(train_feats, test_feats, train_labels, test_labels, out_dir, n_samples=5000):
    rng = np.random.default_rng(42)
    idx_tr = rng.choice(len(train_feats), min(n_samples, len(train_feats)), replace=False)
    idx_te = rng.choice(len(test_feats),  min(n_samples, len(test_feats)),  replace=False)

    # Reduce to 50 PCA dims first to speed up t-SNE
    pca50 = PCA(n_components=50)
    tr50 = pca50.fit_transform(train_feats[idx_tr])
    te50 = pca50.fit_transform(test_feats[idx_te])

    print("Running t-SNE on train subset...")
    tsne_tr = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(tr50)
    print("Running t-SNE on test subset...")
    tsne_te = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(te50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, emb, labels, idx, title in zip(
        axes,
        [tsne_tr, tsne_te],
        [train_labels, test_labels],
        [idx_tr, idx_te],
        ["Train", "Test"],
    ):
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            mask = labels[idx] == cls_id
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=COLORS[cls_id], label=cls_name,
                       alpha=0.5, s=5, rasterized=True)
        ax.set_title(f"t-SNE — {title} (n={len(idx)})")
        ax.legend(markerscale=3)
    plt.tight_layout()
    path = os.path.join(out_dir, "tsne.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_cosine_similarity(train_feats, train_labels, out_dir):
    centroids = np.stack([
        train_feats[train_labels == cls_id].mean(axis=0)
        for cls_id in range(len(CLASS_NAMES))
    ])
    sim = cosine_similarity(centroids)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{sim[i, j]:.3f}", ha="center", va="center", color="white")
    plt.colorbar(im, ax=ax)
    ax.set_title("Cosine similarity between class centroids (train)")
    plt.tight_layout()
    path = os.path.join(out_dir, "cosine_similarity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot DINO feature visualizations.")
    parser.add_argument("--features_dir", required=True,
                        help="Directory containing trainfeat.pth / testfeat.pth / *labels.pth")
    parser.add_argument("--n_samples_tsne", type=int, default=5000,
                        help="Max samples for t-SNE (default: 5000)")
    args = parser.parse_args()

    print(f"\nLoading features from: {args.features_dir}")
    train_feats, test_feats, train_labels, test_labels = load_features(args.features_dir)

    plot_label_distribution(train_labels, test_labels, args.features_dir)
    plot_pca2d(train_feats, test_feats, train_labels, test_labels, args.features_dir)
    plot_pca3d(train_feats, train_labels, args.features_dir)
    plot_tsne(train_feats, test_feats, train_labels, test_labels,
              args.features_dir, args.n_samples_tsne)
    plot_cosine_similarity(train_feats, train_labels, args.features_dir)

    print(f"\nAll plots saved to: {args.features_dir}")


if __name__ == "__main__":
    main()
