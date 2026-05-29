"""Generate green-colormap row-normalized confusion matrices for all three fusion strategies."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "deepfusion": {
        "title": "Deep Fusion Confusion Matrix",
        "labels": ["thick ice", "thin ice", "water"],
        "pred_labels": ["Predicted thick ice", "Predicted thin ice", "Predicted water"],
        # Row-normalized percentages from test_metrics.json cm field
        "cm_norm": np.array([
            [96.20, 3.80, 0.00],
            [7.51,  92.21, 0.28],
            [0.10,  3.86, 96.04],
        ]),
    },
    "latefusion": {
        "title": "Late Fusion Confusion Matrix",
        "labels": ["ice", "thin_ice", "water"],
        "pred_labels": ["Pred ice", "Pred thin_ice", "Pred water"],
        "cm_norm": np.array([
            [97.96, 2.04, 0.00],
            [15.73, 83.93, 0.34],
            [1.79,  4.97, 93.24],
        ]),
    },
    "hybridfusion": {
        "title": "Hybrid Fusion Confusion Matrix",
        "labels": ["ice", "thin_ice", "water"],
        "pred_labels": ["Pred ice", "Pred thin_ice", "Pred water"],
        "cm_norm": np.array([
            [97.44, 2.56, 0.00],
            [11.98, 87.25, 0.77],
            [0.32,  3.72, 95.95],
        ]),
    },
}


def plot_confmat(cm_norm, title, labels, pred_labels, out_path, cmap="Greens"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pred_labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


for name, cfg in MODELS.items():
    out_green = os.path.join(OUTPUT_DIR, f"{name}_green.png")
    plot_confmat(
        cfg["cm_norm"], cfg["title"], cfg["labels"], cfg["pred_labels"],
        out_green, cmap="Greens"
    )
    out_blue = os.path.join(OUTPUT_DIR, f"{name}_blue.png")
    plot_confmat(
        cfg["cm_norm"], cfg["title"], cfg["labels"], cfg["pred_labels"],
        out_blue, cmap="Blues"
    )

print("Done.")
