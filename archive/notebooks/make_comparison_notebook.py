"""
Generate comparison.ipynb -- builds all four comparison figures from the
three completed runs (U-Net, LSTM, Fusion).

Outputs (in runs/comparison/):
  - prediction_grid.png    : 6 test samples x [input, GT, U-Net, LSTM, Fusion]
  - per_class_iou_bar.png  : per-class IoU + mIoU bar chart across the 3 variants
  - training_curves.png    : val mIoU per epoch overlaid for the 3 runs
  - confmat_grid.png       : 3 row-normalized confusion matrices side by side
  - results_table.csv      : the numbers behind the bar chart, ready to drop in the PDF
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "comparison.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Compare U-Net vs Bi-LSTM vs Deep Fusion

Loads the three trained checkpoints and produces the figures we need for
the report:

1. **Prediction grid** -- 6 test samples shown side-by-side under all 3 models.
2. **Per-class IoU bar chart** -- ice / thin / water / mIoU across variants.
3. **Training curves overlay** -- val mIoU vs epoch for the 3 runs.
4. **Confusion matrix grid** -- 3 row-normalized confmats side-by-side.

All figures land in `runs/comparison/`.""")

# -------- Cell: GPU pin --------------------------------------------------
md("## 0. Setup")

code(r"""import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
""")

code(r"""import json, re, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- Cell: config ---------------------------------------------------
md("## 1. Config")

code(r"""PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
RUNS         = PROJECT_ROOT / "runs"
RUN_UNET     = RUNS / "unet_imgonly_v1"
RUN_LSTM     = RUNS / "lstm_csvonly_v1"
RUN_FUSION   = RUNS / "fusion_deep_v1"
OUT_DIR      = RUNS / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIR  = PROJECT_ROOT / "outputs"
MASK_DIR = PROJECT_ROOT / "outputs_segmented"
CSV_DIR  = PROJECT_ROOT / "IS2_Corrected_data"

SEED         = 42
NUM_CLASSES  = 3
PATCH        = 128
WINDOW_K     = 32
HALF         = WINDOW_K // 2
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.2
FUSION_CH    = 16

CLASS_NAMES = ["ice", "thin_ice", "water"]
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("output dir:", OUT_DIR)
""")

# -------- Cell: load shared data ---------------------------------------
md(r"""## 2. Load shared data (manifest, splits, CSV features)

The three runs all use the same manifest and splits, so we only need to
load them once.""")

code(r"""manifest = pd.read_csv(RUN_UNET / "manifest.csv")
print(f"manifest: {len(manifest):,} rows")

# attach csv_id
csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
csv_meta = pd.DataFrame([{
    "csv_path": str(p), "tile": p.stem.split("_")[3], "beam": p.stem.split("_")[4],
} for p in csv_files])
csv_meta["csv_id"] = csv_meta.index
manifest = manifest.merge(csv_meta[["tile", "beam", "csv_path", "csv_id"]],
                          on=["tile", "beam"], how="left")
manifest["csv_id"] = manifest["csv_id"].astype(int)

# splits
tiles_train = ["T02CNA", "T02CNC"]
tiles_test  = ["T03CWT"]
test_df = manifest[manifest["tile"].isin(tiles_test)].reset_index(drop=True)
print(f"test rows: {len(test_df):,}")

# normalized CSV features (cached by the LSTM run)
csv_features = {}
for cid in csv_meta["csv_id"]:
    arr = np.load(RUN_LSTM / "csv_normed" / f"csv_{cid}.npy")
    csv_features[int(cid)] = arr
n_features = next(iter(csv_features.values())).shape[1]
print(f"csv_features: {len(csv_features)} CSVs, {n_features} features")
""")

# -------- Cell: helpers -------------------------------------------------
md("## 3. Helpers (mask decoding, palette)")

code(r"""def mask_rgb_to_int(mask_rgb):
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    out[(mask_rgb == [255, 0, 0]).all(axis=-1)] = 0
    out[(mask_rgb == [0, 0, 255]).all(axis=-1)] = 1
    out[(mask_rgb == [0, 255, 0]).all(axis=-1)] = 2
    return out


def int_mask_to_rgb(m):
    out = np.zeros((*m.shape, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        out[m == c] = color
    return out
""")

# -------- Cell: model definitions --------------------------------------
md(r"""## 4. Model definitions (must match the training notebooks)""")

code(r"""class CSVOnlyModel(nn.Module):
    def __init__(self, n_features, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, num_classes=NUM_CLASSES, patch=PATCH):
        super().__init__()
        self.patch = patch
        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Conv2d(hidden * 2, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x, valid=None):
        h = self.proj(x)
        h, _ = self.lstm(h)
        center = x.size(1) // 2
        feat = h[:, center, :][:, :, None, None].expand(-1, -1, self.patch, self.patch)
        return self.head(feat)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, bottleneck)
        self.fc2 = nn.Linear(bottleneck, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        z = self.gap(x).view(b, c)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return x * z.view(b, c, 1, 1)


class DeepFusionModel(nn.Module):
    def __init__(self, n_features, num_classes=NUM_CLASSES, patch=PATCH,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
                 lstm_dropout=LSTM_DROPOUT, fusion_ch=FUSION_CH):
        super().__init__()
        self.patch = patch
        self.unet = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                             in_channels=3, classes=fusion_ch)
        self.csv_proj = nn.Sequential(
            nn.Linear(n_features, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=lstm_dropout if lstm_layers > 1 else 0.0)
        self.csv_to_chan = nn.Linear(lstm_hidden * 2, fusion_ch)
        self.se = SqueezeExcitation(channels=fusion_ch * 2, reduction=8)
        self.head = nn.Sequential(
            nn.Conv2d(fusion_ch * 2, fusion_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fusion_ch, num_classes, kernel_size=1),
        )

    def forward(self, img, csv_window, valid=None):
        img_feat = self.unet(img)
        h = self.csv_proj(csv_window)
        h, _ = self.lstm(h)
        center = csv_window.size(1) // 2
        csv_feat = self.csv_to_chan(h[:, center, :])
        csv_feat = csv_feat[:, :, None, None].expand(-1, -1, self.patch, self.patch)
        fused = torch.cat([img_feat, csv_feat], dim=1)
        return self.head(self.se(fused))
""")

# -------- Cell: load checkpoints ---------------------------------------
md(r"""## 5. Load all three trained models""")

code(r"""# U-Net image-only
unet = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                in_channels=3, classes=NUM_CLASSES).to(device)
ck = torch.load(RUN_UNET / "best.pt", map_location=device, weights_only=False)
unet.load_state_dict(ck["model_state"]); unet.eval()
print(f"U-Net loaded   (epoch {ck['epoch']}, val mIoU {ck['val_metrics']['miou']:.4f})")

# Bi-LSTM CSV-only
lstm = CSVOnlyModel(n_features=n_features).to(device)
ck = torch.load(RUN_LSTM / "best.pt", map_location=device, weights_only=False)
lstm.load_state_dict(ck["model_state"]); lstm.eval()
print(f"Bi-LSTM loaded (epoch {ck['epoch']}, val mIoU {ck['val_metrics']['miou']:.4f})")

# Deep fusion
fusion = DeepFusionModel(n_features=n_features).to(device)
ck = torch.load(RUN_FUSION / "best.pt", map_location=device, weights_only=False)
fusion.load_state_dict(ck["model_state"]); fusion.eval()
print(f"Fusion loaded  (epoch {ck['epoch']}, val mIoU {ck['val_metrics']['miou']:.4f})")
""")

# -------- Cell: build the side-by-side prediction grid ----------------
md(r"""## 6. Side-by-side prediction grid

6 random test samples, each row shows: input photo | ground truth |
U-Net prediction | LSTM prediction | Fusion prediction.""")

code(r"""def build_image_input(rgb_raw):
    img = ((rgb_raw.astype(np.float32) / 255.0 - IM_MEAN) / IM_STD)
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))[None].to(device)


def build_csv_input(csv_id, row_idx):
    feats = csv_features[int(csv_id)]
    n_rows = feats.shape[0]
    win = np.zeros((WINDOW_K, n_features), dtype=np.float32)
    for k in range(WINDOW_K):
        src = int(row_idx) - HALF + k
        if 0 <= src < n_rows:
            win[k] = feats[src]
    return torch.from_numpy(win)[None].to(device)


# Pick the same 6 test samples each time
sample = test_df.sample(n=6, random_state=SEED).reset_index(drop=True)

n_rows = len(sample)
n_cols = 5  # input, GT, U-Net, LSTM, Fusion
fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.4 * n_rows))
column_titles = ["input", "ground truth", "U-Net (image only)",
                 "Bi-LSTM (CSV only)", "Deep fusion"]

with torch.no_grad():
    for i, r in sample.iterrows():
        rgb = np.array(Image.open(r["image_path"]).convert("RGB"))
        gt  = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))

        img_t = build_image_input(rgb)
        win_t = build_csv_input(r["csv_id"], r["row_idx"])

        p_unet   = unet(img_t).argmax(1)[0].cpu().numpy()
        p_lstm   = lstm(win_t).argmax(1)[0].cpu().numpy()
        p_fusion = fusion(img_t, win_t).argmax(1)[0].cpu().numpy()

        panels = [rgb, int_mask_to_rgb(gt), int_mask_to_rgb(p_unet),
                  int_mask_to_rgb(p_lstm), int_mask_to_rgb(p_fusion)]

        for j, (p, title) in enumerate(zip(panels, column_titles)):
            axes[i, j].imshow(p)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(title, fontsize=10)
        axes[i, 0].set_ylabel(r["filename"], fontsize=6, rotation=0,
                              ha="right", va="center", labelpad=70)

# small legend strip at the bottom
legend_handles = [
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=(1, 0, 0),    markersize=12, label="ice"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=(0, 0, 1),    markersize=12, label="thin ice"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=(0, 1, 0),    markersize=12, label="water"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.005))

plt.tight_layout()
plt.subplots_adjust(bottom=0.04)
fig_path = OUT_DIR / "prediction_grid.png"
plt.savefig(fig_path, dpi=160, bbox_inches="tight")
plt.show()
print(f"saved -> {fig_path}")
""")

# -------- Cell: per-class IoU bar chart --------------------------------
md(r"""## 7. Per-class IoU bar chart""")

code(r"""def load_test_metrics(run_dir):
    with open(run_dir / "test_metrics.json") as f:
        return json.load(f)


m_unet   = load_test_metrics(RUN_UNET)
m_lstm   = load_test_metrics(RUN_LSTM)
m_fusion = load_test_metrics(RUN_FUSION)

variants = ["U-Net\n(image only)", "Bi-LSTM\n(CSV only)", "Deep fusion"]
colors   = ["#4C72B0", "#55A868", "#C44E52"]

categories = ["ice", "thin ice", "water", "mIoU"]
data = np.array([
    [m_unet["per_iou"][0],   m_unet["per_iou"][1],   m_unet["per_iou"][2],   m_unet["miou"]],
    [m_lstm["per_iou"][0],   m_lstm["per_iou"][1],   m_lstm["per_iou"][2],   m_lstm["miou"]],
    [m_fusion["per_iou"][0], m_fusion["per_iou"][1], m_fusion["per_iou"][2], m_fusion["miou"]],
])

x = np.arange(len(categories))
width = 0.27
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for i, (variant, color) in enumerate(zip(variants, colors)):
    bars = ax.bar(x + (i - 1) * width, data[i], width, label=variant, color=color)
    for b, val in zip(bars, data[i]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel("IoU (test set, T03CWT)")
ax.set_ylim(0, 1.05)
ax.set_title("Per-class IoU and mIoU on the held-out test tile")
ax.legend(loc="lower left", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig_path = OUT_DIR / "per_class_iou_bar.png"
plt.savefig(fig_path, dpi=160)
plt.show()
print(f"saved -> {fig_path}")

# Also save the underlying numbers as CSV for easy paste into the report
results_df = pd.DataFrame(data, index=variants, columns=categories)
results_df.to_csv(OUT_DIR / "results_table.csv")
print(results_df.round(4))
""")

# -------- Cell: training curves overlay --------------------------------
md(r"""## 8. Training curves overlay (val mIoU per epoch)""")

code(r"""def load_curve(run_dir):
    return pd.read_csv(run_dir / "metrics.csv")


curves = {
    "U-Net (image only)":   (load_curve(RUN_UNET),   "#4C72B0"),
    "Bi-LSTM (CSV only)":   (load_curve(RUN_LSTM),   "#55A868"),
    "Deep fusion":          (load_curve(RUN_FUSION), "#C44E52"),
}

fig, ax = plt.subplots(figsize=(8.5, 4.5))
for name, (df, color) in curves.items():
    ax.plot(df["epoch"], df["val_miou"], "-o", color=color, label=name,
            linewidth=2, markersize=4)
    # mark the best epoch (peak val mIoU)
    best = df["val_miou"].idxmax()
    ax.scatter(df.loc[best, "epoch"], df.loc[best, "val_miou"],
               s=120, edgecolor=color, facecolor="none", linewidth=2, zorder=5)

ax.set_xlabel("epoch")
ax.set_ylabel("validation mIoU")
ax.set_title("Validation mIoU per epoch (circle = best epoch)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = OUT_DIR / "training_curves.png"
plt.savefig(fig_path, dpi=160)
plt.show()
print(f"saved -> {fig_path}")
""")

# -------- Cell: confusion matrix grid ----------------------------------
md(r"""## 9. Confusion matrix grid (3 side by side)""")

code(r"""def load_test_cm_only(run_dir):
    # The metrics JSON we save excludes 'cm', so re-derive from a fresh test pass...
    # ...actually we have it on disk in confmat.png, but let's recompute exactly the
    # same numbers so we can plot a side-by-side at consistent style.
    # Cheat: read the saved test_metrics.json -- only has miou/per_iou/loss/pix_acc.
    # So we re-run inference once and accumulate. Quick on test set.
    return None  # placeholder -- handled below


# Run inference once across the test set per model, accumulating the confusion matrix.
from torch.utils.data import Dataset, DataLoader


class TestSet(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rgb = np.array(Image.open(r["image_path"]).convert("RGB"))
        mask = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))
        img = (rgb.astype(np.float32) / 255.0 - IM_MEAN) / IM_STD
        img = np.transpose(img, (2, 0, 1))

        feats = csv_features[int(r["csv_id"])]
        n_rows = feats.shape[0]; center = int(r["row_idx"])
        win = np.zeros((WINDOW_K, n_features), dtype=np.float32)
        for k in range(WINDOW_K):
            src = center - HALF + k
            if 0 <= src < n_rows:
                win[k] = feats[src]
        return (torch.from_numpy(img), torch.from_numpy(win),
                torch.from_numpy(mask).long())


loader = DataLoader(TestSet(test_df), batch_size=64, shuffle=False,
                    num_workers=4, pin_memory=True)


def confmat(model_fn):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for img, win, y in loader:
            img = img.to(device, non_blocking=True)
            win = win.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                pred = model_fn(img, win).argmax(1).cpu().numpy().ravel()
            t = y.numpy().ravel()
            idx = NUM_CLASSES * t + pred
            cm += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cm


cms = {
    "U-Net (image only)":  confmat(lambda img, win: unet(img)),
    "Bi-LSTM (CSV only)":  confmat(lambda img, win: lstm(win)),
    "Deep fusion":          confmat(lambda img, win: fusion(img, win)),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (name, cm) in zip(axes, cms.items()):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    if ax is axes[0]:
        ax.set_ylabel("true")
    ax.set_title(name, fontsize=11)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=10)

cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.66])
fig.colorbar(im, cax=cbar_ax)
plt.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.12, wspace=0.25)
fig_path = OUT_DIR / "confmat_grid.png"
plt.savefig(fig_path, dpi=160)
plt.show()
print(f"saved -> {fig_path}")
""")

# -------- Cell: summary --------------------------------------------------
md(r"""## 10. Summary

All four figures saved to `runs/comparison/`:

* `prediction_grid.png`
* `per_class_iou_bar.png`
* `training_curves.png`
* `confmat_grid.png`
* `results_table.csv`

These are ready to embed in the methodology PDF / final report.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
