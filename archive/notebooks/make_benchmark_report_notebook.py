"""
Generate benchmark_report.ipynb -- one-stop benchmark report comparing the
two inference modes of the Deep Fusion model (U-Net + Bi-LSTM, SE-fused)
on the held-out test tile T03CWT:

  * Deep Fusion (no TTA)  -- single forward pass per sample
  * Deep Fusion (+ TTA)   -- 4-view test-time augmentation
                             (identity / hflip / vflip / rot180),
                             averaged via inverse-softmax mean.

Both modes load the SAME trained checkpoint -- the only difference is at
inference, so there is a single training loss curve.

Produces (in runs/benchmark_report/):
  - benchmark_summary.csv          : one row per mode, full metric suite
  - per_class_breakdown.csv        : per-class precision/recall/F1/IoU
  - confmat_no_tta_counts.png      : raw-count confusion matrix (no TTA)
  - confmat_no_tta_percent.png     : prof-style row-% CM (no TTA)
  - confmat_tta_counts.png         : raw-count CM (with TTA)
  - confmat_tta_percent.png        : prof-style row-% CM (with TTA)
  - loss_curves_fusion.png         : 4-panel training diagnostics
  - benchmark_bar.png              : acc / macro-F1 / mIoU bar chart

This mirrors the project's `make_*_notebook.py` -> `*.ipynb` convention.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "benchmark_report.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Benchmark report -- Deep Fusion: no TTA vs with TTA

Single report notebook for the two inference modes we care about.

| short name              | model                                          | inference                                 |
|-------------------------|------------------------------------------------|-------------------------------------------|
| **Deep Fusion (no TTA)**| U-Net (ResNet-18) + Bi-LSTM, SE-attention fused | single forward pass per sample            |
| **Deep Fusion (+ TTA)** | same model -- same weights                     | 4-view TTA: id / hflip / vflip / rot180   |

Both modes load **the same checkpoint** (`runs/fusion_deep_v1/best.pt`).
TTA averages predictions across 4 input orientations and inverse-flips them
back to the original frame before taking the softmax mean. So:

* There is **one training loss curve** (the Fusion model's training history).
* There are **two test-set confusion matrices** -- one per inference mode.

For each inference mode this notebook produces:

1. **Benchmark suite** -- pixel accuracy, mIoU, macro-F1, weighted-F1,
   plus per-class precision / recall / F1 / IoU.
2. **Confusion matrix in two flavours**
   * raw-count matrix (appendix)
   * **professor-style row-normalised percentage matrix** -- Blues colormap,
     `thick ice / thin ice / water` labels, title `Confusion Matrix (Percentages)`.

Plus a single shared **detailed loss-curve panel** for the training run.
""")

# -------------------------------------------------------------------------
md("## 0. Setup")

code(r"""import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
""")

code(r"""import json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------------------------------------------------------------------------
md(r"""## 1. Config

Paths assume the standard project layout. Adjust `PROJECT_ROOT` if running
outside the original training environment.""")

code(r"""PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
RUNS         = PROJECT_ROOT / "runs"
RUN_LSTM     = RUNS / "lstm_csvonly_v1"      # only used for cached CSV features + manifest
RUN_FUSION   = RUNS / "fusion_deep_v1"
OUT_DIR      = RUNS / "benchmark_report"
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

# internal class index order (ice / thin / water) <-> display order (thick ice / thin ice / water)
CLASS_NAMES_INT  = ["ice", "thin_ice", "water"]
CLASS_NAMES_DISP = ["thick ice", "thin ice", "water"]

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("output dir:", OUT_DIR)
""")

# -------------------------------------------------------------------------
md(r"""## 2. Manifest + test split

The manifest is shared across runs. For the test set we keep tile `T03CWT`.""")

code(r"""manifest_path = None
for cand in [RUN_FUSION / "manifest.csv", RUN_LSTM / "manifest.csv",
             RUNS / "unet_imgonly_v1" / "manifest.csv"]:
    if cand.exists():
        manifest_path = cand
        break
assert manifest_path is not None, "could not find manifest.csv in any run dir"
print("using manifest:", manifest_path)

manifest = pd.read_csv(manifest_path)
csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
csv_meta = pd.DataFrame([{
    "csv_path": str(p), "tile": p.stem.split("_")[3], "beam": p.stem.split("_")[4],
} for p in csv_files])
csv_meta["csv_id"] = csv_meta.index
manifest = manifest.merge(csv_meta[["tile", "beam", "csv_path", "csv_id"]],
                          on=["tile", "beam"], how="left")
manifest["csv_id"] = manifest["csv_id"].astype(int)

test_df = manifest[manifest["tile"].isin(["T03CWT"])].reset_index(drop=True)
print(f"test patches: {len(test_df):,}")

# Cached, normalised CSV features (written by the Bi-LSTM training notebook).
csv_features = {}
for cid in csv_meta["csv_id"]:
    arr = np.load(RUN_LSTM / "csv_normed" / f"csv_{cid}.npy")
    csv_features[int(cid)] = arr
n_features = next(iter(csv_features.values())).shape[1]
print(f"csv_features: {len(csv_features)} CSVs, {n_features} features")
""")

# -------------------------------------------------------------------------
md(r"""## 3. Dataset + Deep Fusion model definition""")

code(r"""def mask_rgb_to_int(mask_rgb):
    # color encoding: ice=red, thin_ice=blue, water=green
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    out[(mask_rgb == [255, 0, 0]).all(axis=-1)] = 0
    out[(mask_rgb == [0, 0, 255]).all(axis=-1)] = 1
    out[(mask_rgb == [0, 255, 0]).all(axis=-1)] = 2
    return out


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


test_loader = DataLoader(TestSet(test_df), batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)


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
        self.csv_proj = nn.Sequential(nn.Linear(n_features, lstm_hidden),
                                      nn.LayerNorm(lstm_hidden), nn.ReLU())
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=lstm_dropout if lstm_layers > 1 else 0.0)
        self.csv_to_chan = nn.Linear(lstm_hidden * 2, fusion_ch)
        self.se = SqueezeExcitation(channels=fusion_ch * 2, reduction=8)
        self.head = nn.Sequential(
            nn.Conv2d(fusion_ch * 2, fusion_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
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

# -------------------------------------------------------------------------
md(r"""## 4. Load the Fusion checkpoint (shared by both inference modes)""")

code(r"""fusion = DeepFusionModel(n_features=n_features).to(device)
ck = torch.load(RUN_FUSION / "best.pt", map_location=device, weights_only=False)
fusion.load_state_dict(ck["model_state"]); fusion.eval()
val_miou = ck["val_metrics"]["miou"]
print(f"Deep Fusion loaded  (val mIoU {val_miou:.4f})")
""")

# -------------------------------------------------------------------------
md(r"""## 5. TTA helpers

Four-view test-time augmentation. The image branch sees each of these
orientations; the CSV branch always sees the original photon window (the
along-track photon order is intrinsic data, not a spatial augmentation).
We invert each spatial transform on the model output before averaging so
the predictions are all in the same orientation.""")

code(r"""def view_forward(view, img):
    if view == "id":     return img
    if view == "hflip":  return torch.flip(img, dims=[-1])
    if view == "vflip":  return torch.flip(img, dims=[-2])
    if view == "rot180": return torch.flip(img, dims=[-1, -2])
    raise ValueError(view)


def view_inverse(view, y):
    # spatial flips are self-inverse, so the inverse is identical
    if view == "id":     return y
    if view == "hflip":  return torch.flip(y, dims=[-1])
    if view == "vflip":  return torch.flip(y, dims=[-2])
    if view == "rot180": return torch.flip(y, dims=[-1, -2])
    raise ValueError(view)


VIEWS = ["id", "hflip", "vflip", "rot180"]
print("TTA views:", VIEWS)
""")

# -------------------------------------------------------------------------
md(r"""## 6. Build the two test-set confusion matrices

One sweep through the test loader. For each batch we run:
  * a single forward pass (no-TTA prediction)
  * four forward passes (one per TTA view), inverse-flip them, average their
    softmaxes, then `argmax` for the TTA prediction.""")

code(r"""@torch.no_grad()
def gather_cms():
    cm_no_tta = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_tta    = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for img, win, y in test_loader:
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            # no-TTA: single forward pass
            log_single = fusion(img, win)
            pred_no_tta = log_single.argmax(1)

            # TTA: 4 views, average softmaxes in original orientation
            prob_sum = None
            for v in VIEWS:
                img_v = view_forward(v, img)
                log_v = fusion(img_v, win)
                prob_v = F.softmax(log_v, dim=1)
                prob_v = view_inverse(v, prob_v)
                prob_sum = prob_v if prob_sum is None else prob_sum + prob_v
            prob_avg = prob_sum / len(VIEWS)
            pred_tta = prob_avg.argmax(1)

        t = y.numpy().ravel()
        for cm, pred in [(cm_no_tta, pred_no_tta), (cm_tta, pred_tta)]:
            p = pred.cpu().numpy().ravel()
            idx = NUM_CLASSES * t + p
            cm += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

    return {"Deep Fusion (no TTA)": cm_no_tta, "Deep Fusion (+ TTA)": cm_tta}


cms = gather_cms()
for name, cm in cms.items():
    print(f"{name}: total pixels = {cm.sum():,}")
""")

# -------------------------------------------------------------------------
md(r"""## 7. Metric suite from each CM

All numbers derived from the confusion matrix:

* **pixel accuracy** = `diag.sum() / total`
* **per-class precision** = `TP / (TP + FP)` = `diag[c] / column_sum[c]`
* **per-class recall**    = `TP / (TP + FN)` = `diag[c] / row_sum[c]`
* **F1**                  = `2*P*R / (P+R)`
* **IoU**                 = `TP / (TP + FP + FN)`
* **macro-F1**            = mean of per-class F1
* **weighted-F1**         = support-weighted F1
* **mIoU**                = mean of per-class IoU
""")

code(r"""def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    total = cm.sum()
    diag  = np.diag(cm)
    support  = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)

    pix_acc = diag.sum() / max(total, 1.0)

    precision = np.where(pred_sum > 0, diag / np.maximum(pred_sum, 1), 0.0)
    recall    = np.where(support  > 0, diag / np.maximum(support,  1), 0.0)
    f1_denom  = precision + recall
    f1 = np.where(f1_denom > 0, 2 * precision * recall / np.maximum(f1_denom, 1e-12), 0.0)

    iou_denom = support + pred_sum - diag
    iou = np.where(iou_denom > 0, diag / np.maximum(iou_denom, 1), 0.0)

    macro_f1    = float(f1.mean())
    weighted_f1 = float((f1 * support).sum() / max(total, 1.0))
    miou        = float(iou.mean())

    return {
        "pix_acc": float(pix_acc),
        "miou":    miou,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


metrics = {name: metrics_from_cm(cm) for name, cm in cms.items()}
""")

# -------------------------------------------------------------------------
md(r"""## 8. Benchmark summary table (one row per inference mode)""")

code(r"""rows = []
for name, m in metrics.items():
    rows.append({
        "model":            name,
        "pixel_accuracy":   round(m["pix_acc"], 4),
        "mIoU":             round(m["miou"], 4),
        "macro_precision":  round(float(m["precision"].mean()), 4),
        "macro_recall":     round(float(m["recall"].mean()), 4),
        "macro_F1":         round(m["macro_f1"], 4),
        "weighted_F1":      round(m["weighted_f1"], 4),
        "ice_IoU":          round(float(m["iou"][0]), 4),
        "thin_IoU":         round(float(m["iou"][1]), 4),
        "water_IoU":        round(float(m["iou"][2]), 4),
    })

summary = pd.DataFrame(rows).set_index("model")
summary.to_csv(OUT_DIR / "benchmark_summary.csv")
print(summary.to_string())
""")

# -------------------------------------------------------------------------
md(r"""## 9. Per-class precision / recall / F1 / IoU breakdown""")

code(r"""breakdown_rows = []
for name, m in metrics.items():
    for c, cname in enumerate(CLASS_NAMES_DISP):
        breakdown_rows.append({
            "model":     name,
            "class":     cname,
            "support":   int(m["support"][c]),
            "precision": round(float(m["precision"][c]), 4),
            "recall":    round(float(m["recall"][c]),    4),
            "F1":        round(float(m["f1"][c]),        4),
            "IoU":       round(float(m["iou"][c]),       4),
        })

breakdown = pd.DataFrame(breakdown_rows)
breakdown.to_csv(OUT_DIR / "per_class_breakdown.csv", index=False)
print(breakdown.to_string(index=False))
""")

# -------------------------------------------------------------------------
md(r"""## 10. Confusion matrix -- professor-style percentage view

Row-normalised percentages (each row sums to 100), Blues colormap,
y-ticks `thick ice / thin ice / water`, x-ticks `Predicted thick ice /
Predicted thin ice / Predicted water`, title `Confusion Matrix (Percentages)`.
A raw-count version is also saved for the appendix.""")

code(r"""def plot_cm_percent(cm, title, save_path):
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, cm / np.maximum(row_sums, 1) * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"Predicted {c}" for c in CLASS_NAMES_DISP], fontsize=11)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES_DISP, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = pct[i, j]
            color = "white" if v > 55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_cm_counts(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"Predicted {c}" for c in CLASS_NAMES_DISP], fontsize=11)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES_DISP, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13)

    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = int(cm[i, j])
            color = "white" if v > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color=color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
""")

code(r"""# Deep Fusion (no TTA)
plot_cm_percent(cms["Deep Fusion (no TTA)"],
                "Deep Fusion (no TTA) -- Confusion Matrix (Percentages)",
                OUT_DIR / "confmat_no_tta_percent.png")
plot_cm_counts(cms["Deep Fusion (no TTA)"],
               "Deep Fusion (no TTA) -- Confusion Matrix (Counts)",
               OUT_DIR / "confmat_no_tta_counts.png")
""")

code(r"""# Deep Fusion (+ TTA)
plot_cm_percent(cms["Deep Fusion (+ TTA)"],
                "Deep Fusion (+ TTA) -- Confusion Matrix (Percentages)",
                OUT_DIR / "confmat_tta_percent.png")
plot_cm_counts(cms["Deep Fusion (+ TTA)"],
               "Deep Fusion (+ TTA) -- Confusion Matrix (Counts)",
               OUT_DIR / "confmat_tta_counts.png")
""")

# -------------------------------------------------------------------------
md(r"""## 11. Detailed loss curves

Both inference modes share the same trained weights, so there's a single
training curve. The Fusion training notebook writes a per-epoch
`metrics.csv` with these columns:

```
epoch, train_loss, val_loss, val_miou,
iou_ice, iou_thin, iou_water, pix_acc, lr
```

The panel below shows train/val loss, val mIoU + per-class IoU, pixel
accuracy, and the LR schedule.""")

code(r"""hist_path = RUN_FUSION / "metrics.csv"
assert hist_path.exists(), f"missing training history: {hist_path}"
hist = pd.read_csv(hist_path)
print(f"Deep Fusion training history: {len(hist)} epochs")
print("columns:", list(hist.columns))


def plot_history_detail(h, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    # 1) Train + val loss
    ax = axes[0, 0]
    ax.plot(h["epoch"], h["train_loss"], "-o", label="train loss",
            color="#1f77b4", markersize=4)
    ax.plot(h["epoch"], h["val_loss"],   "-o", label="val loss",
            color="#d62728", markersize=4)
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training vs validation loss")
    ax.grid(alpha=0.3); ax.legend()

    # 2) Validation mIoU + per-class IoU
    ax = axes[0, 1]
    ax.plot(h["epoch"], h["val_miou"], "-o", label="val mIoU",
            color="black", linewidth=2, markersize=4)
    ax.plot(h["epoch"], h["iou_ice"],   "--", label="thick ice IoU", color="#e41a1c")
    ax.plot(h["epoch"], h["iou_thin"],  "--", label="thin ice IoU",  color="#377eb8")
    ax.plot(h["epoch"], h["iou_water"], "--", label="water IoU",     color="#4daf4a")
    ax.set_xlabel("epoch"); ax.set_ylabel("IoU")
    ax.set_title("Validation mIoU and per-class IoU")
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 3) Pixel accuracy
    ax = axes[1, 0]
    ax.plot(h["epoch"], h["pix_acc"], "-o", color="#2ca02c", markersize=4)
    ax.set_xlabel("epoch"); ax.set_ylabel("pixel accuracy")
    ax.set_title("Validation pixel accuracy")
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3)

    # 4) LR schedule
    ax = axes[1, 1]
    ax.plot(h["epoch"], h["lr"], "-o", color="#ff7f0e", markersize=4)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title("LR schedule")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")

    fig.suptitle("Deep Fusion -- training diagnostics", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


plot_history_detail(hist, OUT_DIR / "loss_curves_fusion.png")
""")

# -------------------------------------------------------------------------
md(r"""## 12. Benchmark bar chart

Headline metrics side-by-side -- the visual that goes in the report.""")

code(r"""variants = list(summary.index)
metrics_to_plot = ["pixel_accuracy", "macro_F1", "mIoU"]
nice_labels = {"pixel_accuracy": "pixel accuracy",
               "macro_F1":       "macro F1",
               "mIoU":           "mIoU"}
data = summary[metrics_to_plot].to_numpy()

x = np.arange(len(metrics_to_plot))
width = 0.35
palette = ["#1f77b4", "#d62728"]

fig, ax = plt.subplots(figsize=(8.5, 4.8))
for i, (variant, color) in enumerate(zip(variants, palette)):
    offset = (i - (len(variants) - 1) / 2) * width
    bars = ax.bar(x + offset, data[i], width, label=variant, color=color)
    for b, val in zip(bars, data[i]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.005, f"{val:.4f}",
                ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([nice_labels[m] for m in metrics_to_plot], fontsize=11)
ax.set_ylabel("test-set score")
ax.set_ylim(0, 1.05)
ax.set_title("Deep Fusion -- no TTA vs with TTA (test tile T03CWT)")
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "benchmark_bar.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------------------------------------------------------------------------
md(r"""## 13. What this notebook produced

All artifacts land in `runs/benchmark_report/`:

| file                            | what it is                                                |
|---------------------------------|-----------------------------------------------------------|
| `benchmark_summary.csv`         | one row per mode: acc / mIoU / P / R / F1 / per-class IoU |
| `per_class_breakdown.csv`       | per-class precision/recall/F1/IoU per mode                |
| `confmat_no_tta_percent.png`    | prof-style row-% CM for Deep Fusion (no TTA)              |
| `confmat_no_tta_counts.png`     | raw-count CM for Deep Fusion (no TTA)                     |
| `confmat_tta_percent.png`       | prof-style row-% CM for Deep Fusion (+ TTA)               |
| `confmat_tta_counts.png`        | raw-count CM for Deep Fusion (+ TTA)                      |
| `loss_curves_fusion.png`        | 4-panel training diagnostics (shared)                     |
| `benchmark_bar.png`             | acc / macro-F1 / mIoU side-by-side bar chart              |

These are the figures and tables to drop straight into the report PDF.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
