"""
Generate fusion_report.ipynb -- clean report artifacts for the Deep Fusion
model (U-Net + Bi-LSTM with SE-attention fusion), evaluated WITHOUT TTA.

This is the initial deep-fusion run. The notebook loads the trained
checkpoint at `runs/fusion_deep_v1/best.pt` and saves, into
`runs/fusion_deep_v1/report/`:

  confusion_matrix.png    -- counts + row-normalized
  loss_curves.png         -- train/val loss + val mIoU + per-class IoU
  per_class_iou.png       -- per-class IoU bar chart
  sample_predictions.png  -- 6 test samples: RGB | GT | Pred
  metrics.json            -- pix_acc, mIoU, per-class IoU/P/R/F1,
                             macro_F1, weighted_F1, full confusion matrix
  metrics_summary.txt     -- plain-text one-pager

Pure inference. ~1 minute.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "fusion_report.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Deep Fusion -- Report Artifacts (no TTA)

Loads the trained Deep Fusion checkpoint (U-Net + Bi-LSTM, SE-attention)
and saves clean figures + metrics into `runs/fusion_deep_v1/report/`.

What this notebook produces:

* **confusion_matrix.png** -- counts side-by-side with row-normalized
* **loss_curves.png** -- training/validation loss + val mIoU + per-class IoU
* **per_class_iou.png** -- per-class IoU bar chart
* **sample_predictions.png** -- 6 test samples, RGB / GT / Prediction
* **metrics.json** -- full metric dump
* **metrics_summary.txt** -- text one-pager for the writeup

Pure inference, no retraining. ~1 minute on A6000.""")

# -------- Cell: GPU pin --------------------------------------------------
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
import matplotlib as mpl
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- Cell: config ---------------------------------------------------
md("## 1. Config")

code(r"""PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
RUN_DIR      = PROJECT_ROOT / "runs" / "fusion_deep_v1"
RUN_UNET     = PROJECT_ROOT / "runs" / "unet_imgonly_v1"
RUN_LSTM     = PROJECT_ROOT / "runs" / "lstm_csvonly_v1"
OUT_DIR      = RUN_DIR / "report"
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
BATCH_SIZE   = 32
NUM_WORKERS  = 4

CLASS_NAMES  = ["ice", "thin_ice", "water"]
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mpl.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
print("output dir:", OUT_DIR)
""")

# -------- Cell: manifest + features --------------------------------------
md("## 2. Manifest + test split + cached CSV features")

code(r"""manifest = pd.read_csv(RUN_UNET / "manifest.csv")
csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
csv_meta = pd.DataFrame([{
    "csv_path": str(p), "tile": p.stem.split("_")[3], "beam": p.stem.split("_")[4],
} for p in csv_files])
csv_meta["csv_id"] = csv_meta.index
manifest = manifest.merge(csv_meta[["tile", "beam", "csv_path", "csv_id"]],
                          on=["tile", "beam"], how="left")
manifest["csv_id"] = manifest["csv_id"].astype(int)
test_df = manifest[manifest["tile"].isin(["T03CWT"])].reset_index(drop=True)
print(f"test samples: {len(test_df):,}")

csv_features = {}
for cid in csv_meta["csv_id"]:
    arr = np.load(RUN_LSTM / "csv_normed" / f"csv_{cid}.npy")
    csv_features[int(cid)] = arr
n_features = next(iter(csv_features.values())).shape[1]
print(f"csv features: {n_features}")
""")

# -------- Cell: dataset --------------------------------------------------
md("## 3. Test dataset")

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


test_loader = DataLoader(TestSet(test_df), batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
""")

# -------- Cell: model ----------------------------------------------------
md("## 4. Deep Fusion model + load checkpoint")

code(r"""class SqueezeExcitation(nn.Module):
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


model = DeepFusionModel(n_features=n_features).to(device)
ck = torch.load(RUN_DIR / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(ck["model_state"]); model.eval()
print(f"loaded checkpoint from epoch {ck.get('epoch', '?')}")
""")

# -------- Cell: metrics --------------------------------------------------
md("## 5. Metric helpers")

code(r"""def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=np.float64)
    pix_acc = float(np.diag(cm).sum() / max(cm.sum(), 1))
    per_iou, per_prec, per_rec, per_f1, support = [], [], [], [], []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        sup = cm[c, :].sum()
        denom_iou = tp + fp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_iou.append(tp / denom_iou if denom_iou > 0 else 0.0)
        per_prec.append(prec); per_rec.append(rec); per_f1.append(f1); support.append(sup)
    miou = float(np.mean(per_iou))
    macro_f1 = float(np.mean(per_f1))
    weighted_f1 = float(np.sum(np.array(per_f1) * np.array(support)) / max(np.sum(support), 1))
    return {"pix_acc": pix_acc, "miou": miou,
            "per_iou":  [float(v) for v in per_iou],
            "per_prec": [float(v) for v in per_prec],
            "per_rec":  [float(v) for v in per_rec],
            "per_f1":   [float(v) for v in per_f1],
            "support":  [int(v)  for v in support],
            "macro_f1": macro_f1, "weighted_f1": weighted_f1}


@torch.no_grad()
def evaluate(model, loader):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img, win, y in loader:
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(img, win)
        pred = logits.argmax(1).cpu().numpy().ravel()
        t = y.numpy().ravel()
        idx = NUM_CLASSES * t + pred
        cm += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cm
""")

# -------- Cell: run eval -------------------------------------------------
md("## 6. Run evaluation (no TTA)")

code(r"""cm = evaluate(model, test_loader)
m  = metrics_from_cm(cm)
print(f"pixel accuracy : {m['pix_acc']:.4f}")
print(f"mIoU           : {m['miou']:.4f}")
print(f"macro F1       : {m['macro_f1']:.4f}")
print(f"weighted F1    : {m['weighted_f1']:.4f}")
print("per-class:")
for c, name in enumerate(CLASS_NAMES):
    print(f"  {name:9s}  IoU={m['per_iou'][c]:.4f}  "
          f"P={m['per_prec'][c]:.4f}  R={m['per_rec'][c]:.4f}  "
          f"F1={m['per_f1'][c]:.4f}  support={m['support'][c]:,}")
""")

# -------- Cell: confusion matrix figure ----------------------------------
md("## 7. Confusion matrix figure")

code(r"""cm_arr = np.asarray(cm, dtype=np.float64)
cm_norm = cm_arr / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
panels = [("Counts", cm_arr.astype(np.int64), "Greens"),
          ("Row-normalized", cm_norm, "Blues")]
for ax, (name, mat, cmap) in zip(axes, panels):
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{name}  (mIoU={m['miou']:.3f})" if name == "Counts" else name)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            txt = f"{int(mat[i,j]):,}" if name == "Counts" else f"{cm_norm[i,j]:.3f}"
            color = "white" if cm_norm[i,j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Deep Fusion -- Test confusion (tile T03CWT)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- Cell: loss curves ----------------------------------------------
md("## 8. Loss / mIoU curves from training history")

code(r"""metrics_csv = RUN_DIR / "metrics.csv"
if metrics_csv.exists():
    df = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    axes[0].plot(df["epoch"], df["train_loss"], "-o", label="train", color="#4C72B0", markersize=4)
    axes[0].plot(df["epoch"], df["val_loss"],   "-o", label="val",   color="#C44E52", markersize=4)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Training / validation loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["val_miou"], "-o", label="val mIoU", color="#55A868", markersize=4)
    if {"iou_ice", "iou_thin", "iou_water"} <= set(df.columns):
        axes[1].plot(df["epoch"], df["iou_ice"],   "--", label="ice",      alpha=0.7)
        axes[1].plot(df["epoch"], df["iou_thin"],  "--", label="thin_ice", alpha=0.7)
        axes[1].plot(df["epoch"], df["iou_water"], "--", label="water",    alpha=0.7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("IoU")
    axes[1].set_title("Validation mIoU + per-class IoU")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(loc="lower right"); axes[1].grid(alpha=0.3)

    fig.suptitle("Deep Fusion -- training curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curves.png", dpi=160, bbox_inches="tight")
    plt.show()
else:
    print(f"WARN: {metrics_csv} not found -- skipping loss curves")
""")

# -------- Cell: per-class IoU bar ----------------------------------------
md("## 9. Per-class IoU bar chart")

code(r"""fig, ax = plt.subplots(figsize=(7.5, 4.0))
x = np.arange(NUM_CLASSES)
bars = ax.bar(x, m["per_iou"], color=["#C44E52", "#4C72B0", "#55A868"])
for b, v in zip(bars, m["per_iou"]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
            f"{v:.3f}", ha="center", va="bottom", fontsize=11)
ax.axhline(m["miou"], color="black", linestyle="--", alpha=0.5,
           label=f"mIoU = {m['miou']:.3f}")
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("IoU"); ax.set_ylim(0, 1.05)
ax.set_title("Deep Fusion -- per-class IoU on test tile T03CWT")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_class_iou.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- Cell: sample predictions ---------------------------------------
md("""## 10. Sample predictions

Six random test samples, picked so all three classes are represented across
the panel. Columns: Sentinel-2 RGB, ground truth mask, deep fusion
prediction.""")

code(r"""def pick_sample_indices(df, n=6, seed=SEED):
    rng = np.random.RandomState(seed)
    candidates = list(rng.permutation(len(df))[: min(60, len(df))])
    chosen, seen = [], set()
    for idx in candidates:
        r = df.iloc[idx]
        gt = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))
        present = set(int(c) for c in np.unique(gt) if c in (0, 1, 2))
        if not present.issubset(seen) or len(chosen) < n:
            chosen.append(idx); seen |= present
        if len(chosen) >= n and seen == {0, 1, 2}:
            break
    return chosen[:n]


idxs = pick_sample_indices(test_df, n=6)
fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 2.7 * len(idxs)))
with torch.no_grad():
    for i, idx in enumerate(idxs):
        r = test_df.iloc[idx]
        rgb_raw = np.array(Image.open(r["image_path"]).convert("RGB"))
        gt = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))

        img_norm = (rgb_raw.astype(np.float32) / 255.0 - IM_MEAN) / IM_STD
        img_t = torch.from_numpy(np.transpose(img_norm, (2, 0, 1)))[None].to(device)
        feats = csv_features[int(r["csv_id"])]
        n_rows = feats.shape[0]; center = int(r["row_idx"])
        win = np.zeros((WINDOW_K, n_features), dtype=np.float32)
        for k in range(WINDOW_K):
            src = center - HALF + k
            if 0 <= src < n_rows:
                win[k] = feats[src]
        win_t = torch.from_numpy(win)[None].to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            pred = model(img_t, win_t).argmax(1)[0].cpu().numpy()

        axes[i, 0].imshow(rgb_raw)
        axes[i, 1].imshow(int_mask_to_rgb(gt))
        axes[i, 2].imshow(int_mask_to_rgb(pred))
        if i == 0:
            axes[i, 0].set_title("Sentinel-2 RGB", fontsize=12)
            axes[i, 1].set_title("Ground truth",   fontsize=12)
            axes[i, 2].set_title("Deep Fusion",    fontsize=12)
        axes[i, 0].set_ylabel(r["filename"], fontsize=7)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Deep Fusion -- prediction samples (tile T03CWT)", fontsize=13, y=1.001)
plt.tight_layout()
plt.savefig(OUT_DIR / "sample_predictions.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- Cell: save metrics ---------------------------------------------
md("## 11. Save metrics.json and metrics_summary.txt")

code(r"""payload = {**m,
           "confusion_matrix":         cm.tolist(),
           "confusion_matrix_rownorm": cm_norm.tolist(),
           "class_names": CLASS_NAMES, "tta": False,
           "checkpoint":  str(RUN_DIR / "best.pt"),
           "n_test":      int(len(test_df))}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(payload, f, indent=2)

with open(OUT_DIR / "metrics_summary.txt", "w") as f:
    f.write("Deep Fusion (U-Net + Bi-LSTM, SE-attention) -- test set T03CWT\n")
    f.write("=" * 64 + "\n\n")
    f.write(f"Pixel accuracy   : {m['pix_acc']:.4f}\n")
    f.write(f"mIoU             : {m['miou']:.4f}\n")
    f.write(f"Macro F1         : {m['macro_f1']:.4f}\n")
    f.write(f"Weighted F1      : {m['weighted_f1']:.4f}\n\n")
    f.write("Per-class:\n")
    for c, name in enumerate(CLASS_NAMES):
        f.write(f"  {name:9s}  IoU={m['per_iou'][c]:.4f}  "
                f"P={m['per_prec'][c]:.4f}  R={m['per_rec'][c]:.4f}  "
                f"F1={m['per_f1'][c]:.4f}  support={m['support'][c]:,}\n")
    f.write("\nNo test-time augmentation. Single forward pass per sample.\n")

print(f"saved -> {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")
""")

md(r"""## Done

All artifacts in `runs/fusion_deep_v1/report/`. Drop these straight into
the writeup.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
