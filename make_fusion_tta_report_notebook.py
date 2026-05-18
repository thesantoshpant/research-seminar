"""
Generate fusion_tta_report.ipynb -- clean report artifacts for the Deep
Fusion model evaluated with 4-view Test-Time Augmentation (the best
combination on the project: deep fusion + TTA).

Same checkpoint as fusion_report.ipynb, but every test sample is forwarded
4 times under {identity, H-flip, V-flip, 180-degree rotation}. Each view's
output is inverted back to the canonical orientation before softmax
averaging. The Bi-LSTM CSV branch sees the original (un-flipped) photon
window because the ICESat-2 track has a fixed along-track order.

Saves into `runs/fusion_deep_v1/report_tta/`:
  confusion_matrix.png    -- counts + row-normalized (TTA predictions)
  loss_curves.png         -- training history (shared with no-TTA run;
                              TTA is inference-only, no extra training)
  per_class_iou.png       -- per-class IoU with TTA
  tta_vs_no_tta.png       -- side-by-side mIoU / pix_acc / per-class IoU
  sample_predictions.png  -- 6 samples: RGB | GT | no-TTA | TTA
  metrics.json            -- TTA and no-TTA metrics with delta
  metrics_summary.txt     -- text one-pager

Pure inference, ~4 minutes (4 forward passes per sample).
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "fusion_tta_report.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Deep Fusion + 4-view TTA -- Report Artifacts

Loads the trained Deep Fusion checkpoint, evaluates with TTA (4 views:
identity, H-flip, V-flip, 180-rotation), and saves clean figures + metrics
into `runs/fusion_deep_v1/report_tta/`.

Each view is forwarded through the model, the softmax map is inverted back
to the canonical orientation, and the 4 maps are averaged before argmax.
The CSV branch sees the original photon window every pass (along-track
order is fixed -- flipping photons would not be a valid augmentation).

What this notebook produces:

* **confusion_matrix.png** -- counts + row-normalized (TTA predictions)
* **loss_curves.png** -- training history (same as no-TTA; TTA is inference-only)
* **per_class_iou.png** -- per-class IoU bar (TTA predictions)
* **tta_vs_no_tta.png** -- side-by-side bars: mIoU / pix_acc / macro_F1 / per-class IoU
* **sample_predictions.png** -- 6 test samples, RGB / GT / no-TTA / TTA
* **metrics.json** -- both metric sets + delta
* **metrics_summary.txt** -- text one-pager

Pure inference, ~4 minutes on A6000.""")

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
OUT_DIR      = RUN_DIR / "report_tta"
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

# -------- Cell: TTA core -------------------------------------------------
md("""## 5. TTA core

Four views: identity, H-flip, V-flip, 180-degree rotation. Each view has a
forward transform (applied to the input image) and an inverse transform
(applied to the output probability map). H-flip, V-flip, and rot180 are
all self-inverse, so the inverse is the same operation as the forward.""")

code(r"""def view_pair(view):
    if view == "id":
        return (lambda x: x), (lambda y: y)
    if view == "hflip":
        return (lambda x: torch.flip(x, dims=[-1])), (lambda y: torch.flip(y, dims=[-1]))
    if view == "vflip":
        return (lambda x: torch.flip(x, dims=[-2])), (lambda y: torch.flip(y, dims=[-2]))
    if view == "rot180":
        return (lambda x: torch.flip(x, dims=[-1, -2])), (lambda y: torch.flip(y, dims=[-1, -2]))
    raise ValueError(view)


VIEWS = ["id", "hflip", "vflip", "rot180"]
""")

# -------- Cell: metrics + eval -------------------------------------------
md("## 6. Metric helpers + evaluation loop")

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
def evaluate(model, loader, use_tta):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img, win, y in loader:
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)
        if not use_tta:
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(img, win)
            probs = F.softmax(logits, dim=1)
        else:
            acc = None
            for v in VIEWS:
                fwd, inv = view_pair(v)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits_v = model(fwd(img), win)
                p_v = inv(F.softmax(logits_v, dim=1))
                acc = p_v if acc is None else acc + p_v
            probs = acc / len(VIEWS)
        pred = probs.argmax(1).cpu().numpy().ravel()
        t = y.numpy().ravel()
        idx = NUM_CLASSES * t + pred
        cm += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cm
""")

# -------- Cell: run both eval --------------------------------------------
md("## 7. Run both evaluations")

code(r"""print("no TTA ...")
cm_base = evaluate(model, test_loader, use_tta=False)
m_base  = metrics_from_cm(cm_base)
print(f"  mIoU={m_base['miou']:.4f}  pix_acc={m_base['pix_acc']:.4f}")

print("4-view TTA ...")
cm_tta = evaluate(model, test_loader, use_tta=True)
m_tta  = metrics_from_cm(cm_tta)
print(f"  mIoU={m_tta['miou']:.4f}  pix_acc={m_tta['pix_acc']:.4f}")

print()
print(f"delta mIoU (TTA - no TTA) : {m_tta['miou'] - m_base['miou']:+.4f}")
print("per-class IoU (TTA):")
for c, name in enumerate(CLASS_NAMES):
    print(f"  {name:9s}  IoU={m_tta['per_iou'][c]:.4f}  "
          f"P={m_tta['per_prec'][c]:.4f}  R={m_tta['per_rec'][c]:.4f}  "
          f"F1={m_tta['per_f1'][c]:.4f}  support={m_tta['support'][c]:,}")
""")

# -------- Cell: confusion matrix -----------------------------------------
md("## 8. Confusion matrix figure (TTA predictions)")

code(r"""cm_arr  = np.asarray(cm_tta, dtype=np.float64)
cm_norm = cm_arr / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
panels = [("Counts", cm_arr.astype(np.int64), "Greens"),
          ("Row-normalized", cm_norm, "Blues")]
for ax, (name, mat, cmap) in zip(axes, panels):
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{name}  (mIoU={m_tta['miou']:.3f})" if name == "Counts" else name)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            txt = f"{int(mat[i,j]):,}" if name == "Counts" else f"{cm_norm[i,j]:.3f}"
            color = "white" if cm_norm[i,j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Deep Fusion + TTA -- Test confusion (tile T03CWT)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- Cell: loss curves ----------------------------------------------
md("## 9. Loss / mIoU curves (from training history)")

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

    fig.suptitle("Deep Fusion -- training curves (TTA is inference-only)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curves.png", dpi=160, bbox_inches="tight")
    plt.show()
else:
    print(f"WARN: {metrics_csv} not found -- skipping loss curves")
""")

# -------- Cell: per-class IoU --------------------------------------------
md("## 10. Per-class IoU bar (TTA)")

code(r"""fig, ax = plt.subplots(figsize=(7.5, 4.0))
x = np.arange(NUM_CLASSES)
bars = ax.bar(x, m_tta["per_iou"], color=["#C44E52", "#4C72B0", "#55A868"])
for b, v in zip(bars, m_tta["per_iou"]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
            f"{v:.3f}", ha="center", va="bottom", fontsize=11)
ax.axhline(m_tta["miou"], color="black", linestyle="--", alpha=0.5,
           label=f"mIoU = {m_tta['miou']:.3f}")
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("IoU"); ax.set_ylim(0, 1.05)
ax.set_title("Deep Fusion + TTA -- per-class IoU on test tile T03CWT")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_class_iou.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- Cell: TTA vs no-TTA --------------------------------------------
md("## 11. TTA vs no-TTA side-by-side")

code(r"""labels = ["mIoU", "pix acc", "macro F1", "ice IoU", "thin IoU", "water IoU"]
base_vals = [m_base["miou"], m_base["pix_acc"], m_base["macro_f1"],
             m_base["per_iou"][0], m_base["per_iou"][1], m_base["per_iou"][2]]
tta_vals  = [m_tta ["miou"], m_tta ["pix_acc"], m_tta ["macro_f1"],
             m_tta ["per_iou"][0], m_tta ["per_iou"][1], m_tta ["per_iou"][2]]

fig, ax = plt.subplots(figsize=(9.5, 4.5))
x = np.arange(len(labels)); width = 0.35
b1 = ax.bar(x - width/2, base_vals, width, label="no TTA",         color="#4C72B0")
b2 = ax.bar(x + width/2, tta_vals,  width, label="with TTA (4-view)", color="#C44E52")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05); ax.set_ylabel("score")
ax.set_title("Effect of 4-view TTA on Deep Fusion (test tile T03CWT)")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "tta_vs_no_tta.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- Cell: sample predictions ---------------------------------------
md("""## 12. Sample predictions: RGB / GT / no-TTA / TTA

Six random test samples, picked so all three classes are represented. The
no-TTA and TTA columns side-by-side make the smoothing effect of TTA
visible at thin-ice / water boundaries.""")

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


@torch.no_grad()
def predict_one(rgb_raw, csv_window, use_tta):
    img_norm = (rgb_raw.astype(np.float32) / 255.0 - IM_MEAN) / IM_STD
    img_t = torch.from_numpy(np.transpose(img_norm, (2, 0, 1)))[None].to(device)
    win_t = torch.from_numpy(csv_window)[None].to(device)
    if not use_tta:
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(img_t, win_t)
        probs = F.softmax(logits, dim=1)
    else:
        acc = None
        for v in VIEWS:
            fwd, inv = view_pair(v)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits_v = model(fwd(img_t), win_t)
            p_v = inv(F.softmax(logits_v, dim=1))
            acc = p_v if acc is None else acc + p_v
        probs = acc / len(VIEWS)
    return probs.argmax(1)[0].cpu().numpy()


idxs = pick_sample_indices(test_df, n=6)
fig, axes = plt.subplots(len(idxs), 4, figsize=(12, 2.7 * len(idxs)))
for i, idx in enumerate(idxs):
    r = test_df.iloc[idx]
    rgb_raw = np.array(Image.open(r["image_path"]).convert("RGB"))
    gt = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))

    feats = csv_features[int(r["csv_id"])]
    n_rows = feats.shape[0]; center = int(r["row_idx"])
    win = np.zeros((WINDOW_K, n_features), dtype=np.float32)
    for k in range(WINDOW_K):
        src = center - HALF + k
        if 0 <= src < n_rows:
            win[k] = feats[src]

    pred_base = predict_one(rgb_raw, win, use_tta=False)
    pred_tta  = predict_one(rgb_raw, win, use_tta=True)

    axes[i, 0].imshow(rgb_raw)
    axes[i, 1].imshow(int_mask_to_rgb(gt))
    axes[i, 2].imshow(int_mask_to_rgb(pred_base))
    axes[i, 3].imshow(int_mask_to_rgb(pred_tta))
    if i == 0:
        axes[i, 0].set_title("Sentinel-2 RGB",   fontsize=12)
        axes[i, 1].set_title("Ground truth",     fontsize=12)
        axes[i, 2].set_title("Fusion (no TTA)",  fontsize=12)
        axes[i, 3].set_title("Fusion + TTA",     fontsize=12)
    axes[i, 0].set_ylabel(r["filename"], fontsize=7)
    for ax in axes[i]:
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Deep Fusion -- prediction samples (no-TTA vs TTA)", fontsize=13, y=1.001)
plt.tight_layout()
plt.savefig(OUT_DIR / "sample_predictions.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- Cell: save metrics ---------------------------------------------
md("## 13. Save metrics.json and metrics_summary.txt")

code(r"""def _cm_payload(cm):
    cm = np.asarray(cm, dtype=np.float64)
    return {"confusion_matrix":         cm.astype(np.int64).tolist(),
            "confusion_matrix_rownorm": (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist()}

payload = {"tta":    {**m_tta,  **_cm_payload(cm_tta)},
           "no_tta": {**m_base, **_cm_payload(cm_base)},
           "delta_miou": float(m_tta["miou"] - m_base["miou"]),
           "views":      VIEWS,
           "class_names": CLASS_NAMES,
           "checkpoint": str(RUN_DIR / "best.pt"),
           "n_test":     int(len(test_df))}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(payload, f, indent=2)

with open(OUT_DIR / "metrics_summary.txt", "w") as f:
    f.write("Deep Fusion + 4-view TTA -- test set T03CWT\n")
    f.write("=" * 64 + "\n\n")
    f.write(f"Pixel accuracy   : {m_tta['pix_acc']:.4f}\n")
    f.write(f"mIoU             : {m_tta['miou']:.4f}    "
            f"(delta vs no-TTA: {m_tta['miou']-m_base['miou']:+.4f})\n")
    f.write(f"Macro F1         : {m_tta['macro_f1']:.4f}\n")
    f.write(f"Weighted F1      : {m_tta['weighted_f1']:.4f}\n\n")
    f.write("Per-class:\n")
    for c, name in enumerate(CLASS_NAMES):
        f.write(f"  {name:9s}  IoU={m_tta['per_iou'][c]:.4f}  "
                f"P={m_tta['per_prec'][c]:.4f}  R={m_tta['per_rec'][c]:.4f}  "
                f"F1={m_tta['per_f1'][c]:.4f}  support={m_tta['support'][c]:,}\n")
    f.write("\nTTA views: identity, H-flip, V-flip, 180-degree rotation.\n")
    f.write("Each view's softmax is inverted back to the canonical\n")
    f.write("orientation before averaging. The CSV branch sees the same\n")
    f.write("photon window every pass (along-track order is fixed).\n")

print(f"saved -> {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")
""")

md(r"""## Done

All artifacts in `runs/fusion_deep_v1/report_tta/`. The headline number for
the writeup: **Deep Fusion + 4-view TTA** on test tile T03CWT.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
