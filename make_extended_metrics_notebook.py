"""
Generate extended_metrics.ipynb -- compute pixel accuracy, macro-F1,
weighted-F1, per-class precision/recall/F1, and IoU for ALL four
models on the held-out test tile.

Models included:
  1. U-Net (image only)
  2. Bi-LSTM (CSV only)
  3. Deep fusion
  4. Late fusion (best-alpha scalar blend -- loaded from late_fusion/ if available;
                  otherwise computed inline by re-blending probs)

Outputs (in runs/extended_metrics/):
  - extended_metrics.csv       : one row per model, all metrics in one place
  - per_class_breakdown.csv    : per-class precision/recall/F1 table
  - accuracy_vs_f1_bar.png     : bar chart of pixel acc vs macro-F1
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "extended_metrics.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Extended metrics for all four models

Currently we only report mIoU and per-class IoU. To compare head-to-head with
Lechamo et al. 2025 (PolDS '25) -- which reports **accuracy / precision /
recall / F1** -- we also need:

* **pixel accuracy** = correct_pixels / total_pixels
* **macro-F1**       = mean of per-class F1 (balanced, not biased by majority class)
* **weighted-F1**    = per-class F1 weighted by class support
* **per-class precision/recall/F1**

All numbers come from the test-set confusion matrices, one pass per model.""")

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
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

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
RUN_LATE     = RUNS / "late_fusion"          # produced by late_fusion.ipynb
OUT_DIR      = RUNS / "extended_metrics"
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
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("output dir:", OUT_DIR)
""")

# -------- Cell: manifest + test split -----------------------------------
md(r"""## 2. Manifest + test split""")

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
print(f"test: {len(test_df):,}")

csv_features = {}
for cid in csv_meta["csv_id"]:
    arr = np.load(RUN_LSTM / "csv_normed" / f"csv_{cid}.npy")
    csv_features[int(cid)] = arr
n_features = next(iter(csv_features.values())).shape[1]
print(f"csv_features: {len(csv_features)} CSVs, {n_features} features")
""")

# -------- Cell: dataset + model defs -----------------------------------
md(r"""## 3. Dataset + model definitions""")

code(r"""def mask_rgb_to_int(mask_rgb):
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


class CSVOnlyModel(nn.Module):
    def __init__(self, n_features, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, num_classes=NUM_CLASSES, patch=PATCH):
        super().__init__()
        self.patch = patch
        self.proj = nn.Sequential(nn.Linear(n_features, hidden),
                                  nn.LayerNorm(hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Conv2d(hidden * 2, 64, kernel_size=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.Conv2d(64, num_classes, kernel_size=1),
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

# -------- Cell: load checkpoints ---------------------------------------
md(r"""## 4. Load the three trained models""")

code(r"""unet = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                in_channels=3, classes=NUM_CLASSES).to(device)
ck = torch.load(RUN_UNET / "best.pt", map_location=device, weights_only=False)
unet.load_state_dict(ck["model_state"]); unet.eval()
print(f"U-Net loaded (val mIoU {ck['val_metrics']['miou']:.4f})")

lstm = CSVOnlyModel(n_features=n_features).to(device)
ck = torch.load(RUN_LSTM / "best.pt", map_location=device, weights_only=False)
lstm.load_state_dict(ck["model_state"]); lstm.eval()
print(f"Bi-LSTM loaded (val mIoU {ck['val_metrics']['miou']:.4f})")

fusion = DeepFusionModel(n_features=n_features).to(device)
ck = torch.load(RUN_FUSION / "best.pt", map_location=device, weights_only=False)
fusion.load_state_dict(ck["model_state"]); fusion.eval()
print(f"Fusion loaded (val mIoU {ck['val_metrics']['miou']:.4f})")
""")

# -------- Cell: confusion matrix accumulator ---------------------------
md(r"""## 5. Confusion matrix for each model

We loop the test set once per model. For the late-fusion entry we capture
the softmax probabilities of U-Net + LSTM and blend at the best alpha that
was picked in `late_fusion.ipynb` (falling back to alpha=0.5 if that
notebook hasn't been run yet).""")

code(r"""# pull the best alpha from late_fusion.json if available, otherwise default to 0.5
late_meta_path = RUN_LATE / "late_fusion_metrics.json"
if late_meta_path.exists():
    with open(late_meta_path) as f:
        late_meta = json.load(f)
    BEST_ALPHA = float(late_meta["strategy_B_best_scalar_alpha"]["best_alpha_on_val"])
    print(f"using best alpha from disk: {BEST_ALPHA:.2f}")
else:
    BEST_ALPHA = 0.5
    print(f"late_fusion_metrics.json not found -- defaulting alpha to {BEST_ALPHA}")
""")

code(r"""@torch.no_grad()
def gather_cms():
    cms = {name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
           for name in ["U-Net (image only)", "Bi-LSTM (CSV only)",
                        "Deep fusion", "Late fusion (alpha)"]}
    for img, win, y in test_loader:
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            log_u = unet(img)
            log_l = lstm(win)
            log_f = fusion(img, win)
            p_u = F.softmax(log_u, dim=1)
            p_l = F.softmax(log_l, dim=1)

        preds = {
            "U-Net (image only)":   log_u.argmax(1),
            "Bi-LSTM (CSV only)":   log_l.argmax(1),
            "Deep fusion":          log_f.argmax(1),
            "Late fusion (alpha)":  (BEST_ALPHA * p_u + (1 - BEST_ALPHA) * p_l).argmax(1),
        }
        t = y.numpy().ravel()
        for name, pred in preds.items():
            p = pred.cpu().numpy().ravel()
            idx = NUM_CLASSES * t + p
            cms[name] += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cms


cms = gather_cms()
for name, cm in cms.items():
    print(f"{name}: total pixels = {cm.sum():,}")
""")

# -------- Cell: metrics from CM ----------------------------------------
md(r"""## 6. Compute the metric suite from each CM""")

code(r"""def metrics_from_cm(cm):
    # Return dict of all the metrics we care about.
    cm = cm.astype(np.float64)
    total = cm.sum()
    diag  = np.diag(cm)
    support = cm.sum(axis=1)      # number of true pixels per class
    pred_sum = cm.sum(axis=0)     # number of predicted pixels per class

    pix_acc = diag.sum() / max(total, 1.0)

    # per-class precision, recall, F1, IoU
    precision = np.where(pred_sum > 0, diag / np.maximum(pred_sum, 1), 0.0)
    recall    = np.where(support  > 0, diag / np.maximum(support,  1), 0.0)
    f1_denom  = (precision + recall)
    f1 = np.where(f1_denom > 0, 2 * precision * recall / np.maximum(f1_denom, 1e-12), 0.0)

    iou_denom = (support + pred_sum - diag)
    iou = np.where(iou_denom > 0, diag / np.maximum(iou_denom, 1), 0.0)

    macro_f1    = float(f1.mean())
    weighted_f1 = float((f1 * support).sum() / max(total, 1.0))
    miou        = float(iou.mean())

    return {
        "pix_acc": float(pix_acc),
        "miou":    miou,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


metrics = {name: metrics_from_cm(cm) for name, cm in cms.items()}
""")

# -------- Cell: assemble summary table ---------------------------------
md(r"""## 7. Summary table (one row per model)""")

code(r"""rows = []
for name, m in metrics.items():
    rows.append({
        "model":         name,
        "pix_acc":       round(m["pix_acc"], 4),
        "mIoU":          round(m["miou"], 4),
        "macro_F1":      round(m["macro_f1"], 4),
        "weighted_F1":   round(m["weighted_f1"], 4),
        "ice IoU":       round(m["iou"][0], 4),
        "thin IoU":      round(m["iou"][1], 4),
        "water IoU":     round(m["iou"][2], 4),
    })

summary = pd.DataFrame(rows).set_index("model")
print(summary)
summary.to_csv(OUT_DIR / "extended_metrics.csv")
""")

# -------- Cell: per-class breakdown -----------------------------------
md(r"""## 8. Per-class precision / recall / F1 breakdown""")

code(r"""breakdown_rows = []
for name, m in metrics.items():
    for c, cname in enumerate(CLASS_NAMES):
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

# -------- Cell: bar chart ----------------------------------------------
md(r"""## 9. Bar chart -- pixel accuracy vs macro-F1 vs mIoU""")

code(r"""variants = list(summary.index)
metrics_to_plot = ["pix_acc", "macro_F1", "mIoU"]
nice_labels = {"pix_acc": "pixel accuracy", "macro_F1": "macro F1", "mIoU": "mIoU"}
data = summary[metrics_to_plot].to_numpy()

x = np.arange(len(metrics_to_plot))
width = 0.18
palette = ["#4C72B0", "#55A868", "#5C5C5C", "#C44E52"]

fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, (variant, color) in enumerate(zip(variants, palette)):
    offset = (i - (len(variants) - 1) / 2) * width
    bars = ax.bar(x + offset, data[i], width, label=variant, color=color)
    for b, val in zip(bars, data[i]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels([nice_labels[m] for m in metrics_to_plot], fontsize=11)
ax.set_ylabel("test-set score")
ax.set_ylim(0, 1.05)
ax.set_title("All metrics, one figure (test tile T03CWT)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_vs_f1_bar.png", dpi=160)
plt.show()
""")

# -------- Cell: notes for the writeup ----------------------------------
md(r"""## 10. What this gives us

* **Comparable-to-Paper-2 numbers.** Lechamo et al. reports accuracy 97.70%
  for Gated Fusion; ours sits in `summary.loc["Deep fusion", "pix_acc"]`.
* **Macro-F1**: paper 2 doesn't compute it directly but the imbalance in
  their dataset means their accuracy is inflated. Macro-F1 is the honest
  comparison number.
* **mIoU**: our headline.
* All raw numbers in `extended_metrics.csv` and `per_class_breakdown.csv`.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
