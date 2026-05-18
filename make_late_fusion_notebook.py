"""
Generate late_fusion.ipynb -- evaluates three late-fusion strategies on top
of the already-trained U-Net (image-only) and Bi-LSTM (CSV-only) models,
and then produces a full benchmark-report bundle (precision/recall/F1/
accuracy, prof-style confusion matrices, loss curves of the components,
bar chart, etc.) for sharing.

Three strategies:
  1. Uniform average of softmax probs  (P_unet + P_lstm) / 2
  2. Best-alpha scalar blend           alpha * P_unet + (1 - alpha) * P_lstm
                                       (alpha picked by val mIoU sweep)
  3. Best-alpha per-class blend        alpha_c picked per class on val

Outputs (in runs/late_fusion/):
  - alpha_sweep.png                  : val mIoU vs alpha
  - results_table.csv                : IoU table -- baselines + 3 strategies + deep fusion
  - late_fusion_metrics.json         : raw numbers
  - late_fusion_bar.png              : per-class IoU bar
  benchmark report:
  - benchmark_summary.csv            : one row per strategy + baselines
  - per_class_breakdown.csv          : per-class precision/recall/F1/IoU
  - confmat_<strategy>_percent.png   : prof-style row-% CM
  - confmat_<strategy>_counts.png    : raw-count CM
  - loss_curves_components.png       : U-Net + Bi-LSTM training diagnostics
  - benchmark_bar.png                : acc / macro-F1 / mIoU bar chart
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "late_fusion.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Late fusion -- ablation + benchmark report

Two halves in one notebook:

**Part A -- Ablation.** Three late-fusion strategies on top of the
already-trained image-only U-Net and CSV-only Bi-LSTM:

1. **Uniform average** of softmax probabilities.
2. **Best-alpha scalar blend** -- pick a single `alpha in [0,1]` by validation mIoU.
3. **Best-alpha per-class** -- pick `alpha_c` separately for each class.

**Part B -- Benchmark report.** Same metric suite + figures we produced for
Deep Fusion -- pixel accuracy, macro/weighted F1, per-class precision/
recall/F1/IoU, and a *professor-style* row-normalised percentage confusion
matrix -- for each late-fusion strategy. Plus the training loss curves of
the component models (U-Net, Bi-LSTM), because late fusion itself has no
training step.""")

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
OUT_DIR      = RUNS / "late_fusion"
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

# internal class index order vs display order
CLASS_NAMES_INT  = ["ice", "thin_ice", "water"]
CLASS_NAMES_DISP = ["thick ice", "thin ice", "water"]

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("output dir:", OUT_DIR)
""")

# -------- Cell: load shared data, reconstruct splits --------------------
md(r"""## 2. Manifest + splits (reconstructed with the same seed)""")

code(r"""manifest = pd.read_csv(RUN_UNET / "manifest.csv")
csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
csv_meta = pd.DataFrame([{
    "csv_path": str(p), "tile": p.stem.split("_")[3], "beam": p.stem.split("_")[4],
} for p in csv_files])
csv_meta["csv_id"] = csv_meta.index
manifest = manifest.merge(csv_meta[["tile", "beam", "csv_path", "csv_id"]],
                          on=["tile", "beam"], how="left")
manifest["csv_id"] = manifest["csv_id"].astype(int)

tiles_train = ["T02CNA", "T02CNC"]
tiles_test  = ["T03CWT"]
train_pool = manifest[manifest["tile"].isin(tiles_train)].reset_index(drop=True)
test_df    = manifest[manifest["tile"].isin(tiles_test)].reset_index(drop=True)

# Reproduce the same val split used at training time.
rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train_pool), size=int(0.10 * len(train_pool)), replace=False)
val_mask_arr = np.zeros(len(train_pool), dtype=bool); val_mask_arr[val_idx] = True
val_df = train_pool[val_mask_arr].reset_index(drop=True)

print(f"val: {len(val_df):,}   test: {len(test_df):,}")

# Normalized CSV features (cached by the LSTM run).
csv_features = {}
for cid in csv_meta["csv_id"]:
    arr = np.load(RUN_LSTM / "csv_normed" / f"csv_{cid}.npy")
    csv_features[int(cid)] = arr
n_features = next(iter(csv_features.values())).shape[1]
print(f"csv_features: {len(csv_features)} CSVs, {n_features} features")
""")

# -------- Cell: helpers + dataset ---------------------------------------
md(r"""## 3. Dataset (image + CSV window + GT mask)""")

code(r"""def mask_rgb_to_int(mask_rgb):
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    out[(mask_rgb == [255, 0, 0]).all(axis=-1)] = 0
    out[(mask_rgb == [0, 0, 255]).all(axis=-1)] = 1
    out[(mask_rgb == [0, 255, 0]).all(axis=-1)] = 2
    return out


class FusionEvalSet(Dataset):
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


val_loader  = DataLoader(FusionEvalSet(val_df),  batch_size=64,
                         shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(FusionEvalSet(test_df), batch_size=64,
                         shuffle=False, num_workers=4, pin_memory=True)
print("loaders ready")
""")

# -------- Cell: model defs ---------------------------------------------
md(r"""## 4. Model definitions""")

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
""")

# -------- Cell: load checkpoints ---------------------------------------
md(r"""## 5. Load trained U-Net + Bi-LSTM""")

code(r"""unet = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                in_channels=3, classes=NUM_CLASSES).to(device)
ck = torch.load(RUN_UNET / "best.pt", map_location=device, weights_only=False)
unet.load_state_dict(ck["model_state"]); unet.eval()
print(f"U-Net   loaded (val mIoU at train time {ck['val_metrics']['miou']:.4f})")

lstm = CSVOnlyModel(n_features=n_features).to(device)
ck = torch.load(RUN_LSTM / "best.pt", map_location=device, weights_only=False)
lstm.load_state_dict(ck["model_state"]); lstm.eval()
print(f"Bi-LSTM loaded (val mIoU at train time {ck['val_metrics']['miou']:.4f})")
""")

# -------- Cell: helpers --------------------------------------------------
md(r"""## 6. Late-fusion helpers

Run the models *once* and cache the softmax probabilities, then evaluate
many blending strategies on the same probability tensors.""")

code(r"""@torch.no_grad()
def collect_probs(loader, desc):
    probs_u, probs_l, ys = [], [], []
    for img, win, y in loader:
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            pu = F.softmax(unet(img), dim=1)
            pl = F.softmax(lstm(win), dim=1)
        probs_u.append(pu.cpu())
        probs_l.append(pl.cpu())
        ys.append(y)
    print(f"  {desc}: collected {len(probs_u)} batches")
    return probs_u, probs_l, ys


def confmat_from_probs(probs_u_batches, probs_l_batches, ys_batches, alpha):
    # alpha is either a scalar in [0,1] or a tensor of shape (NUM_CLASSES,)
    if isinstance(alpha, (int, float)):
        a_u = float(alpha); a_l = 1.0 - a_u
        a_tensor = None
    else:
        a_tensor = alpha.view(1, NUM_CLASSES, 1, 1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for pu, pl, y in zip(probs_u_batches, probs_l_batches, ys_batches):
        if a_tensor is None:
            blended = a_u * pu + a_l * pl
        else:
            blended = a_tensor * pu + (1.0 - a_tensor) * pl
        pred = blended.argmax(1).numpy().ravel()
        t = y.numpy().ravel()
        idx = NUM_CLASSES * t + pred
        cm += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cm


def iou_from_cm(cm):
    iou = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou.append(tp / denom if denom > 0 else 0.0)
    return np.array(iou), float(np.mean(iou))
""")

# -------- Cell: collect probs once --------------------------------------
md(r"""## 7. Collect probabilities for val and test (one pass each)""")

code(r"""print("collecting val probs ...")
val_pu, val_pl, val_y = collect_probs(val_loader,  "val")
print("collecting test probs ...")
test_pu, test_pl, test_y = collect_probs(test_loader, "test")
""")

# -------- Cell: uniform average ----------------------------------------
md(r"""## 8. Strategy A -- uniform average""")

code(r"""cm_A = confmat_from_probs(test_pu, test_pl, test_y, alpha=0.5)
iou_A, miou_A = iou_from_cm(cm_A)
print(f"uniform average mIoU = {miou_A:.4f}")
for c, name in enumerate(CLASS_NAMES_INT):
    print(f"  {name:9s} IoU = {iou_A[c]:.4f}")
""")

# -------- Cell: best scalar alpha --------------------------------------
md(r"""## 9. Strategy B -- best scalar alpha (swept on val)""")

code(r"""alphas = np.linspace(0.0, 1.0, 21)
val_miou_curve = []
for a in alphas:
    cm = confmat_from_probs(val_pu, val_pl, val_y, alpha=float(a))
    _, m = iou_from_cm(cm)
    val_miou_curve.append(m)
val_miou_curve = np.array(val_miou_curve)
best_a = float(alphas[val_miou_curve.argmax()])
print(f"best alpha on val = {best_a:.2f} (val mIoU {val_miou_curve.max():.4f})")

cm_B = confmat_from_probs(test_pu, test_pl, test_y, alpha=best_a)
iou_B, miou_B = iou_from_cm(cm_B)
print(f"best-alpha test mIoU = {miou_B:.4f}")
for c, name in enumerate(CLASS_NAMES_INT):
    print(f"  {name:9s} IoU = {iou_B[c]:.4f}")

plt.figure(figsize=(7, 3.4))
plt.plot(alphas, val_miou_curve, "-o", linewidth=2, markersize=4)
plt.axvline(best_a, color="red", linestyle="--", alpha=0.5,
            label=f"best alpha = {best_a:.2f}")
plt.xlabel("alpha (weight on U-Net)")
plt.ylabel("validation mIoU")
plt.title("Scalar-alpha sweep on validation")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "alpha_sweep.png", dpi=160); plt.show()
""")

# -------- Cell: per-class alpha ----------------------------------------
md(r"""## 10. Strategy C -- per-class alpha (3 alphas, picked on val)""")

code(r"""best_per_class = np.zeros(NUM_CLASSES)
for c in range(NUM_CLASSES):
    best_m, best_a_c = -1, 0.5
    for a in alphas:
        alpha_vec = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
        alpha_vec[c] = float(a)
        cm = confmat_from_probs(val_pu, val_pl, val_y,
                                alpha=torch.tensor(alpha_vec))
        _, m = iou_from_cm(cm)
        if m > best_m:
            best_m, best_a_c = m, float(a)
    best_per_class[c] = best_a_c
    print(f"class {CLASS_NAMES_INT[c]:9s}: alpha = {best_a_c:.2f} (val mIoU {best_m:.4f})")

cm_C = confmat_from_probs(test_pu, test_pl, test_y,
                          alpha=torch.tensor(best_per_class.astype(np.float32)))
iou_C, miou_C = iou_from_cm(cm_C)
print(f"per-class-alpha test mIoU = {miou_C:.4f}")
for c, name in enumerate(CLASS_NAMES_INT):
    print(f"  {name:9s} IoU = {iou_C[c]:.4f}")
""")

# -------- Cell: pull existing test mIoU from disk ----------------------
md(r"""## 11. Pull baseline numbers + assemble headline IoU table""")

code(r"""def load_test_metrics(run_dir):
    with open(run_dir / "test_metrics.json") as f:
        return json.load(f)

m_unet   = load_test_metrics(RUN_UNET)
m_lstm   = load_test_metrics(RUN_LSTM)
m_fusion = load_test_metrics(RUN_FUSION)

table = pd.DataFrame({
    "ice IoU":     [m_unet["per_iou"][0],   m_lstm["per_iou"][0],
                    iou_A[0], iou_B[0], iou_C[0], m_fusion["per_iou"][0]],
    "thin IoU":    [m_unet["per_iou"][1],   m_lstm["per_iou"][1],
                    iou_A[1], iou_B[1], iou_C[1], m_fusion["per_iou"][1]],
    "water IoU":   [m_unet["per_iou"][2],   m_lstm["per_iou"][2],
                    iou_A[2], iou_B[2], iou_C[2], m_fusion["per_iou"][2]],
    "mIoU":        [m_unet["miou"],         m_lstm["miou"],
                    miou_A, miou_B, miou_C, m_fusion["miou"]],
}, index=["U-Net (img only)", "Bi-LSTM (CSV only)",
          "late A: uniform avg", f"late B: alpha={best_a:.2f}",
          "late C: per-class alpha", "Deep fusion"])

print(table.round(4))
table.to_csv(OUT_DIR / "results_table.csv")
""")

# -------- Cell: ablation bar chart -------------------------------------
md(r"""## 12. Ablation bar chart -- late vs deep fusion""")

code(r"""variants = list(table.index)
categories = ["ice IoU", "thin IoU", "water IoU", "mIoU"]
data = table[categories].to_numpy()

palette = ["#4C72B0", "#55A868", "#8C8C8C", "#5C5C5C", "#2C2C2C", "#C44E52"]

x = np.arange(len(categories))
width = 0.13
fig, ax = plt.subplots(figsize=(10.5, 4.8))
for i, (variant, color) in enumerate(zip(variants, palette)):
    offset = (i - (len(variants) - 1) / 2) * width
    bars = ax.bar(x + offset, data[i], width, label=variant, color=color)
    for b, val in zip(bars, data[i]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel("IoU (test set, T03CWT)")
ax.set_ylim(0, 1.05)
ax.set_title("Late fusion strategies vs deep fusion")
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig_path = OUT_DIR / "late_fusion_bar.png"
plt.savefig(fig_path, dpi=160)
plt.show()
print(f"saved -> {fig_path}")
""")

# -------- Cell: save json ----------------------------------------------
md(r"""## 13. Save raw ablation numbers""")

code(r"""payload = {
    "strategy_A_uniform_avg": {
        "per_iou": iou_A.tolist(), "miou": miou_A,
    },
    "strategy_B_best_scalar_alpha": {
        "best_alpha_on_val": best_a,
        "per_iou": iou_B.tolist(), "miou": miou_B,
    },
    "strategy_C_per_class_alpha": {
        "best_alphas_on_val": best_per_class.tolist(),
        "per_iou": iou_C.tolist(), "miou": miou_C,
    },
    "baselines": {
        "U-Net":  {"per_iou": m_unet["per_iou"],   "miou": m_unet["miou"]},
        "LSTM":   {"per_iou": m_lstm["per_iou"],   "miou": m_lstm["miou"]},
        "Fusion": {"per_iou": m_fusion["per_iou"], "miou": m_fusion["miou"]},
    },
}
with open(OUT_DIR / "late_fusion_metrics.json", "w") as f:
    json.dump(payload, f, indent=2)
print("saved metrics json")
""")

# =========================================================================
# PART B -- BENCHMARK REPORT
# =========================================================================
md(r"""# Part B -- Benchmark report (the share-with-friends bundle)

Below we produce the same outputs we did for Deep Fusion, but for the late
fusion model. Specifically, for each of the three late-fusion strategies:

* **Full metric suite** -- pixel accuracy, mIoU, macro-F1, weighted-F1,
  per-class precision / recall / F1 / IoU.
* **Professor-style confusion matrix** -- row-normalised percentages, Blues
  colormap, `thick ice / thin ice / water` axis labels, title
  `Confusion Matrix (Percentages)`.
* **Raw-count confusion matrix** -- for the appendix.

And once for the bundle:

* **Loss curves of the component models** (U-Net, Bi-LSTM) -- late fusion
  itself isn't trained, so the loss curves that go in the report are from
  the two models being blended.
* **Headline benchmark bar chart** -- pixel accuracy / macro-F1 / mIoU for
  baselines + the 3 late-fusion variants + deep fusion.""")

# -------- Cell: collect CMs into a dict --------------------------------
md(r"""## 14. Collect all relevant confusion matrices

We already have the late-fusion CMs (`cm_A`, `cm_B`, `cm_C`). For the
baselines (U-Net, Bi-LSTM) we'd want CMs too, but they're not stored as
the matrix on disk -- only per-class IoU. We pick them up at `alpha=1.0`
(pure U-Net) and `alpha=0.0` (pure Bi-LSTM) from our cached softmax probs,
which is mathematically identical to argmaxing each model's logits.""")

code(r"""cm_unet  = confmat_from_probs(test_pu, test_pl, test_y, alpha=1.0)
cm_lstm  = confmat_from_probs(test_pu, test_pl, test_y, alpha=0.0)

all_cms = {
    "U-Net (img only)":             cm_unet,
    "Bi-LSTM (CSV only)":           cm_lstm,
    "Late fusion -- uniform avg":   cm_A,
    f"Late fusion -- best alpha={best_a:.2f}": cm_B,
    "Late fusion -- per-class alpha": cm_C,
}
for name, cm in all_cms.items():
    print(f"{name:42s}  total pixels = {cm.sum():,}")
""")

# -------- Cell: metric helper ------------------------------------------
md(r"""## 15. Full metric suite from each CM

Same helper we used for Deep Fusion -- everything derives from the CM.""")

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

    return {
        "pix_acc": float(pix_acc),
        "miou":    float(iou.mean()),
        "macro_f1":    float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / max(total, 1.0)),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


metrics_all = {name: metrics_from_cm(cm) for name, cm in all_cms.items()}
""")

# -------- Cell: summary table ------------------------------------------
md(r"""## 16. Benchmark summary table (one row per model / strategy)""")

code(r"""rows = []
for name, m in metrics_all.items():
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

# also add deep fusion from its on-disk test_metrics.json
m_fus_per = m_fusion["per_iou"]
rows.append({
    "model":            "Deep fusion (reference)",
    "pixel_accuracy":   round(float(m_fusion.get("pix_acc", 0.0)), 4),
    "mIoU":             round(float(m_fusion["miou"]), 4),
    "macro_precision":  np.nan,
    "macro_recall":     np.nan,
    "macro_F1":         round(float(m_fusion.get("macro_f1", 0.0)), 4),
    "weighted_F1":      round(float(m_fusion.get("weighted_f1", 0.0)), 4),
    "ice_IoU":          round(float(m_fus_per[0]), 4),
    "thin_IoU":         round(float(m_fus_per[1]), 4),
    "water_IoU":        round(float(m_fus_per[2]), 4),
})

summary = pd.DataFrame(rows).set_index("model")
summary.to_csv(OUT_DIR / "benchmark_summary.csv")
print(summary.to_string())
""")

# -------- Cell: per-class breakdown ------------------------------------
md(r"""## 17. Per-class precision / recall / F1 / IoU breakdown""")

code(r"""breakdown_rows = []
for name, m in metrics_all.items():
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

# -------- Cell: prof-style confusion matrix helpers --------------------
md(r"""## 18. Confusion matrix -- professor-style percentage view

Row-normalised percentages (each row sums to 100), Blues colormap, ticks
`thick ice / thin ice / water`, x-ticks `Predicted thick ice / Predicted
thin ice / Predicted water`, title `Confusion Matrix (Percentages)`.
Raw-count versions are saved alongside for the appendix.""")

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

code(r"""# Strategy A -- uniform average
plot_cm_percent(cm_A,
                "Late fusion (uniform avg) -- Confusion Matrix (Percentages)",
                OUT_DIR / "confmat_uniform_percent.png")
plot_cm_counts(cm_A,
               "Late fusion (uniform avg) -- Confusion Matrix (Counts)",
               OUT_DIR / "confmat_uniform_counts.png")
""")

code(r"""# Strategy B -- best scalar alpha
plot_cm_percent(cm_B,
                f"Late fusion (alpha={best_a:.2f}) -- Confusion Matrix (Percentages)",
                OUT_DIR / "confmat_alpha_percent.png")
plot_cm_counts(cm_B,
               f"Late fusion (alpha={best_a:.2f}) -- Confusion Matrix (Counts)",
               OUT_DIR / "confmat_alpha_counts.png")
""")

code(r"""# Strategy C -- per-class alpha
plot_cm_percent(cm_C,
                "Late fusion (per-class alpha) -- Confusion Matrix (Percentages)",
                OUT_DIR / "confmat_perclass_percent.png")
plot_cm_counts(cm_C,
               "Late fusion (per-class alpha) -- Confusion Matrix (Counts)",
               OUT_DIR / "confmat_perclass_counts.png")
""")

# -------- Cell: component loss curves -----------------------------------
md(r"""## 19. Loss curves -- the component models

Late fusion has no trainable weights of its own (the only learnable thing
is the scalar `alpha`, picked by validation sweep). The training curves
that go in the report are from the two models being blended -- U-Net and
Bi-LSTM -- both of which logged a per-epoch `metrics.csv` during their
training.""")

code(r"""hist_unet = pd.read_csv(RUN_UNET / "metrics.csv")
hist_lstm = pd.read_csv(RUN_LSTM / "metrics.csv")
print(f"U-Net training history:   {len(hist_unet)} epochs")
print(f"Bi-LSTM training history: {len(hist_lstm)} epochs")


def plot_two_histories(h1, label1, h2, label2, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    ax.plot(h1["epoch"], h1["train_loss"], "-o",  label=f"{label1} (train)",
            color="#1f77b4", markersize=4)
    ax.plot(h1["epoch"], h1["val_loss"],   "-s",  label=f"{label1} (val)",
            color="#1f77b4", markersize=4, markerfacecolor="none")
    ax.plot(h2["epoch"], h2["train_loss"], "-o",  label=f"{label2} (train)",
            color="#d62728", markersize=4)
    ax.plot(h2["epoch"], h2["val_loss"],   "-s",  label=f"{label2} (val)",
            color="#d62728", markersize=4, markerfacecolor="none")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss -- component models")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(h1["epoch"], h1["val_miou"], "-o", label=label1,
            color="#1f77b4", markersize=4)
    ax.plot(h2["epoch"], h2["val_miou"], "-o", label=label2,
            color="#d62728", markersize=4)
    ax.set_xlabel("epoch"); ax.set_ylabel("val mIoU")
    ax.set_title("Validation mIoU -- component models")
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(h1["epoch"], h1["pix_acc"], "-o", label=label1,
            color="#1f77b4", markersize=4)
    ax.plot(h2["epoch"], h2["pix_acc"], "-o", label=label2,
            color="#d62728", markersize=4)
    ax.set_xlabel("epoch"); ax.set_ylabel("val pixel accuracy")
    ax.set_title("Pixel accuracy -- component models")
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


plot_two_histories(hist_unet, "U-Net", hist_lstm, "Bi-LSTM",
                   OUT_DIR / "loss_curves_components.png")
""")

# -------- Cell: headline benchmark bar chart ---------------------------
md(r"""## 20. Headline benchmark bar chart

Side-by-side test-set comparison: baselines + 3 late-fusion variants +
deep fusion reference. The visual that goes in the report.""")

code(r"""variants = list(summary.index)
metrics_to_plot = ["pixel_accuracy", "macro_F1", "mIoU"]
nice_labels = {"pixel_accuracy": "pixel accuracy",
               "macro_F1":       "macro F1",
               "mIoU":           "mIoU"}
data = summary[metrics_to_plot].to_numpy()

x = np.arange(len(metrics_to_plot))
n = len(variants)
width = 0.85 / n
palette = ["#4C72B0", "#55A868", "#8C8C8C", "#5C5C5C", "#2C2C2C", "#C44E52"]
palette = (palette * ((n + len(palette) - 1) // len(palette)))[:n]

fig, ax = plt.subplots(figsize=(11.5, 5.0))
for i, (variant, color) in enumerate(zip(variants, palette)):
    offset = (i - (n - 1) / 2) * width
    vals = data[i]
    bars = ax.bar(x + offset, vals, width, label=variant, color=color)
    for b, val in zip(bars, vals):
        if np.isnan(val):
            continue
        ax.text(b.get_x() + b.get_width() / 2, val + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([nice_labels[m] for m in metrics_to_plot], fontsize=11)
ax.set_ylabel("test-set score")
ax.set_ylim(0, 1.08)
ax.set_title("Late fusion benchmark -- baselines + 3 strategies + deep fusion (test tile T03CWT)")
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, ncol=2)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "benchmark_bar.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- Cell: summary -------------------------------------------------
md(r"""## 21. Done -- what landed in `runs/late_fusion/`

**Ablation outputs:**
- `alpha_sweep.png`           -- scalar-alpha sweep on validation
- `late_fusion_bar.png`       -- per-class IoU bar for all variants
- `results_table.csv`         -- IoU table (baselines + 3 strategies + deep fusion)
- `late_fusion_metrics.json`  -- raw numbers

**Benchmark report outputs (the share-with-friends bundle):**
- `benchmark_summary.csv`            -- one row per model: acc / P / R / F1 / mIoU / per-class IoU
- `per_class_breakdown.csv`          -- per-class precision/recall/F1/IoU per model
- `confmat_uniform_percent.png`      -- prof-style row-% CM, Strategy A
- `confmat_uniform_counts.png`       -- raw-count CM, Strategy A
- `confmat_alpha_percent.png`        -- prof-style row-% CM, Strategy B (best alpha)
- `confmat_alpha_counts.png`         -- raw-count CM, Strategy B
- `confmat_perclass_percent.png`     -- prof-style row-% CM, Strategy C (per-class alpha)
- `confmat_perclass_counts.png`      -- raw-count CM, Strategy C
- `loss_curves_components.png`       -- U-Net + Bi-LSTM training diagnostics
- `benchmark_bar.png`                -- acc / macro-F1 / mIoU side-by-side bar chart

The expected story: late fusion strategies sit between the U-Net baseline
and the deep-fusion model -- i.e. simple probability blending recovers some
of the gain, but not all of it, motivating the feature-level architecture.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
