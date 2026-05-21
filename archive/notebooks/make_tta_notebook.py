"""
Generate tta_eval.ipynb -- Test-Time Augmentation for U-Net and Deep Fusion.

For each model we run 4 forward passes per test sample using these views:
  - identity
  - horizontal flip
  - vertical flip
  - 180 deg rotation (= horizontal flip then vertical flip)
We invert each view in probability space, average the softmaxes, then argmax.

Bi-LSTM is image-blind (CSV only) so TTA on it is a no-op -- we report its
non-TTA number for completeness.

Outputs (in runs/tta/):
  - tta_metrics.json       : per-model with/without TTA
  - tta_bar.png            : compare with vs without TTA
  - tta_results_table.csv  : numbers behind the chart
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "tta_eval.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Test-Time Augmentation (TTA)

Free inference-only improvement: run each test sample under 4 views
(identity, H-flip, V-flip, 180 rotate), invert each view back, average the
softmax probabilities, then argmax.

* U-Net: 4 forward passes on the image.
* Deep fusion: 4 forward passes; the image branch sees flipped views, the
  CSV branch sees the same window each time (the LSTM does not depend on
  image orientation), and the **inverse spatial transform is applied to the
  output probability map** for each view before averaging.
* Bi-LSTM: image-blind, so TTA is a no-op. Reported as-is for completeness.

The CSV branch always receives the *original* photon window because the
ICESat-2 track has a fixed along-track order; flipping the photons would
not be a valid augmentation here.""")

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
OUT_DIR      = RUNS / "tta"
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

# -------- Cell: manifest + test split + features -----------------------
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

# -------- Cell: dataset --------------------------------------------------
md(r"""## 3. Test dataset""")

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


test_loader = DataLoader(TestSet(test_df), batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)
""")

# -------- Cell: model defs ---------------------------------------------
md(r"""## 4. Model definitions""")

code(r"""class CSVOnlyModel(nn.Module):
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
md(r"""## 5. Load trained models""")

code(r"""unet = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                in_channels=3, classes=NUM_CLASSES).to(device)
ck = torch.load(RUN_UNET / "best.pt", map_location=device, weights_only=False)
unet.load_state_dict(ck["model_state"]); unet.eval()

lstm = CSVOnlyModel(n_features=n_features).to(device)
ck = torch.load(RUN_LSTM / "best.pt", map_location=device, weights_only=False)
lstm.load_state_dict(ck["model_state"]); lstm.eval()

fusion = DeepFusionModel(n_features=n_features).to(device)
ck = torch.load(RUN_FUSION / "best.pt", map_location=device, weights_only=False)
fusion.load_state_dict(ck["model_state"]); fusion.eval()
print("all three models loaded")
""")

# -------- Cell: TTA helpers ---------------------------------------------
md(r"""## 6. TTA helpers

Four views: identity, H-flip, V-flip, 180 deg rotate.

Each view has a forward transform (applied to the image input) and a
matching inverse transform (applied to the output probability map). The
inverse is the *same* operation for all four of these since H-flip and
V-flip are self-inverse.""")

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

# -------- Cell: inference + cm runner ----------------------------------
md(r"""## 7. Per-model evaluation (no-TTA and with-TTA)""")

code(r"""@torch.no_grad()
def evaluate(model_fn, use_tta: bool):
    # model_fn(img, win) -> logits.
    # If use_tta=True, average softmax probabilities across the 4 views.
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img, win, y in test_loader:
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)

        if not use_tta:
            with torch.amp.autocast("cuda"):
                logits = model_fn(img, win)
            probs = F.softmax(logits, dim=1)
        else:
            acc = None
            for v in VIEWS:
                fwd, inv = view_pair(v)
                with torch.amp.autocast("cuda"):
                    logits_v = model_fn(fwd(img), win)
                p_v = inv(F.softmax(logits_v, dim=1))
                acc = p_v if acc is None else acc + p_v
            probs = acc / len(VIEWS)

        pred = probs.argmax(1).cpu().numpy().ravel()
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

# -------- Cell: run all combos -----------------------------------------
md(r"""## 8. Run all combinations""")

code(r"""configs = [
    ("U-Net (image only)", lambda i, w: unet(i)),
    ("Bi-LSTM (CSV only)", lambda i, w: lstm(w)),
    ("Deep fusion",        lambda i, w: fusion(i, w)),
]

results = {}
for name, fn in configs:
    print(f"--- {name} ---")
    for use_tta in [False, True]:
        # Bi-LSTM is image-blind: TTA is a no-op, just skip to save time.
        if name == "Bi-LSTM (CSV only)" and use_tta:
            results[(name, True)] = results[(name, False)]
            print(f"  TTA skipped (model is image-blind)")
            continue
        cm = evaluate(fn, use_tta=use_tta)
        iou, miou = iou_from_cm(cm)
        results[(name, use_tta)] = {"cm": cm, "iou": iou, "miou": miou}
        tag = "with TTA" if use_tta else "no TTA  "
        print(f"  {tag}: mIoU={miou:.4f}  per-class={iou.round(4).tolist()}")
""")

# -------- Cell: assemble table -----------------------------------------
md(r"""## 9. Results table""")

code(r"""rows = []
for name, _ in configs:
    base = results[(name, False)]
    tta  = results[(name, True)]
    rows.append({
        "model":             name,
        "mIoU (no TTA)":     round(base["miou"], 4),
        "mIoU (with TTA)":   round(tta["miou"], 4),
        "delta mIoU":        round(tta["miou"] - base["miou"], 4),
        "ice IoU (TTA)":     round(float(tta["iou"][0]), 4),
        "thin IoU (TTA)":    round(float(tta["iou"][1]), 4),
        "water IoU (TTA)":   round(float(tta["iou"][2]), 4),
    })

table = pd.DataFrame(rows).set_index("model")
print(table)
table.to_csv(OUT_DIR / "tta_results_table.csv")
""")

# -------- Cell: bar chart ----------------------------------------------
md(r"""## 10. Bar chart -- with vs without TTA""")

code(r"""variants = list(table.index)
x = np.arange(len(variants))
width = 0.35

fig, ax = plt.subplots(figsize=(8.5, 4.5))
b1 = ax.bar(x - width/2, table["mIoU (no TTA)"].values, width,
            label="no TTA", color="#4C72B0")
b2 = ax.bar(x + width/2, table["mIoU (with TTA)"].values, width,
            label="with TTA (4 views)", color="#C44E52")

for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=10)
ax.set_ylabel("mIoU (test tile T03CWT)")
ax.set_ylim(0, 1.05)
ax.set_title("Effect of 4-view test-time augmentation")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "tta_bar.png", dpi=160)
plt.show()
""")

# -------- Cell: save metrics json --------------------------------------
md(r"""## 11. Save metrics json""")

code(r"""payload = {}
for (name, use_tta), m in results.items():
    key = f"{name} | {'TTA' if use_tta else 'noTTA'}"
    payload[key] = {
        "per_iou": m["iou"].tolist(),
        "miou":    m["miou"],
    }
with open(OUT_DIR / "tta_metrics.json", "w") as f:
    json.dump(payload, f, indent=2)
print("saved tta_metrics.json")
""")

md(r"""## 12. Notes for the writeup

* TTA averages predictions across 4 input orientations -- a standard, free
  inference-only trick. No retraining.
* Bi-LSTM does not benefit because it ignores the image; reported as-is for
  completeness.
* Expect deep fusion to gain the most (cleaner edge predictions near
  thin-ice / water boundaries).""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
