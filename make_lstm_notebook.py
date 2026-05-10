"""
Generate lstm_baseline.ipynb -- the CSV-only Bi-LSTM baseline (ablation #2).
Same task, same splits, same loss, same metric as the U-Net notebook;
model is a Bi-LSTM that consumes a window of K=32 CSV rows and tiles the
result across the 128x128 patch.

Run this script and it writes the notebook to disk.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "lstm_baseline.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Sea Ice Segmentation -- Bi-LSTM (CSV-only) baseline

This is **variant #2** of our ablation: train a Bi-LSTM on a window of
ICESat-2 photon features only (no image), then tile the LSTM output across
the 128x128 patch and predict per-pixel.

* **Input:** for each labeled point, a window of K=32 consecutive CSV rows
  centered on that row (16 before + the row + 15 after, ordered by row index
  in its CSV).
* **Target:** the same 128x128 segmented mask as variant #1.
* **Loss / metric / splits:** identical to the U-Net baseline so numbers
  are directly comparable.

Expected outcome: **much lower mIoU than U-Net**. Because the LSTM has no
spatial information, it can only emit a single class per patch, so per-pixel
mIoU is fundamentally capped. That's the *point* of this ablation -- it
quantifies what the photon features alone are worth.

Designed to run on a single A6000. Light model -- expect ~10-20 min total.
""")

# -------- Cell: GPU pin --------------------------------------------------
md("## 0. Setup")

code(r"""# Pin to a specific GPU on a shared box. MUST run before `import torch`.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
""")

code(r"""import sys, subprocess
for pkg in ["tqdm"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
""")

code(r"""import json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- Cell: config ---------------------------------------------------
md(r"""## 1. Config""")

code(r"""# >>> EDIT ME <<< -- match wherever your data lives on the GPU box.
PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
EXP_NAME     = "lstm_csvonly_v1"

IMG_DIR     = PROJECT_ROOT / "outputs"
MASK_DIR    = PROJECT_ROOT / "outputs_segmented"
CSV_DIR     = PROJECT_ROOT / "IS2_Corrected_data"
RUN_DIR     = PROJECT_ROOT / "runs" / EXP_NAME
PRIOR_RUN   = PROJECT_ROOT / "runs" / "unet_imgonly_v1"   # to reuse manifest + class weights
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
SEED         = 42
NUM_CLASSES  = 3
PATCH        = 128
WINDOW_K     = 32         # context window: 16 rows before + center + 15 after
BATCH_SIZE   = 256        # LSTM is tiny; we can fit much bigger batches than U-Net
EPOCHS       = 30
LR           = 1e-3       # higher than U-Net since model is fully from scratch
WEIGHT_DECAY = 1e-4
PATIENCE     = 5
NUM_WORKERS  = 4
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.2
USE_AMP      = True

CLASS_NAMES = ["ice", "thin_ice", "water"]
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}

# CSV feature selection
# Drop: index, ID, raw timestamp, geometry, pixel coords, label, geographic coords,
# along-track distance, sun angles (these are nearly constant per CSV and would
# leak tile identity).
DROP_COLS = {
    "Unnamed: 0", "Ori_Id",
    "year", "month", "day", "hour", "minute", "second",
    "geometry", "pix_x", "pix_y", "label",
    "lat", "lon", "x", "y",
    "x_atc",
    "s_azi", "s_ele",
}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Project root:", PROJECT_ROOT)
print("Run dir:     ", RUN_DIR)
""")

# -------- Cell: manifest -------------------------------------------------
md(r"""## 2. Manifest

Reuse the manifest built by the U-Net notebook (saved under
`runs/unet_imgonly_v1/manifest.csv`). If it isn't there, we rebuild it.""")

code(r"""import re

manifest_path_prior = PRIOR_RUN / "manifest.csv"
manifest_path_local = RUN_DIR / "manifest.csv"

if manifest_path_prior.exists():
    manifest = pd.read_csv(manifest_path_prior)
    print(f"Loaded existing manifest from prior run: {len(manifest):,} rows")
elif manifest_path_local.exists():
    manifest = pd.read_csv(manifest_path_local)
    print(f"Loaded existing manifest: {len(manifest):,} rows")
else:
    pat = re.compile(r"^row(\d+)_(\d{8}T\d{6})_(T\d+[A-Z]+)_(gt[12]r)\.png$")
    rows = []
    for p in sorted(IMG_DIR.iterdir()):
        m = pat.match(p.name)
        if not m:
            continue
        rows.append({
            "filename": p.name,
            "row_idx":  int(m.group(1)),
            "date":     m.group(2),
            "tile":     m.group(3),
            "beam":     m.group(4),
            "image_path": str(p),
            "mask_path":  str(MASK_DIR / p.name),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path_local, index=False)
    print(f"Built manifest: {len(manifest):,} rows")

manifest.head(3)
""")

# -------- Cell: link manifest -> CSVs -----------------------------------
md(r"""## 3. Map every manifest row to its source CSV file

Each (tile, beam, date) triple has exactly one CSV. We match by filename
pattern (the prof's standard naming) and add a `csv_id` column.""")

code(r"""csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
csv_meta = []
for p in csv_files:
    # Filename: ATL03_<date>_<orbit>_<tile>_<beam>_labeled_10m_done.csv
    parts = p.stem.split("_")
    csv_meta.append({
        "csv_path": str(p),
        "csv_name": p.name,
        "tile":     parts[3],
        "beam":     parts[4],
    })
csv_meta = pd.DataFrame(csv_meta)
csv_meta["csv_id"] = csv_meta.index
print(csv_meta[["csv_id", "tile", "beam", "csv_name"]])

manifest = manifest.merge(csv_meta[["tile", "beam", "csv_path", "csv_id"]],
                          on=["tile", "beam"], how="left")
assert manifest["csv_id"].notna().all(), "some manifest rows have no matching CSV"
manifest["csv_id"] = manifest["csv_id"].astype(int)
print(f"manifest now has csv_id linking to its source CSV; "
      f"{manifest['csv_id'].nunique()} unique CSVs")
""")

# -------- Cell: split ---------------------------------------------------
md(r"""## 4. Tile-based train/val/test split (same as U-Net notebook)""")

code(r"""tiles_train = ["T02CNA", "T02CNC"]
tiles_test  = ["T03CWT"]

train_pool = manifest[manifest["tile"].isin(tiles_train)].reset_index(drop=True)
test_df    = manifest[manifest["tile"].isin(tiles_test)].reset_index(drop=True)

rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train_pool), size=int(0.10 * len(train_pool)), replace=False)
val_mask_arr = np.zeros(len(train_pool), dtype=bool); val_mask_arr[val_idx] = True
train_df = train_pool[~val_mask_arr].reset_index(drop=True)
val_df   = train_pool[ val_mask_arr].reset_index(drop=True)

print(f"Train: {len(train_df):,}   Val: {len(val_df):,}   Test: {len(test_df):,}")
""")

# -------- Cell: CSV preprocessing -- features ---------------------------
md(r"""## 5. Pre-process CSVs: pick features, normalize, cache as numpy arrays

We load each CSV once, drop columns we don't want (see `DROP_COLS`),
fit a StandardScaler on **training-tile rows only**, and save the
normalized features per CSV. Indexing later is O(1) per sample.

Z-scoring is fit on training tiles only so the test set's feature
distribution doesn't leak.""")

code(r"""# Read every CSV once, decide on the feature column order from the first one
raw_csvs = {}
for _, row in csv_meta.iterrows():
    df = pd.read_csv(row["csv_path"])
    raw_csvs[row["csv_id"]] = df

first_id = next(iter(raw_csvs))
all_cols = list(raw_csvs[first_id].columns)
feature_cols = [c for c in all_cols if c not in DROP_COLS]
n_features = len(feature_cols)
print(f"Using {n_features} features:")
print("  " + ", ".join(feature_cols))
""")

code(r"""# Build a big numpy array of training-tile features for fitting the scaler
train_arrays = []
for _, row in csv_meta.iterrows():
    if row["tile"] not in tiles_train:
        continue
    df = raw_csvs[row["csv_id"]]
    arr = df[feature_cols].to_numpy(dtype=np.float32)
    train_arrays.append(arr)

train_concat = np.concatenate(train_arrays, axis=0)
print(f"training rows: {train_concat.shape[0]:,}, features: {train_concat.shape[1]}")
print(f"NaN count in training rows: {int(np.isnan(train_concat).sum())}")
""")

code(r"""# Fit z-score on training rows only; replace any NaN with 0 (post-normalization)
mu = np.nanmean(train_concat, axis=0).astype(np.float32)
sd = np.nanstd (train_concat, axis=0).astype(np.float32)
sd[sd < 1e-6] = 1.0  # avoid divide-by-zero on constant columns

stats_path = RUN_DIR / "feature_stats.json"
with open(stats_path, "w") as f:
    json.dump({"feature_cols": feature_cols,
               "mean": mu.tolist(), "std": sd.tolist()}, f, indent=2)
print(f"saved feature stats -> {stats_path}")

# Apply to every CSV and cache as .npy
csv_features = {}   # csv_id -> (n_rows, n_features) float32
csv_dir_norm = RUN_DIR / "csv_normed"
csv_dir_norm.mkdir(exist_ok=True)
for cid, df in raw_csvs.items():
    raw = df[feature_cols].to_numpy(dtype=np.float32)
    z = (raw - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    csv_features[cid] = z
    np.save(csv_dir_norm / f"csv_{cid}.npy", z)
print(f"normalized + cached {len(csv_features)} CSVs to {csv_dir_norm}")
for cid, arr in csv_features.items():
    print(f"  csv_{cid}: shape={arr.shape}")
""")

# -------- Cell: mask decoding -------------------------------------------
md("## 6. Mask color -> integer label (same helper as before)")

code(r"""def mask_rgb_to_int(mask_rgb):
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    out[(mask_rgb == [255, 0, 0]).all(axis=-1)] = 0
    out[(mask_rgb == [0, 0, 255]).all(axis=-1)] = 1
    out[(mask_rgb == [0, 255, 0]).all(axis=-1)] = 2
    return out
""")

# -------- Cell: class weights ------------------------------------------
md(r"""## 7. Class weights (reuse the U-Net run if present)""")

code(r"""prior_weights = PRIOR_RUN / "class_weights.json"
local_weights = RUN_DIR / "class_weights.json"

if prior_weights.exists():
    with open(prior_weights) as f:
        d = json.load(f)
    print(f"Reusing class weights from {prior_weights}")
elif local_weights.exists():
    with open(local_weights) as f:
        d = json.load(f)
else:
    n_sample = min(5000, len(train_df))
    sampled = train_df.sample(n=n_sample, random_state=SEED)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in tqdm(sampled["mask_path"], desc="counting class pixels"):
        m = mask_rgb_to_int(np.array(Image.open(p).convert("RGB")))
        for c in range(NUM_CLASSES):
            counts[c] += int((m == c).sum())
    total = counts.sum()
    weights_arr = (total / (NUM_CLASSES * counts.astype(np.float64))).astype(np.float32)
    d = {"counts": counts.tolist(), "weights": weights_arr.tolist(),
         "n_sample": int(n_sample)}
    with open(local_weights, "w") as f:
        json.dump(d, f, indent=2)

counts  = np.array(d["counts"],  dtype=np.int64)
weights = np.array(d["weights"], dtype=np.float32)
for c, name in enumerate(CLASS_NAMES):
    pct = 100 * counts[c] / counts.sum()
    print(f"  {name:8s}  {counts[c]:>14,d} px ({pct:5.2f}%)  weight={weights[c]:.3f}")
""")

# -------- Cell: dataset --------------------------------------------------
md(r"""## 8. Dataset

Each sample returns:

* `csv_window`: `(K, F)` float32 — the K rows centered on the labeled row,
  padded with zeros at edges of the track.
* `valid_mask`: `(K,)` float32 — 1 where the timestep is a real CSV row,
  0 where padded. (We don't use this in the loss right now, but it's there
  if you want to mask the LSTM later.)
* `mask`: `(H, W)` long — the segmentation target.

No image is returned -- this is the CSV-only baseline.""")

code(r"""HALF = WINDOW_K // 2

class SeaIceCSVDataset(Dataset):
    def __init__(self, df, csv_features, mask_dir):
        self.df = df.reset_index(drop=True)
        self.csv_features = csv_features
        self.mask_dir = Path(mask_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        feats = self.csv_features[int(r["csv_id"])]
        n_rows, n_feat = feats.shape
        center = int(r["row_idx"])

        win = np.zeros((WINDOW_K, n_feat), dtype=np.float32)
        valid = np.zeros((WINDOW_K,), dtype=np.float32)

        # window: rows [center-HALF, center+HALF) -> WINDOW_K positions
        for k in range(WINDOW_K):
            src = center - HALF + k
            if 0 <= src < n_rows:
                win[k]   = feats[src]
                valid[k] = 1.0

        mask = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))
        return (torch.from_numpy(win),
                torch.from_numpy(valid),
                torch.from_numpy(mask).long())


# Spot-check
_ds = SeaIceCSVDataset(train_df.head(8), csv_features, MASK_DIR)
_w, _v, _m = _ds[0]
print(f"window:  {_w.shape}, dtype={_w.dtype}, mean={_w.mean():.3f}, std={_w.std():.3f}")
print(f"valid:   {_v.shape}, sum={_v.sum().item():.0f}/{WINDOW_K}")
print(f"mask:    {_m.shape}, unique={torch.unique(_m).tolist()}")
""")

# -------- Cell: dataloaders --------------------------------------------
md("## 9. DataLoaders")

code(r"""train_ds = SeaIceCSVDataset(train_df, csv_features, MASK_DIR)
val_ds   = SeaIceCSVDataset(val_df,   csv_features, MASK_DIR)
test_ds  = SeaIceCSVDataset(test_df,  csv_features, MASK_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                          persistent_workers=NUM_WORKERS > 0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=NUM_WORKERS > 0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=NUM_WORKERS > 0)

print(f"steps/epoch -- train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
""")

# -------- Cell: model ---------------------------------------------------
md(r"""## 10. Model -- Bi-LSTM, then tile across the patch

The Bi-LSTM produces one feature vector at the center timestep. We
broadcast that vector over the 128x128 grid and run a small conv head
to produce per-pixel logits. The output is necessarily uniform across
the patch (same prediction for every pixel) -- that's the design.""")

code(r"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSVOnlyModel(nn.Module):
    def __init__(self, n_features, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, num_classes=NUM_CLASSES, patch=PATCH):
        super().__init__()
        self.patch = patch

        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden * 2, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x, valid=None):
        # x: (B, K, F) -- valid is (B, K) but unused in the loss for now.
        h = self.proj(x)             # (B, K, H)
        h, _ = self.lstm(h)          # (B, K, 2H)
        center = x.size(1) // 2
        feat = h[:, center, :]       # (B, 2H)  -- pooled at center timestep
        feat = feat[:, :, None, None]                     # (B, 2H, 1, 1)
        feat = feat.expand(-1, -1, self.patch, self.patch) # (B, 2H, 128, 128)
        return self.head(feat)       # (B, num_classes, 128, 128)


model = CSVOnlyModel(n_features=n_features).to(device)
with torch.no_grad():
    dummy_w = torch.zeros(2, WINDOW_K, n_features, device=device)
    out = model(dummy_w)
print(f"model output: {tuple(out.shape)}  dtype={out.dtype}")
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
""")

# -------- Cell: training utilities -------------------------------------
md("## 11. Training utilities (mIoU, evaluate -- same shape as U-Net notebook)")

code(r"""class IoUAccumulator:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.inter = np.zeros(self.num_classes, dtype=np.int64)
        self.union = np.zeros(self.num_classes, dtype=np.int64)
        self.cm    = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        p = preds.detach().cpu().numpy().ravel()
        t = targets.detach().cpu().numpy().ravel()
        for c in range(self.num_classes):
            pc = (p == c); tc = (t == c)
            self.inter[c] += int(np.logical_and(pc, tc).sum())
            self.union[c] += int(np.logical_or (pc, tc).sum())
        idx = self.num_classes * t + p
        self.cm += np.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes)

    def per_class_iou(self):
        return self.inter / np.maximum(self.union, 1)
    def miou(self):           return float(self.per_class_iou().mean())
    def pixel_accuracy(self): return float(np.diag(self.cm).sum() / max(self.cm.sum(), 1))


def evaluate(model, loader, criterion, device):
    model.eval()
    acc = IoUAccumulator(NUM_CLASSES)
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for w, v, y in loader:
            w = w.to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(w, v)
                loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            acc.update(preds, y)
            loss_sum += loss.item() * w.size(0)
            n += w.size(0)
    return {"loss": loss_sum / max(n, 1),
            "miou": acc.miou(), "per_iou": acc.per_class_iou().tolist(),
            "pix_acc": acc.pixel_accuracy(), "cm": acc.cm.tolist()}
""")

# -------- Cell: training loop -------------------------------------------
md("## 12. Training loop")

code(r"""criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

best_miou = -1.0
patience_left = PATIENCE
log_path = RUN_DIR / "metrics.csv"
ckpt_path = RUN_DIR / "best.pt"
log = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum, n = 0.0, 0
    pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}/{EPOCHS}", leave=False)
    for w, v, y in pbar:
        w = w.to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(w, v)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * w.size(0)
        n += w.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    train_loss = train_loss_sum / max(n, 1)
    val = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
          f"val_loss={val['loss']:.4f}  val_mIoU={val['miou']:.4f}  "
          f"per_iou={[f'{v:.3f}' for v in val['per_iou']]}  "
          f"pix_acc={val['pix_acc']:.4f}  "
          f"lr={optimizer.param_groups[0]['lr']:.2e}")

    log.append({"epoch": epoch, "train_loss": train_loss,
                "val_loss": val["loss"], "val_miou": val["miou"],
                "iou_ice": val["per_iou"][0], "iou_thin": val["per_iou"][1],
                "iou_water": val["per_iou"][2], "pix_acc": val["pix_acc"],
                "lr": optimizer.param_groups[0]["lr"]})
    pd.DataFrame(log).to_csv(log_path, index=False)

    if val["miou"] > best_miou:
        best_miou = val["miou"]
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "val_metrics": val, "weights": weights.tolist(),
                    "n_features": n_features}, ckpt_path)
        patience_left = PATIENCE
        print(f"  -> saved best ({best_miou:.4f}) to {ckpt_path}")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"  -> early stopping (no val_mIoU improvement for {PATIENCE} epochs)")
            break

print(f"\nBest val mIoU: {best_miou:.4f}")
""")

# -------- Cell: final test eval -----------------------------------------
md(r"""## 13. Final test evaluation""")

code(r"""ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])

test_metrics = evaluate(model, test_loader, criterion, device)
print(f"TEST   mIoU={test_metrics['miou']:.4f}   pix_acc={test_metrics['pix_acc']:.4f}")
print("per-class IoU:")
for n, v in zip(CLASS_NAMES, test_metrics["per_iou"]):
    print(f"  {n:8s}  {v:.4f}")

cm = np.array(test_metrics["cm"])
cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Test confusion (row-normalized)\nmIoU={test_metrics['miou']:.3f}")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(RUN_DIR / "confmat.png", dpi=150)
plt.show()

with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump({k: v for k, v in test_metrics.items() if k != "cm"}, f, indent=2)
""")

# -------- Cell: visualizations -----------------------------------------
md(r"""## 14. Visualize predictions

Pick 6 random test samples. Because the LSTM-only model emits a uniform
prediction across the patch, the "predicted" column will be a single solid
color per sample.""")

code(r"""def int_mask_to_rgb(m):
    out = np.zeros((*m.shape, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        out[m == c] = color
    return out


model.eval()
sample = test_df.sample(n=6, random_state=SEED).reset_index(drop=True)
fig, axes = plt.subplots(6, 3, figsize=(7.5, 14))
with torch.no_grad():
    for i, r in sample.iterrows():
        rgb  = np.array(Image.open(r["image_path"]).convert("RGB"))
        gt   = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))

        # Build the same window the dataset would
        feats = csv_features[int(r["csv_id"])]
        n_rows = feats.shape[0]
        center = int(r["row_idx"])
        win = np.zeros((WINDOW_K, n_features), dtype=np.float32)
        for k in range(WINDOW_K):
            src = center - HALF + k
            if 0 <= src < n_rows:
                win[k] = feats[src]
        x = torch.from_numpy(win)[None].to(device)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            pred = model(x).argmax(1)[0].cpu().numpy()

        axes[i, 0].imshow(rgb);                axes[i, 0].set_title("input"        if i == 0 else "")
        axes[i, 1].imshow(int_mask_to_rgb(gt)); axes[i, 1].set_title("ground truth" if i == 0 else "")
        axes[i, 2].imshow(int_mask_to_rgb(pred)); axes[i, 2].set_title("LSTM only"   if i == 0 else "")
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(r["filename"], fontsize=7)

plt.tight_layout()
plt.savefig(RUN_DIR / "sample_predictions.png", dpi=150)
plt.show()
print(f"Saved -> {RUN_DIR / 'sample_predictions.png'}")
""")

# =========================================================================
md(r"""## Done.

Artifacts in `runs/lstm_csvonly_v1/`:
* `feature_stats.json`, `csv_normed/csv_*.npy`
* `metrics.csv`, `best.pt`, `test_metrics.json`
* `confmat.png`, `sample_predictions.png`

Compare:

| Variant | Test mIoU |
|---|---|
| U-Net (image only) | **0.8704** |
| Bi-LSTM (CSV only) | _fill in after this run_ |

Once we have both numbers, we know what each modality contributes alone.
Then variant #3 (early), #4 (late), #5 (deep concat), #6 (deep + attention)
all build on these two branches.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
