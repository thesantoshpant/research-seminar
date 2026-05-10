"""
Generate fusion_deep.ipynb -- the deep fusion model (U-Net + Bi-LSTM with
SE-attention feature-level fusion).

This is the main event. Trains end-to-end:
  - U-Net (ResNet-18 encoder, ImageNet pretrained) sees the RGB photo and
    produces a 16-channel feature map at full 128x128 resolution.
  - Bi-LSTM consumes a window of K=32 CSV rows and produces a 256-D vector
    summarizing the local along-track context.
  - The CSV vector is projected to 16 channels and broadcast across the
    spatial grid; concatenated with the U-Net features (32 total channels);
    SE-attention rescales channels; a small conv head outputs per-pixel
    3-class logits.

Both branches train together. Gradients flow through both during the same
backward pass -- that's what 'deep fusion' means.

Run this script and it writes the notebook to disk.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "fusion_deep.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Sea Ice Segmentation -- Deep Fusion (U-Net + Bi-LSTM)

The main model. Two branches, fused inside the network so they help each
other during training.

* **Image branch:** U-Net with ResNet-18 encoder (ImageNet pretrained),
  outputs a 16-channel feature map at full 128x128 resolution.
* **CSV branch:** Bi-LSTM (2 layers, hidden=128, bidirectional) over a
  K=32 window of consecutive CSV rows; center-pooled to a 256-D vector.
* **Fusion:** the CSV vector is projected to 16 channels, broadcast across
  the 128x128 grid, concatenated with the image features (32 channels total),
  then channel-wise rescaled by an SE attention block before a 3x3 +
  1x1 conv head produces per-pixel 3-class logits.

Same loss, same metric, same splits as the U-Net baseline so the numbers
sit cleanly next to each other in the ablation table.

Expected wall time on A6000: **~1-2 hours** for 30 epochs.""")

# -------- Cell: GPU pin --------------------------------------------------
md("## 0. Setup")

code(r"""# Pin to a specific GPU on a shared box. MUST run before `import torch`.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
""")

code(r"""import sys, subprocess
for pkg in ["segmentation_models_pytorch", "tqdm"]:
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
import segmentation_models_pytorch as smp

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- Cell: config ---------------------------------------------------
md("## 1. Config")

code(r"""# >>> EDIT ME <<< -- match wherever your data lives on the GPU box.
PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
EXP_NAME     = "fusion_deep_v1"

IMG_DIR     = PROJECT_ROOT / "outputs"
MASK_DIR    = PROJECT_ROOT / "outputs_segmented"
CSV_DIR     = PROJECT_ROOT / "IS2_Corrected_data"
RUN_DIR     = PROJECT_ROOT / "runs" / EXP_NAME
PRIOR_UNET  = PROJECT_ROOT / "runs" / "unet_imgonly_v1"
PRIOR_LSTM  = PROJECT_ROOT / "runs" / "lstm_csvonly_v1"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
SEED         = 42
NUM_CLASSES  = 3
PATCH        = 128
WINDOW_K     = 32
BATCH_SIZE   = 32         # bigger model than U-Net alone; 32 is safe on A6000
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 5
NUM_WORKERS  = 4
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.2
FUSION_CH    = 16          # both branches project to this channel count for concat
USE_AMP      = True

CLASS_NAMES = ["ice", "thin_ice", "water"]
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}

# CSV feature selection (same as LSTM baseline)
DROP_COLS = {
    "Unnamed: 0", "Ori_Id",
    "year", "month", "day", "hour", "minute", "second",
    "geometry", "pix_x", "pix_y", "label",
    "lat", "lon", "x", "y",
    "x_atc",
    "s_azi", "s_ele",
}

# ImageNet normalization for the ResNet-18 encoder
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Project root:", PROJECT_ROOT)
print("Run dir:     ", RUN_DIR)
""")

# -------- Cell: manifest -------------------------------------------------
md(r"""## 2. Manifest

Reuse the manifest built by the U-Net notebook (it's the same set of
samples). Falls back to rebuilding if absent.""")

code(r"""import re

manifest_path_prior = PRIOR_UNET / "manifest.csv"
manifest_path_local = RUN_DIR / "manifest.csv"

if manifest_path_prior.exists():
    manifest = pd.read_csv(manifest_path_prior)
    print(f"Loaded manifest from {manifest_path_prior}: {len(manifest):,} rows")
elif manifest_path_local.exists():
    manifest = pd.read_csv(manifest_path_local)
else:
    pat = re.compile(r"^row(\d+)_(\d{8}T\d{6})_(T\d+[A-Z]+)_(gt[12]r)\.png$")
    rows = []
    for p in sorted(IMG_DIR.iterdir()):
        m = pat.match(p.name)
        if not m:
            continue
        rows.append({
            "filename": p.name, "row_idx": int(m.group(1)),
            "date": m.group(2), "tile": m.group(3), "beam": m.group(4),
            "image_path": str(p), "mask_path": str(MASK_DIR / p.name),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path_local, index=False)
    print(f"Built manifest: {len(manifest):,} rows")
""")

# -------- Cell: link CSVs -----------------------------------------------
md("## 3. Link manifest rows to source CSVs")

code(r"""csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
csv_meta = []
for p in csv_files:
    parts = p.stem.split("_")
    csv_meta.append({"csv_path": str(p), "csv_name": p.name,
                     "tile": parts[3], "beam": parts[4]})
csv_meta = pd.DataFrame(csv_meta)
csv_meta["csv_id"] = csv_meta.index

manifest = manifest.merge(csv_meta[["tile", "beam", "csv_path", "csv_id"]],
                          on=["tile", "beam"], how="left")
assert manifest["csv_id"].notna().all()
manifest["csv_id"] = manifest["csv_id"].astype(int)
print(f"{manifest['csv_id'].nunique()} CSVs linked to {len(manifest):,} rows")
""")

# -------- Cell: split ---------------------------------------------------
md("## 4. Tile-based train/val/test split (identical to baselines)")

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

# -------- Cell: CSV preprocessing ---------------------------------------
md(r"""## 5. CSV features: load, normalize, cache

Reuse the cached normalized arrays from the LSTM run if present
(`runs/lstm_csvonly_v1/csv_normed/csv_*.npy`). Otherwise compute fresh
using train-tile statistics so test features stay invisible to training.""")

code(r"""# Load each CSV once and pick feature columns
raw_csvs = {}
for _, row in csv_meta.iterrows():
    raw_csvs[int(row["csv_id"])] = pd.read_csv(row["csv_path"])

first_id = next(iter(raw_csvs))
feature_cols = [c for c in raw_csvs[first_id].columns if c not in DROP_COLS]
n_features = len(feature_cols)
print(f"Using {n_features} features")
""")

code(r"""prior_npy_dir   = PRIOR_LSTM / "csv_normed"
prior_stats     = PRIOR_LSTM / "feature_stats.json"
local_npy_dir   = RUN_DIR / "csv_normed"
local_npy_dir.mkdir(exist_ok=True)

if prior_stats.exists() and prior_npy_dir.exists() and \
   all((prior_npy_dir / f"csv_{cid}.npy").exists() for cid in raw_csvs):
    print(f"Loading cached normalized CSVs from {prior_npy_dir}")
    csv_features = {cid: np.load(prior_npy_dir / f"csv_{cid}.npy")
                    for cid in raw_csvs}
    with open(prior_stats) as f:
        d = json.load(f)
    assert d["feature_cols"] == feature_cols, "feature column mismatch with prior run"
else:
    print("Computing CSV normalization from scratch")
    train_arrays = []
    for _, row in csv_meta.iterrows():
        if row["tile"] not in tiles_train:
            continue
        arr = raw_csvs[int(row["csv_id"])][feature_cols].to_numpy(dtype=np.float32)
        train_arrays.append(arr)
    train_concat = np.concatenate(train_arrays, axis=0)
    mu = np.nanmean(train_concat, axis=0).astype(np.float32)
    sd = np.nanstd (train_concat, axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    csv_features = {}
    for cid, df in raw_csvs.items():
        z = (df[feature_cols].to_numpy(dtype=np.float32) - mu) / sd
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        csv_features[cid] = z
        np.save(local_npy_dir / f"csv_{cid}.npy", z)
    with open(RUN_DIR / "feature_stats.json", "w") as f:
        json.dump({"feature_cols": feature_cols,
                   "mean": mu.tolist(), "std": sd.tolist()}, f, indent=2)

for cid, arr in csv_features.items():
    print(f"  csv_{cid}: shape={arr.shape}")
""")

# -------- Cell: mask decoder --------------------------------------------
md("## 6. Mask color -> integer label")

code(r"""def mask_rgb_to_int(mask_rgb):
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    out[(mask_rgb == [255, 0, 0]).all(axis=-1)] = 0
    out[(mask_rgb == [0, 0, 255]).all(axis=-1)] = 1
    out[(mask_rgb == [0, 255, 0]).all(axis=-1)] = 2
    return out
""")

# -------- Cell: class weights ------------------------------------------
md(r"""## 7. Class weights (reuse from prior runs)""")

code(r"""prior_weights = PRIOR_UNET / "class_weights.json"
local_weights = RUN_DIR / "class_weights.json"
if prior_weights.exists():
    with open(prior_weights) as f: d = json.load(f)
    print(f"Reused class weights from {prior_weights}")
elif local_weights.exists():
    with open(local_weights) as f: d = json.load(f)
else:
    n_sample = min(5000, len(train_df))
    sampled = train_df.sample(n=n_sample, random_state=SEED)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in tqdm(sampled["mask_path"], desc="counting class pixels"):
        m = mask_rgb_to_int(np.array(Image.open(p).convert("RGB")))
        for c in range(NUM_CLASSES):
            counts[c] += int((m == c).sum())
    weights_arr = (counts.sum() / (NUM_CLASSES * counts.astype(np.float64))).astype(np.float32)
    d = {"counts": counts.tolist(), "weights": weights_arr.tolist(), "n_sample": int(n_sample)}
    with open(local_weights, "w") as f: json.dump(d, f, indent=2)

counts  = np.array(d["counts"],  dtype=np.int64)
weights = np.array(d["weights"], dtype=np.float32)
for c, name in enumerate(CLASS_NAMES):
    print(f"  {name:8s}  {counts[c]:>14,d} px ({100*counts[c]/counts.sum():5.2f}%)  weight={weights[c]:.3f}")
""")

# -------- Cell: dataset --------------------------------------------------
md(r"""## 8. Dataset

Returns `(image, csv_window, valid, mask)`. Image and mask get the same
random spatial augmentation (flips + 90deg rotations); the CSV window does
not (it's a sequence of measurements, not a spatial grid).""")

code(r"""HALF = WINDOW_K // 2

def random_flip_rotate(img, mask):
    if random.random() < 0.5:
        img = img[:, ::-1, :]; mask = mask[:, ::-1]
    if random.random() < 0.5:
        img = img[::-1, :, :]; mask = mask[::-1, :]
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k, axes=(0, 1)); mask = np.rot90(mask, k, axes=(0, 1))
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class SeaIceFusionDataset(Dataset):
    def __init__(self, df, csv_features, augment=False):
        self.df = df.reset_index(drop=True)
        self.csv_features = csv_features
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        img  = np.array(Image.open(r["image_path"]).convert("RGB"))
        mask = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))
        if self.augment:
            img, mask = random_flip_rotate(img, mask)
        img = (img.astype(np.float32) / 255.0 - IM_MEAN) / IM_STD
        img = np.transpose(img, (2, 0, 1))

        feats = self.csv_features[int(r["csv_id"])]
        n_rows, n_feat = feats.shape
        center = int(r["row_idx"])
        win = np.zeros((WINDOW_K, n_feat), dtype=np.float32)
        valid = np.zeros((WINDOW_K,), dtype=np.float32)
        for k in range(WINDOW_K):
            src = center - HALF + k
            if 0 <= src < n_rows:
                win[k] = feats[src]; valid[k] = 1.0

        return (torch.from_numpy(img),
                torch.from_numpy(win),
                torch.from_numpy(valid),
                torch.from_numpy(mask).long())


# Spot-check
_ds = SeaIceFusionDataset(train_df.head(8), csv_features, augment=True)
_x, _w, _v, _m = _ds[0]
print(f"image:   {_x.shape}, range [{_x.min():.2f}, {_x.max():.2f}]")
print(f"window:  {_w.shape}, mean={_w.mean():.3f}, std={_w.std():.3f}")
print(f"valid:   {_v.shape}, sum={_v.sum().item():.0f}/{WINDOW_K}")
print(f"mask:    {_m.shape}, unique={torch.unique(_m).tolist()}")
""")

# -------- Cell: loaders -------------------------------------------------
md("## 9. DataLoaders")

code(r"""train_ds = SeaIceFusionDataset(train_df, csv_features, augment=True)
val_ds   = SeaIceFusionDataset(val_df,   csv_features, augment=False)
test_ds  = SeaIceFusionDataset(test_df,  csv_features, augment=False)

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
md(r"""## 10. Deep fusion model

```
RGB (B,3,128,128) ---> U-Net (ResNet-18 enc/dec) ---> img_feat (B,16,128,128)
                                                                  |
                                                                  v
                                                       cat -> SE -> head -> logits (B,3,128,128)
                                                                  ^
                                                                  |
CSV (B,32,F) ---> proj(F->128) -> Bi-LSTM ---> center pool -> Linear(256->16) -> tile -> (B,16,128,128)
```

The SE block (Squeeze-and-Excitation) is the "attention" piece -- it
learns per-channel scaling weights conditioned on a global summary of
both branches' features, so the network can dynamically up- or down-weight
either modality on a sample-by-sample basis.""")

code(r"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # Image branch -- full U-Net but output `fusion_ch` feature channels (not class logits yet)
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=fusion_ch,
        )

        # CSV branch -- Bi-LSTM
        self.csv_proj = nn.Sequential(
            nn.Linear(n_features, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=lstm_hidden, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.csv_to_chan = nn.Linear(lstm_hidden * 2, fusion_ch)

        # Fusion head: SE attention -> conv -> head
        self.se = SqueezeExcitation(channels=fusion_ch * 2, reduction=8)
        self.head = nn.Sequential(
            nn.Conv2d(fusion_ch * 2, fusion_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fusion_ch, num_classes, kernel_size=1),
        )

    def forward(self, img, csv_window, valid=None):
        img_feat = self.unet(img)                                       # (B, fusion_ch, 128, 128)

        h = self.csv_proj(csv_window)                                   # (B, K, lstm_hidden)
        h, _ = self.lstm(h)                                              # (B, K, 2*lstm_hidden)
        center = csv_window.size(1) // 2
        csv_feat = self.csv_to_chan(h[:, center, :])                    # (B, fusion_ch)
        csv_feat = csv_feat[:, :, None, None].expand(-1, -1, self.patch, self.patch)

        fused = torch.cat([img_feat, csv_feat], dim=1)                  # (B, 2*fusion_ch, 128, 128)
        fused = self.se(fused)
        return self.head(fused)                                          # (B, num_classes, 128, 128)


model = DeepFusionModel(n_features=n_features).to(device)

with torch.no_grad():
    dummy_img = torch.zeros(2, 3, PATCH, PATCH, device=device)
    dummy_csv = torch.zeros(2, WINDOW_K, n_features, device=device)
    out = model(dummy_img, dummy_csv)
print(f"output shape: {tuple(out.shape)}  dtype={out.dtype}")
def n_params(m):
    return sum(p.numel() for p in m.parameters())
print(f"params: total={n_params(model)/1e6:.2f}M  "
      f"unet={n_params(model.unet)/1e6:.2f}M  "
      f"lstm={n_params(model.lstm)/1e6:.2f}M  "
      f"head={(n_params(model.se)+n_params(model.head)+n_params(model.csv_proj)+n_params(model.csv_to_chan))/1e6:.2f}M")
""")

# -------- Cell: training utilities -------------------------------------
md("## 11. Training utilities (mIoU accumulator + evaluate)")

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

    def per_class_iou(self):  return self.inter / np.maximum(self.union, 1)
    def miou(self):           return float(self.per_class_iou().mean())
    def pixel_accuracy(self): return float(np.diag(self.cm).sum() / max(self.cm.sum(), 1))


def evaluate(model, loader, criterion, device):
    model.eval()
    acc = IoUAccumulator(NUM_CLASSES)
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for img, win, val, y in loader:
            img = img.to(device, non_blocking=True)
            win = win.to(device, non_blocking=True)
            val = val.to(device, non_blocking=True)
            y   = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(img, win, val)
                loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            acc.update(preds, y)
            loss_sum += loss.item() * img.size(0)
            n += img.size(0)
    return {"loss": loss_sum / max(n, 1),
            "miou": acc.miou(), "per_iou": acc.per_class_iou().tolist(),
            "pix_acc": acc.pixel_accuracy(), "cm": acc.cm.tolist()}
""")

# -------- Cell: training loop -------------------------------------------
md(r"""## 12. Training loop

End-to-end. Both branches train at the same time on the same loss.""")

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
    for img, win, val, y in pbar:
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)
        val = val.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(img, win, val)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * img.size(0)
        n += img.size(0)
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
md("## 13. Final test evaluation")

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
ax.set_title(f"Test confusion (row-normalized)\nDeep Fusion -- mIoU={test_metrics['miou']:.3f}")
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

Pick 6 random test samples and plot RGB | ground truth | fusion prediction.""")

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
        rgb_raw = np.array(Image.open(r["image_path"]).convert("RGB"))
        gt      = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))

        # build inputs
        img_norm = ((rgb_raw.astype(np.float32) / 255.0 - IM_MEAN) / IM_STD)
        img_t = torch.from_numpy(np.transpose(img_norm, (2, 0, 1)))[None].to(device)

        feats = csv_features[int(r["csv_id"])]
        n_rows = feats.shape[0]; center = int(r["row_idx"])
        win = np.zeros((WINDOW_K, n_features), dtype=np.float32)
        for k in range(WINDOW_K):
            src = center - HALF + k
            if 0 <= src < n_rows: win[k] = feats[src]
        win_t = torch.from_numpy(win)[None].to(device)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            pred = model(img_t, win_t).argmax(1)[0].cpu().numpy()

        axes[i, 0].imshow(rgb_raw);                axes[i, 0].set_title("input"        if i == 0 else "")
        axes[i, 1].imshow(int_mask_to_rgb(gt));    axes[i, 1].set_title("ground truth" if i == 0 else "")
        axes[i, 2].imshow(int_mask_to_rgb(pred));  axes[i, 2].set_title("deep fusion"  if i == 0 else "")
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

Artifacts in `runs/fusion_deep_v1/`:
* `metrics.csv`, `best.pt`, `test_metrics.json`
* `confmat.png`, `sample_predictions.png`

The ablation table once this finishes:

| Variant | Test mIoU | ice / thin / water |
|---|---|---|
| #1 U-Net (image only) | 0.8704 | 0.930 / 0.768 / 0.913 |
| #2 Bi-LSTM (CSV only) | 0.2420 | ~0.73 / 0.00 / 0.00 |
| **#5/6 Deep fusion (this run)** | _fill in_ | _fill in_ |

The fusion number is the headline result. If it beats 0.8704 -- especially
on **thin ice IoU**, where the U-Net struggled most -- the photon data
contributed something the image alone couldn't see, and the project's
thesis is validated.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
