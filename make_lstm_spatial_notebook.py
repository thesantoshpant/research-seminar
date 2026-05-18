"""
Generate lstm_spatial.ipynb -- a spatial-aware Bi-LSTM that:

  1. Runs the same 32-photon Bi-LSTM we already trained, but keeps ALL
     hidden states (not just the center one).
  2. For each photon in the window, computes its (px, py) pixel position
     inside the 128x128 patch from pix_x/pix_y in the source CSV.
  3. Places each photon's hidden state at its real pixel position
     (scatter_add) and builds a "presence" mask channel.
  4. Runs a small spatial decoder (7x7 then 5x5 convs) that spreads the
     line of photon features outward into the rest of the patch.

It also uses **median-frequency-balanced** class weights instead of plain
inverse-frequency, per the professor's note on the LSTM mode-collapsing
to thick ice.

Outputs (in runs/lstm_spatial_v1/):
  - best.pt              : best-by-val-mIoU checkpoint
  - metrics.csv          : per-epoch train/val loss + mIoU
  - test_metrics.json    : final test-set metrics (matches the format used
                           by extended_metrics.ipynb so it can be re-ingested)
  - confmat.png          : test-set confusion matrix
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "lstm_spatial.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Spatial-aware Bi-LSTM (LSTM-S)

The plain Bi-LSTM collapses to all-ice on test because it produces **one
vector per patch** and tiles it: every pixel in the patch gets the same
prediction. ICESat-2 photons cover only a thin 1-D line through each
128x128 patch (~32 photons in a window), so the LSTM literally has no
information about the off-track pixels.

This notebook fixes that by:

1. Running the Bi-LSTM as before but **keeping every photon's hidden state**.
2. Placing each photon's hidden state at its **actual (pix_x, pix_y)
   location inside the patch** (recovered from the source CSV).
3. Adding a **presence mask** channel that marks where photons actually landed.
4. Running a small **spatial decoder** (7x7 and 5x5 conv layers) that spreads
   the line of photon features outward into the rest of the patch.

Pixels near the photon track get confident predictions from real data;
pixels far from the track get extrapolations from the nearest photons.

It also switches the class weights from plain inverse-frequency to
**median-frequency balancing** (the standard fix for imbalanced
segmentation), per the professor's note on the LSTM's mode collapse.""")

# -------- Cell: GPU pin --------------------------------------------------
md("## 0. Setup")

code(r"""import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
""")

code(r"""import json, random, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- Cell: config ---------------------------------------------------
md("## 1. Config")

code(r"""PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
RUNS         = PROJECT_ROOT / "runs"
RUN_UNET     = RUNS / "unet_imgonly_v1"     # for the manifest
RUN_LSTM     = RUNS / "lstm_csvonly_v1"     # for the cached csv_normed/
OUT_DIR      = RUNS / "lstm_spatial_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIR  = PROJECT_ROOT / "outputs"
MASK_DIR = PROJECT_ROOT / "outputs_segmented"
CSV_DIR  = PROJECT_ROOT / "IS2_Corrected_data"

SEED         = 42
NUM_CLASSES  = 3
PATCH        = 128
HALF         = PATCH // 2
WINDOW_K     = 32
HALF_K       = WINDOW_K // 2
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.2
DECODER_CH   = 64

EPOCHS       = 30
BATCH        = 16   # smaller than the plain LSTM run -- decoder uses more memory
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 5

CLASS_NAMES = ["ice", "thin_ice", "water"]
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("output dir:", OUT_DIR)
""")

# -------- Cell: manifest + splits + csv normed --------------------------
md(r"""## 2. Manifest + splits + cached normalized CSV features""")

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

rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train_pool), size=int(0.10 * len(train_pool)), replace=False)
val_mask_arr = np.zeros(len(train_pool), dtype=bool); val_mask_arr[val_idx] = True

train_df = train_pool[~val_mask_arr].reset_index(drop=True)
val_df   = train_pool[ val_mask_arr].reset_index(drop=True)
print(f"Train: {len(train_df):,}   Val: {len(val_df):,}   Test: {len(test_df):,}")

# normalized 23-feature arrays cached by the LSTM run
csv_features = {}
for cid in csv_meta["csv_id"]:
    arr = np.load(RUN_LSTM / "csv_normed" / f"csv_{cid}.npy")
    csv_features[int(cid)] = arr
n_features = next(iter(csv_features.values())).shape[1]
print(f"csv_features: {len(csv_features)} CSVs, {n_features} features")
""")

# -------- Cell: load raw (pix_x, pix_y) per CSV -------------------------
md(r"""## 3. Recover per-photon (pix_x, pix_y) from raw CSVs

The cached `csv_normed/csv_{cid}.npy` arrays dropped the positional
columns. Re-load `pix_x`/`pix_y` from the source CSV so we can place each
photon at its real pixel inside the patch.""")

code(r"""csv_positions = {}
for cid, row in csv_meta.iterrows():
    df_p = pd.read_csv(row["csv_path"], usecols=["pix_x", "pix_y"])
    csv_positions[int(cid)] = df_p[["pix_x", "pix_y"]].to_numpy(dtype=np.int32)
print(f"loaded positions for {len(csv_positions)} CSVs")
for cid, arr in csv_positions.items():
    print(f"  csv {cid}: {arr.shape[0]:,} rows  "
          f"x in [{arr[:,0].min()},{arr[:,0].max()}]  "
          f"y in [{arr[:,1].min()},{arr[:,1].max()}]")
""")

# -------- Cell: mask helper ---------------------------------------------
md(r"""## 4. Mask color -> integer label""")

code(r"""def mask_rgb_to_int(mask_rgb):
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    out[(mask_rgb == [255, 0, 0]).all(axis=-1)] = 0
    out[(mask_rgb == [0, 0, 255]).all(axis=-1)] = 1
    out[(mask_rgb == [0, 255, 0]).all(axis=-1)] = 2
    return out
""")

# -------- Cell: median-frequency class weights -------------------------
md(r"""## 5. Median-frequency-balanced class weights

Standard imbalanced-segmentation fix: weight class c by
`median(freq) / freq(c)` so rare classes get larger weights.

We compute on TRAINING masks only (no peeking at val/test).""")

code(r"""def compute_class_pixel_counts(df, sample=2000):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    rows = df if len(df) <= sample else df.sample(sample, random_state=SEED)
    for _, r in rows.iterrows():
        m = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))
        for c in range(NUM_CLASSES):
            counts[c] += (m == c).sum()
    return counts


print("counting training pixels (sampled subset for speed)...")
t0 = time.perf_counter()
class_counts = compute_class_pixel_counts(train_df, sample=2000)
print(f"  {time.perf_counter() - t0:.1f}s   counts = {class_counts.tolist()}")

freq = class_counts / class_counts.sum()
median_freq = np.median(freq)
class_weights = (median_freq / freq).astype(np.float32)
print(f"freq            = {freq.round(4).tolist()}")
print(f"median freq     = {median_freq:.4f}")
print(f"class weights   = {class_weights.round(3).tolist()}")
print(f"(plain inv-freq = {(1.0/freq).round(3).tolist()})")
""")

# -------- Cell: dataset --------------------------------------------------
md(r"""## 6. Dataset: (features, positions, valid mask, GT mask)

`positions` is per-photon (px, py) inside the patch, computed from the
photon's source-tile pixel coords minus the patch center photon's coords,
plus HALF (so center photon ends up at (64, 64)).

`valid` is 1.0 if the photon exists in the CSV AND its computed (px, py)
falls inside [0, 128).""")

code(r"""class SpatialLSTMDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        mask = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))

        cid = int(r["csv_id"])
        feats = csv_features[cid]
        positions = csv_positions[cid]
        n_rows = feats.shape[0]
        center = int(r["row_idx"])
        cx, cy = int(positions[center, 0]), int(positions[center, 1])

        win_feats = np.zeros((WINDOW_K, n_features), dtype=np.float32)
        win_pos   = np.zeros((WINDOW_K, 2),           dtype=np.int64)
        valid     = np.zeros((WINDOW_K,),             dtype=np.float32)

        for k in range(WINDOW_K):
            src = center - HALF_K + k
            if 0 <= src < n_rows:
                win_feats[k] = feats[src]
                fx, fy = int(positions[src, 0]), int(positions[src, 1])
                px = fx - cx + HALF
                py = fy - cy + HALF
                if 0 <= px < PATCH and 0 <= py < PATCH:
                    win_pos[k, 0] = px
                    win_pos[k, 1] = py
                    valid[k] = 1.0

        # Augmentation: random h-flip / v-flip / 90deg rotations (synced for mask + positions)
        if self.augment:
            if random.random() < 0.5:
                mask = mask[:, ::-1].copy()
                # mirror x: px -> PATCH-1 - px
                win_pos[:, 0] = (PATCH - 1 - win_pos[:, 0]) * valid.astype(np.int64) \
                                + win_pos[:, 0] * (1 - valid.astype(np.int64))
            if random.random() < 0.5:
                mask = mask[::-1, :].copy()
                win_pos[:, 1] = (PATCH - 1 - win_pos[:, 1]) * valid.astype(np.int64) \
                                + win_pos[:, 1] * (1 - valid.astype(np.int64))
            k_rot = random.choice([0, 1, 2, 3])
            if k_rot:
                mask = np.rot90(mask, k_rot).copy()
                for _ in range(k_rot):
                    # 90 deg ccw: (x, y) -> (y, PATCH-1-x)
                    new_x = win_pos[:, 1].copy()
                    new_y = PATCH - 1 - win_pos[:, 0]
                    win_pos[:, 0] = new_x
                    win_pos[:, 1] = new_y

        return (torch.from_numpy(win_feats),
                torch.from_numpy(win_pos),
                torch.from_numpy(valid),
                torch.from_numpy(mask.astype(np.int64)))


train_ds = SpatialLSTMDataset(train_df, augment=True)
val_ds   = SpatialLSTMDataset(val_df,   augment=False)
test_ds  = SpatialLSTMDataset(test_df,  augment=False)
print(f"datasets ready: {len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}")

# quick sanity check on first sample
xf, xp, v, m = train_ds[0]
print(f"feats {xf.shape}  pos {xp.shape}  valid sum {v.sum().item():.0f}/{WINDOW_K}  "
      f"mask uniq {np.unique(m.numpy()).tolist()}")
""")

# -------- Cell: model ---------------------------------------------------
md(r"""## 7. SpatialLSTMModel""")

code(r"""class SpatialLSTMModel(nn.Module):
    def __init__(self, n_features, num_classes=NUM_CLASSES, patch=PATCH,
                 hidden=LSTM_HIDDEN, layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, decoder_ch=DECODER_CH):
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
        in_ch = hidden * 2 + 1   # +1 = presence mask channel
        # Big receptive fields up front so a single photon influences a wide neighbourhood,
        # then smaller kernels refine.
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, decoder_ch, kernel_size=7, padding=3),
            nn.BatchNorm2d(decoder_ch), nn.ReLU(inplace=True),
            nn.Conv2d(decoder_ch, decoder_ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(decoder_ch), nn.ReLU(inplace=True),
            nn.Conv2d(decoder_ch, decoder_ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(decoder_ch), nn.ReLU(inplace=True),
            nn.Conv2d(decoder_ch, num_classes, kernel_size=1),
        )

    def forward(self, feats, positions, valid):
        # feats:     [B, K, F]
        # positions: [B, K, 2] integer (px, py) in [0, patch)
        # valid:     [B, K] 1.0 if photon exists and is in-patch
        B, K, _ = feats.shape
        h = self.proj(feats)
        h, _ = self.lstm(h)              # [B, K, 2*hidden]
        D = h.size(-1)

        # mask out invalid photons before scattering
        h_m = h * valid.unsqueeze(-1)     # zeros for invalid

        # flat index per photon (invalid ones get clamped to 0 but contribute zeros)
        px = positions[..., 0].clamp(0, self.patch - 1).long()
        py = positions[..., 1].clamp(0, self.patch - 1).long()
        flat_idx = py * self.patch + px   # [B, K]

        # scatter_add into [B, D, H*W]
        spatial = torch.zeros(B, D, self.patch * self.patch,
                              device=feats.device, dtype=h_m.dtype)
        idx_d = flat_idx.unsqueeze(1).expand(-1, D, -1)        # [B, D, K]
        spatial.scatter_add_(2, idx_d, h_m.transpose(1, 2))    # [B, D, K]

        # presence count, then clamp to {0,1}
        presence = torch.zeros(B, 1, self.patch * self.patch,
                               device=feats.device, dtype=h_m.dtype)
        presence.scatter_add_(2, flat_idx.unsqueeze(1),
                              valid.unsqueeze(1).to(h_m.dtype))
        presence = (presence > 0).to(h_m.dtype)

        # normalize spatial by counts where there were multiple hits, so that
        # one-hit pixels and N-hit pixels are on the same scale.
        counts = torch.zeros_like(presence)
        counts.scatter_add_(2, flat_idx.unsqueeze(1),
                            valid.unsqueeze(1).to(h_m.dtype))
        spatial = spatial / counts.clamp(min=1.0)

        spatial = spatial.view(B, D, self.patch, self.patch)
        presence = presence.view(B, 1, self.patch, self.patch)
        feat_map = torch.cat([spatial, presence], dim=1)        # [B, D+1, H, W]
        return self.decoder(feat_map)


model = SpatialLSTMModel(n_features=n_features).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"SpatialLSTMModel parameters: {n_params:,}")

# tiny sanity-check forward pass
xf, xp, v, _ = next(iter(DataLoader(train_ds, batch_size=2, shuffle=False)))
with torch.no_grad():
    y = model(xf.to(device), xp.to(device), v.to(device))
print(f"forward output: {tuple(y.shape)}")
""")

# -------- Cell: training loop ------------------------------------------
md(r"""## 8. Training loop""")

code(r"""train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

ce_weights = torch.tensor(class_weights, device=device)
criterion = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda")


def cm_accum(cm, logits, targets):
    pred = logits.argmax(1).cpu().numpy().ravel()
    t = targets.cpu().numpy().ravel()
    keep = t != 255
    pred, t = pred[keep], t[keep]
    idx = NUM_CLASSES * t + pred
    cm += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_cm(cm):
    iou = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou.append(tp / denom if denom > 0 else 0.0)
    iou = np.array(iou)
    pix_acc = float(np.diag(cm).sum() / max(cm.sum(), 1))
    return float(iou.mean()), iou, pix_acc


best_val = -1.0
patience_left = PATIENCE
log = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.perf_counter()

    # --- train ---
    model.train()
    train_loss = 0.0; n_seen = 0
    train_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for feats, pos, val, mask in train_loader:
        feats = feats.to(device, non_blocking=True)
        pos   = pos.to(device, non_blocking=True)
        val   = val.to(device, non_blocking=True)
        mask  = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(feats, pos, val)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * feats.size(0)
        n_seen += feats.size(0)
        cm_accum(train_cm, logits.detach(), mask)
    scheduler.step()
    train_loss /= max(n_seen, 1)
    train_miou, _, train_acc = metrics_from_cm(train_cm)

    # --- val ---
    model.eval()
    val_loss = 0.0; n_seen = 0
    val_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for feats, pos, val, mask in val_loader:
            feats = feats.to(device, non_blocking=True)
            pos   = pos.to(device, non_blocking=True)
            val   = val.to(device, non_blocking=True)
            mask  = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(feats, pos, val)
                loss = criterion(logits, mask)
            val_loss += loss.item() * feats.size(0); n_seen += feats.size(0)
            cm_accum(val_cm, logits, mask)
    val_loss /= max(n_seen, 1)
    val_miou, val_iou, val_acc = metrics_from_cm(val_cm)

    log.append({"epoch": epoch,
                "train_loss": train_loss, "train_miou": train_miou, "train_acc": train_acc,
                "val_loss":   val_loss,   "val_miou":   val_miou,   "val_acc":   val_acc,
                "val_ice": val_iou[0], "val_thin": val_iou[1], "val_water": val_iou[2]})
    print(f"epoch {epoch:02d}  "
          f"train_loss {train_loss:.4f}  train_miou {train_miou:.4f}  |  "
          f"val_loss {val_loss:.4f}  val_miou {val_miou:.4f}  "
          f"per-class {val_iou.round(3).tolist()}  "
          f"({time.perf_counter() - t0:.0f}s)")

    if val_miou > best_val + 1e-4:
        best_val = val_miou
        patience_left = PATIENCE
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "val_metrics": {"miou": val_miou, "per_iou": val_iou.tolist(),
                                    "pix_acc": val_acc, "loss": val_loss}},
                   OUT_DIR / "best.pt")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"early stop at epoch {epoch} (best val mIoU {best_val:.4f})")
            break

pd.DataFrame(log).to_csv(OUT_DIR / "metrics.csv", index=False)
print(f"best val mIoU: {best_val:.4f}")
""")

# -------- Cell: test eval -----------------------------------------------
md(r"""## 9. Test set evaluation (load best checkpoint)""")

code(r"""ck = torch.load(OUT_DIR / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(ck["model_state"]); model.eval()
print(f"loaded epoch {ck['epoch']}  val mIoU {ck['val_metrics']['miou']:.4f}")

test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                         num_workers=4, pin_memory=True)
test_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
test_loss = 0.0; n_seen = 0
with torch.no_grad():
    for feats, pos, val, mask in test_loader:
        feats = feats.to(device, non_blocking=True)
        pos   = pos.to(device, non_blocking=True)
        val   = val.to(device, non_blocking=True)
        mask  = mask.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(feats, pos, val)
            loss = criterion(logits, mask)
        test_loss += loss.item() * feats.size(0); n_seen += feats.size(0)
        cm_accum(test_cm, logits, mask)
test_loss /= max(n_seen, 1)
test_miou, test_iou, test_acc = metrics_from_cm(test_cm)

print(f"TEST  mIoU {test_miou:.4f}  pix_acc {test_acc:.4f}  "
      f"per-class {test_iou.round(4).tolist()}  loss {test_loss:.4f}")

with open(OUT_DIR / "test_metrics.json", "w") as f:
    json.dump({"miou": test_miou, "per_iou": test_iou.tolist(),
               "pix_acc": test_acc, "loss": test_loss}, f, indent=2)
""")

# -------- Cell: confmat -------------------------------------------------
md(r"""## 10. Confusion matrix""")

code(r"""cm_norm = test_cm / np.maximum(test_cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(5.5, 4.4))
im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Spatial-aware Bi-LSTM (test)")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUT_DIR / "confmat.png", dpi=160)
plt.show()
""")

# -------- Cell: side by side ABOUT a few test samples -------------------
md(r"""## 11. A handful of sample predictions""")

code(r"""def int_mask_to_rgb(m):
    out = np.zeros((*m.shape, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        out[m == c] = color
    return out


sample = test_df.sample(n=6, random_state=SEED).reset_index(drop=True)
fig, axes = plt.subplots(len(sample), 3, figsize=(8, 2.5 * len(sample)))
column_titles = ["input", "ground truth", "spatial LSTM"]

with torch.no_grad():
    for i, r in sample.iterrows():
        rgb = np.array(Image.open(r["image_path"]).convert("RGB"))
        gt  = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))
        feats, pos, val, _ = test_ds[test_df.index[test_df["filename"] == r["filename"]][0]]
        logits = model(feats[None].to(device), pos[None].to(device), val[None].to(device))
        pred = logits.argmax(1)[0].cpu().numpy()

        for j, panel in enumerate([rgb, int_mask_to_rgb(gt), int_mask_to_rgb(pred)]):
            axes[i, j].imshow(panel)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(column_titles[j], fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "samples.png", dpi=160)
plt.show()
""")

md(r"""## 12. Done

Files written to `runs/lstm_spatial_v1/`:

* `best.pt`            -- best-by-val checkpoint
* `metrics.csv`        -- per-epoch curves
* `test_metrics.json`  -- final test mIoU / per-class IoU / pix_acc
* `confmat.png`        -- test confusion matrix
* `samples.png`        -- 6 random qualitative samples

If `test_miou` jumps from ~0.24 (plain LSTM) to ~0.5+ this confirms the
architectural hypothesis: the LSTM was capped not just by class weights
but by its uniform-per-patch output. Spreading photons spatially makes the
LSTM-alone score meaningful, even if the deep-fusion number is still
the headline.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
