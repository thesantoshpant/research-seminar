"""
Generate lstm_focal.ipynb -- retrain the plain (non-spatial) Bi-LSTM with
Focal Loss, per the professor's request.

She pointed at her own Keras notebook
(`3_ATL03_prepare_data_LSTM_training_2025_cor_label.ipynb`) which uses
`tf.keras.losses.CategoricalFocalCrossentropy(alpha=..., gamma=2.0)`.

After reading her notebook end-to-end (her final `model.compile` is at
cell line 12587 / 12593, and `model.fit` at 12606), the *actually run*
recipe is:

    alpha = [0.05, 0.45, 0.60]     # the active vector in her final cell
    gamma = 2.0
    optimizer = Adam(lr=0.0008886176350890356)   # Keras-Tuner result
    epochs = 50, batch_size = 32, EarlyStopping defined but not passed
    dropout = 0.4 (twice after the LSTM)
    dense head after LSTM:  Dense(16, elu) -> Dropout(0.4)
                            Dense(16, elu) -> Dropout(0.4)
                            Dense(3,  softmax)

This generator mirrors all of those hyperparameters in our PyTorch
per-pixel pipeline (input is still our 32-row window of normalized CSV
features, output is still 128x128 -- so deep-fusion can still consume
this checkpoint). Only the loss + head + LR + dropout + epoch count are
swapped to match hers.

Alpha vectors she experimented with (indexed [ice, thin_ice, water]):

    [0.02, 0.44, 0.54]
    [0.05, 0.45, 0.50]
    [0.05, 0.50, 0.45]
    [0.041, 0.409, 0.550]
    [0.05, 0.45, 0.60]    <-- her *active* vector in the final cell

Outputs (in runs/lstm_focal_v1/):
  - best.pt              : best-by-val-mIoU checkpoint
  - metrics.csv          : per-epoch curves
  - test_metrics.json    : final test metrics (same schema as the other runs)
  - confmat.png          : test confusion matrix
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "lstm_focal.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Bi-LSTM with Focal Loss (aligned to prof's recipe)

The plain Bi-LSTM run collapsed to all-ice on test (mIoU 0.2420).
Per the professor's note, we swap the weighted cross-entropy for **Focal
Loss** with the alpha / gamma she ran in her own Keras notebook
(`3_ATL03_prepare_data_LSTM_training_2025_cor_label.ipynb`).

After reading that notebook, her *actually run* settings are:

| setting | her value | now used here |
|---|---|---|
| loss | `CategoricalFocalCrossentropy(alpha=[0.05,0.45,0.60], gamma=2.0)` | same |
| optimizer | `Adam(lr=8.886e-4)` (Keras-Tuner pick) | same |
| dropout | 0.4 (twice after LSTM) | same |
| head | Dense(16,elu) -> Dense(16,elu) -> Dense(3,softmax) | same (as 1x1 Conv2d) |
| epochs | 50 (EarlyStopping defined but not passed to fit) | same |
| LSTM | uni-directional, 48 hidden, 1 layer | we keep our Bi-LSTM(128, 2 layers); see note below |

**What stays our own:** the input pipeline (32-row window of normalized
CSV features) and per-pixel 128x128 output (tiled head). This is
*deliberate* -- the downstream deep-fusion model needs a per-pixel CSV
branch to fuse against the U-Net feature map. Her notebook is
per-segment-classification only, so we can't slot her exact model into
fusion as-is. The faithful structural mirror lives in a separate
notebook (`lstm_prof_style.ipynb`).

Alpha vectors she experimented with -- swap `ALPHA` to ablate:

    [0.02, 0.44, 0.54]
    [0.05, 0.45, 0.50]
    [0.05, 0.50, 0.45]
    [0.041, 0.409, 0.550]
    [0.05, 0.45, 0.60]    <-- her active vector in the final cell""")

# -------- Cell: GPU pin --------------------------------------------------
md("## 0. Setup")

code(r"""import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
""")

code(r"""import json, math, random, time, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- Cell: config ---------------------------------------------------
md("## 1. Config")

code(r"""PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
EXP_NAME     = "lstm_focal_v1"

IMG_DIR     = PROJECT_ROOT / "outputs"
MASK_DIR    = PROJECT_ROOT / "outputs_segmented"
CSV_DIR     = PROJECT_ROOT / "IS2_Corrected_data"
RUN_DIR     = PROJECT_ROOT / "runs" / EXP_NAME
PRIOR_UNET  = PROJECT_ROOT / "runs" / "unet_imgonly_v1"     # manifest source
PRIOR_LSTM  = PROJECT_ROOT / "runs" / "lstm_csvonly_v1"     # cached csv_normed/
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
NUM_CLASSES  = 3
PATCH        = 128
WINDOW_K     = 32
BATCH_SIZE   = 256
EPOCHS       = 50                  # prof: 50, no early stop
LR           = 8.886176350890356e-4 # prof's Keras-Tuner pick
WEIGHT_DECAY = 1e-4
PATIENCE     = EPOCHS              # effectively disables early stopping
NUM_WORKERS  = 4
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.4                 # prof: 0.4 (was 0.2)
HEAD_DROPOUT = 0.4                 # prof: 0.4 after each Dense(16, elu)
HEAD_HIDDEN  = 16                  # prof: Dense(16, elu) x2

# ------- focal loss settings (per prof's notebook, final cell) ----
# alpha indexes are [ice, thin_ice, water]. Swap to one of the variants below.
ALPHA   = [0.05, 0.45, 0.60]   # <-- her *active* vector in final compile cell
# ALPHA = [0.02, 0.44, 0.54]
# ALPHA = [0.05, 0.45, 0.50]
# ALPHA = [0.05, 0.50, 0.45]
# ALPHA = [0.041, 0.409, 0.550]
GAMMA   = 2.0
# ---------------------------------------------------------

CLASS_NAMES = ["ice", "thin_ice", "water"]
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}

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
print("Run dir:", RUN_DIR)
print(f"Focal loss: alpha={ALPHA}, gamma={GAMMA}")
""")

# -------- Cell: manifest + splits ---------------------------------------
md(r"""## 2. Manifest, CSV mapping, splits""")

code(r"""manifest = pd.read_csv(PRIOR_UNET / "manifest.csv")
print(f"manifest: {len(manifest):,} rows")

csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
csv_meta = []
for p in csv_files:
    parts = p.stem.split("_")
    csv_meta.append({"csv_path": str(p), "csv_name": p.name,
                     "tile": parts[3], "beam": parts[4]})
csv_meta = pd.DataFrame(csv_meta)
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
""")

# -------- Cell: load cached normalized CSV features --------------------
md(r"""## 3. Reuse the normalized CSV features cached by the plain LSTM run""")

code(r"""csv_features = {}
for cid in csv_meta["csv_id"]:
    arr = np.load(PRIOR_LSTM / "csv_normed" / f"csv_{cid}.npy")
    csv_features[int(cid)] = arr
n_features = next(iter(csv_features.values())).shape[1]
print(f"csv_features: {len(csv_features)} CSVs, {n_features} features")
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

# -------- Cell: dataset --------------------------------------------------
md(r"""## 5. Dataset (same as the plain LSTM)""")

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
        for k in range(WINDOW_K):
            src = center - HALF + k
            if 0 <= src < n_rows:
                win[k]   = feats[src]
                valid[k] = 1.0
        mask = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))
        return (torch.from_numpy(win),
                torch.from_numpy(valid),
                torch.from_numpy(mask).long())


train_ds = SeaIceCSVDataset(train_df, csv_features, MASK_DIR)
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

# -------- Cell: model (identical to plain LSTM) -------------------------
md(r"""## 6. Model (same `CSVOnlyModel` as the plain LSTM run)""")

code(r"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSVOnlyModel(nn.Module):
    def __init__(self, n_features, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, num_classes=NUM_CLASSES, patch=PATCH,
                 head_hidden=HEAD_HIDDEN, head_dropout=HEAD_DROPOUT):
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
        # Mirror prof's head: Dropout(0.4) -> Dense(16,elu) -> Dropout(0.4)
        #                    -> Dense(16,elu) -> Dropout(0.4) -> Dense(3,softmax)
        # (softmax is implicit since the focal-loss op takes logits and
        #  applies log_softmax internally)
        self.head = nn.Sequential(
            nn.Dropout2d(head_dropout),
            nn.Conv2d(hidden * 2, head_hidden, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(head_hidden, head_hidden, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(head_hidden, num_classes, kernel_size=1),
        )

    def forward(self, x, valid=None):
        h = self.proj(x)
        h, _ = self.lstm(h)
        center = x.size(1) // 2
        feat = h[:, center, :][:, :, None, None].expand(-1, -1, self.patch, self.patch)
        return self.head(feat)


model = CSVOnlyModel(n_features=n_features).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"CSVOnlyModel parameters: {n_params:,}")
""")

# -------- Cell: focal loss --------------------------------------------
md(r"""## 7. Focal Loss (PyTorch port of Keras's `CategoricalFocalCrossentropy`)

Formula (matches Keras when `alpha` is given as a per-class vector):

    FL(p_t) = -alpha[true_class] * (1 - p_t)^gamma * log(p_t)

where `p_t` is the predicted probability for the true class. We average
over all non-ignored pixels (equivalent to Keras's
`reduction='sum_over_batch_size'` when each pixel is a sample).""")

code(r"""class CategoricalFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, ignore_index=255):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = float(gamma)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: (B, C, H, W)   target: (B, H, W) int64 in {0..C-1, ignore}
        valid = target != self.ignore_index
        # clamp ignore index to 0 so gather works; we mask those out later
        target_safe = target.clamp_min(0)

        log_probs = F.log_softmax(logits, dim=1)
        log_p_t = log_probs.gather(1, target_safe.unsqueeze(1)).squeeze(1)  # (B, H, W)
        p_t = log_p_t.exp().clamp(min=1e-8, max=1.0 - 1e-8)

        alpha_t = self.alpha.to(logits.device)[target_safe]                  # (B, H, W)
        focal_term = (1.0 - p_t).pow(self.gamma)
        per_pixel = -alpha_t * focal_term * log_p_t                          # (B, H, W)

        per_pixel = per_pixel * valid.float()
        denom = valid.float().sum().clamp_min(1.0)
        return per_pixel.sum() / denom


criterion = CategoricalFocalLoss(alpha=ALPHA, gamma=GAMMA).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda")
print("loss + optimizer ready")
""")

# -------- Cell: training loop ------------------------------------------
md(r"""## 8. Training loop""")

code(r"""def cm_accum(cm, logits, targets):
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
    tr_loss = 0.0; n_seen = 0
    tr_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for win, valid, mask in train_loader:
        win = win.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(win)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * win.size(0); n_seen += win.size(0)
        cm_accum(tr_cm, logits.detach(), mask)
    scheduler.step()
    tr_loss /= max(n_seen, 1)
    tr_miou, _, tr_acc = metrics_from_cm(tr_cm)

    # --- val ---
    model.eval()
    va_loss = 0.0; n_seen = 0
    va_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for win, valid, mask in val_loader:
            win = win.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(win)
                loss = criterion(logits, mask)
            va_loss += loss.item() * win.size(0); n_seen += win.size(0)
            cm_accum(va_cm, logits, mask)
    va_loss /= max(n_seen, 1)
    va_miou, va_iou, va_acc = metrics_from_cm(va_cm)

    log.append({"epoch": epoch,
                "train_loss": tr_loss, "train_miou": tr_miou, "train_acc": tr_acc,
                "val_loss":   va_loss, "val_miou":   va_miou, "val_acc":   va_acc,
                "val_ice": va_iou[0], "val_thin": va_iou[1], "val_water": va_iou[2]})
    print(f"epoch {epoch:02d}  "
          f"train_loss {tr_loss:.4f}  train_miou {tr_miou:.4f}  |  "
          f"val_loss {va_loss:.4f}  val_miou {va_miou:.4f}  "
          f"per-class {va_iou.round(3).tolist()}  "
          f"({time.perf_counter() - t0:.0f}s)")

    if va_miou > best_val + 1e-4:
        best_val = va_miou
        patience_left = PATIENCE
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "alpha": ALPHA, "gamma": GAMMA,
                    "val_metrics": {"miou": va_miou, "per_iou": va_iou.tolist(),
                                    "pix_acc": va_acc, "loss": va_loss}},
                   RUN_DIR / "best.pt")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"early stop at epoch {epoch} (best val mIoU {best_val:.4f})")
            break

pd.DataFrame(log).to_csv(RUN_DIR / "metrics.csv", index=False)
print(f"best val mIoU: {best_val:.4f}")
""")

# -------- Cell: test eval -----------------------------------------------
md(r"""## 9. Test evaluation (load best checkpoint)""")

code(r"""ck = torch.load(RUN_DIR / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(ck["model_state"]); model.eval()
print(f"loaded epoch {ck['epoch']}  val mIoU {ck['val_metrics']['miou']:.4f}")

test_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
test_loss = 0.0; n_seen = 0
with torch.no_grad():
    for win, valid, mask in test_loader:
        win = win.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(win)
            loss = criterion(logits, mask)
        test_loss += loss.item() * win.size(0); n_seen += win.size(0)
        cm_accum(test_cm, logits, mask)
test_loss /= max(n_seen, 1)
test_miou, test_iou, test_acc = metrics_from_cm(test_cm)

print(f"TEST  mIoU {test_miou:.4f}  pix_acc {test_acc:.4f}  "
      f"per-class {test_iou.round(4).tolist()}  loss {test_loss:.4f}")

with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump({"miou": test_miou, "per_iou": test_iou.tolist(),
               "pix_acc": test_acc, "loss": test_loss,
               "alpha": ALPHA, "gamma": GAMMA}, f, indent=2)
""")

# -------- Cell: confmat -------------------------------------------------
md(r"""## 10. Confusion matrix""")

code(r"""cm_norm = test_cm / np.maximum(test_cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(5.5, 4.4))
im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Bi-LSTM + Focal Loss (alpha={ALPHA}, gamma={GAMMA})")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(RUN_DIR / "confmat.png", dpi=160)
plt.show()
""")

md(r"""## 11. Done

Files in `runs/lstm_focal_v1/`:

* `best.pt`            -- best-by-val checkpoint
* `metrics.csv`        -- training curves
* `test_metrics.json`  -- final test mIoU / per-class IoU / pix_acc
* `confmat.png`        -- test-set confusion matrix

**What was changed in this iteration (vs the original plain-LSTM run)**:

* Loss: CE -> CategoricalFocalLoss(alpha=[0.05,0.45,0.60], gamma=2.0)
* LR:   1e-3 -> 8.886e-4 (her Keras-Tuner pick)
* Dropout in LSTM: 0.2 -> 0.4
* Head: 1x1 Conv(64)+BN+ReLU+Conv(3) -> 1x1 Conv(16,elu) x 2 + Dropout(0.4)
  + Conv(3) (mirrors her Dense(16,elu)*2 + Dense(3,softmax))
* Epochs: 30 + early-stop -> 50, early-stop effectively disabled

If the LSTM-alone mIoU jumps from ~0.24 to something meaningfully higher
(and the confusion matrix shows non-trivial thin-ice / water diagonals),
the focal-loss switch worked as the prof predicted. Next step would be to
retrain the deep-fusion model with the same loss to see if fusion picks
up additional mIoU on thin-ice.

For a structurally faithful mirror of her *exact* architecture
(uni-LSTM(48), 5-segment window over 8 engineered features, per-segment
output), see `lstm_prof_style.ipynb`.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
