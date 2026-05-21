"""
Generate fusion_winner.ipynb -- like v4 (hot-load sweep-winner LSTM weights)
but DON'T freeze. Fine-tune the pretrained CSV branch at 0.1x the
U-Net's LR (1e-5) so it can adapt to fusion context without overwriting
what it learned on the LSTM-only task.

Tests the "fine-tune both pretrained experts" recipe:
  - Sweep-winner LSTM weights at LR=1e-5 (slow adapt)
  - U-Net (ImageNet) + new csv_head.7 + SE + head at LR=1e-4

This is the discriminator between v4 (frozen, 0.8982 mIoU) and "we let
the LSTM evolve a bit during fusion". If v5 > v4, fine-tuning helps;
if v5 ~= v4, frozen is sufficient; if v5 < v4, fine-tuning is
overwriting the sweep winner's good features.

    Loss: CategoricalFocalLoss(alpha=[0.05,0.45,0.60], gamma=2.0)

Outputs in runs/fusion_winner/:
  - best.pt, final.pt, metrics.csv, test_metrics.json
  - confmat.png (prof-style %)
  - loss_curve.png
  - summary_vs_all.csv -- v5 vs v4 vs v3 vs v2 vs v1 vs LSTM winner
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "fusion_winner.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Deep Fusion v5 -- hot-load sweep-winner LSTM + fine-tune at 0.1x LR

v4 (frozen sweep-winner LSTM) hit test mIoU 0.8982. v5 keeps the
hot-loading but **doesn't freeze** the pretrained branch. Instead, we
use **parameter groups**:

* Loaded params (LSTM + first two Dense layers) -> **LR=1e-5**
* Fresh params (U-Net + csv_head.7 + SE + fusion head) -> **LR=1e-4**

So the pretrained CSV branch gets to adapt slowly to the fusion task
without catastrophically overwriting the discriminative features it
learned during the LSTM-only sweep. The U-Net + fusion glue learn at
full LR as usual.

If v5 > v4 -> fine-tuning helps.
If v5 ~= v4 -> the pretrained features were good enough as-is.
If v5 < v4 -> we're losing the sweep winner's edge by letting it move.

```
RGB (B,3,128,128) ---> U-Net (ResNet-18) ----> img_feat (B,16,128,128)
                                                       |
                                                       v
                                           cat -> SE -> head -> logits
                                                       ^
                                                       |
CSV (B,5,8) --> uni-LSTM(96) --> last hidden -> Dense(96->16,elu)+drop
                                              -> Dense(16->16,elu)+drop
                                              -> tile -> (B,16,128,128)
```

| component | v4 (current best, 0.8982) | v5 (this notebook) |
|---|---|---|
| CSV branch arch | uni-LSTM(96)x1 | same |
| CSV branch init | sweep winner | sweep winner |
| CSV branch trains | **FROZEN** | **LR=1e-5 (0.1x)** |
| CSV window | 5 segments | 5 segments |
| CSV features | 8 engineered | 8 engineered |
| Loss | Focal alpha=[.05,.45,.60] gamma=2.0 | same |
| Main LR | 1e-4 | 1e-4 |
| Image branch (U-Net) | ImageNet pretrain | same |
| Fusion head | SE + 3x3 conv | same |
""")

# -------- 0. Setup ------------------------------------------------------
md("## 0. Setup")

code(r"""import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
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
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm

print("torch:", torch.__version__,
      "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- 1. Config -----------------------------------------------------
md("## 1. Config")

code(r"""PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
EXP_NAME     = "fusion_winner"

IMG_DIR     = PROJECT_ROOT / "outputs"
MASK_DIR    = PROJECT_ROOT / "outputs_segmented"
CSV_DIR     = PROJECT_ROOT / "IS2_Corrected_data"
RUN_DIR     = PROJECT_ROOT / "runs" / EXP_NAME
PRIOR_UNET  = PROJECT_ROOT / "runs" / "unet_imgonly_v1"
PRIOR_FUSION_V1 = PROJECT_ROOT / "runs" / "fusion_deep_v1"
SWEEP_DIR   = PROJECT_ROOT / "runs" / "lstm_sweep"
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
NUM_CLASSES  = 3
PATCH        = 128
SEQ_LEN      = 5            # prof: 5 segments
NEARBY       = SEQ_LEN // 2
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4                   # for fresh params (U-Net + csv_head.7 + SE + head)
LR_PRETRAINED= 1e-5                   # for pretrained params (LSTM + csv_head.1/4) -- 0.1x
WEIGHT_DECAY = 1e-4
PATIENCE     = 8
NUM_WORKERS  = 4

LSTM_HIDDEN  = 96           # MUST match sweep winner so we can load its weights
LSTM_DROPOUT = 0.4
HEAD_HIDDEN  = 16
HEAD_DROPOUT = 0.4

# Sweep-winner checkpoint to hot-load
SWEEP_WINNER_CKPT = PROJECT_ROOT / "runs" / "lstm_sweep" / "hidden_96" / "best.pt"

FUSION_CH    = 16
USE_AMP      = True

# Focal loss (prof active vector + sweep winner)
ALPHA = [0.05, 0.45, 0.60]
GAMMA = 2.0

CLASS_NAMES      = ["ice", "thin_ice", "water"]
CLASS_NAMES_DISP = ["thick ice", "thin ice", "water"]

FEATURES = ["h_cor_mean", "h_diff", "rel_height_min_elev", "height_sd",
            "pcnth_mean", "pcnt_mean", "bcnt_mean", "brate_mean"]
N_FEATS  = len(FEATURES)

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print("Run dir:", RUN_DIR)
print(f"Focal loss alpha={ALPHA}  gamma={GAMMA}  LR={LR:.4g}")
""")

# -------- 2. Manifest --------------------------------------------------
md(r"""## 2. Manifest (reuse U-Net's)""")

code(r"""manifest_path_prior = PRIOR_UNET / "manifest.csv"
manifest = pd.read_csv(manifest_path_prior)
print(f"Loaded manifest: {len(manifest):,} rows")

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
assert manifest["csv_id"].notna().all()
manifest["csv_id"] = manifest["csv_id"].astype(int)

tiles_train = ["T02CNA", "T02CNC"]
tiles_test  = ["T03CWT"]
train_pool = manifest[manifest["tile"].isin(tiles_train)].reset_index(drop=True)
test_df    = manifest[manifest["tile"].isin(tiles_test)].reset_index(drop=True)

rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train_pool), size=int(0.10 * len(train_pool)), replace=False)
mask_val = np.zeros(len(train_pool), dtype=bool); mask_val[val_idx] = True
train_df = train_pool[~mask_val].reset_index(drop=True)
val_df   = train_pool[ mask_val].reset_index(drop=True)
print(f"Train: {len(train_df):,}   Val: {len(val_df):,}   Test: {len(test_df):,}")
""")

# -------- 3. Build per-CSV 8-feature segment arrays --------------------
md(r"""## 3. Build per-CSV 8-feature arrays (prof's recipe)

Each `ATL03_*_done.csv` is already a per-10m-segment table. For each
CSV we:

1. Keep the 8 prof features (`h_cor_mean`, `h_cor_med` -> `h_diff`,
   `rel_height_min_elev` computed as `h_cor_mean - min(h_cor_mean)`,
   `height_sd`, `pcnth_mean`, `pcnt_mean`, `bcnt_mean`, `brate_mean`).
2. **Don't reorder rows** -- the manifest's `row_idx` indexes the
   original CSV row order, so we keep that.

We then z-score-normalize using means/stds computed on the **train
tiles only** so test-tile information never leaks into training. Cached
to `RUN_DIR/csv_segments/csv_{cid}.npy` for fast loading next epoch.""")

code(r"""seg_dir = RUN_DIR / "csv_segments"
seg_dir.mkdir(exist_ok=True)
stats_path = RUN_DIR / "feature_stats.json"

def build_seg_array(df):
    df = df.copy()
    if "h_cor_mean" not in df or "h_cor_med" not in df:
        raise ValueError("CSV missing h_cor_mean/h_cor_med")
    df["h_diff"] = df["h_cor_mean"] - df["h_cor_med"]
    df["rel_height_min_elev"] = df["h_cor_mean"] - df["h_cor_mean"].min()
    arr = df[FEATURES].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

# Pass 1: read and build raw segment arrays
raw_segs = {}
for _, row in csv_meta.iterrows():
    cid = int(row["csv_id"])
    df = pd.read_csv(row["csv_path"])
    raw_segs[cid] = build_seg_array(df)
    print(f"  csv_{cid:02d}  {row['tile']}/{row['beam']}  rows={len(df):,}")

# Train-tile statistics
train_arrays = []
for _, row in csv_meta.iterrows():
    if row["tile"] in tiles_train:
        train_arrays.append(raw_segs[int(row["csv_id"])])
train_concat = np.concatenate(train_arrays, axis=0)
mu = np.nanmean(train_concat, axis=0).astype(np.float32)
sd = np.nanstd (train_concat, axis=0).astype(np.float32)
sd[sd < 1e-6] = 1.0
print("\ntrain means:", mu.round(3))
print("train stds: ", sd.round(3))

# Normalize and cache
csv_features = {}
for cid, arr in raw_segs.items():
    z = (arr - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    csv_features[cid] = z
    np.save(seg_dir / f"csv_{cid}.npy", z)

with open(stats_path, "w") as f:
    json.dump({"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist()}, f, indent=2)
print(f"\nNormalized arrays cached in {seg_dir}")
""")

# -------- 4. Mask decoder + class weights (for reference only) ---------
md("## 4. Mask color -> integer label")

code(r"""def mask_rgb_to_int(mask_rgb):
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    out[(mask_rgb == [255, 0, 0]).all(axis=-1)] = 0
    out[(mask_rgb == [0, 0, 255]).all(axis=-1)] = 1
    out[(mask_rgb == [0, 255, 0]).all(axis=-1)] = 2
    return out
""")

# -------- 5. Dataset ----------------------------------------------------
md(r"""## 5. Dataset (image + 5-segment window + mask)

Returns `(image, csv_window, mask)`. CSV window is shape `(SEQ_LEN, 8)`
with the manifest's `row_idx` as the *center* segment. Edge rows get
zero-padded.""")

code(r"""def random_flip_rotate(img, mask):
    if random.random() < 0.5:
        img = img[:, ::-1, :]; mask = mask[:, ::-1]
    if random.random() < 0.5:
        img = img[::-1, :, :]; mask = mask[::-1, :]
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k, axes=(0, 1)); mask = np.rot90(mask, k, axes=(0, 1))
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class SeaIceFusionV2Dataset(Dataset):
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
        n_rows = feats.shape[0]
        center = int(r["row_idx"])
        win = np.zeros((SEQ_LEN, N_FEATS), dtype=np.float32)
        for k in range(SEQ_LEN):
            src = center - NEARBY + k
            if 0 <= src < n_rows:
                win[k] = feats[src]
        return (torch.from_numpy(img),
                torch.from_numpy(win),
                torch.from_numpy(mask).long())


train_ds = SeaIceFusionV2Dataset(train_df, csv_features, augment=True)
val_ds   = SeaIceFusionV2Dataset(val_df,   csv_features, augment=False)
test_ds  = SeaIceFusionV2Dataset(test_df,  csv_features, augment=False)

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

# -------- 6. Model -----------------------------------------------------
md(r"""## 6. Deep fusion model (prof-style CSV branch)""")

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


class FusionV5Model(nn.Module):
    def __init__(self, n_features=N_FEATS, num_classes=NUM_CLASSES, patch=PATCH,
                 lstm_hidden=LSTM_HIDDEN, lstm_dropout=LSTM_DROPOUT,
                 head_hidden=HEAD_HIDDEN, head_dropout=HEAD_DROPOUT,
                 fusion_ch=FUSION_CH):
        super().__init__()
        self.patch = patch

        # Image branch -- U-Net ResNet-18 returning `fusion_ch` features
        self.unet = smp.Unet(encoder_name="resnet18",
                             encoder_weights="imagenet",
                             in_channels=3,
                             classes=fusion_ch)

        # CSV branch -- prof-style uni-LSTM(96, 1 layer)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True, bidirectional=False)
        # Prof's head, swapped final dense to produce `fusion_ch` instead of 3 logits
        self.csv_head = nn.Sequential(
            nn.Dropout(lstm_dropout),
            nn.Linear(lstm_hidden, head_hidden), nn.ELU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden), nn.ELU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, fusion_ch),
        )

        # Fusion head: SE -> 3x3 -> BN -> ReLU -> drop -> 1x1 logits
        self.se = SqueezeExcitation(channels=fusion_ch * 2, reduction=8)
        self.head = nn.Sequential(
            nn.Conv2d(fusion_ch * 2, fusion_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fusion_ch, num_classes, kernel_size=1),
        )

    def forward(self, img, csv_window):
        img_feat = self.unet(img)                          # (B, fusion_ch, 128, 128)

        h, _ = self.lstm(csv_window)                        # (B, SEQ_LEN, lstm_hidden)
        csv_vec = self.csv_head(h[:, -1, :])               # (B, fusion_ch)
        csv_feat = csv_vec[:, :, None, None].expand(-1, -1, self.patch, self.patch)

        fused = torch.cat([img_feat, csv_feat], dim=1)
        fused = self.se(fused)
        return self.head(fused)


model = FusionV5Model().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"FusionV5Model parameters: {n_params:,}")
""")

# -------- 6b. Hot-load sweep winner weights + tag for low-LR ----------
md(r"""## 6b. Hot-load sweep-winner LSTM weights (no freeze)

We load `runs/lstm_sweep/hidden_96/best.pt` and copy its weights into:

| sweep state_dict key | fusion model dest |
|---|---|
| `lstm.*` (4 tensors) | `lstm.*` |
| `head.1.weight/bias` (Linear 96->16) | `csv_head.1.weight/bias` |
| `head.4.weight/bias` (Linear 16->16) | `csv_head.4.weight/bias` |
| `head.7.weight/bias` (Linear 16->3 logits) | *dropped* |

Unlike v4, we do **NOT** freeze. Instead, we'll later put these loaded
params in a separate optimizer parameter group with LR=1e-5 (0.1x), so
they can adapt slowly while the U-Net + fusion glue learn at full
LR=1e-4.""")

code(r"""print(f"Loading sweep-winner checkpoint from {SWEEP_WINNER_CKPT}")
ck = torch.load(SWEEP_WINNER_CKPT, map_location="cpu", weights_only=False)
sweep_sd = ck["model_state"]

# Map sweep state_dict -> fusion CSV-branch state_dict
load_sd = {}
for k, v in sweep_sd.items():
    if k.startswith("lstm."):
        load_sd[k] = v
    elif k in ("head.1.weight", "head.1.bias"):
        load_sd["csv_head." + k.split("head.", 1)[1]] = v
    elif k in ("head.4.weight", "head.4.bias"):
        load_sd["csv_head." + k.split("head.", 1)[1]] = v
    # head.7.* is the 16->3 classifier; we drop it because our
    # fusion CSV branch ends in 16->fusion_ch (different semantics)

missing, unexpected = model.load_state_dict(load_sd, strict=False)
print(f"loaded {len(load_sd)} tensors from sweep winner")
print(f"  missing keys (expected: U-Net + csv_head.7 + SE + head): {len(missing)}")
print(f"  unexpected keys (expected: 0): {len(unexpected)}")
if unexpected:
    print("  unexpected:", unexpected)

# Sanity-check at least one shape transfer worked
import torch.nn.functional as _F
ref_w = sweep_sd["lstm.weight_ih_l0"]
cur_w = dict(model.named_parameters())["lstm.weight_ih_l0"].detach().cpu()
print(f"lstm.weight_ih_l0 transfer check: max abs diff = "
      f"{(ref_w - cur_w).abs().max().item():.3e}  (should be ~0)")

# Tag pretrained params for the low-LR group; everything else trains at LR.
PRETRAINED_PREFIXES = ("lstm.", "csv_head.1.", "csv_head.4.")
pretrained_params = []
fresh_params = []
for name, p in model.named_parameters():
    if name.startswith(PRETRAINED_PREFIXES):
        pretrained_params.append(p)
    else:
        fresh_params.append(p)

n_pre  = sum(p.numel() for p in pretrained_params)
n_fresh = sum(p.numel() for p in fresh_params)
print(f"pretrained (LR=1e-5):  {n_pre:>12,}  ({len(pretrained_params)} tensors)")
print(f"fresh      (LR=1e-4):  {n_fresh:>12,}  ({len(fresh_params)} tensors)")
""")

# -------- 7. Focal loss + optimizer ------------------------------------
md(r"""## 7. Focal Loss + Adam + cosine LR""")

code(r"""class CategoricalFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, ignore_index=255):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = float(gamma)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        valid = target != self.ignore_index
        target_safe = target.clamp_min(0)
        log_probs = F.log_softmax(logits, dim=1)
        log_p_t = log_probs.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp().clamp(min=1e-8, max=1.0 - 1e-8)
        alpha_t = self.alpha.to(logits.device)[target_safe]
        focal_term = (1.0 - p_t).pow(self.gamma)
        per_pixel = -alpha_t * focal_term * log_p_t
        per_pixel = per_pixel * valid.float()
        denom = valid.float().sum().clamp_min(1.0)
        return per_pixel.sum() / denom


criterion = CategoricalFocalLoss(alpha=ALPHA, gamma=GAMMA).to(device)
optimizer = torch.optim.Adam([
    {"params": pretrained_params, "lr": LR_PRETRAINED},
    {"params": fresh_params,      "lr": LR},
], weight_decay=WEIGHT_DECAY)
print(f"optimizer param groups: pretrained @ {LR_PRETRAINED:.0e}, fresh @ {LR:.0e}")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)
print("loss + optimizer + scheduler + AMP ready")
""")

# -------- 8. Training --------------------------------------------------
md(r"""## 8. Training loop""")

code(r"""def cm_accum(cm, logits, targets):
    pred = logits.argmax(1).detach().cpu().numpy().ravel()
    t = targets.detach().cpu().numpy().ravel()
    keep = t != 255
    pred, t = pred[keep], t[keep]
    idx = NUM_CLASSES * t + pred
    cm += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_cm(cm):
    iou, prec, rec, f1 = [], [], [], []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou.append(tp / denom if denom > 0 else 0.0)
        prec.append(tp / (tp + fp) if (tp + fp) else 0.0)
        rec.append(tp / (tp + fn) if (tp + fn) else 0.0)
        f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1])
                  if (prec[-1] + rec[-1]) else 0.0)
    iou = np.array(iou); prec = np.array(prec); rec = np.array(rec); f1 = np.array(f1)
    return {"miou": float(iou.mean()), "per_iou": iou.tolist(),
            "pix_acc": float(np.diag(cm).sum() / max(cm.sum(), 1)),
            "precision": prec.tolist(), "recall": rec.tolist(),
            "f1": f1.tolist(), "macro_f1": float(f1.mean())}


best_val = -1.0
patience_left = PATIENCE
log = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.perf_counter()
    model.train()
    tr_loss = 0.0; n = 0
    tr_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img, win, mask in train_loader:
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(img, win)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * img.size(0); n += img.size(0)
        cm_accum(tr_cm, logits.detach(), mask)
    scheduler.step()
    tr_loss /= max(n, 1)
    tr_m = metrics_from_cm(tr_cm)

    model.eval()
    va_loss = 0.0; n = 0
    va_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for img, win, mask in val_loader:
            img = img.to(device, non_blocking=True)
            win = win.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(img, win)
                loss = criterion(logits, mask)
            va_loss += loss.item() * img.size(0); n += img.size(0)
            cm_accum(va_cm, logits, mask)
    va_loss /= max(n, 1)
    va_m = metrics_from_cm(va_cm)

    log.append({"epoch": epoch,
                "train_loss": tr_loss, "train_miou": tr_m["miou"],
                "val_loss":   va_loss, "val_miou":   va_m["miou"],
                "val_iou_ice": va_m["per_iou"][0],
                "val_iou_thin": va_m["per_iou"][1],
                "val_iou_water": va_m["per_iou"][2],
                "val_pix_acc": va_m["pix_acc"],
                "lr": optimizer.param_groups[0]["lr"]})
    print(f"epoch {epoch:02d}  tr_loss {tr_loss:.4f}  tr_miou {tr_m['miou']:.4f}  |  "
          f"va_loss {va_loss:.4f}  va_miou {va_m['miou']:.4f}  "
          f"per-class {[round(x,3) for x in va_m['per_iou']]}  "
          f"({time.perf_counter() - t0:.0f}s)")

    if va_m["miou"] > best_val + 1e-4:
        best_val = va_m["miou"]
        patience_left = PATIENCE
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "alpha": ALPHA, "gamma": GAMMA,
                    "val_metrics": va_m},
                   RUN_DIR / "best.pt")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"early stop at epoch {epoch} (best val mIoU {best_val:.4f})")
            break

torch.save({"epoch": epoch, "model_state": model.state_dict(),
            "alpha": ALPHA, "gamma": GAMMA},
           RUN_DIR / "final.pt")
pd.DataFrame(log).to_csv(RUN_DIR / "metrics.csv", index=False)
print(f"\nbest val mIoU: {best_val:.4f}  (best.pt)")
""")

# -------- 9. Test eval -------------------------------------------------
md(r"""## 9. Test evaluation (best.pt)""")

code(r"""ck = torch.load(RUN_DIR / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(ck["model_state"]); model.eval()
print(f"loaded epoch {ck['epoch']}  val mIoU {ck['val_metrics']['miou']:.4f}")

test_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
with torch.no_grad():
    for img, win, mask in tqdm(test_loader, desc="test"):
        img = img.to(device, non_blocking=True)
        win = win.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(img, win)
        cm_accum(test_cm, logits, mask)
test_m = metrics_from_cm(test_cm)
test_m["cm"] = test_cm.tolist()

print(f"\nTEST  mIoU {test_m['miou']:.4f}  pix_acc {test_m['pix_acc']:.4f}")
print(f"per-class IoU       {[round(x, 4) for x in test_m['per_iou']]}")
print(f"per-class precision {[round(x, 4) for x in test_m['precision']]}")
print(f"per-class recall    {[round(x, 4) for x in test_m['recall']]}")
print(f"per-class F1        {[round(x, 4) for x in test_m['f1']]}")
print(f"macro F1            {test_m['macro_f1']:.4f}")

with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump({**test_m, "alpha": ALPHA, "gamma": GAMMA,
               "lstm_hidden": LSTM_HIDDEN, "seq_len": SEQ_LEN,
               "epoch_best": ck["epoch"]}, f, indent=2)
""")

# -------- 10. Confusion matrix -----------------------------------------
md(r"""## 10. Confusion matrix (prof-style %)""")

code(r"""cm = np.array(test_m["cm"], dtype=np.float64)
rs = cm.sum(axis=1, keepdims=True)
pct = np.where(rs > 0, cm / np.maximum(rs, 1) * 100.0, 0.0)
fig, ax = plt.subplots(figsize=(7.2, 5.8))
im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels([f"Predicted {c}" for c in CLASS_NAMES_DISP], fontsize=11)
ax.set_yticklabels(CLASS_NAMES_DISP, fontsize=11)
ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix (Percentages)", fontsize=13)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        v = pct[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v > 55 else "black", fontsize=13)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(RUN_DIR / "confmat.png", dpi=180, bbox_inches="tight")
plt.show()
""")

# -------- 11. Loss curve ----------------------------------------------
md(r"""## 11. Loss + val mIoU curves""")

code(r"""hist = pd.read_csv(RUN_DIR / "metrics.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
axes[0].plot(hist["epoch"], hist["val_loss"],   label="val")
axes[0].set_title("Focal loss"); axes[0].set_xlabel("epoch")
axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(hist["epoch"], hist["train_miou"], label="train")
axes[1].plot(hist["epoch"], hist["val_miou"],   label="val")
axes[1].set_title("mIoU"); axes[1].set_xlabel("epoch")
axes[1].legend(); axes[1].grid(alpha=.3)
axes[2].plot(hist["epoch"], hist["val_iou_ice"],   label="ice")
axes[2].plot(hist["epoch"], hist["val_iou_thin"],  label="thin")
axes[2].plot(hist["epoch"], hist["val_iou_water"], label="water")
axes[2].set_title("Per-class val IoU"); axes[2].set_xlabel("epoch")
axes[2].legend(); axes[2].grid(alpha=.3)
plt.tight_layout()
plt.savefig(RUN_DIR / "loss_curve.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- 12. Compare vs v1 + LSTM-only winner -------------------------
md(r"""## 12. Side-by-side: fusion v2 vs fusion v1 vs LSTM-only winner""")

code(r"""rows = []
def add_row(name, tm_path):
    if not Path(tm_path).exists():
        rows.append({"model": name, "test_miou": float("nan")})
        return
    tm = json.loads(Path(tm_path).read_text())
    rows.append({"model": name,
                 "test_miou":  tm.get("miou", float("nan")),
                 "test_acc":   tm.get("pix_acc", float("nan")),
                 "iou_ice":    tm.get("per_iou", [None]*3)[0],
                 "iou_thin":   tm.get("per_iou", [None]*3)[1],
                 "iou_water":  tm.get("per_iou", [None]*3)[2],
                 "macro_f1":   tm.get("macro_f1", float("nan"))})

add_row("fusion_winner (fine-tune sweep-winner LSTM @ 0.1x LR)", RUN_DIR / "test_metrics.json")
add_row("fusion_v4 (frozen sweep-winner LSTM + U-Net)",
        PROJECT_ROOT / "runs" / "fusion_v4" / "test_metrics.json")
add_row("fusion_v3 (Bi-LSTM(128)x2 + prof data + focal)",
        PROJECT_ROOT / "runs" / "fusion_v3" / "test_metrics.json")
add_row("fusion_v2 (uni-LSTM(96) random init + prof data)",
        PROJECT_ROOT / "runs" / "fusion_v2" / "test_metrics.json")
add_row("fusion_deep_v1 (Bi-LSTM + weighted CE)", PRIOR_FUSION_V1 / "test_metrics.json")
add_row("lstm_sweep winner (hidden_96, CSV-only)",
        SWEEP_DIR / "hidden_96" / "test_metrics.json")

cmp = pd.DataFrame(rows)
cmp.to_csv(RUN_DIR / "summary_vs_all.csv", index=False)
print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
""")

# -------- 13. Done ----------------------------------------------------
md(r"""## 13. Done

Files in `runs/fusion_v2/`:

* `best.pt`, `final.pt`       -- checkpoints (best-by-val and last-epoch)
* `metrics.csv`               -- per-epoch train/val curves
* `test_metrics.json`         -- test mIoU, per-class P/R/F1/IoU, macro-F1
* `confmat.png`               -- prof-style percentage CM
* `loss_curve.png`            -- 3-panel diagnostics
* `summary_vs_v1.csv`         -- side-by-side table with fusion v1 + sweep winner
* `csv_segments/csv_*.npy`    -- cached normalized 8-feature arrays
* `feature_stats.json`        -- mean/std used (train-only)

If `iou_thin` and `iou_water` here are both higher than in
`fusion_deep_v1`, the prof-style LSTM swap is a win and we should
promote `fusion_v2` to the new baseline.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
