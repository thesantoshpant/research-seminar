"""
Generate lstm_prof_style.ipynb -- a structurally faithful mirror of the
professor's Keras LSTM notebook
(`3_ATL03_prepare_data_LSTM_training_2025_cor_label.ipynb`).

Differences from our existing `lstm_focal.ipynb`:

  * **Per-segment classification**, not per-pixel 128x128 segmentation.
  * **Uni-directional LSTM(48)**, single layer (her arch).
  * **5-segment sliding window** along x_atc with the label of the
    center segment. Features per step come from 8 aggregated 10 m
    columns (h_cor_mean, h_diff, rel_height_min_elev, height_sd,
    pcnth_mean, pcnt_mean, bcnt_mean, brate_mean).
  * **Head** = Dense(16, elu) -> Dropout(0.4) -> Dense(16, elu) ->
    Dropout(0.4) -> Dense(3, softmax) (here as Linear/ELU stack).
  * **Loss**  = CategoricalFocalLoss(alpha=[0.05, 0.45, 0.60], gamma=2.0)
  * **Optimizer** = Adam(lr=8.886e-4), 50 epochs, batch=32, no early stop.
  * **Split**  = tile-grouped (T02CNA+T02CNC train, T03CWT test) so the
                 numbers aren't inflated by the random-leakage split she
                 used. Friends can flip `GROUPED_SPLIT=False` to match
                 hers exactly.

Outputs in `runs/lstm_prof_style_v1/`:
  - best.pt              : best-by-val-mIoU checkpoint
  - metrics.csv          : per-epoch curves
  - test_metrics.json    : final test metrics
  - confmat.png          : test confusion matrix (counts + percent)
  - loss_curve.png       : train/val loss + val mIoU
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "lstm_prof_style.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Prof-style Bi-LSTM (faithful mirror)

This notebook replicates the architecture and hyperparameters of the
professor's Keras notebook
(`notebook_output/3_ATL03_prepare_data_LSTM_training_2025_cor_label (1).ipynb`).

**Why a separate notebook?** Our other LSTM notebooks
(`lstm_baseline.ipynb`, `lstm_focal.ipynb`) produce a per-pixel
128x128 segmentation so they can plug into the deep-fusion model. The
prof's LSTM is a *per-segment classifier* over 5 consecutive 10 m
along-track segments -- it predicts the class of the center segment
only. This notebook follows that design exactly so the numbers can be
compared apples-to-apples with hers, and so we can confirm the
focal-loss collapse fix she predicted.

**Architecture (per her final `model.compile` cell):**

```
LSTM(units=48, activation='tanh')          # uni-directional, single layer
Dropout(0.4)
Dense(16, 'elu')  -> Dropout(0.4)
Dense(16, 'elu')  -> Dropout(0.4)
Dense(3,  'softmax')
```

**Loss / optimizer:**
`CategoricalFocalCrossentropy(alpha=[0.05, 0.45, 0.60], gamma=2.0)`,
`Adam(lr=0.0008886176350890356)`, 50 epochs, batch=32, no early stop.

**Features fed at each step** (8 columns, in this order):
`h_cor_mean, h_diff, rel_height_min_elev, height_sd, pcnth_mean,
 pcnt_mean, bcnt_mean, brate_mean`.

`h_diff` is engineered: `h_cor_mean - h_cor_med`.
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
import matplotlib.pyplot as plt

print("torch:", torch.__version__,
      "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- 1. Config -----------------------------------------------------
md("## 1. Config")

code(r"""PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
EXP_NAME     = "lstm_prof_style_v1"

CSV_DIR     = PROJECT_ROOT / "IS2_Corrected_data"
RUN_DIR     = PROJECT_ROOT / "runs" / EXP_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
NUM_CLASSES  = 3
SEQ_LEN      = 5                  # prof: 5-segment sliding window
NEARBY       = SEQ_LEN // 2       # 2 segments before + 2 after the center
BATCH_SIZE   = 32                 # prof: 32
EPOCHS       = 50                 # prof: 50 (early stop defined but not passed)
LR           = 8.886176350890356e-4
WEIGHT_DECAY = 0.0                # prof: no L2 reg
PATIENCE     = EPOCHS             # effectively no early stop
NUM_WORKERS  = 2

LSTM_HIDDEN  = 48                 # prof: 48
LSTM_DROPOUT = 0.4                # prof: 0.4 right after LSTM
HEAD_HIDDEN  = 16                 # prof: Dense(16,elu)
HEAD_DROPOUT = 0.4                # prof: 0.4 between dense layers

# ------- focal loss (prof's final compile cell) ----------
ALPHA = [0.05, 0.45, 0.60]
GAMMA = 2.0
# ---------------------------------------------------------

# Train/test split. Defaults to her sklearn `train_test_split` recipe
# (60/20/20 random shuffle, random_state=20) so the result is directly
# comparable to her diagonal CM. Flip to True for our tile-grouped split
# (T02CN* train / T03CWT test) when you want the honest no-leakage run.
GROUPED_SPLIT = False

CLASS_NAMES      = ["ice", "thin_ice", "water"]
CLASS_NAMES_DISP = ["thick ice", "thin ice", "water"]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print("Run dir:", RUN_DIR)
print(f"Focal loss: alpha={ALPHA}, gamma={GAMMA}")
print(f"Grouped split (T02CN* train / T03CWT test): {GROUPED_SPLIT}")
""")

# -------- 2. Load the 10 m segment CSVs --------------------------------
md(r"""## 2. Load the 10 m segment CSVs

The `ATL03_*_done.csv` files are **already aggregated to 10 m segments** --
each row is one segment, not one photon. We just need to:

1. Read them in
2. Drop rows with bad / missing labels
3. Sort each track by `x_atc` (so the sliding window is in along-track order)
4. Engineer the two derived features `h_diff` and `rel_height_min_elev`

The 8 features per segment we feed to the LSTM (in this order):

| feature | source | meaning |
|---|---|---|
| h_cor_mean | column | mean corrected photon height (m) |
| h_diff | derived = h_cor_mean - h_cor_med | asymmetry of the segment's photons |
| rel_height_min_elev | derived = h_cor_mean - min(h_cor_mean) over track | relative elevation along track |
| height_sd | column | std of photon heights in segment |
| pcnth_mean | column | mean photon-count height per segment |
| pcnt_mean | column | mean photon count per segment |
| bcnt_mean | column | mean background photon count |
| brate_mean | column | mean background rate |

This is the exact feature set she uses in her notebook (the one piece we
have to derive ourselves is `rel_height_min_elev`, since the raw CSV
doesn't carry a `h_cor_min` column).""")

code(r"""def load_segment_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    needed = ["h_cor_mean", "h_cor_med", "x_atc", "label",
              "height_sd", "pcnth_mean", "pcnt_mean", "bcnt_mean", "brate_mean"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")

    # keep only labelled segments (0/1/2)
    df = df[df["label"].isin([0, 1, 2])].copy()
    df = df.dropna(subset=needed)
    df = df.sort_values("x_atc").reset_index(drop=True)

    # engineered features
    df["h_diff"] = df["h_cor_mean"] - df["h_cor_med"]
    df["rel_height_min_elev"] = df["h_cor_mean"] - df["h_cor_mean"].min()
    return df


csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
print(f"raw CSVs: {len(csv_files)}")

records = []  # list of (tile, beam, segment_df)
for p in csv_files:
    parts = p.stem.split("_")
    tile  = parts[3]
    beam  = parts[4]
    seg   = load_segment_csv(p)
    seg["tile"] = tile
    seg["beam"] = beam
    seg["src"]  = p.name
    records.append((tile, beam, seg))
    print(f"  {p.name}: {len(seg):,} segments, "
          f"label counts = {seg['label'].value_counts().to_dict()}")
print(f"total: {sum(len(s) for _, _, s in records):,} segments")
""")

# -------- 3. Build per-track sliding windows ----------------------------
md(r"""## 3. Build (N, 5, 8) sliding windows

Each sample is the 8 features for [center-2, ..., center+2] segments
within the *same track* (same tile/beam/source CSV). The label is the
center segment's label. Border segments are skipped.""")

code(r"""FEATURES = [
    "h_cor_mean", "h_diff", "rel_height_min_elev", "height_sd",
    "pcnth_mean", "pcnt_mean", "bcnt_mean", "brate_mean",
]
N_FEATS = len(FEATURES)
print("features:", FEATURES)


def sliding_windows(seg_df, seq_len=SEQ_LEN, nearby=NEARBY):
    arr = seg_df[FEATURES].to_numpy(dtype=np.float32)   # (T, F)
    lab = seg_df["label"].to_numpy(dtype=np.int64)      # (T,)
    if len(arr) < seq_len:
        return np.empty((0, seq_len, N_FEATS), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)
    n = len(arr) - 2 * nearby
    X = np.zeros((n, seq_len, N_FEATS), dtype=np.float32)
    for k in range(seq_len):
        X[:, k, :] = arr[k : k + n]
    y = lab[nearby : nearby + n]
    valid = (y >= 0) & (y < NUM_CLASSES)
    return X[valid], y[valid]


bundles = []
for tile, beam, seg in records:
    X, y = sliding_windows(seg)
    if len(X) == 0:
        continue
    bundles.append({"tile": tile, "beam": beam,
                    "X": X, "y": y, "src": seg["src"].iloc[0]})

total_n = sum(len(b["y"]) for b in bundles)
print(f"prepared {len(bundles)} tracks, {total_n:,} samples total")
""")

# -------- 4. Split ------------------------------------------------------
md(r"""## 4. Train / val / test split

* `GROUPED_SPLIT=True` (default): train on `T02CNA + T02CNC`, test on
  `T03CWT`. Val = 10 % random subset of the train tiles.
* `GROUPED_SPLIT=False`: replicate her sklearn `train_test_split`
  (60/20/20 random shuffle, `random_state=20`). Use this only when you
  want a numbers-match with her notebook, not for honest reporting.""")

code(r"""if GROUPED_SPLIT:
    train_tiles, test_tiles = {"T02CNA", "T02CNC"}, {"T03CWT"}
    X_tr = np.concatenate([b["X"] for b in bundles if b["tile"] in train_tiles])
    y_tr = np.concatenate([b["y"] for b in bundles if b["tile"] in train_tiles])
    X_te = np.concatenate([b["X"] for b in bundles if b["tile"] in test_tiles])
    y_te = np.concatenate([b["y"] for b in bundles if b["tile"] in test_tiles])
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_tr))
    cut = int(0.10 * len(idx))
    val_idx, tr_idx = idx[:cut], idx[cut:]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]
    X_tr,  y_tr  = X_tr[tr_idx],  y_tr[tr_idx]
else:
    from sklearn.model_selection import train_test_split
    X_all = np.concatenate([b["X"] for b in bundles])
    y_all = np.concatenate([b["y"] for b in bundles])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, random_state=20, shuffle=True
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=20, shuffle=True
    )

print(f"train: {len(X_tr):,}   val: {len(X_val):,}   test: {len(X_te):,}")
for split_name, y in [("train", y_tr), ("val", y_val), ("test", y_te)]:
    counts = np.bincount(y, minlength=NUM_CLASSES)
    pct = counts / max(counts.sum(), 1) * 100
    print(f"  {split_name}: " +
          "  ".join(f"{n}={c} ({p:.1f}%)" for n, c, p in zip(CLASS_NAMES, counts, pct)))
""")

# -------- 5. Standardize features --------------------------------------
md(r"""## 5. Standardize features

We use proper z-score standardization (`(x - mean) / std`) computed on
the training set only. The prof's notebook uses `(x - mean) / (1 - std)`
which is almost certainly a typo; we use the conventional formula so
the numbers are defensible. If you want bit-for-bit parity with her
code, replace `denom = stds` with `denom = (1.0 - stds)`.""")

code(r"""flat_tr = X_tr.reshape(-1, N_FEATS)
means = flat_tr.mean(axis=0)
stds  = flat_tr.std(axis=0) + 1e-6
denom = stds   # set to (1.0 - stds) to mirror prof exactly (not recommended)


def standardize(X):
    return (X - means[None, None, :]) / denom[None, None, :]


X_tr  = standardize(X_tr).astype(np.float32)
X_val = standardize(X_val).astype(np.float32)
X_te  = standardize(X_te).astype(np.float32)
print("means:", means.round(3))
print("stds: ", stds.round(3))
""")

# -------- 6. Dataset / loader ------------------------------------------
md(r"""## 6. Dataset + loaders""")

code(r"""class SegmentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_loader = DataLoader(SegmentDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(SegmentDataset(X_val, y_val), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(SegmentDataset(X_te,  y_te),  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
print(f"steps -- train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
""")

# -------- 7. Model -----------------------------------------------------
md(r"""## 7. Model = uni-directional LSTM(48) + Dense(16, elu) x2 + Dense(3)

Mirrors her Keras stack exactly. Softmax is implicit -- the focal loss
takes raw logits and applies log_softmax internally.""")

code(r"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProfStyleLSTM(nn.Module):
    def __init__(self, n_features=N_FEATS, hidden=LSTM_HIDDEN,
                 head_hidden=HEAD_HIDDEN, head_dropout=HEAD_DROPOUT,
                 lstm_dropout=LSTM_DROPOUT, num_classes=NUM_CLASSES):
        super().__init__()
        # nn.LSTM applies tanh on hidden, matching Keras LSTM default.
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=1, batch_first=True,
                            bidirectional=False)
        self.head = nn.Sequential(
            nn.Dropout(lstm_dropout),
            nn.Linear(hidden, head_hidden), nn.ELU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden), nn.ELU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]      # Keras LSTM(return_sequences=False) = last step
        return self.head(last)    # (B, num_classes)


model = ProfStyleLSTM().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"ProfStyleLSTM parameters: {n_params:,}")
""")

# -------- 8. Focal loss + optimizer ------------------------------------
md(r"""## 8. Focal Loss + Adam

`FL = -alpha[true] * (1 - p_true)^gamma * log(p_true)`, averaged over the
batch. Equivalent to Keras's `CategoricalFocalCrossentropy(from_logits=False,
reduction='sum_over_batch_size')` -- here we take logits and do the
softmax inside the loss for numerical stability.""")

code(r"""class CategoricalFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = float(gamma)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)         # (B, C)
        log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp().clamp(min=1e-8, max=1.0 - 1e-8)
        alpha_t = self.alpha.to(logits.device)[target]
        focal_term = (1.0 - p_t).pow(self.gamma)
        per_sample = -alpha_t * focal_term * log_p_t
        return per_sample.mean()


criterion = CategoricalFocalLoss(alpha=ALPHA, gamma=GAMMA).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
print("loss + optimizer ready")
""")

# -------- 9. Training loop --------------------------------------------
md(r"""## 9. Training loop""")

code(r"""def cm_accum(cm, logits, targets):
    pred = logits.argmax(1).detach().cpu().numpy().ravel()
    t = targets.detach().cpu().numpy().ravel()
    idx = NUM_CLASSES * t + pred
    cm += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_cm(cm):
    iou = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou.append(tp / denom if denom > 0 else 0.0)
    iou = np.array(iou, dtype=np.float64)
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
    for X, y in train_loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * X.size(0); n_seen += X.size(0)
        cm_accum(tr_cm, logits, y)
    tr_loss /= max(n_seen, 1)
    tr_miou, _, tr_acc = metrics_from_cm(tr_cm)

    # --- val ---
    model.eval()
    va_loss = 0.0; n_seen = 0
    va_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(X)
            loss = criterion(logits, y)
            va_loss += loss.item() * X.size(0); n_seen += X.size(0)
            cm_accum(va_cm, logits, y)
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
          f"({time.perf_counter()-t0:.0f}s)")

    if va_miou > best_val + 1e-4:
        best_val = va_miou
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "alpha": ALPHA, "gamma": GAMMA,
                    "means": means, "stds": stds,
                    "val_metrics": {"miou": va_miou, "per_iou": va_iou.tolist(),
                                    "pix_acc": va_acc, "loss": va_loss}},
                   RUN_DIR / "best.pt")

# Match the prof's recipe: she uses the FINAL epoch's weights (no early
# stop, no checkpoint callback), so we save those too and use them for
# the test eval below.
torch.save({"epoch": EPOCHS, "model_state": model.state_dict(),
            "alpha": ALPHA, "gamma": GAMMA,
            "means": means, "stds": stds},
           RUN_DIR / "final.pt")

pd.DataFrame(log).to_csv(RUN_DIR / "metrics.csv", index=False)
print(f"best val mIoU: {best_val:.4f}  (best.pt)")
print(f"final epoch saved -> final.pt")
""")

# -------- 10. Test evaluation -----------------------------------------
md(r"""## 10. Test evaluation""")

code(r"""# Default to final.pt (matches her no-early-stop recipe). Change to
# "best.pt" for best-by-val.
CKPT_NAME = "final.pt"
ck = torch.load(RUN_DIR / CKPT_NAME, map_location=device, weights_only=False)
model.load_state_dict(ck["model_state"]); model.eval()
print(f"loaded {CKPT_NAME} (epoch {ck['epoch']})")

test_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
test_loss = 0.0; n_seen = 0
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)
        test_loss += loss.item() * X.size(0); n_seen += X.size(0)
        cm_accum(test_cm, logits, y)
test_loss /= max(n_seen, 1)
test_miou, test_iou, test_acc = metrics_from_cm(test_cm)

# Per-class precision/recall/F1
prec = np.zeros(NUM_CLASSES); rec = np.zeros(NUM_CLASSES); f1 = np.zeros(NUM_CLASSES)
for c in range(NUM_CLASSES):
    tp = test_cm[c, c]
    fp = test_cm[:, c].sum() - tp
    fn = test_cm[c, :].sum() - tp
    prec[c] = tp / (tp + fp) if (tp + fp) else 0.0
    rec[c]  = tp / (tp + fn) if (tp + fn) else 0.0
    f1[c]   = 2 * prec[c] * rec[c] / (prec[c] + rec[c]) if (prec[c] + rec[c]) else 0.0

print(f"TEST  mIoU {test_miou:.4f}  pix_acc {test_acc:.4f}")
print("per-class IoU:      ", test_iou.round(4).tolist())
print("per-class precision:", prec.round(4).tolist())
print("per-class recall:   ", rec.round(4).tolist())
print("per-class F1:       ", f1.round(4).tolist())

with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump({
        "miou": test_miou, "per_iou": test_iou.tolist(),
        "pix_acc": test_acc, "loss": test_loss,
        "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
        "macro_f1": float(f1.mean()),
        "alpha": ALPHA, "gamma": GAMMA,
        "grouped_split": GROUPED_SPLIT,
    }, f, indent=2)
""")

# -------- 11. Confusion matrix (prof-style) ---------------------------
md(r"""## 11. Confusion matrix (prof-style)""")

code(r"""def plot_cm_percent(cm, title, save_path):
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, cm / np.maximum(row_sums, 1) * 100.0, 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"Predicted {c}" for c in CLASS_NAMES_DISP], fontsize=11)
    ax.set_yticklabels(CLASS_NAMES_DISP, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = pct[i, j]
            color = "white" if v > 55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


plot_cm_percent(test_cm,
                "Confusion Matrix (Percentages)",
                RUN_DIR / "confmat.png")
# Also save a copy whose filename + subtitle records the run hyperparams,
# so we can compare ablations later without losing context.
plot_cm_percent(test_cm,
                f"Confusion Matrix (Percentages)  alpha={ALPHA}, gamma={GAMMA}",
                RUN_DIR / f"confmat_alpha_{'_'.join(str(a).replace('.','p') for a in ALPHA)}.png")
""")

# -------- 12. Loss curves ---------------------------------------------
md(r"""## 12. Loss + val-mIoU curves""")

code(r"""hist = pd.read_csv(RUN_DIR / "metrics.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
axes[0].plot(hist["epoch"], hist["val_loss"],   label="val")
axes[0].set_title("Focal loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(hist["epoch"], hist["train_miou"], label="train")
axes[1].plot(hist["epoch"], hist["val_miou"],   label="val")
axes[1].set_title("mIoU"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig(RUN_DIR / "loss_curve.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- 13. Done ----------------------------------------------------
md(r"""## 13. Done

Files in `runs/lstm_prof_style_v1/`:

* `best.pt`            -- best-by-val checkpoint (includes feature
  mean/std for inference)
* `metrics.csv`        -- per-epoch train/val curves
* `test_metrics.json`  -- test mIoU, pix_acc, per-class P/R/F1/IoU
* `confmat.png`        -- prof-style percentage confusion matrix
* `loss_curve.png`     -- training diagnostics

This is the structurally faithful sibling of `lstm_focal.ipynb`:

| | lstm_focal.ipynb | lstm_prof_style.ipynb |
|---|---|---|
| direction | Bi-LSTM | **Uni-LSTM (her arch)** |
| hidden / layers | 128 / 2 | **48 / 1 (hers)** |
| input | per-photon window tiled to 128x128 | **5-segment window over 8 engineered features (hers)** |
| output | per-pixel 128x128 mask | **per-segment class (hers)** |
| consumable by deep-fusion? | yes (per-pixel) | no (single class out) |
| comparable to prof's results? | indirectly | **yes, apples-to-apples** |

If the test-set per-class IoU here shows thin-ice + water both
non-trivial (i.e. NOT collapsed-to-ice), the focal-loss recipe is
working and we can carry the same alpha/gamma into our per-pixel
focal notebook + deep-fusion retraining.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
