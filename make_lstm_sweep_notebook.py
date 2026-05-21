"""
Generate lstm_sweep.ipynb -- runs a curated grid of focal-loss /
LSTM / training hyperparameters on top of the prof-style LSTM
recipe and ranks them by test mIoU.

Baseline (the run that gave 0.6811 test mIoU):
    alpha=[0.05, 0.45, 0.60], gamma=2.0, lstm_hidden=48,
    lr=8.886e-4, dropout=0.4, seq_len=5, epochs=50, batch=32, seed=42

The sweep varies *one axis at a time* from the baseline:

  * Alpha vectors (her 5 variants)
  * Gamma (1.0, 2.0, 3.0, 5.0)
  * LSTM hidden (32, 48, 64, 96)
  * Learning rate (5e-4, 8.886e-4, 1e-3, 2e-3)
  * Dropout (0.2, 0.3, 0.4, 0.5)
  * Sequence length (3, 5, 7)
  * Random seed (42, 7, 123) -- triple-seeded baseline for variance

Total: ~19 distinct runs (the baseline is shared, so we don't run it
twice for each axis -- the loop deduplicates configs by name).

Each run saves into `runs/lstm_sweep/<run_name>/`:
  - final.pt, best.pt
  - metrics.csv
  - test_metrics.json
  - confmat.png
Plus a master `runs/lstm_sweep/sweep_results.csv` table, and at the end
a `confmat_winner.png` for the top-scoring config.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "lstm_sweep.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Prof-style LSTM -- Hyperparameter Sweep

This notebook runs ~19 ablations of the prof-style LSTM + focal-loss
recipe and ranks them. It is the "super notebook" version of
`lstm_prof_style.ipynb` -- same model, same data loader, but
parameterized so we can sweep a curated grid in one go.

**Baseline** (gave us test mIoU 0.6811):
* alpha = [0.05, 0.45, 0.60]
* gamma = 2.0
* lstm_hidden = 48
* lr = 8.886e-4
* dropout = 0.4
* seq_len = 5
* epochs = 50, batch = 32, seed = 42

**One axis varied at a time:**

| axis | values |
|---|---|
| alpha | [0.05,0.45,0.60], [0.02,0.44,0.54], [0.05,0.45,0.50], [0.05,0.50,0.45], [0.041,0.409,0.550] |
| gamma | 1.0, **2.0**, 3.0, 5.0 |
| lstm_hidden | 32, **48**, 64, 96 |
| lr | 5e-4, **8.886e-4**, 1e-3, 2e-3 |
| dropout | 0.2, 0.3, **0.4**, 0.5 |
| seq_len | 3, **5**, 7 |
| seed | **42**, 7, 123 |

Each run writes its own folder under `runs/lstm_sweep/<run_name>/`, and
a row to `runs/lstm_sweep/sweep_results.csv`. The notebook is safe to
re-run: completed runs are skipped (it checks for `test_metrics.json`).
""")

# -------- 0. Setup ------------------------------------------------------
md("## 0. Setup")

code(r"""import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
""")

code(r"""import json, math, random, time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# -------- 1. Paths ------------------------------------------------------
md("## 1. Paths")

code(r"""PROJECT_ROOT = Path("/home/spant/Research Seminar/Project")
CSV_DIR      = PROJECT_ROOT / "IS2_Corrected_data"
SWEEP_DIR    = PROJECT_ROOT / "runs" / "lstm_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES  = 3
NUM_WORKERS  = 2
CLASS_NAMES      = ["ice", "thin_ice", "water"]
CLASS_NAMES_DISP = ["thick ice", "thin ice", "water"]
FEATURES = ["h_cor_mean", "h_diff", "rel_height_min_elev", "height_sd",
            "pcnth_mean", "pcnt_mean", "bcnt_mean", "brate_mean"]
N_FEATS  = len(FEATURES)

# Split recipe -- match the prof's `train_test_split(test_size=0.20,
# random_state=20, shuffle=True)` then 25% val of remaining.
GROUPED_SPLIT = False    # flip to True for the honest tile-grouped split
print("sweep dir:", SWEEP_DIR)
""")

# -------- 2. Data load (same as prof_style) ----------------------------
md(r"""## 2. Load + cache segment data (one time)

This is identical to `lstm_prof_style.ipynb` Section 2. We do it once
here and reuse the cached arrays for every sweep config.""")

code(r"""def load_segment_csv(csv_path):
    df = pd.read_csv(csv_path)
    needed = ["h_cor_mean", "h_cor_med", "x_atc", "label",
              "height_sd", "pcnth_mean", "pcnt_mean", "bcnt_mean", "brate_mean"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing {missing}")
    df = df[df["label"].isin([0, 1, 2])].copy()
    df = df.dropna(subset=needed)
    df = df.sort_values("x_atc").reset_index(drop=True)
    df["h_diff"] = df["h_cor_mean"] - df["h_cor_med"]
    df["rel_height_min_elev"] = df["h_cor_mean"] - df["h_cor_mean"].min()
    return df


csv_files = sorted(CSV_DIR.glob("ATL03_*_done.csv"))
print(f"raw CSVs: {len(csv_files)}")
records = []
for p in csv_files:
    parts = p.stem.split("_")
    seg = load_segment_csv(p)
    seg["tile"] = parts[3]; seg["beam"] = parts[4]; seg["src"] = p.name
    records.append((parts[3], parts[4], seg))
print(f"total segments: {sum(len(s) for _, _, s in records):,}")
""")

# -------- 3. Window-builder (parameterized on seq_len) ------------------
md(r"""## 3. Sliding-window builder (parameterized on `seq_len`)

For each track, we build windows of length `seq_len` with the center
segment's label. `nearby = seq_len // 2` so seq_len=5 -> 2 above + 2
below + center. `seq_len=3` -> 1 above + 1 below + center, etc.""")

code(r"""def build_windows(records, seq_len):
    nearby = seq_len // 2
    bundles = []
    for tile, beam, seg in records:
        arr = seg[FEATURES].to_numpy(dtype=np.float32)
        lab = seg["label"].to_numpy(dtype=np.int64)
        if len(arr) < seq_len:
            continue
        n = len(arr) - 2 * nearby
        X = np.zeros((n, seq_len, N_FEATS), dtype=np.float32)
        for k in range(seq_len):
            X[:, k, :] = arr[k : k + n]
        y = lab[nearby : nearby + n]
        valid = (y >= 0) & (y < NUM_CLASSES)
        bundles.append({"tile": tile, "beam": beam,
                        "X": X[valid], "y": y[valid]})
    return bundles


# Cache one bundle per distinct seq_len (3, 5, 7) so we don't rebuild
WINDOW_CACHE = {}
for sl in [3, 5, 7]:
    bundles = build_windows(records, sl)
    WINDOW_CACHE[sl] = bundles
    total = sum(len(b["y"]) for b in bundles)
    print(f"  seq_len={sl}: {total:,} samples")
""")

# -------- 4. Split + standardize (parameterized on seq_len, seed) -------
md(r"""## 4. Per-config split + standardize

Standardization mean/std comes from the *train* split only and is
applied to val/test. Standardize stats depend on `seq_len`, so we
recompute per config.""")

code(r"""from sklearn.model_selection import train_test_split as _tts


def split_and_standardize(seq_len, seed, grouped=GROUPED_SPLIT):
    bundles = WINDOW_CACHE[seq_len]
    if grouped:
        train_tiles = {"T02CNA", "T02CNC"}
        test_tiles  = {"T03CWT"}
        X_tr = np.concatenate([b["X"] for b in bundles if b["tile"] in train_tiles])
        y_tr = np.concatenate([b["y"] for b in bundles if b["tile"] in train_tiles])
        X_te = np.concatenate([b["X"] for b in bundles if b["tile"] in test_tiles])
        y_te = np.concatenate([b["y"] for b in bundles if b["tile"] in test_tiles])
        rng  = np.random.RandomState(seed)
        idx  = rng.permutation(len(X_tr))
        cut  = int(0.10 * len(idx))
        X_val, y_val = X_tr[idx[:cut]], y_tr[idx[:cut]]
        X_tr,  y_tr  = X_tr[idx[cut:]], y_tr[idx[cut:]]
    else:
        X_all = np.concatenate([b["X"] for b in bundles])
        y_all = np.concatenate([b["y"] for b in bundles])
        # Mirror her sklearn recipe; seed lets us vary the shuffle
        X_tr, X_te, y_tr, y_te = _tts(X_all, y_all,
                                      test_size=0.20, random_state=seed, shuffle=True)
        X_tr, X_val, y_tr, y_val = _tts(X_tr, y_tr,
                                        test_size=0.25, random_state=seed, shuffle=True)
    flat = X_tr.reshape(-1, N_FEATS)
    means = flat.mean(axis=0)
    stds  = flat.std(axis=0) + 1e-6
    def stz(X): return ((X - means) / stds).astype(np.float32)
    return (stz(X_tr), y_tr.astype(np.int64),
            stz(X_val), y_val.astype(np.int64),
            stz(X_te),  y_te.astype(np.int64),
            means, stds)


print("split helper ready")
""")

# -------- 5. Model factory + focal loss --------------------------------
md(r"""## 5. Model + focal loss factories

Same architecture as `lstm_prof_style.ipynb` (uni-LSTM + Dense(16,elu)x2
+ Dense(3)) but parameterized on `hidden`, `dropout`, `head_hidden`.""")

code(r"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProfStyleLSTM(nn.Module):
    def __init__(self, n_features, hidden, head_hidden=16,
                 dropout=0.4, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, head_hidden), nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden), nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class CategoricalFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = float(gamma)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp().clamp(min=1e-8, max=1.0 - 1e-8)
        alpha_t = self.alpha.to(logits.device)[target]
        focal_term = (1.0 - p_t).pow(self.gamma)
        return (-alpha_t * focal_term * log_p_t).mean()


print("model + loss factories ready")
""")

# -------- 6. Train + eval (one config) --------------------------------
md(r"""## 6. Train + eval one config

Returns the final-epoch test metrics. Saves every run's artifacts to
`runs/lstm_sweep/<run_name>/`.""")

code(r"""def cm_accum(cm, logits, target):
    pred = logits.argmax(1).detach().cpu().numpy().ravel()
    t = target.detach().cpu().numpy().ravel()
    idx = NUM_CLASSES * t + pred
    cm += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_cm(cm):
    iou = []; prec = []; rec = []; f1 = []
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
    pix_acc = float(np.diag(cm).sum() / max(cm.sum(), 1))
    return {"miou": float(iou.mean()), "per_iou": iou.tolist(),
            "pix_acc": pix_acc,
            "precision": prec.tolist(), "recall": rec.tolist(),
            "f1": f1.tolist(), "macro_f1": float(f1.mean())}


def run_one(cfg, verbose=True):
    run_dir = SWEEP_DIR / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    tm_path = run_dir / "test_metrics.json"
    if tm_path.exists():
        if verbose: print(f"  [skip] {cfg['name']} (already done)")
        return json.loads(tm_path.read_text())

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    X_tr, y_tr, X_val, y_val, X_te, y_te, means, stds = \
        split_and_standardize(cfg["seq_len"], cfg["seed"])

    def loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                          num_workers=NUM_WORKERS, drop_last=shuffle)

    tr_ld  = loader(X_tr,  y_tr,  True)
    val_ld = loader(X_val, y_val, False)
    te_ld  = loader(X_te,  y_te,  False)

    model = ProfStyleLSTM(N_FEATS, hidden=cfg["hidden"],
                          head_hidden=cfg["head_hidden"],
                          dropout=cfg["dropout"]).to(device)
    crit  = CategoricalFocalLoss(cfg["alpha"], gamma=cfg["gamma"]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val = -1.0; log = []
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        # train
        model.train()
        tr_loss = 0.0; n = 0
        tr_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for X, y in tr_ld:
            X = X.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            tr_loss += loss.item() * X.size(0); n += X.size(0)
            cm_accum(tr_cm, logits, y)
        tr_loss /= max(n, 1)
        tr_m = metrics_from_cm(tr_cm)

        # val
        model.eval()
        va_loss = 0.0; n = 0
        va_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for X, y in val_ld:
                X = X.to(device); y = y.to(device)
                logits = model(X)
                va_loss += crit(logits, y).item() * X.size(0); n += X.size(0)
                cm_accum(va_cm, logits, y)
        va_loss /= max(n, 1)
        va_m = metrics_from_cm(va_cm)

        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                    "train_miou": tr_m["miou"], "val_miou": va_m["miou"]})
        if va_m["miou"] > best_val + 1e-4:
            best_val = va_m["miou"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "means": means, "stds": stds, **cfg},
                       run_dir / "best.pt")
        if verbose and (ep == 1 or ep % 10 == 0 or ep == cfg["epochs"]):
            print(f"    ep{ep:02d}  tr_loss {tr_loss:.4f}  val_miou {va_m['miou']:.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)")
    # final ckpt
    torch.save({"epoch": cfg["epochs"], "model_state": model.state_dict(),
                "means": means, "stds": stds, **cfg},
               run_dir / "final.pt")
    pd.DataFrame(log).to_csv(run_dir / "metrics.csv", index=False)

    # test (use final.pt to match her no-early-stop recipe)
    model.eval()
    te_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for X, y in te_ld:
            X = X.to(device); y = y.to(device)
            cm_accum(te_cm, model(X), y)
    test_m = metrics_from_cm(te_cm)
    test_m["cm"] = te_cm.tolist()
    test_m["best_val_miou"] = best_val
    test_m["config"] = cfg

    with open(tm_path, "w") as f:
        json.dump(test_m, f, indent=2)

    # also save a quick CM image per run
    _save_cm_quick(te_cm, run_dir / "confmat.png",
                   subtitle=cfg["name"])
    return test_m


def _save_cm_quick(cm, path, subtitle=""):
    cm = cm.astype(np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    pct = np.where(rs > 0, cm / np.maximum(rs, 1) * 100.0, 0.0)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"Predicted {c}" for c in CLASS_NAMES_DISP])
    ax.set_yticklabels(CLASS_NAMES_DISP)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Percentages)\n{subtitle}", fontsize=11)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = pct[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 55 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
""")

# -------- 7. Sweep definitions -----------------------------------------
md(r"""## 7. Sweep grid (one axis at a time)

We start from a **baseline** and vary one axis per run. The notebook
walks a flat list of configs, so total time = `len(configs) * ~3min`
(actual depends on `seq_len` and `hidden`).""")

code(r"""BASELINE = dict(
    alpha=[0.05, 0.45, 0.60],
    gamma=2.0,
    hidden=48,
    head_hidden=16,
    lr=8.886176350890356e-4,
    dropout=0.4,
    seq_len=5,
    epochs=50,
    batch_size=32,
    seed=42,
)


def _fmt(v):
    if isinstance(v, list):
        return "[" + ",".join(f"{x:.3f}".rstrip("0").rstrip(".") for x in v) + "]"
    if isinstance(v, float):
        if v < 1e-2: return f"{v:.0e}"
        return f"{v:g}"
    return str(v)


def _make(name, **overrides):
    cfg = deepcopy(BASELINE)
    cfg.update(overrides)
    cfg["name"] = name
    return cfg


configs = []
seen = set()

def add(cfg):
    if cfg["name"] in seen: return
    seen.add(cfg["name"])
    configs.append(cfg)

# Baseline
add(_make("baseline"))

# 1. Alpha sweep
for a in [[0.02, 0.44, 0.54], [0.05, 0.45, 0.50],
          [0.05, 0.50, 0.45], [0.041, 0.409, 0.550]]:
    add(_make(f"alpha_{_fmt(a)}", alpha=a))

# 2. Gamma sweep
for g in [1.0, 3.0, 5.0]:
    add(_make(f"gamma_{g}", gamma=g))

# 3. Hidden sweep
for h in [32, 64, 96]:
    add(_make(f"hidden_{h}", hidden=h))

# 4. LR sweep
for lr in [5e-4, 1e-3, 2e-3]:
    add(_make(f"lr_{_fmt(lr)}", lr=lr))

# 5. Dropout sweep
for d in [0.2, 0.3, 0.5]:
    add(_make(f"dropout_{d}", dropout=d))

# 6. Sequence length sweep
for sl in [3, 7]:
    add(_make(f"seq_{sl}", seq_len=sl))

# 7. Seed reproducibility
for s in [7, 123]:
    add(_make(f"seed_{s}", seed=s))

print(f"Total configs: {len(configs)}")
for c in configs:
    print(f"  {c['name']}")
""")

# -------- 8. Run sweep --------------------------------------------------
md(r"""## 8. Run the sweep

Each run takes a few minutes. The notebook caches results -- if you
re-run after a crash, completed configs are skipped.""")

code(r"""results = []
t0_all = time.perf_counter()
for i, cfg in enumerate(configs):
    print(f"[{i+1}/{len(configs)}]  {cfg['name']}")
    r = run_one(cfg, verbose=True)
    row = {"name": cfg["name"],
           "alpha":   _fmt(cfg["alpha"]),
           "gamma":   cfg["gamma"],
           "hidden":  cfg["hidden"],
           "lr":      cfg["lr"],
           "dropout": cfg["dropout"],
           "seq_len": cfg["seq_len"],
           "seed":    cfg["seed"],
           "test_miou":   r["miou"],
           "test_acc":    r["pix_acc"],
           "test_iou_ice":   r["per_iou"][0],
           "test_iou_thin":  r["per_iou"][1],
           "test_iou_water": r["per_iou"][2],
           "test_f1_macro":  r["macro_f1"],
           "best_val_miou":  r.get("best_val_miou", float("nan"))}
    results.append(row)
    # save incrementally so we keep progress on crash
    pd.DataFrame(results).to_csv(SWEEP_DIR / "sweep_results.csv", index=False)

print(f"\nTotal sweep time: {(time.perf_counter() - t0_all)/60:.1f} min")
""")

# -------- 9. Rank --------------------------------------------------------
md(r"""## 9. Ranked results""")

code(r"""df = pd.read_csv(SWEEP_DIR / "sweep_results.csv").sort_values("test_miou",
                                                                 ascending=False)
display_cols = ["name", "test_miou", "test_acc", "test_iou_ice",
                "test_iou_thin", "test_iou_water", "test_f1_macro",
                "alpha", "gamma", "hidden", "lr", "dropout", "seq_len", "seed"]
print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
""")

# -------- 10. Per-axis plots -------------------------------------------
md(r"""## 10. Per-axis comparison plots""")

code(r"""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()

def _bar(ax, sub, label_col, title):
    sub = sub.sort_values("test_miou", ascending=False)
    x = np.arange(len(sub))
    ax.bar(x, sub["test_miou"], color="#4C78A8")
    ax.set_xticks(x); ax.set_xticklabels(sub[label_col], rotation=20, fontsize=8)
    ax.set_title(title); ax.set_ylabel("test mIoU")
    ax.axhline(df.loc[df["name"] == "baseline", "test_miou"].iloc[0],
               color="red", linestyle="--", alpha=0.6, label="baseline")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    for xi, v in zip(x, sub["test_miou"]):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

_bar(axes[0], df[df["name"].str.startswith(("baseline", "alpha_"))], "alpha",  "Alpha")
_bar(axes[1], df[df["name"].str.startswith(("baseline", "gamma_"))], "gamma",  "Gamma")
_bar(axes[2], df[df["name"].str.startswith(("baseline", "hidden_"))], "hidden", "Hidden")
_bar(axes[3], df[df["name"].str.startswith(("baseline", "lr_"))],     "lr",     "LR")
_bar(axes[4], df[df["name"].str.startswith(("baseline", "dropout_"))],"dropout","Dropout")
_bar(axes[5], df[df["name"].str.startswith(("baseline", "seq_"))],    "seq_len","Seq len")
plt.tight_layout()
plt.savefig(SWEEP_DIR / "sweep_per_axis.png", dpi=160, bbox_inches="tight")
plt.show()
""")

# -------- 11. Seed variance --------------------------------------------
md(r"""## 11. Seed variance (baseline + seed_7 + seed_123)""")

code(r"""seed_runs = df[df["name"].isin(["baseline", "seed_7", "seed_123"])]
if len(seed_runs) >= 2:
    m = seed_runs["test_miou"]
    print(f"baseline seed=42: {df.loc[df['name']=='baseline', 'test_miou'].iloc[0]:.4f}")
    print(f"seed mean={m.mean():.4f}  std={m.std():.4f}  range=[{m.min():.4f}, {m.max():.4f}]")
else:
    print("(seed runs not complete yet)")
""")

# -------- 12. Winner CM ------------------------------------------------
md(r"""## 12. Winner -- prof-style CM""")

code(r"""winner = df.iloc[0]
print(f"Winner: {winner['name']}  test mIoU {winner['test_miou']:.4f}")
print(f"  alpha={winner['alpha']}  gamma={winner['gamma']}  "
      f"hidden={winner['hidden']}  lr={winner['lr']}  "
      f"dropout={winner['dropout']}  seq_len={winner['seq_len']}  seed={winner['seed']}")

tm = json.loads((SWEEP_DIR / winner["name"] / "test_metrics.json").read_text())
cm_winner = np.array(tm["cm"], dtype=np.int64)

# Plot in EXACTLY the prof's style: percentages, Blues, thick/thin/water
cm = cm_winner.astype(np.float64)
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
plt.savefig(SWEEP_DIR / "confmat_winner.png", dpi=180, bbox_inches="tight")
plt.show()
""")

# -------- 13. Done -----------------------------------------------------
md(r"""## 13. Done

Files in `runs/lstm_sweep/`:

* `sweep_results.csv`   -- one row per config (test mIoU, per-class IoU,
                           F1 macro, all hyperparams)
* `<run_name>/best.pt`, `final.pt`, `metrics.csv`, `test_metrics.json`,
  `confmat.png` -- per-run artifacts
* `sweep_per_axis.png`  -- 6-panel bar chart, mIoU vs each varied axis
* `confmat_winner.png`  -- top config's confusion matrix in the prof's
                           exact style

What to do next:

1. Find the winner from Section 12, eyeball its `confmat.png`
2. If a single axis stands out (e.g. `gamma=3.0` clearly dominates),
   pin that and bring it into deep-fusion
3. If two axes interact (e.g. higher hidden likes a different alpha),
   run a small 2-axis grid by hand on top of this notebook
4. Triple-seed the winner config (already partially done by `seed_7`
   and `seed_123`) to report mean +- std""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
