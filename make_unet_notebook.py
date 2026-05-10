"""
Generate unet_baseline.ipynb -- the image-only U-Net baseline notebook.
Run this script and it writes the notebook to disk.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "unet_baseline.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================
md(r"""# Sea Ice Segmentation -- U-Net (image-only) baseline

This is **variant #1** of our ablation: train a U-Net to predict the per-pixel
3-class mask from the RGB image alone, no CSV features.

* **Input:** `outputs/row{i}_{date}_{tile}_{beam}.png` (128x128 RGB)
* **Target:** `outputs_segmented/row{i}_{date}_{tile}_{beam}.png` decoded to {0=ice, 1=thin_ice, 2=water}
* **Loss:** pixel-wise weighted cross-entropy
* **Metric:** mIoU (matches Zhao et al. 2023)

Designed to run on a single A6000 (48 GB). Default config completes in ~30-60 min.
""")

# -------- Cell: setup ----------------------------------------------------
md("## 0. Setup")

code(r"""# Pin to a specific GPU on a shared box. MUST run before `import torch`.
# Edit the index to whichever GPU has free memory (run `nvidia-smi` to check).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
""")

code(r"""# Pip-install only the things likely missing on a fresh PyTorch box.
# Skip this cell if you already have the deps.
import sys, subprocess
for pkg in ["segmentation_models_pytorch", "tqdm"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
""")

code(r"""import os, json, math, random, time
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
md(r"""## 1. Config

Edit `PROJECT_ROOT` to wherever you put the data on the GPU box.
Everything else has sensible defaults.""")

code(r"""# >>> EDIT ME <<<
PROJECT_ROOT = Path("/home/spant/sea_ice")    # change to your path on the GPU
EXP_NAME     = "unet_imgonly_v1"

# Derived paths
IMG_DIR  = PROJECT_ROOT / "outputs"             # 128x128 RGB photos
MASK_DIR = PROJECT_ROOT / "outputs_segmented"   # 128x128 colored ground-truth
RUN_DIR  = PROJECT_ROOT / "runs" / EXP_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
SEED         = 42
NUM_CLASSES  = 3
PATCH        = 128
BATCH_SIZE   = 64        # A6000 (48 GB) handles this easily for ResNet-18 U-Net
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 5         # early stopping on val mIoU
NUM_WORKERS  = 4
USE_AMP      = True

# Class definitions (matches what the segmentation pipeline writes)
CLASS_NAMES = ["ice", "thin_ice", "water"]
CLASS_COLORS = {
    0: (255, 0, 0),    # ice -- red
    1: (0, 0, 255),    # thin ice -- blue
    2: (0, 255, 0),    # water -- green
}

# ImageNet normalization (because we use an ImageNet-pretrained ResNet-18)
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Project root:", PROJECT_ROOT)
print("Run dir:     ", RUN_DIR)
""")

# -------- Cell: manifest -------------------------------------------------
md(r"""## 2. Build manifest

Walk `outputs/`, parse the filename `row{N}_{date}_{tile}_{beam}.png`,
and assemble a DataFrame indexing every (image, mask) pair.
Saved to disk so reruns are instant.""")

code(r"""import re

manifest_path = RUN_DIR / "manifest.csv"

if manifest_path.exists():
    manifest = pd.read_csv(manifest_path)
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
    manifest.to_csv(manifest_path, index=False)
    print(f"Built manifest: {len(manifest):,} rows -> {manifest_path}")

# Sanity: every mask must exist
missing = [m for m in manifest["mask_path"].head(5) if not Path(m).exists()]
assert not missing, f"missing masks: {missing}"
manifest.head(3)
""")

# -------- Cell: split ---------------------------------------------------
md(r"""## 3. Tile-based train / val / test split

* **Train:** T02CNA + T02CNC (both beams)
* **Val:**   10% randomly held out from train (seeded)
* **Test:**  T03CWT (different date, different scene)""")

code(r"""tiles_train = ["T02CNA", "T02CNC"]
tiles_test  = ["T03CWT"]

train_pool = manifest[manifest["tile"].isin(tiles_train)].reset_index(drop=True)
test_df    = manifest[manifest["tile"].isin(tiles_test)].reset_index(drop=True)

# 10% of train_pool -> val (seeded)
rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train_pool), size=int(0.10 * len(train_pool)), replace=False)
val_mask = np.zeros(len(train_pool), dtype=bool); val_mask[val_idx] = True

train_df = train_pool[~val_mask].reset_index(drop=True)
val_df   = train_pool[ val_mask].reset_index(drop=True)

print(f"Train: {len(train_df):,}   Val: {len(val_df):,}   Test: {len(test_df):,}")
print(f"  by tile (train): {train_df['tile'].value_counts().to_dict()}")
print(f"  by tile (test):  {test_df['tile'].value_counts().to_dict()}")
""")

# -------- Cell: mask decoding -------------------------------------------
md(r"""## 4. Mask color -> integer label

The professor's segmentation pipeline writes RGB PNGs where every pixel is
exactly one of three colors. We decode them into a `(H, W)` int label map.""")

code(r"""def mask_rgb_to_int(mask_rgb):
    # (H,W,3) RGB uint8 -> (H,W) uint8 in {0,1,2}; 255 = unmapped pixel.
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    is_ice   = (mask_rgb == [255, 0, 0]).all(axis=-1)
    is_thin  = (mask_rgb == [0, 0, 255]).all(axis=-1)
    is_water = (mask_rgb == [0, 255, 0]).all(axis=-1)
    out[is_ice]   = 0
    out[is_thin]  = 1
    out[is_water] = 2
    return out


# Sanity check on 200 random masks: no unmapped pixels expected
sample = manifest.sample(n=200, random_state=SEED)
total, unmapped = 0, 0
for p in tqdm(sample["mask_path"], desc="verifying masks"):
    arr = np.array(Image.open(p).convert("RGB"))
    out = mask_rgb_to_int(arr)
    total    += out.size
    unmapped += int((out == 255).sum())
print(f"checked {len(sample)} masks, {total:,} pixels, {unmapped} unmapped "
      f"({100*unmapped/total:.4f}%)")
""")

# -------- Cell: class weights ------------------------------------------
md(r"""## 5. Class weights

Our data is heavily imbalanced (ice dominates). Weighted cross-entropy
prevents the model from collapsing to "predict ice everywhere".""")

code(r"""weights_path = RUN_DIR / "class_weights.json"

if weights_path.exists():
    with open(weights_path) as f:
        d = json.load(f)
    counts  = np.array(d["counts"], dtype=np.int64)
    weights = np.array(d["weights"], dtype=np.float32)
else:
    n_sample = min(5000, len(train_df))
    sampled = train_df.sample(n=n_sample, random_state=SEED)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in tqdm(sampled["mask_path"], desc="counting class pixels"):
        arr = np.array(Image.open(p).convert("RGB"))
        m = mask_rgb_to_int(arr)
        for c in range(NUM_CLASSES):
            counts[c] += int((m == c).sum())
    total = counts.sum()
    weights = (total / (NUM_CLASSES * counts.astype(np.float64))).astype(np.float32)
    with open(weights_path, "w") as f:
        json.dump({"counts": counts.tolist(),
                   "weights": weights.tolist(),
                   "n_sample": int(n_sample)}, f, indent=2)

for c, name in enumerate(CLASS_NAMES):
    pct = 100 * counts[c] / counts.sum()
    print(f"  {name:8s}  {counts[c]:>14,d} px  ({pct:5.2f}%)  weight={weights[c]:.3f}")
""")

# -------- Cell: dataset -------------------------------------------------
md(r"""## 6. Dataset

Lazy-loads each `(image, mask)` pair from disk. ImageNet normalization is
applied because the encoder is ImageNet-pretrained. Augmentation: random
flips and 90-degree rotations applied identically to image and mask.""")

code(r"""def random_flip_rotate(img: np.ndarray, mask: np.ndarray):
    if random.random() < 0.5:
        img  = img[:, ::-1, :]
        mask = mask[:, ::-1]
    if random.random() < 0.5:
        img  = img[::-1, :, :]
        mask = mask[::-1, :]
    k = random.randint(0, 3)
    if k:
        img  = np.rot90(img,  k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class SeaIceImageDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img  = np.array(Image.open(r["image_path"]).convert("RGB"))   # H,W,3 uint8
        mask = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))
        if self.augment:
            img, mask = random_flip_rotate(img, mask)
        img = img.astype(np.float32) / 255.0
        img = (img - IM_MEAN) / IM_STD
        img = np.transpose(img, (2, 0, 1))                            # C,H,W
        return torch.from_numpy(img), torch.from_numpy(mask).long()


# Quick spot-check
_ds = SeaIceImageDataset(train_df.head(8), augment=True)
_x, _y = _ds[0]
print(f"image: {_x.shape}, {_x.dtype}, range [{_x.min():.2f}, {_x.max():.2f}]")
print(f"mask:  {_y.shape}, {_y.dtype}, unique={torch.unique(_y).tolist()}")
""")

# -------- Cell: dataloaders --------------------------------------------
md("## 7. DataLoaders")

code(r"""train_ds = SeaIceImageDataset(train_df, augment=True)
val_ds   = SeaIceImageDataset(val_df,   augment=False)
test_ds  = SeaIceImageDataset(test_df,  augment=False)

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
md(r"""## 8. Model

U-Net with a ResNet-18 backbone, ImageNet-pretrained.
Output: `(B, 3, 128, 128)` logits.""")

code(r"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(device)

# Sanity forward
with torch.no_grad():
    dummy = torch.zeros(2, 3, PATCH, PATCH, device=device)
    out = model(dummy)
print(f"model output: {tuple(out.shape)}  dtype={out.dtype}")
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
""")

# -------- Cell: training utilities -------------------------------------
md(r"""## 9. Training utilities

`IoUAccumulator` keeps running per-class intersection and union counts,
so mIoU is computed correctly across the entire eval set
(not averaged per batch).""")

code(r"""class IoUAccumulator:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.inter = np.zeros(self.num_classes, dtype=np.int64)
        self.union = np.zeros(self.num_classes, dtype=np.int64)
        self.cm    = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.detach().cpu().numpy().ravel()
        t = targets.detach().cpu().numpy().ravel()
        for c in range(self.num_classes):
            pc = (p == c); tc = (t == c)
            self.inter[c] += int(np.logical_and(pc, tc).sum())
            self.union[c] += int(np.logical_or (pc, tc).sum())
        # Confusion matrix (rows = true, cols = predicted)
        idx = self.num_classes * t + p
        self.cm += np.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes)

    def per_class_iou(self):
        return self.inter / np.maximum(self.union, 1)

    def miou(self):
        return float(self.per_class_iou().mean())

    def pixel_accuracy(self):
        return float(np.diag(self.cm).sum() / max(self.cm.sum(), 1))


def evaluate(model, loader, criterion, device):
    model.eval()
    acc = IoUAccumulator(NUM_CLASSES)
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            acc.update(preds, y)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
    return {
        "loss":     loss_sum / max(n, 1),
        "miou":     acc.miou(),
        "per_iou":  acc.per_class_iou().tolist(),
        "pix_acc":  acc.pixel_accuracy(),
        "cm":       acc.cm.tolist(),
    }
""")

# -------- Cell: training loop -------------------------------------------
md(r"""## 10. Training loop

Trains with AMP (mixed precision), AdamW, cosine schedule.
Saves the best checkpoint by val mIoU.""")

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
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    train_loss = train_loss_sum / max(n, 1)
    val = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
          f"val_loss={val['loss']:.4f}  val_mIoU={val['miou']:.4f}  "
          f"per_iou={[f'{v:.3f}' for v in val['per_iou']]}  "
          f"pix_acc={val['pix_acc']:.4f}  "
          f"lr={optimizer.param_groups[0]['lr']:.2e}")

    log.append({
        "epoch": epoch, "train_loss": train_loss,
        "val_loss": val["loss"], "val_miou": val["miou"],
        "iou_ice": val["per_iou"][0], "iou_thin": val["per_iou"][1],
        "iou_water": val["per_iou"][2], "pix_acc": val["pix_acc"],
        "lr": optimizer.param_groups[0]["lr"],
    })
    pd.DataFrame(log).to_csv(log_path, index=False)

    if val["miou"] > best_miou:
        best_miou = val["miou"]
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "val_metrics": val, "weights": weights.tolist(),
        }, ckpt_path)
        patience_left = PATIENCE
        print(f"  -> saved best ({best_miou:.4f}) to {ckpt_path}")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"  -> early stopping (no val_mIoU improvement for {PATIENCE} epochs)")
            break

print(f"\nBest val mIoU: {best_miou:.4f}")
""")

# -------- Cell: final test eval ----------------------------------------
md(r"""## 11. Final test evaluation

Load the best checkpoint and evaluate on the held-out **T03CWT** tile.""")

code(r"""ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])

test_metrics = evaluate(model, test_loader, criterion, device)
print(f"TEST   mIoU={test_metrics['miou']:.4f}   pix_acc={test_metrics['pix_acc']:.4f}")
print(f"per-class IoU:")
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

# persist final metrics
with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump({k: v for k, v in test_metrics.items() if k != "cm"}, f, indent=2)
""")

# -------- Cell: visualizations -----------------------------------------
md(r"""## 12. Visualize predictions

Pick 6 random test samples and plot RGB | ground-truth | prediction
side-by-side.""")

code(r"""def int_mask_to_rgb(m: np.ndarray) -> np.ndarray:
    out = np.zeros((*m.shape, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        out[m == c] = color
    return out


model.eval()
sample = test_df.sample(n=6, random_state=SEED).reset_index(drop=True)
fig, axes = plt.subplots(6, 3, figsize=(7.5, 14))
with torch.no_grad():
    for i, r in sample.iterrows():
        rgb = np.array(Image.open(r["image_path"]).convert("RGB"))
        gt  = mask_rgb_to_int(np.array(Image.open(r["mask_path"]).convert("RGB")))

        x = ((rgb.astype(np.float32) / 255.0 - IM_MEAN) / IM_STD)
        x = torch.from_numpy(np.transpose(x, (2, 0, 1)))[None].to(device)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            pred = model(x).argmax(1)[0].cpu().numpy()

        axes[i, 0].imshow(rgb); axes[i, 0].set_title("input" if i == 0 else "")
        axes[i, 1].imshow(int_mask_to_rgb(gt)); axes[i, 1].set_title("ground truth" if i == 0 else "")
        axes[i, 2].imshow(int_mask_to_rgb(pred)); axes[i, 2].set_title("prediction" if i == 0 else "")
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

Artifacts saved to `runs/unet_imgonly_v1/`:
* `manifest.csv`, `class_weights.json`
* `metrics.csv` (per-epoch log)
* `best.pt` (best checkpoint by val mIoU)
* `test_metrics.json` (final test numbers)
* `confmat.png`, `sample_predictions.png`

Next steps:
1. Inspect `metrics.csv` and `sample_predictions.png` -- does the model look sane?
2. If yes, this becomes our **image-only ablation row** in the comparison table.
3. Then we add the LSTM branch and the fusion variants.""")

# =========================================================================
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
