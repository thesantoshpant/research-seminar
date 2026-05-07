"""
Single-point crop test: take first row of one CSV, crop a 128x128 patch
from the matching TIFF centered on (pix_x, pix_y), save as PNG.

Edge case: if the 128x128 box extends past the image bounds, the missing
area is zero-padded so the output stays 128x128 with the point centered.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

PROJECT = Path(r"C:\Users\Santosh\Desktop\0-100\Research Seminar\Project")
CSV_PATH = PROJECT / "IS2_Corrected_data" / "ATL03_20191104195311_05940510_T02CNA_gt1r_labeled_10m_done.csv"
TIF_PATH = PROJECT / "S2_tiff" / "S2_tiff" / "s2_vis_04_20191104T194529_20191104T194523_T02CNA.tif"
OUT_DIR = PROJECT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PATCH = 128
HALF = PATCH // 2  # 64

df = pd.read_csv(CSV_PATH)
row = df.iloc[0]
pix_x = int(row["pix_x"])  # column
pix_y = int(row["pix_y"])  # row
label = int(row["label"])

# Desired (possibly out-of-bounds) box in image coords
col_start = pix_x - HALF
col_end = pix_x + HALF  # exclusive
row_start = pix_y - HALF
row_end = pix_y + HALF  # exclusive

with rasterio.open(TIF_PATH) as src:
    H, W = src.height, src.width
    bands = src.count

    # Clamp the read window to image bounds
    rs = max(row_start, 0)
    re = min(row_end, H)
    cs = max(col_start, 0)
    ce = min(col_end, W)

    window = rasterio.windows.Window(cs, rs, ce - cs, re - rs)
    data = src.read(window=window)  # shape: (bands, h, w)

# Place the read region into a zero-padded 128x128 canvas at the right offset
canvas = np.zeros((bands, PATCH, PATCH), dtype=data.dtype)
dst_r0 = rs - row_start  # how far down inside the canvas the read region starts
dst_c0 = cs - col_start
canvas[:, dst_r0:dst_r0 + (re - rs), dst_c0:dst_c0 + (ce - cs)] = data

# Convert (bands, H, W) -> (H, W, bands) for PIL; assume 3-band RGB
img = np.transpose(canvas, (1, 2, 0))
if img.shape[2] == 1:
    pil = Image.fromarray(img[..., 0])
else:
    pil = Image.fromarray(img[..., :3])

out_name = f"{CSV_PATH.stem}_idx0_pixx{pix_x}_pixy{pix_y}_label{label}.png"
out_path = OUT_DIR / out_name
pil.save(out_path)

print(f"TIFF size: {W} x {H} (cols x rows), bands={bands}")
print(f"Center pixel: pix_x={pix_x}, pix_y={pix_y}, label={label}")
print(f"Desired box: rows [{row_start}, {row_end}), cols [{col_start}, {col_end})")
print(f"Read window: rows [{rs}, {re}), cols [{cs}, {ce})")
print(f"Canvas offset: dst_r0={dst_r0}, dst_c0={dst_c0}")
print(f"Saved: {out_path}")
