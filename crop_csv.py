"""
Crop a 128x128 patch around every (pix_x, pix_y) in a CSV from its matching TIFF
and save each as a PNG. Edge pixels are zero-padded so output is always 128x128
with the labeled point at the center.

Output layout (lets you trace any PNG back to its CSV, TIFF, and row):
    outputs/{csv_stem}__{tif_stem}/
        row{N:06d}_pixx{X}_pixy{Y}_label{L}.png
        manifest.csv  (row_index, pix_x, pix_y, label, png_name)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

PROJECT = Path(r"C:\Users\Santosh\Desktop\0-100\Research Seminar\Project")
CSV_PATH = PROJECT / "IS2_Corrected_data" / "ATL03_20191104195311_05940510_T02CNA_gt1r_labeled_10m_done.csv"
TIF_PATH = PROJECT / "S2_tiff" / "S2_tiff" / "s2_vis_04_20191104T194529_20191104T194523_T02CNA.tif"
OUT_ROOT = PROJECT / "outputs"

PATCH = 128
HALF = PATCH // 2  # 64


def crop_patch(img_hwc: np.ndarray, pix_x: int, pix_y: int) -> np.ndarray:
    """Return a (PATCH, PATCH, C) crop centered at (pix_x, pix_y), zero-padded if needed."""
    H, W, C = img_hwc.shape
    row_start, row_end = pix_y - HALF, pix_y + HALF
    col_start, col_end = pix_x - HALF, pix_x + HALF

    rs, re = max(row_start, 0), min(row_end, H)
    cs, ce = max(col_start, 0), min(col_end, W)

    canvas = np.zeros((PATCH, PATCH, C), dtype=img_hwc.dtype)
    if rs < re and cs < ce:
        dst_r0 = rs - row_start
        dst_c0 = cs - col_start
        canvas[dst_r0:dst_r0 + (re - rs), dst_c0:dst_c0 + (ce - cs)] = img_hwc[rs:re, cs:ce]
    return canvas


def main():
    out_dir = OUT_ROOT / f"{CSV_PATH.stem}__{TIF_PATH.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV : {CSV_PATH.name}")
    print(f"TIFF: {TIF_PATH.name}")
    print(f"Out : {out_dir}")

    t0 = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV: {len(df)} rows in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    with rasterio.open(TIF_PATH) as src:
        arr = src.read()  # (bands, H, W)
    img_hwc = np.transpose(arr, (1, 2, 0))  # (H, W, bands)
    if img_hwc.shape[2] > 3:
        img_hwc = img_hwc[..., :3]
    print(f"Loaded TIFF: shape={img_hwc.shape}, dtype={img_hwc.dtype} in {time.perf_counter() - t0:.2f}s")

    manifest_rows = []
    t0 = time.perf_counter()
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        pix_x = int(row.pix_x)
        pix_y = int(row.pix_y)
        label = int(row.label)

        patch = crop_patch(img_hwc, pix_x, pix_y)
        png_name = f"row{i:06d}_pixx{pix_x}_pixy{pix_y}_label{label}.png"
        Image.fromarray(patch).save(out_dir / png_name)
        manifest_rows.append((i, pix_x, pix_y, label, png_name))

        if (i + 1) % 1000 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  {i + 1}/{n}  rate={rate:.1f} img/s  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    pd.DataFrame(manifest_rows, columns=["row_index", "pix_x", "pix_y", "label", "png_name"]).to_csv(
        out_dir / "manifest.csv", index=False
    )

    total = time.perf_counter() - t0
    print(f"Done. {n} patches in {total:.1f}s ({n / total:.1f} img/s). Output: {out_dir}")


if __name__ == "__main__":
    main()
