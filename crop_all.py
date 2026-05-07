"""
Crop 128x128 patches around every (pix_x, pix_y) in each of the 6 CSVs from
its matching TIFF and save as PNGs into outputs/.

Filename format: row{N}_{first_timestamp}_{tile}_{beam}.png
Example:         row5_20191104T194529_T02CNA_gt1r.png

Edge handling: if the 128x128 box extends past image bounds, the missing area
is zero-padded so the output is always 128x128 with the labeled point centered.

TIFFs are loaded once and reused across the two CSVs (gt1r, gt2r) that share
the same tile.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

PROJECT = Path(r"C:\Users\Santosh\Desktop\0-100\Research Seminar\Project")
CSV_DIR = PROJECT / "IS2_Corrected_data"
TIF_DIR = PROJECT / "S2_tiff" / "S2_tiff"
OUT_DIR = PROJECT / "outputs"
PATCH = 128
HALF = PATCH // 2  # 64

# Each job: one TIFF + the two CSVs (gt1r, gt2r) that map to it.
JOBS = [
    {
        "tif": "s2_vis_04_20191104T194529_20191104T194523_T02CNA.tif",
        "date": "20191104T194529",
        "tile": "T02CNA",
        "csvs": [
            ("ATL03_20191104195311_05940510_T02CNA_gt1r_labeled_10m_done.csv", "gt1r"),
            ("ATL03_20191104195311_05940510_T02CNA_gt2r_labeled_10m_done.csv", "gt2r"),
        ],
    },
    {
        "tif": "s2_vis_06_20191104T194529_20191104T194523_T02CNC.tif",
        "date": "20191104T194529",
        "tile": "T02CNC",
        "csvs": [
            ("ATL03_20191104195311_05940510_T02CNC_gt1r_labeled_10m_done.csv", "gt1r"),
            ("ATL03_20191104195311_05940510_T02CNC_gt2r_labeled_10m_done.csv", "gt2r"),
        ],
    },
    {
        "tif": "s2_vis_63_20191126T184459_20191126T184514_T03CWT.tif",
        "date": "20191126T184459",
        "tile": "T03CWT",
        "csvs": [
            ("ATL03_20191126182014_09290510_T03CWT_gt1r_labeled_10m_done.csv", "gt1r"),
            ("ATL03_20191126182014_09290510_T03CWT_gt2r_labeled_10m_done.csv", "gt2r"),
        ],
    },
]


def crop_patch(img_hwc: np.ndarray, pix_x: int, pix_y: int) -> np.ndarray:
    """128x128 RGB crop centered at (pix_x=col, pix_y=row), zero-padded at edges."""
    H, W, C = img_hwc.shape
    rs0, re0 = pix_y - HALF, pix_y + HALF  # exclusive end
    cs0, ce0 = pix_x - HALF, pix_x + HALF
    rs, re = max(rs0, 0), min(re0, H)
    cs, ce = max(cs0, 0), min(ce0, W)
    canvas = np.zeros((PATCH, PATCH, C), dtype=img_hwc.dtype)
    if rs < re and cs < ce:
        canvas[rs - rs0:(rs - rs0) + (re - rs), cs - cs0:(cs - cs0) + (ce - cs)] = img_hwc[rs:re, cs:ce]
    return canvas


def main():
    OUT_DIR.mkdir(exist_ok=True)
    grand_total = 0
    grand_t0 = time.perf_counter()

    for job in JOBS:
        tif_path = TIF_DIR / job["tif"]
        print(f"\n=== TIFF: {job['tif']} ===")
        t0 = time.perf_counter()
        with rasterio.open(tif_path) as src:
            arr = src.read()  # (bands, H, W)
        img = np.transpose(arr, (1, 2, 0))  # (H, W, bands)
        if img.shape[2] > 3:
            img = img[..., :3]
        print(f"  loaded in {time.perf_counter() - t0:.2f}s, shape={img.shape}, dtype={img.dtype}")

        for csv_name, beam in job["csvs"]:
            csv_path = CSV_DIR / csv_name
            df = pd.read_csv(csv_path, usecols=["pix_x", "pix_y"])
            n = len(df)
            xs = df["pix_x"].astype(int).values
            ys = df["pix_y"].astype(int).values
            print(f"  CSV {beam}: {n} rows -> writing PNGs...")

            t0 = time.perf_counter()
            prefix = f"_{job['date']}_{job['tile']}_{beam}.png"
            for i in range(n):
                patch = crop_patch(img, int(xs[i]), int(ys[i]))
                Image.fromarray(patch).save(OUT_DIR / f"row{i}{prefix}")
                if (i + 1) % 5000 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed
                    eta = (n - i - 1) / rate if rate > 0 else 0
                    print(f"    {i + 1}/{n}  rate={rate:.0f} img/s  eta={eta:.0f}s")
            elapsed = time.perf_counter() - t0
            print(f"    done {n} patches in {elapsed:.1f}s ({n / elapsed:.0f} img/s)")
            grand_total += n

    elapsed = time.perf_counter() - grand_t0
    print(f"\nALL DONE. {grand_total} patches in {elapsed:.1f}s ({grand_total / elapsed:.0f} img/s)")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
