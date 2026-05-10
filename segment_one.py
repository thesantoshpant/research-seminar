"""
Single-image test for the professor's shadow/cloud removal + segmentation.
Reads ONE PNG from outputs/, runs the pipeline, and saves ONLY seg_res
to outputs_segmented/ with the same filename.

The pipeline functions (color_segmentation, shadow_cloud_removal) are kept
identical to the professor's code, except:
  - shadow_cloud_removal returns only seg_res (no plotting, no extra returns)
"""

from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(r"C:\Users\Santosh\Desktop\0-100\Research Seminar\Project")
IN_DIR = PROJECT / "outputs"
OUT_DIR = PROJECT / "outputs_segmented"
OUT_DIR.mkdir(exist_ok=True)

# Pick any image to test with — change this filename to try a different one.
TEST_IMG = "row0_20191104T194529_T02CNA_gt1r.png"


def color_segmentation(img):
    # Get a "mask" over the image for each pixel
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_ice = (0, 0, 205)
    upper_ice = (185, 255, 255)
    mask_ice = cv2.inRange(hsv_img, lower_ice, upper_ice)

    lower_tice = (0, 0, 31)
    upper_tice = (185, 255, 204)
    mask_tice = cv2.inRange(hsv_img, lower_tice, upper_tice)

    lower_water = (0, 0, 0)
    upper_water = (185, 255, 30)
    mask_water = cv2.inRange(hsv_img, lower_water, upper_water)

    seg_img = img.copy()
    seg_img[mask_ice == 255] = [255, 0, 0]
    seg_img[mask_tice == 255] = [0, 0, 255]
    seg_img[mask_water == 255] = [0, 255, 0]

    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    return seg_img


def shadow_cloud_removal(ori):
    ### separate open water
    lower_water = (0, 0, 0)
    upper_water = (185, 255, 30)
    hsv_img = cv2.cvtColor(ori, cv2.COLOR_RGB2HSV)
    mask_water = cv2.inRange(hsv_img, lower_water, upper_water)

    without_water_img = ori.copy()
    without_water_img[mask_water == 255] = [255, 255, 255]

    img = cv2.cvtColor(without_water_img, cv2.COLOR_RGB2GRAY)

    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 155)
    diff_img = 255 - cv2.absdiff(img, bg_img)

    ret2, outs2 = cv2.threshold(
        src=diff_img, thresh=0, maxval=255,
        type=cv2.THRESH_OTSU + cv2.THRESH_BINARY,
    )
    diff_img2 = cv2.bitwise_and(diff_img, outs2)

    norm_img = cv2.normalize(
        diff_img2, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1,
    )
    _, thr_img = cv2.threshold(norm_img, 235, 0, cv2.THRESH_TRUNC)
    thr_img = cv2.normalize(
        thr_img, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1,
    )

    ### separate thin and old ice
    old_thin_ice = cv2.cvtColor(thr_img, cv2.COLOR_GRAY2RGB)
    hsv_img = cv2.cvtColor(old_thin_ice, cv2.COLOR_RGB2HSV)

    lower_tice = (0, 0, 0)
    upper_tice = (185, 255, 204)
    mask_tice = cv2.inRange(hsv_img, lower_tice, upper_tice)

    lower_ice = (0, 0, 205)
    upper_ice = (185, 255, 255)
    mask_ice = cv2.inRange(hsv_img, lower_ice, upper_ice)
    mask_ice = cv2.bitwise_xor(mask_water, mask_ice)

    shadow_free = old_thin_ice.copy()
    shadow_free[mask_ice == 255] = [255, 255, 255]
    shadow_free[mask_tice == 255] = [155, 155, 155]
    shadow_free[mask_water == 255] = [0, 0, 0]
    shadow_free = cv2.cvtColor(shadow_free, cv2.COLOR_BGR2RGB)

    # final segmentation on the shadow-free image — this is what we save
    seg_res = color_segmentation(shadow_free)
    return seg_res


def main():
    in_path = IN_DIR / TEST_IMG
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    # cv2.imread gives BGR, which is what the professor's reference pipeline
    # is fed (matches the original `ori = cv2.imread(...)` call).
    ori = cv2.imread(str(in_path))
    if ori is None:
        raise RuntimeError(f"cv2 failed to read: {in_path}")

    seg_res = shadow_cloud_removal(ori)

    out_path = OUT_DIR / TEST_IMG
    cv2.imwrite(str(out_path), seg_res)
    print(f"Input : {in_path}")
    print(f"Saved : {out_path}")


if __name__ == "__main__":
    main()
