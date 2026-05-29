# Project Summary: Multimodal Deep Fusion for Antarctic Sea Ice Classification

## Overview

This project develops a multimodal deep-learning framework for per-pixel classification of Antarctic sea ice into three categories: **thick ice**, **thin ice**, and **open water**. The framework fuses two complementary remote sensing modalities: Sentinel-2 optical imagery and ICESat-2 ATL03 photon-altimetry data.

The best model (Deep Fusion) achieves a mean Intersection-over-Union (mIoU) of **0.9010** and a macro-averaged F1 score of **0.9468** on the geographically held-out test tile T03CWT, outperforming both unimodal baselines.

---

## Research Context

Sea ice monitoring is critical for understanding climate change in polar regions. Manual classification is impractical at scale, so automated methods are essential. This project addresses the challenge by combining:

- **Visual texture** from Sentinel-2 optical imagery, which captures spatial context and reflectance patterns
- **Surface height profiles** from ICESat-2 ATL03 photon altimetry, which distinguishes ice types by elevation signature

The fusion approach improves especially on minority classes (thin ice and open water) where optical information alone is insufficient.

---

## Data Sources

| Source | Description | Spatial Resolution |
|:--|:--|:--|
| Sentinel-2 Level-1C | Optical RGB imagery over the Ross Sea region | 10 m |
| ICESat-2 ATL03 | Geolocated photon clouds aggregated to along-track segments | 10 m along-track |
| Ground-truth masks | Per-pixel labels via HSV color-thresholding with cloud and shadow removal | 10 m |

**Training tiles:** T02CNA and T02CNC  
**Test tile:** T03CWT (geographically separated, providing a strict test of generalization)

**Class encoding in masks:** red = thick ice, blue = thin ice, green = open water

---

## Model Architecture

### Image Branch (U-Net)

Each 128 x 128 RGB patch is processed by a U-Net with a ResNet-18 encoder pretrained on ImageNet. The decoder restores the original spatial resolution and produces a 16-channel feature map of shape (16, 128, 128).

### Photon Branch (LSTM)

ATL03 records are aggregated into 10-meter along-track segments. A sliding window of five consecutive segments forms a sequence, and eight engineered features are extracted per segment:

| Feature | Description |
|:--|:--|
| h_cor_mean | Mean corrected photon height |
| h_cor_med | Median corrected photon height |
| h_diff | Difference between mean and median height (within-segment asymmetry) |
| rel_height_min_elev | Mean height relative to the per-track minimum |
| height_sd | Standard deviation of photon heights |
| pcnth_mean | Mean photon-count height |
| pcnt_mean | Mean photon count |
| bcnt_mean, brate_mean | Mean background photon count and background rate |

The recurrent network uses a single-layer LSTM (hidden dimension 96, dropout 0.4) followed by fully connected layers and a softmax head. Training uses categorical focal loss (alpha = [0.05, 0.45, 0.60], gamma = 2.0) to prevent prediction collapse onto the dominant thick-ice class. The best configuration was selected by a 21-run hyperparameter sweep.

### Fusion Stage

The photon feature vector is projected to 16 channels and broadcast across the spatial grid. It is concatenated with the U-Net feature map to form a 32-channel tensor. A Squeeze-and-Excitation block (reduction ratio 8) performs channel-wise recalibration. A 1 x 1 convolution produces the three-class per-pixel logits. The pretrained photon branch is fine-tuned at one-tenth of the base learning rate, allowing adaptation to the fusion context while retaining representations learned during standalone training.

---

## LSTM Hyperparameter Sweep

A 21-configuration sweep determined the optimal photon branch settings. Parameters swept:

- Focal loss alpha weights: [0.02, 0.44, 0.54], [0.041, 0.409, 0.55], [0.05, 0.45, 0.5], [0.05, 0.5, 0.45]
- Focal loss gamma: 1.0, 2.0, 3.0, 5.0
- Hidden dimension: 32, 64, 96
- Learning rate: 5e-4, 8.9e-4, 1e-3, 2e-3
- Dropout: 0.2, 0.3, 0.4, 0.5
- Sequence length: 3, 5, 7
- Random seed: 7, 42, 123

**Winning configuration:** hidden = 96, dropout = 0.4, alpha = [0.05, 0.45, 0.60], gamma = 2.0, sequence length = 5, learning rate = 8.9e-4, seed = 42.

---

## Results

All models are trained on tiles T02CNA and T02CNC and evaluated on the geographically separated tile T03CWT.

### Per-Model Summary

| Model | Input Modality | mIoU | Pixel Accuracy | Macro F1 |
|:--|:--|:--:|:--:|:--:|
| U-Net | Sentinel-2 optical | 0.8704 | 0.9429 | N/A |
| LSTM | ICESat-2 photon | 0.6978 | 0.9594 | 0.8080 |
| **Deep Fusion** | **optical + photon** | **0.9010** | **0.9530** | **0.9468** |

### Per-Class IoU

| Model | Thick Ice | Thin Ice | Open Water |
|:--|:--:|:--:|:--:|
| U-Net | 0.9299 | 0.7683 | 0.9130 |
| LSTM | 0.9671 | 0.5427 | 0.5836 |
| **Deep Fusion** | **0.9403** | **0.8138** | **0.9489** |

The fusion model yields its largest improvements on the two minority classes: thin-ice IoU improves from 0.768 (U-Net) to 0.814 (+4.5 percentage points); water IoU improves from 0.913 to 0.949 (+3.6 percentage points).

### Deep Fusion Per-Class Precision, Recall, and F1

| Class | Precision | Recall | F1 |
|:--|:--:|:--:|:--:|
| Thick Ice | 0.9766 | 0.9620 | 0.9692 |
| Thin Ice | 0.8739 | 0.9221 | 0.8973 |
| Open Water | 0.9876 | 0.9604 | 0.9738 |

### LSTM Per-Class Precision, Recall, and F1

| Class | Precision | Recall | F1 |
|:--|:--:|:--:|:--:|
| Thick Ice | 0.9922 | 0.9745 | 0.9833 |
| Thin Ice | 0.6555 | 0.7592 | 0.7036 |
| Open Water | 0.6512 | 0.8489 | 0.7370 |

### Confusion Matrices

Row-normalized confusion matrices (percentage of true-class pixels predicted per class) on T03CWT:

<p align="center">
  <img src="confusion_matrices/unet_green.png" width="310" alt="U-Net confusion matrix"/>
  <img src="confusion_matrices/lstm_green.png" width="310" alt="LSTM confusion matrix"/>
  <img src="confusion_matrices/deepfusion_green.png" width="310" alt="Deep Fusion confusion matrix"/>
</p>
<p align="center"><em>Left: U-Net | Center: LSTM | Right: Deep Fusion</em></p>

---

## Ablation Study

The following variants quantify the contribution of key design decisions. All are evaluated on T03CWT.

| Variant | mIoU | Macro F1 | Configuration |
|:--|:--:|:--:|:--|
| fusion_v2 | 0.8020 | 0.8882 | Photon branch trained from random initialization (collapsed to dominant class) |
| fusion_v3 | 0.8949 | 0.9431 | Larger recurrent branch (hidden=128), random initialization |
| fusion_v4 | 0.8982 | 0.9453 | Pretrained photon branch, frozen during fusion training |
| **deep_fusion** | **0.9010** | **0.9468** | Pretrained photon branch, fine-tuned at 0.1x base learning rate |

**Key finding:** a smaller pretrained-and-fine-tuned recurrent branch (hidden=96) outperforms a larger randomly initialized branch (hidden=128), demonstrating that transferred representations contribute more than raw model capacity for this task.

---

## Key Findings

1. Deep fusion of optical and photon data achieves mIoU 0.9010, surpassing both unimodal baselines by a significant margin.
2. The largest gains are on minority classes: thin-ice IoU improves by +4.5 pp over the optical baseline; water IoU improves by +3.6 pp.
3. The LSTM photon branch alone struggles with thin ice (IoU 0.543) due to the sparse along-track coverage of ICESat-2, but provides strong height discrimination when fused with image features.
4. Transfer learning from the standalone photon branch is more valuable than increasing recurrent capacity.
5. Focal loss with asymmetric class weights is essential to prevent prediction collapse on the imbalanced ice-type distribution.

---

## Repository Structure

```
.
+-- deep_fusion.ipynb              Deep-fusion model (primary result)
+-- lstm_sweep.ipynb               21-configuration LSTM hyperparameter sweep
+-- requirements.txt               Python dependencies
+-- project_summary.pdf            Full technical report with figures
+-- project_summary.md             This document
|
+-- crop_all.py, crop_csv.py, crop_one_point.py    Patch extraction
+-- segment_all.py, segment_one.py                 Ground-truth mask generation
|
+-- confusion_matrices/            Green and blue confusion matrix PNGs for all three models
|
+-- runs/
|   +-- deep_fusion/               Deep-fusion outputs (mIoU 0.9010)
|   |   +-- test_metrics.json
|   |   +-- confmat.png
|   |   +-- loss_curve.png
|   |   \-- summary_vs_all.csv
|   \-- bilstm/                    Photon-only LSTM outputs (mIoU 0.6978)
|       +-- test_metrics.json
|       +-- confmat.png
|       \-- metrics.csv
|
+-- archive/                       Superseded experiments and earlier fusion variants
+-- notebook_output/               Intermediate outputs from exploratory notebooks
\-- papers/                        Reference literature
```

---

## Requirements

- Python 3.9 or later
- A CUDA-capable GPU with at least 10 GB of VRAM (experiments used an NVIDIA RTX A6000)
- Python packages listed in `requirements.txt`: torch, torchvision, segmentation-models-pytorch, numpy, pandas, pillow, matplotlib, scikit-learn, tqdm, jupyter, nbconvert

---

## Acknowledgments

This work was conducted as part of the Research Seminar at Knox College. We thank Prof. Iqrah for guidance throughout the project.

- ICESat-2 ATL03 products: NASA National Snow and Ice Data Center (NSIDC)
- Sentinel-2 imagery: ESA Copernicus Programme

A corresponding manuscript is in preparation.
