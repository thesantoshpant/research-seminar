# Multimodal Fusion Strategies for Antarctic Sea Ice Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![mIoU](https://img.shields.io/badge/mIoU-0.8896-success)](#)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.9468-success)](#)
[![License](https://img.shields.io/badge/License-Academic-blue)](#)

## Authors

| Name | Role |
|:--|:--|
| **Chau Tran** | Model Development & Technical Writing |
| **Dipsha Budhathoki** | Research Writing & Literature Review |
| **Santosh Pant** | Model Implementation |
| **Arito Nakamichi** | Figure Design & Visualization |
| **Professor Iqrah** | Faculty Advisor |

---

This repository contains the source code, trained-model artifacts, and documentation for a multimodal deep-learning framework that performs per-pixel classification of Antarctic sea ice into three classes (**thick ice**, **thin ice**, and **open water**) by fusing Sentinel-2 optical imagery with ICESat-2 ATL03 photon-altimetry data.

Three fusion strategies are evaluated: **Late Fusion**, **Hybrid Fusion**, and **Deep Fusion**. Each integrates the two modalities at a different level of abstraction. Deep feature-level fusion achieves the best result, attaining a mean Intersection-over-Union (mIoU) of **0.8896** and a macro-averaged F1 score of **0.9468** on a geographically held-out test tile (T03CWT), improving over both unimodal baselines and the other two fusion approaches.

<table align="center">
  <tr>
    <td align="center">
      <img src="confusion_matrices/unet_green.png" width="300" alt="U-Net Confusion Matrix"/>
      <br/><b>(a) U-Net</b>
    </td>
    <td align="center">
      <img src="confusion_matrices/lstm_green.png" width="300" alt="BiLSTM Confusion Matrix"/>
      <br/><b>(b) BiLSTM</b>
    </td>
  </tr>
</table>

*Fig. 1: Row-normalized confusion matrices for the two unimodal baseline models evaluated on the geographically held-out tile T03CWT.*

<table align="center">
  <tr>
    <td align="center">
      <img src="confusion_matrices/deepfusion_green.png" width="300" alt="Deep Fusion Confusion Matrix"/>
      <br/><b>(a) Deep Fusion</b>
    </td>
    <td align="center">
      <img src="confusion_matrices/latefusion_green.png" width="300" alt="Late Fusion Confusion Matrix"/>
      <br/><b>(b) Late Fusion</b>
    </td>
    <td align="center">
      <img src="confusion_matrices/hybridfusion_green.png" width="300" alt="Hybrid Fusion Confusion Matrix"/>
      <br/><b>(c) Hybrid Fusion</b>
    </td>
  </tr>
</table>

*Fig. 2: Row-normalized confusion matrices for the three multimodal fusion strategies evaluated on the geographically held-out tile T03CWT.*

---

## Key Results

All models are trained on tiles T02CNA and T02CNC and evaluated on the geographically separated tile T03CWT. The split is performed by tile rather than by random patches, providing a strict test of geographic generalization.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|:--|:--:|:--:|:--:|:--:|
| U-Net | 0.9429 | 0.9402 | 0.9188 | 0.9291 |
| BiLSTM | 0.9594 | 0.7663 | 0.8609 | 0.8080 |
| Late Fusion | 0.9460 | 0.9504 | 0.9171 | 0.9329 |
| Hybrid Fusion | 0.9509 | 0.9449 | 0.9355 | 0.9401 |
| **Deep Fusion** | **0.9530** | **0.9460** | **0.9481** | **0.9468** |

### Per-Class IoU

| Model | Input Modality | Test mIoU | IoU (Ice) | IoU (Thin Ice) | IoU (Open Water) | Macro F1 |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| U-Net | Sentinel-2 optical | 0.8704 | 0.9299 | 0.7683 | 0.9130 | 0.9291 |
| BiLSTM | ICESat-2 photon | 0.6978 | 0.9671 | 0.5427 | 0.5836 | 0.8080 |
| Late Fusion | optical + photon | 0.8770 | 0.9334 | 0.7785 | 0.9189 | 0.9329 |
| Hybrid Fusion | optical + photon | 0.8891 | 0.9396 | 0.7988 | 0.9290 | 0.9401 |
| **Deep Fusion** | **optical + photon** | **0.8896** | **0.9383** | **0.7962** | **0.9344** | **0.9468** |

Deep Fusion achieves the highest overall mIoU and the largest gain in open water (+2.1 pp over U-Net). Hybrid Fusion achieves the largest improvement in thin-ice segmentation (+3.0 pp over U-Net), ranking closely behind Deep Fusion overall. Both fusion models surpass all unimodal baselines.

A detailed report with per-class precision/recall/F1, confusion matrices, training curves, and sample predictions is provided in [`project_summary.pdf`](project_summary.pdf).

---

## Fusion Strategy Comparison

### Architectures

All three strategies share the same two modality-specific branches (a U-Net image branch and a recurrent photon branch) but differ in **where and how** their representations are combined.

| Strategy | Integration Level | Mechanism |
|:--|:--|:--|
| Late Fusion | Decision level | Softmax predictions from each branch are averaged pixel-wise |
| Hybrid Fusion | Feature + Decision level | Intermediate U-Net features are concatenated with photon embeddings; a second prediction head is also averaged at the decision level |
| Deep Fusion | Deep feature level | The photon embedding is projected to 16 channels, broadcast spatially, concatenated with the U-Net decoder feature map, recalibrated by a Squeeze-and-Excitation block, and classified by a 1×1 convolution |

### Model Configurations

| Component | U-Net | Bi-LSTM | Deep Fusion | Hybrid Fusion | Late Fusion |
|:--|:--:|:--:|:--:|:--:|:--:|
| Input modality | Optical image | ATL03 features | Image + ATL03 features | Image + ATL03 features | Image + ATL03 features |
| Image encoder | U-Net ResNet-18 | — | U-Net ResNet-18 | U-Net ResNet-18 | U-Net ResNet-18 |
| Encoder weights | ImageNet | — | ImageNet | ImageNet | ImageNet |
| Input channels | 3 | — | 3 + ATL03 | 3 + ATL03 | 3 + ATL03 |
| Number of classes | 3 | 3 | 3 | 3 | 3 |
| Sequence length | — | 5 | 5 | 5 | 5 |
| LSTM hidden dim | — | 96 | 96 | 48 | 48 |
| LSTM layers | — | 1 | 1 | 1 | 1 |
| LSTM direction | — | Unidirectional | Unidirectional | Unidirectional | Unidirectional |
| LSTM dropout | — | 0.4 | 0.4 | 0.4 | 0.4 |
| Head hidden dim | — | 16 | 16 | 16 | 16 |
| Head dropout | — | 0.4 | 0.4 | 0.4 | 0.4 |
| Fusion channels | — | — | 16 | 16 | 16 |
| Fusion strategy | — | — | Feature-level fusion | Feature + decision fusion | Decision-level fusion |
| Fusion block | — | — | SE attention | SE attention + learned blending | Learned logit blending |
| SE reduction | — | — | 8 | 8 | — |
| Optimizer | AdamW | Adam | Adam | AdamW | AdamW |
| Learning rate | 1e-4 | 8.886e-4 | 1e-4 / 1e-5 | 1e-4 | 1e-4 |
| Weight decay | 1e-4 | — | 1e-4 | 1e-4 | 1e-4 |
| Batch size | 64 | 32 | 32 | 32 | 32 |
| Max epochs | 30 | 50 | 30 | 30 | 30 |
| Early stopping patience | 5 | — | 8 | 7 | 7 |
| Loss function | Weighted CE | Focal loss | Focal loss | Focal loss | Focal loss |
| Focal alpha | — | [0.05, 0.45, 0.60] | [0.05, 0.45, 0.60] | [0.05, 0.45, 0.60] | [0.05, 0.45, 0.60] |
| Focal gamma | — | 2.0 | 2.0 | 2.0 | 2.0 |
| LR scheduler | Cosine annealing | — | Cosine annealing | Cosine annealing | Cosine annealing |
| Mixed precision | AMP | — | AMP | AMP | AMP |
| Gradient clipping | — | — | — | 1.0 | 1.0 |

### Performance Analysis

**Late Fusion (mIoU 0.8770)** is the most modular approach: each branch is trained and evaluated independently, and their outputs are combined only at inference time. While straightforward to implement and debug, late fusion cannot capture cross-modal feature interactions; the two branches remain "unaware" of each other during training and during intermediate computations. This limits its ability to learn complementary representations.

| Metric | Value |
|:--|:--:|
| Accuracy / Pixel Accuracy | 0.9460 |
| Precision | 0.9504 |
| Recall | 0.9171 |
| F1-score | 0.9329 |
| mIoU | 0.8770 |

| Class | IoU | Precision | Recall | F1 |
|:--|:--:|:--:|:--:|:--:|
| Ice | 0.9334 | 0.9519 | 0.9796 | 0.9656 |
| Thin Ice | 0.7785 | 0.9149 | 0.8393 | 0.8755 |
| Water | 0.9189 | 0.9845 | 0.9324 | 0.9577 |

**Hybrid Fusion (mIoU 0.8891)** improves on late fusion by introducing a mid-network fusion path: photon embeddings are injected into the U-Net decoder at an intermediate spatial scale, allowing the image branch to condition its feature extraction on photon cues. The retained decision-level averaging provides a regularization effect. The result is a +1.2 pp mIoU gain over late fusion, with the largest improvement in thin-ice IoU (+2.0 pp).

| Metric | Value |
|:--|:--:|
| Accuracy / Pixel Accuracy | 0.9509 |
| Precision | 0.9449 |
| Recall | 0.9355 |
| F1-score | 0.9461 |
| mIoU | 0.8891 |

| Class | IoU | Precision | Recall | F1 |
|:--|:--:|:--:|:--:|:--:|
| Ice | 0.9396 | 0.9634 | 0.9744 | 0.9639 |
| Thin Ice | 0.7988 | 0.9043 | 0.8725 | 0.8881 |
| Water | 0.9290 | 0.9669 | 0.9595 | 0.9632 |

**Deep Fusion (mIoU 0.8896)** achieves the best result by fusing modalities entirely at the deep feature level and removing the late-fusion averaging path. Broadcasting the photon feature vector across the full 128×128 spatial grid allows every pixel to be informed by the along-track altimetry reading. The Squeeze-and-Excitation recalibration then selectively amplifies photon-consistent channels. Fine-tuning the pretrained photon branch at one-tenth the base learning rate within the fusion model allows the branch to adapt to the fusion context while retaining transferred representations.

| Metric | Value |
|:--|:--:|
| Accuracy / Pixel Accuracy | 0.9502 |
| Precision | 0.9484 |
| Recall | 0.9325 |
| F1-score | 0.9402 |
| mIoU | 0.8896 |

| Class | IoU | Precision | Recall | F1 |
|:--|:--:|:--:|:--:|:--:|
| Ice | 0.9383 | 0.9615 | 0.9750 | 0.9682 |
| Thin Ice | 0.7962 | 0.9029 | 0.8689 | 0.8865 |
| Water | 0.9344 | 0.9787 | 0.9537 | 0.9661 |

### Per-Class Breakdown

| Class | U-Net | BiLSTM | Late Fusion | Hybrid Fusion | Deep Fusion |
|:--|:--:|:--:|:--:|:--:|:--:|
| Ice IoU | 0.9299 | 0.9671 | 0.9334 | **0.9396** | 0.9383 |
| Thin Ice IoU  | 0.7683 | 0.5427 | 0.7785 | **0.7988** | 0.7962 |
| Open Water IoU | 0.9130 | 0.5836 | 0.9189 | 0.9290 | **0.9344** |

Thin ice is the most challenging class across all models. Hybrid Fusion achieves the best thin-ice IoU (+3.0 pp over U-Net), while Deep Fusion leads on open water (+2.1 pp over U-Net) and attains the highest overall mIoU.

### Key Takeaways

- **Feature-level fusion outperforms decision-level fusion**: Combining modalities before the final classifier consistently improves performance.
- **Spatial broadcast of photon features is effective**: Broadcasting the per-point photon embedding over the full spatial grid allows altimetry to inform every pixel, not just photon-footprint pixels.
- **Pretrained-and-fine-tuned photon branch beats frozen or random initialization**: Transferred representations from standalone LSTM training provide a better starting point than random weights, even with a smaller model capacity.
- **All fusion strategies outperform both unimodal baselines**: Even the weakest fusion variant (late fusion) surpasses the U-Net optical baseline in mIoU.

---

## Repository Structure

```
.
├── deep_fusion.ipynb                  Deep-fusion model (primary result)
├── fusion_late_unet.ipynb             Late-fusion model
├── fusion_hybrid_unet.ipynb           Hybrid-fusion model
├── lstm_sweep.ipynb                   21-configuration LSTM hyperparameter sweep
├── requirements.txt                   Python dependencies
├── project_summary.pdf                Technical report with figures
├── results_summary_public.pdf         Public summary
│
├── crop_all.py, crop_csv.py, crop_one_point.py    Patch extraction
├── segment_all.py, segment_one.py                 Ground-truth mask generation
│
├── confusion_matrices/                Row-normalized confusion matrices (green + blue colormaps)
│   ├── deepfusion_green.png / deepfusion_blue.png
│   ├── latefusion_green.png  / latefusion_blue.png
│   ├── hybridfusion_green.png / hybridfusion_blue.png
│   ├── unet_green.png        / unet_blue.png
│   ├── lstm_green.png        / lstm_blue.png
│   └── README.md
│
├── runs/
│   ├── deep_fusion/                   Deep-fusion outputs (mIoU 0.8896)
│   │   ├── test_metrics.json
│   │   ├── confmat.png
│   │   ├── loss_curve.png
│   │   └── summary_vs_all.csv
│   └── bilstm/                        Photon-only BiLSTM outputs (mIoU 0.6978)
│       ├── test_metrics.json
│       ├── confmat.png
│       └── metrics.csv
│
├── archive/                           Superseded experiments
│   ├── notebooks/                     Earlier fusion variants and baselines
│   └── runs/                          Per-run metrics for archived experiments
│
├── notebook_output/                   Auxiliary notebook outputs
│   └── atl03_lstm_data_preparation_2025.ipynb
│
├── IS2_Corrected_data/                ICESat-2 ATL03 photon CSV files (input)
├── S2_tiff/                           Sentinel-2 GeoTIFF scenes (large; not tracked)
├── outputs/                           Extracted 128×128 RGB patches (not tracked)
├── outputs_segmented/                 Ground-truth segmentation masks (not tracked)
└── papers/                            Reference literature
```

Large datasets, model checkpoints (`*.pt`), and intermediate caches are excluded from version control via `.gitignore` and must be regenerated or supplied locally.

---

## Requirements

- Python 3.9 or later
- A CUDA-capable GPU with at least 10 GB of memory (the reported experiments used an NVIDIA RTX A6000)
- The Python packages listed in [`requirements.txt`](requirements.txt):

```
torch
torchvision
segmentation-models-pytorch
numpy
pandas
pillow
matplotlib
scikit-learn
tqdm
jupyter
nbconvert
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Santoshpant23/research-seminar.git
cd research-seminar

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

For GPU acceleration, install the build of PyTorch that matches your CUDA version, following the official instructions at https://pytorch.org/get-started/locally/.

---

## Usage

The pipeline runs in three stages: data preparation, photon-branch training, and fusion-model training. Each notebook defines its input and output paths in a configuration cell near the top; adjust these to match your environment before execution.

### 1. Prepare the dataset

Place the ICESat-2 ATL03 photon CSV files in `IS2_Corrected_data/` and the Sentinel-2 GeoTIFF scenes in `S2_tiff/`, then extract aligned image patches and generate the corresponding segmentation masks:

```bash
python crop_all.py          # extract 128x128 RGB patches centered on labeled points
python segment_all.py       # generate per-pixel ground-truth masks
```

This produces the paired patches and masks under `outputs/` and `outputs_segmented/`.

### 2. Train the photon-only BiLSTM

Execute the hyperparameter sweep, which trains the recurrent photon branch and selects the best configuration:

```bash
jupyter nbconvert --to notebook --execute lstm_sweep.ipynb
```

Alternatively, open `lstm_sweep.ipynb` in Jupyter and run all cells interactively. The selected configuration and its checkpoint are written under `runs/`.

### 3. Train the fusion models

Execute any of the three fusion notebooks to train and evaluate that strategy:

```bash
# Deep Fusion (best result)
jupyter nbconvert --to notebook --execute deep_fusion.ipynb

# Late Fusion
jupyter nbconvert --to notebook --execute fusion_late_unet.ipynb

# Hybrid Fusion
jupyter nbconvert --to notebook --execute fusion_hybrid_unet.ipynb
```

Outputs (confusion matrices, loss curves, metrics) are written to the respective `runs/` subdirectory. The final per-class metrics are recorded in `test_metrics.json`.

> **Note.** The notebooks are configured to use a single GPU. Set the device with the `CUDA_VISIBLE_DEVICES` environment variable (for example, `CUDA_VISIBLE_DEVICES=0`) before launching. One fusion training run of 30 epochs requires approximately 60–90 minutes on an RTX A6000.

---

## Methodology

The framework comprises two modality-specific branches whose representations are integrated through one of three fusion strategies.

### Image Branch (U-Net)

Each 128×128 RGB patch is processed by a U-Net with a ResNet-18 encoder pretrained on ImageNet. The decoder restores the original spatial resolution and produces a 16-channel feature map of shape (16, 128, 128).

### Photon Branch (BiLSTM)

The ATL03 records are aggregated into 10-meter along-track segments. For each labeled location, a sliding window of five consecutive segments (the center segment and two neighbors on each side) is formed, and eight engineered features are extracted per segment:

| Feature | Description |
|:--|:--|
| `h_cor_mean` | Mean corrected photon height |
| `h_cor_med` | Median corrected photon height |
| `h_diff` | Difference between mean and median height (within-segment asymmetry) |
| `rel_height_min_elev` | Mean height relative to the per-track minimum |
| `height_sd` | Standard deviation of photon heights |
| `pcnth_mean` | Mean photon-count height |
| `pcnt_mean` | Mean photon count |
| `bcnt_mean`, `brate_mean` | Mean background photon count and background rate |

The sequence is processed by a single-layer recurrent network (hidden dimension 96, dropout 0.4) followed by fully connected layers and a softmax classification head. The branch is trained with categorical focal loss (alpha = [0.05, 0.45, 0.60], gamma = 2.0), which prevents the model from collapsing onto the dominant thick-ice class. The configuration was selected by a 21-run sweep over the loss weights, gamma, hidden dimension, learning rate, dropout, sequence length, and random seed.

### Fusion Stage

Refer to the [Fusion Strategy Comparison](#fusion-strategy-comparison) section above for per-strategy details. In all cases the pretrained photon branch is fine-tuned within the fusion model at one-tenth of the base learning rate, allowing it to adapt to the fusion context while retaining the representations learned during standalone training.

---

## Ablation Study

Throughout development we explored different model architectures, loss configurations, and hyperparameter settings before arriving at the final design. The variants stored under `archive/` represent these earlier iterations; the current notebooks outside the archive folder reflect the final configurations used to produce the reported results.

---

## Dataset

| Source | Description |
|:--|:--|
| Sentinel-2 Level-1C | Optical RGB imagery at 10-meter resolution over the Ross Sea region |
| ICESat-2 ATL03 | Geolocated photon point clouds aggregated to 10-meter along-track segments |
| Ground-truth masks | Per-pixel labels generated by an HSV color-thresholding pipeline with cloud and shadow removal |

Class encoding in the masks: red = thick ice, blue = thin ice, green = open water.

---

## Citation and Acknowledgments

This work was conducted as part of the Research Seminar at Knox College. We thank Prof. Iqrah for guidance throughout the project.

- ICESat-2 ATL03 products: NASA National Snow and Ice Data Center (NSIDC)
- Sentinel-2 imagery: ESA Copernicus Programme

A corresponding manuscript is in preparation. Please cite that work if you build upon this repository.

---

## Future Directions

### 1. SAR Modality Integration

The current framework relies on Sentinel-2 optical imagery, which is unavailable under cloud cover, a frequent occurrence over polar regions. A natural next step is to incorporate Sentinel-1 C-band synthetic aperture radar (SAR) backscatter as a third input modality. Unlike optical sensors, SAR penetrates clouds and operates independently of solar illumination, making it well-suited for year-round polar monitoring. At the architecture level, SAR features could be introduced through a third branch analogous to the photon branch, with its output projected and fused at the feature-concatenation stage alongside the U-Net and LSTM representations. Because SAR backscatter encodes surface roughness and dielectric properties, it carries complementary ice-structural information that may help resolve thin-ice and nilas categories that are spectrally ambiguous in optical bands.

### 2. Temporal Sequence Modeling

The current model treats each 128×128 patch as an independent snapshot, discarding the temporal context available from repeat satellite passes. Sentinel-2 revisits the same tile every five days and ICESat-2 follows a 91-day repeat cycle, making multi-date fusion a tractable extension. A temporal model could stack patches from several consecutive overpasses as additional input channels to the U-Net, or apply a convolutional LSTM across the time dimension to propagate spatial-temporal hidden states. This would allow the model to distinguish between ice classes that look similar in a single image but evolve differently over days or weeks. Beyond classification accuracy, temporal modeling opens the door to change-detection outputs: identifying pixels that transition between classes across acquisitions and quantifying the rate and spatial pattern of ice-cover change.

### 3. Geographic Transfer to the Arctic

All training and evaluation in this study used Ross Sea tiles (T02CNA, T02CNC, T03CWT). Antarctic and Arctic sea ice differ substantially in age distribution, surface roughness, melt-pond coverage, and sensor viewing geometry, so out-of-region generalization cannot be assumed. A systematic transfer study would evaluate the trained model in a zero-shot setting on labeled Arctic acquisitions and compare it with models fine-tuned on small Arctic target sets. Successful transfer would establish the framework as a general polar ice-classification tool rather than a region-specific one, increasing its utility for operational agencies that monitor both hemispheres.

---

## Project Status

| Component | Status |
|:--|:--|
| Data preparation pipeline | Complete |
| U-Net optical baseline | Complete |
| BiLSTM photon baseline and hyperparameter sweep | Complete |
| Late-fusion model | Complete |
| Hybrid-fusion model | Complete |
| Deep-fusion model and ablation study | Complete |
| Technical report (`project_summary.pdf`) | Complete |
| Manuscript | In preparation |
