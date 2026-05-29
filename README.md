# Multimodal Deep Fusion for Antarctic Sea Ice Classification

This repository contains the source code, trained-model artifacts, and documentation for a multimodal deep-learning framework that performs per-pixel classification of Antarctic sea ice into three classes (**thick ice**, **thin ice**, and **open water**) by fusing Sentinel-2 optical imagery with ICESat-2 ATL03 photon-altimetry data.

The framework combines a U-Net image branch (ResNet-18 encoder) with a recurrent photon branch and integrates the two modalities through a deep feature-level fusion stage. On a geographically held-out test tile (T03CWT), the fused model attains a mean Intersection-over-Union (mIoU) of **0.9010** and a macro-averaged F1 score of **0.9468**, improving over both unimodal baselines.

<p align="center">
  <img src="confusion_matrices/unet_green.png" width="310" alt="Confusion matrix – U-Net baseline"/>
  <img src="confusion_matrices/lstm_green.png" width="310" alt="Confusion matrix – LSTM baseline"/>
  <img src="confusion_matrices/deepfusion_green.png" width="310" alt="Confusion matrix – Deep Fusion (best model)"/>
</p>
<p align="center"><em>Left: U-Net &nbsp;|&nbsp; Centre: LSTM &nbsp;|&nbsp; Right: Deep Fusion</em></p>

---

## Key Results

All models are trained on tiles T02CNA and T02CNC and evaluated on the geographically separated tile T03CWT. The split is performed by tile rather than by random patches, providing a strict test of geographic generalization.

| Model | Input modality | Test mIoU | IoU (thick ice) | IoU (thin ice) | IoU (water) |
|:--|:--|:--:|:--:|:--:|:--:|
| U-Net | Sentinel-2 optical | 0.8704 | 0.9299 | 0.7683 | 0.9130 |
| LSTM | ICESat-2 photon | 0.6978 | 0.9671 | 0.5427 | 0.5836 |
| **Deep Fusion** | **optical + photon** | **0.9010** | **0.9403** | **0.8138** | **0.9489** |

The fusion model yields its largest improvements on the two minority classes, for which the image-only baseline is weakest: thin-ice IoU increases from 0.768 to 0.814 (+4.5 percentage points) and water IoU increases from 0.913 to 0.949 (+3.6 percentage points).

A detailed report with per-class precision/recall/F1, confusion matrices, training curves, and sample predictions is provided in [`project_summary.pdf`](project_summary.pdf).

---

## Repository Structure

```
.
├── deep_fusion.ipynb              Deep-fusion model (primary result)
├── lstm_sweep.ipynb               21-configuration LSTM hyperparameter sweep
├── requirements.txt               Python dependencies
├── project_summary.pdf            Technical report with figures
│
├── crop_all.py, crop_csv.py, crop_one_point.py    Patch extraction
├── segment_all.py, segment_one.py                 Ground-truth mask generation
│
├── runs/
│   ├── deep_fusion/               Deep-fusion outputs (mIoU 0.9010)
│   │   ├── test_metrics.json
│   │   ├── confmat.png
│   │   ├── loss_curve.png
│   │   └── summary_vs_all.csv
│   └── bilstm/                    Photon-only LSTM outputs (mIoU 0.6978)
│       ├── test_metrics.json
│       ├── confmat.png
│       └── metrics.csv
│
├── archive/                       Superseded experiments
│   ├── notebooks/                 Earlier fusion variants and baselines
│   └── runs/                      Per-run metrics for archived experiments
│
├── IS2_Corrected_data/            ICESat-2 ATL03 photon CSV files (input)
├── S2_tiff/                       Sentinel-2 GeoTIFF scenes (large; not tracked)
├── outputs/                       Extracted 128x128 RGB patches (not tracked)
├── outputs_segmented/             Ground-truth segmentation masks (not tracked)
└── papers/                        Reference literature
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

### 2. Train the photon-only LSTM

Execute the hyperparameter sweep, which trains the recurrent photon branch and selects the best configuration:

```bash
jupyter nbconvert --to notebook --execute lstm_sweep.ipynb
```

Alternatively, open `lstm_sweep.ipynb` in Jupyter and run all cells interactively. The selected configuration and its checkpoint are written under `runs/`.

### 3. Train the deep-fusion model

Execute the fusion notebook, which loads the trained LSTM branch, combines it with the U-Net image branch, and fine-tunes the complete model:

```bash
jupyter nbconvert --to notebook --execute deep_fusion.ipynb
```

The trained model, confusion matrix, loss curves, and evaluation metrics are written to `runs/deep_fusion/`. The final per-class metrics are recorded in `runs/deep_fusion/test_metrics.json`.

> **Note.** The notebooks are configured to use a single GPU. Set the device with the `CUDA_VISIBLE_DEVICES` environment variable (for example, `CUDA_VISIBLE_DEVICES=0`) before launching. One fusion training run of 30 epochs requires approximately 60–90 minutes on an RTX A6000.

---

## Methodology

The framework comprises two modality-specific branches whose representations are integrated through a deep fusion stage.

### Image branch (U-Net)

Each 128x128 RGB patch is processed by a U-Net with a ResNet-18 encoder pretrained on ImageNet. The decoder restores the original spatial resolution and produces a 16-channel feature map of shape (16, 128, 128).

### Photon branch (LSTM)

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

### Fusion stage

The photon feature vector is projected to 16 channels and broadcast across the spatial grid, then concatenated with the U-Net feature map to form a 32-channel tensor. A Squeeze-and-Excitation block (reduction ratio 8) performs channel-wise recalibration, and a final 1x1 convolution produces the three-class per-pixel logits. The pretrained photon branch is fine-tuned within the fusion model at one-tenth of the base learning rate, allowing it to adapt to the fusion context while retaining the representations learned during standalone training.

---

## Ablation Study

The following variants quantify the contribution of each design decision. All are evaluated on the held-out tile T03CWT.

| Variant | mIoU | Configuration |
|:--|:--:|:--|
| `fusion_v2` | 0.8020 | Photon branch trained from random initialization (unstable) |
| `fusion_v3` | 0.8949 | Higher-capacity recurrent branch, random initialization |
| `fusion_v4` | 0.8982 | Pretrained photon branch, frozen during fusion |
| **`deep_fusion`** | **0.9010** | Pretrained photon branch, fine-tuned at 0.1x learning rate |

The strongest result is obtained with the smaller pretrained-and-fine-tuned recurrent branch rather than the larger randomly initialized one, indicating that transferred representations contribute more than additional model capacity for this task. The archived variants and their metrics are available under `archive/`.

---

## Dataset

| Source | Description |
|:--|:--|
| Sentinel-2 Level-1C | Optical RGB imagery at 10-meter resolution over the Ross Sea region |
| ICESat-2 ATL03 | Geolocated photon point clouds aggregated to 10-meter along-track segments |
| Ground-truth masks | Per-pixel labels generated by an HSV color-thresholding pipeline with cloud and shadow removal |

Class encoding in the masks: red = thick ice, blue = thin ice, green = open water.

---

## Future Directions

### 1. SAR Modality Integration

The current framework relies on Sentinel-2 optical imagery, which is unavailable under cloud cover — a frequent occurrence over polar regions. A natural next step is to incorporate Sentinel-1 C-band synthetic aperture radar (SAR) backscatter as a third input modality. Unlike optical sensors, SAR penetrates clouds and operates independently of solar illumination, making it well-suited for year-round polar monitoring. At the architecture level, SAR features could be introduced through a third branch analogous to the photon branch, with its output projected and fused at the feature-concatenation stage alongside the U-Net and LSTM representations. Because SAR backscatter encodes surface roughness and dielectric properties, it carries complementary ice-structural information that may help resolve thin-ice and nilas categories that are spectrally ambiguous in optical bands. The primary challenge is co-registration: Sentinel-1 and Sentinel-2 acquisitions are not simultaneous, so temporal offsets must be accounted for during patch extraction.

### 2. Temporal Sequence Modeling

The current model treats each 128×128 patch as an independent snapshot, discarding the temporal context available from repeat satellite passes. Sentinel-2 revisits the same tile every five days and ICESat-2 follows a 91-day repeat cycle, making multi-date fusion a tractable extension. A temporal model could stack patches from several consecutive overpasses as additional input channels to the U-Net, or apply a convolutional LSTM across the time dimension to propagate spatial–temporal hidden states. This would allow the model to distinguish between ice classes that look similar in a single image but evolve differently over days or weeks — for example, new ice forming over open water versus persistent first-year ice. Beyond classification accuracy, temporal modeling opens the door to change-detection outputs: identifying pixels that transition between classes across acquisitions and quantifying the rate and spatial pattern of ice-cover change, which is directly relevant to climate-monitoring applications.

### 3. Geographic Transfer to the Arctic

All training and evaluation in this study used Ross Sea tiles (T02CNA, T02CNC, T03CWT). Antarctic and Arctic sea ice differ substantially in age distribution, surface roughness, melt-pond coverage, and sensor viewing geometry, so out-of-region generalization cannot be assumed. A systematic transfer study would evaluate the trained deep-fusion model in a zero-shot setting on labeled Arctic acquisitions and compare it with models fine-tuned on small Arctic target sets. If labeled Arctic data are scarce, domain-adaptation techniques — such as adversarial feature alignment or self-supervised pre-training on unlabeled Arctic imagery — could bridge the gap. Successful transfer would establish the framework as a general polar ice-classification tool rather than a region-specific one, increasing its utility for operational agencies such as the National Ice Center and the Norwegian Ice Service that monitor both hemispheres.

---

## Citation and Acknowledgments

This work was conducted as part of the Research Seminar at Knox College. We thank Prof. Iqrah for guidance throughout the project.

- ICESat-2 ATL03 products: NASA National Snow and Ice Data Center (NSIDC)
- Sentinel-2 imagery: ESA Copernicus Programme

A corresponding manuscript is in preparation. Please cite that work if you build upon this repository.

---

## Project Status

| Component | Status |
|:--|:--|
| Data preparation pipeline | Complete |
| U-Net optical baseline | Complete |
| LSTM photon baseline and hyperparameter sweep | Complete |
| Deep-fusion model and ablation study | Complete |
| Technical report (`project_summary.pdf`) | Complete |
| Manuscript | In preparation |

---

## Releases

No releases published.

---

## Packages

No packages published.
