# Confusion Matrices (Green + Blue Colormaps)

Row-normalized (percentage) confusion matrices on the held-out test tile T03CWT, provided in both **green** and **blue** colormaps. The styling (title, "True Label" / "Predicted label" axes, colorbar, and large bold cell values) matches the paper figures; the only difference between the green and blue versions is the colormap. The numbers are identical.

| Model | mIoU | Green | Blue |
|---|:---:|---|---|
| Deep Fusion | 0.9010 | `deepfusion_green.png` | `deepfusion_blue.png` |
| Hybrid Fusion | 0.8891 | `hybridfusion_green.png` | `hybridfusion_blue.png` |
| Late Fusion | 0.8770 | `latefusion_green.png` | `latefusion_blue.png` |
| BiLSTM (photon only) | 0.6978 | `lstm_green.png` | `lstm_blue.png` |
| U-Net (optical only) | 0.8704 | `unet_green.png` | `unet_blue.png` |

All values are row-normalized percentages (each row sums to 100 %).

Source data:
- Deep Fusion: `runs/deep_fusion/test_metrics.json`
- Hybrid Fusion: `archive/runs/fusion_hybrid_unet_profstyle_v1/test_metrics.json`
- Late Fusion: `archive/runs/fusion_late_unet_profstyle_v1/test_metrics.json`
- BiLSTM / U-Net: `runs/bilstm/test_metrics.json` and the U-Net baseline figure
