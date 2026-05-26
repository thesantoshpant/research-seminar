# Confusion Matrices (green + blue)

Row-normalized (percentage) confusion matrices on the held-out test tile
T03CWT, provided in both **green** and **blue** colormaps. The styling
(title, "True Label" / "Predicted label" axes, colorbar, and large bold
cell values) matches the paper figures; the only difference between the
green and blue versions is the colormap. The numbers are identical.

| Model | Green | Blue |
|---|---|---|
| Deep Fusion | `deepfusion_green.png` | `deepfusion_blue.png` |
| BiLSTM (photon only) | `lstm_green.png` | `lstm_blue.png` |
| U-Net (optical only) | `unet_green.png` | `unet_blue.png` |

All values are row-normalized percentages (each row sums to 100%).
Source data: `runs/deep_fusion/test_metrics.json`,
`runs/bilstm/test_metrics.json`, and the U-Net baseline figure.
