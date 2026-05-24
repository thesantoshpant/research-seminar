# Confusion Matrices (green + blue)

Row-normalized (percentage) confusion matrices on the held-out test tile
T03CWT, provided in both **green** and **blue** colormaps. The green
versions match the color scheme requested for the paper figures; the blue
versions are kept alongside for comparison. Numbers are identical between
the two colors — only the colormap and font size differ.

| Model | Green | Blue |
|---|---|---|
| Deep Fusion (winner) | `deepfusion_green.png` | `deepfusion_blue.png` |
| LSTM (photon only) | `lstm_green.png` | `lstm_blue.png` |
| U-Net (optical only) | `unet_green.png` | `unet_blue.png` |

All values are row-normalized percentages (each row sums to 100%).
Source data: `runs/fusion_winner/test_metrics.json`,
`runs/lstm_winner/test_metrics.json`, and the U-Net baseline figure.
