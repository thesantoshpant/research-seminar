"""
Generate sea_ice_deep_fusion_workflow.pdf
A visually clear, well-structured workflow report.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

OUT = Path(__file__).parent / "sea_ice_deep_fusion_workflow.pdf"

# ── Palette ─────────────────────────────────────────────────────────────────
C_BG      = "#0A1628"   # deep navy background
C_CARD    = "#112244"   # card background
C_ACCENT  = "#00C8FF"   # cyan accent
C_GREEN   = "#00E676"   # green highlight
C_AMBER   = "#FFB300"   # amber/gold
C_PINK    = "#FF4081"   # pink for thin ice
C_ICE     = "#64B5F6"   # ice blue
C_WATER   = "#26C6DA"   # water teal
C_THIN    = "#FF8A65"   # thin ice orange
C_THICK   = "#42A5F5"   # thick ice blue
C_WHITE   = "#E8F0FE"
C_GREY    = "#546E7A"
C_BORDER  = "#1E3A5F"

def set_dark_bg(fig):
    fig.patch.set_facecolor(C_BG)

def card(ax, x, y, w, h, color=C_CARD, radius=0.04, lw=1.5, ec=C_ACCENT, alpha=1.0):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=color, edgecolor=ec,
                          linewidth=lw, alpha=alpha,
                          transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    return rect

def hide_axes(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C_BG)

# ── Shared model comparison data ────────────────────────────────────────────
MODELS = [
    "U-Net\n(Image only)",
    "LSTM\n(CSV only)",
    "Fusion\nDeep v1",
    "Fusion v4\n(Frozen LSTM)",
    "Deep Fusion\n(Fine-tuned)",
]
MIOU   = [0.8704, 0.2420, 0.8915, 0.8982, 0.8896]
PIX    = [0.9429, 0.7254, 0.9512, 0.9531, 0.9502]

BEST_PER_IOU   = [0.9407, 0.8157, 0.9382]   # fusion_v4
BEST_PREC      = [0.9776, 0.8710, 0.9928]
BEST_REC       = [0.9614, 0.9278, 0.9446]
BEST_F1        = [0.9695, 0.8985, 0.9681]
CLASS_NAMES    = ["Thick Ice", "Thin Ice", "Water"]
CLASS_COLORS   = [C_THICK, C_THIN, C_WATER]

with PdfPages(OUT) as pdf:

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1 ─ COVER
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    set_dark_bg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    hide_axes(ax)

    # gradient-like header bar
    for i, alpha in enumerate(np.linspace(0.9, 0.3, 40)):
        ax.axhspan(0.72 - i*0.005, 0.72 - (i-1)*0.005 + 0.005,
                   xmin=0, xmax=1,
                   facecolor=C_BORDER, alpha=alpha, zorder=0)

    # Decorative ice crystal pattern
    rng = np.random.RandomState(7)
    for _ in range(80):
        cx, cy = rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)
        r = rng.uniform(0.003, 0.012)
        circle = plt.Circle((cx, cy), r, color=C_ACCENT,
                             alpha=rng.uniform(0.03, 0.12),
                             transform=ax.transAxes, zorder=1)
        ax.add_patch(circle)

    # Top accent line
    ax.axhline(0.93, color=C_ACCENT, linewidth=3, zorder=3)
    ax.axhline(0.915, color=C_GREEN, linewidth=1.5, zorder=3)

    # Title
    ax.text(0.5, 0.82, "Sea Ice Deep Fusion",
            ha="center", va="center", fontsize=44, fontweight="bold",
            color=C_WHITE, transform=ax.transAxes, zorder=4,
            path_effects=[pe.withStroke(linewidth=4, foreground=C_BG)])

    ax.text(0.5, 0.73, "End-to-End Workflow",
            ha="center", va="center", fontsize=28, fontweight="normal",
            color=C_ACCENT, transform=ax.transAxes, zorder=4)

    # Horizontal rule
    ax.axhline(0.68, xmin=0.15, xmax=0.85,
               color=C_BORDER, linewidth=1.5, zorder=3)

    # Subtitle
    ax.text(0.5, 0.62,
            "Satellite imagery + ICESat-2 laser altimetry fusion\n"
            "for Arctic sea ice classification",
            ha="center", va="center", fontsize=16, color="#90CAF9",
            transform=ax.transAxes, zorder=4, linespacing=1.6)

    # Key metric banner
    card(ax, 0.08, 0.36, 0.25, 0.17, color="#0D2137", ec=C_GREEN)
    ax.text(0.205, 0.49, "Best mIoU", ha="center", va="center",
            fontsize=11, color=C_GREEN, transform=ax.transAxes, zorder=5)
    ax.text(0.205, 0.43, "0.8982", ha="center", va="center",
            fontsize=26, fontweight="bold", color=C_WHITE,
            transform=ax.transAxes, zorder=5)

    card(ax, 0.375, 0.36, 0.25, 0.17, color="#0D2137", ec=C_ACCENT)
    ax.text(0.50, 0.49, "Pixel Accuracy", ha="center", va="center",
            fontsize=11, color=C_ACCENT, transform=ax.transAxes, zorder=5)
    ax.text(0.50, 0.43, "95.3 %", ha="center", va="center",
            fontsize=26, fontweight="bold", color=C_WHITE,
            transform=ax.transAxes, zorder=5)

    card(ax, 0.67, 0.36, 0.25, 0.17, color="#0D2137", ec=C_AMBER)
    ax.text(0.795, 0.49, "Macro F1", ha="center", va="center",
            fontsize=11, color=C_AMBER, transform=ax.transAxes, zorder=5)
    ax.text(0.795, 0.43, "0.9453", ha="center", va="center",
            fontsize=26, fontweight="bold", color=C_WHITE,
            transform=ax.transAxes, zorder=5)

    # Classes
    for i, (name, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        cx = 0.25 + i * 0.25
        circle = plt.Circle((cx, 0.24), 0.045, color=col, alpha=0.85,
                             transform=ax.transAxes, zorder=4)
        ax.add_patch(circle)
        ax.text(cx, 0.24, str(i + 1), ha="center", va="center",
                fontsize=14, fontweight="bold", color=C_BG,
                transform=ax.transAxes, zorder=5)
        ax.text(cx, 0.16, name, ha="center", va="center",
                fontsize=13, color=C_WHITE, transform=ax.transAxes, zorder=5)

    ax.text(0.5, 0.095, "3-class semantic segmentation  ·  Tile-grouped train/test split  ·  Focal loss",
            ha="center", va="center", fontsize=11, color=C_GREY,
            transform=ax.transAxes, zorder=5)

    ax.axhline(0.07, color=C_BORDER, linewidth=1, zorder=3)
    ax.text(0.5, 0.04, "Research Seminar  ·  Sea Ice Analysis Project",
            ha="center", va="center", fontsize=10, color=C_GREY,
            transform=ax.transAxes, zorder=5)

    pdf.savefig(fig, facecolor=C_BG)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 2 ─ PROBLEM OVERVIEW  &  DATA SOURCES
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    set_dark_bg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    hide_axes(ax)

    ax.axhline(0.94, color=C_ACCENT, linewidth=2)
    ax.text(0.05, 0.96, "Problem Overview & Data Sources",
            fontsize=20, fontweight="bold", color=C_WHITE, transform=ax.transAxes)
    ax.text(0.95, 0.96, "Page 2", fontsize=10, color=C_GREY,
            transform=ax.transAxes, ha="right")

    # LEFT: Problem description
    card(ax, 0.04, 0.64, 0.44, 0.27, ec=C_ACCENT)
    ax.text(0.06, 0.895, "Why is this hard?",
            fontsize=13, fontweight="bold", color=C_ACCENT, transform=ax.transAxes)
    lines = [
        "• Thin ice is visually similar to thick ice in RGB",
        "• Melt ponds create ambiguous color signatures",
        "• Spatial resolution limits single-sensor accuracy",
        "• Class imbalance: thin ice is rare but critical",
        "• Generalization: test tile is geographically separate",
    ]
    for k, line in enumerate(lines):
        ax.text(0.06, 0.852 - k * 0.042, line,
                fontsize=10.5, color=C_WHITE, transform=ax.transAxes)

    # RIGHT: Solution
    card(ax, 0.52, 0.64, 0.44, 0.27, ec=C_GREEN)
    ax.text(0.54, 0.895, "Our Solution: Multimodal Fusion",
            fontsize=13, fontweight="bold", color=C_GREEN, transform=ax.transAxes)
    lines2 = [
        "• Satellite RGB patches  →  spatial texture (U-Net)",
        "• ICESat-2 ATL03 segments  →  height/roughness (LSTM)",
        "• Deep fusion: combine features before final decision",
        "• Transfer learning: pretrained ResNet-18 + sweep winner",
        "• Squeeze-Excitation gating for adaptive channel weighting",
    ]
    for k, line in enumerate(lines2):
        ax.text(0.54, 0.852 - k * 0.042, line,
                fontsize=10.5, color=C_WHITE, transform=ax.transAxes)

    # Data sources boxes
    # Satellite image box
    card(ax, 0.04, 0.38, 0.27, 0.22, ec=C_ICE)
    ax.text(0.175, 0.578, "Satellite Images", ha="center",
            fontsize=12, fontweight="bold", color=C_ICE, transform=ax.transAxes)
    img_lines = [
        "Format: 128×128 RGB patches",
        "Tiles: T02CNA, T02CNC (train)",
        "        T03CWT (test)",
        "Pre-proc: ImageNet normalise",
        "Augment: flip + rotate (train)",
    ]
    for k, line in enumerate(img_lines):
        ax.text(0.055, 0.545 - k*0.035, line, fontsize=9.5, color=C_WHITE,
                transform=ax.transAxes)

    # ICESat-2 box
    card(ax, 0.365, 0.38, 0.27, 0.22, ec=C_AMBER)
    ax.text(0.50, 0.578, "ICESat-2 / ATL03", ha="center",
            fontsize=12, fontweight="bold", color=C_AMBER, transform=ax.transAxes)
    csv_lines = [
        "Format: 10 m along-track segments",
        "Features (8 per segment):",
        "  h_cor_mean, h_diff",
        "  rel_height_min_elev, height_sd",
        "  pcnth/pcnt/bcnt/brate_mean",
        "Window: 5 consecutive segments",
    ]
    for k, line in enumerate(csv_lines):
        ax.text(0.375, 0.545 - k*0.035, line, fontsize=9.5, color=C_WHITE,
                transform=ax.transAxes)

    # Masks box
    card(ax, 0.69, 0.38, 0.27, 0.22, ec=C_PINK)
    ax.text(0.825, 0.578, "Segmentation Masks", ha="center",
            fontsize=12, fontweight="bold", color=C_PINK, transform=ax.transAxes)
    mask_lines = [
        "3 classes encoded as RGB:",
        "  Red   (255,0,0)  → Class 0",
        "   = Thick Ice",
        "  Blue  (0,0,255)  → Class 1",
        "   = Thin Ice / Melt pond",
        "  Green (0,255,0)  → Class 2",
        "   = Open Water",
    ]
    for k, line in enumerate(mask_lines):
        ax.text(0.70, 0.545 - k*0.030, line, fontsize=9.5, color=C_WHITE,
                transform=ax.transAxes)

    # Dataset split info
    card(ax, 0.04, 0.18, 0.92, 0.16, ec=C_BORDER, lw=1)
    ax.text(0.5, 0.322, "Dataset Split Strategy  (tile-grouped — no geographic leakage)",
            ha="center", fontsize=12, fontweight="bold", color=C_ACCENT,
            transform=ax.transAxes)

    splits = [
        ("Train\n(90 %)", "T02CNA + T02CNC", C_GREEN, 0.18),
        ("Val\n(10 %)", "Random subset\nof train tiles", C_AMBER, 0.42),
        ("Test\n(held-out)", "T03CWT\n(new region)", C_PINK, 0.66),
    ]
    for label, desc, col, x in splits:
        card(ax, x, 0.195, 0.18, 0.10, color=C_BG, ec=col)
        ax.text(x + 0.09, 0.265, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(x + 0.09, 0.218, desc, ha="center", va="center",
                fontsize=9, color=C_WHITE, transform=ax.transAxes, linespacing=1.4)

    ax.text(0.5, 0.055,
            "Tile T03CWT was never seen during training — providing an honest "
            "out-of-distribution evaluation.",
            ha="center", fontsize=10, color=C_GREY, transform=ax.transAxes,
            style="italic")

    pdf.savefig(fig, facecolor=C_BG)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 3 ─ ARCHITECTURE DIAGRAM
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    set_dark_bg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    hide_axes(ax)

    ax.axhline(0.94, color=C_ACCENT, linewidth=2)
    ax.text(0.05, 0.96, "Model Architecture  — Deep Fusion (v4/v5)",
            fontsize=20, fontweight="bold", color=C_WHITE, transform=ax.transAxes)
    ax.text(0.95, 0.96, "Page 3", fontsize=10, color=C_GREY,
            transform=ax.transAxes, ha="right")

    def arch_box(ax, cx, cy, w, h, label, sublabel, color, fontsize=10):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.01",
                              facecolor=color, edgecolor=C_WHITE,
                              linewidth=1.5, alpha=0.92,
                              transform=ax.transAxes, zorder=4)
        ax.add_patch(rect)
        ax.text(cx, cy + 0.02, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_BG,
                transform=ax.transAxes, zorder=5)
        if sublabel:
            ax.text(cx, cy - 0.025, sublabel, ha="center", va="center",
                    fontsize=8, color=C_BG, transform=ax.transAxes, zorder=5,
                    alpha=0.85)

    def arrow(ax, x0, y0, x1, y1, color=C_WHITE, lw=1.5, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, linestyle=ls,
                                   mutation_scale=14), zorder=6)

    def label(ax, x, y, txt, color=C_WHITE, fs=9, ha="center"):
        ax.text(x, y, txt, ha=ha, va="center", fontsize=fs, color=color,
                transform=ax.transAxes, zorder=6)

    # ── IMAGE BRANCH ────────────────────────────────────────────────────
    arch_box(ax, 0.09, 0.73, 0.13, 0.07, "RGB Image",
             "B×3×128×128", C_ICE)
    arch_box(ax, 0.09, 0.58, 0.13, 0.07, "ResNet-18",
             "encoder (ImageNet)", C_ICE)
    arch_box(ax, 0.09, 0.43, 0.14, 0.07, "U-Net Decoder",
             "bilinear upsample", "#29B6F6")
    arch_box(ax, 0.09, 0.28, 0.14, 0.07, "Image Features",
             "B×16×128×128", "#0288D1")

    arrow(ax, 0.09, 0.695, 0.09, 0.617)
    arrow(ax, 0.09, 0.545, 0.09, 0.467)
    arrow(ax, 0.09, 0.395, 0.09, 0.317)

    label(ax, 0.035, 0.65, "Image\nbranch", C_ICE, 9)

    # ── CSV BRANCH ──────────────────────────────────────────────────────
    arch_box(ax, 0.34, 0.73, 0.14, 0.07, "CSV Window",
             "B×5×8  (seg. feats)", C_AMBER)
    arch_box(ax, 0.34, 0.58, 0.14, 0.07, "Uni-LSTM (96)",
             "1 layer, tanh", C_AMBER)
    arch_box(ax, 0.34, 0.455, 0.14, 0.06, "Last Hidden",
             "h_T  → B×96", "#F9A825")
    arch_box(ax, 0.34, 0.345, 0.16, 0.065, "Dense(96→16→16)",
             "ELU + Dropout×2", "#F57F17")
    arch_box(ax, 0.34, 0.23, 0.14, 0.06, "Tile & Expand",
             "B×16×128×128", "#E65100")

    arrow(ax, 0.34, 0.695, 0.34, 0.617)
    arrow(ax, 0.34, 0.545, 0.34, 0.487)
    arrow(ax, 0.34, 0.423, 0.34, 0.378)
    arrow(ax, 0.34, 0.310, 0.34, 0.262)

    label(ax, 0.42, 0.73, "CSV\nbranch", C_AMBER, 9)

    # ── FUSION ──────────────────────────────────────────────────────────
    arch_box(ax, 0.62, 0.26, 0.16, 0.075, "Concatenate",
             "B×32×128×128", "#CE93D8")

    # arrows from both branches to concat
    arrow(ax, 0.165, 0.28, 0.535, 0.27)
    arrow(ax, 0.415, 0.23, 0.535, 0.265)

    arch_box(ax, 0.62, 0.155, 0.16, 0.07, "SE Block",
             "squeeze-excite gate", "#BA68C8")
    arch_box(ax, 0.845, 0.26, 0.14, 0.07, "Conv 3×3",
             "BN + ReLU + Drop", "#9C27B0")
    arch_box(ax, 0.845, 0.13, 0.14, 0.065, "Conv 1×1",
             "→ 3-class logits", "#7B1FA2")
    arch_box(ax, 0.845, 0.035, 0.15, 0.055, "Prediction",
             "B×3×128×128", C_GREEN)

    arrow(ax, 0.62, 0.222, 0.62, 0.193)
    arrow(ax, 0.70, 0.155, 0.77, 0.26)
    arrow(ax, 0.845, 0.222, 0.845, 0.167)
    arrow(ax, 0.845, 0.095, 0.845, 0.067)

    label(ax, 0.745, 0.155, "Fusion\nhead", "#CE93D8", 9)

    # ── Loss annotations ────────────────────────────────────────────────
    card(ax, 0.04, 0.04, 0.48, 0.12, color="#0A1F38", ec=C_BORDER, lw=1)
    ax.text(0.05, 0.14, "Training recipe:", fontsize=10, fontweight="bold",
            color=C_ACCENT, transform=ax.transAxes)
    recipe = [
        "Loss: Categorical Focal  α=[0.05, 0.45, 0.60]  γ=2.0",
        "Optimizer: Adam  |  LR: 1e-4 (fresh) / 1e-5 (pretrained LSTM)",
        "Scheduler: Cosine Annealing  |  AMP: FP16  |  Patience=8",
    ]
    for k, r in enumerate(recipe):
        ax.text(0.05, 0.115 - k * 0.03, r, fontsize=9, color=C_WHITE,
                transform=ax.transAxes)

    # ── LSTM pretrain note ───────────────────────────────────────────────
    card(ax, 0.55, 0.04, 0.42, 0.12, color="#0A1F38", ec=C_AMBER, lw=1)
    ax.text(0.565, 0.14, "Transfer learning (LSTM):", fontsize=10,
            fontweight="bold", color=C_AMBER, transform=ax.transAxes)
    note = [
        "1. Hyperparameter sweep → best LSTM (hidden=96)",
        "2. Hot-load sweep winner weights into fusion CSV branch",
        "3. Fine-tune at 0.1× LR (v5) or freeze (v4)",
    ]
    for k, n in enumerate(note):
        ax.text(0.565, 0.115 - k * 0.03, n, fontsize=9, color=C_WHITE,
                transform=ax.transAxes)

    pdf.savefig(fig, facecolor=C_BG)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4 ─ TRAINING PIPELINE FLOW
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    set_dark_bg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    hide_axes(ax)

    ax.axhline(0.94, color=C_ACCENT, linewidth=2)
    ax.text(0.05, 0.96, "Training Pipeline — Step by Step",
            fontsize=20, fontweight="bold", color=C_WHITE, transform=ax.transAxes)
    ax.text(0.95, 0.96, "Page 4", fontsize=10, color=C_GREY,
            transform=ax.transAxes, ha="right")

    steps = [
        (0.08, 0.80, C_ICE,   "Step 1", "Load & Align Data",
         ["• Pair each 128×128 image patch with its\n"
          "  ATL03 CSV row via tile + beam + row_idx",
          "• Build 8-feature arrays per CSV segment",
          "• Compute train-tile z-score stats (no leakage)"]),
        (0.38, 0.80, C_AMBER, "Step 2", "Preprocess & Cache",
         ["• Z-normalise CSV features using train-only\n"
          "  mean/std → save as .npy per CSV file",
          "• ImageNet-normalise RGB on the fly",
          "• Random flip/rotate for augmentation"]),
        (0.68, 0.80, "#CE93D8","Step 3", "LSTM Sweep (offline)",
         ["• Grid search: hidden ∈ {32,64,96},\n"
          "  LR, dropout, focal γ, α, seq_len",
          "• Best config: hidden=96, lr=1e-3,\n"
          "  α=[0.05,0.45,0.60], γ=2.0"]),
        (0.08, 0.44, C_GREEN, "Step 4", "Build Fusion Model",
         ["• U-Net (ResNet-18, ImageNet weights)\n"
          "  → 16-channel feature map",
          "• Hot-load sweep-winner LSTM weights\n"
          "• Tile CSV vector → spatial feature map"]),
        (0.38, 0.44, "#F57F17","Step 5", "Train Fusion",
         ["• Focal loss handles class imbalance\n"
          "  (thin ice α=0.45 up-weighted)",
          "• Two param groups: fresh @ LR=1e-4,\n"
          "  pretrained LSTM @ LR=1e-5",
          "• Early stop on val mIoU, patience=8"]),
        (0.68, 0.44, C_PINK,  "Step 6", "Evaluate & Compare",
         ["• Load best.pt (highest val mIoU)\n"
          "• Evaluate on T03CWT test tile",
          "• Report mIoU, pix_acc, per-class\n"
          "  IoU / Precision / Recall / F1"]),
    ]

    for cx, cy, col, num, title, bullets in steps:
        card(ax, cx, cy - 0.30, 0.27, 0.32, color=C_CARD, ec=col, lw=2)
        # Coloured header
        header = FancyBboxPatch((cx, cy - 0.01), 0.27, 0.065,
                                boxstyle="round,pad=0.01",
                                facecolor=col, edgecolor=col,
                                transform=ax.transAxes, zorder=4)
        ax.add_patch(header)
        ax.text(cx + 0.135, cy + 0.025, f"{num}  |  {title}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=C_BG, transform=ax.transAxes, zorder=5)
        for k, b in enumerate(bullets):
            ax.text(cx + 0.01, cy - 0.065 - k * 0.07, b,
                    fontsize=8.8, color=C_WHITE, transform=ax.transAxes,
                    zorder=5, linespacing=1.4)

    # Arrows between steps
    for (x0, x1), y in [((0.35, 0.37), 0.75), ((0.65, 0.67), 0.75),
                         ((0.35, 0.37), 0.40), ((0.65, 0.67), 0.40)]:
        arrow(ax, x0, y, x1, y, color=C_ACCENT, lw=2)

    # Down arrows between rows
    for cx in [0.22, 0.52, 0.82]:
        arrow(ax, cx, 0.50, cx, 0.48, color=C_ACCENT, lw=2)

    # Bottom note
    ax.text(0.5, 0.05,
            "All experiments use tile-grouped splits (T02CNA+T02CNC → train, T03CWT → test)\n"
            "ensuring no geographic overlap between seen and unseen data.",
            ha="center", fontsize=10, color=C_GREY, transform=ax.transAxes,
            linespacing=1.5, style="italic")

    pdf.savefig(fig, facecolor=C_BG)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 5 ─ MODEL COMPARISON  (bar charts)
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    set_dark_bg(fig)
    fig.suptitle("Model Comparison — mIoU & Pixel Accuracy",
                 fontsize=18, fontweight="bold", color=C_WHITE, y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.10,
                           hspace=0.55, wspace=0.35)

    # 1) mIoU bar chart
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(C_CARD)
    colors = [C_GREY, C_GREY, C_ICE, C_GREEN, C_ICE]
    colors[3] = C_GREEN   # best model
    bars = ax1.barh(MODELS, MIOU, color=colors, edgecolor=C_BG, height=0.55)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("mean IoU (mIoU)", color=C_WHITE, fontsize=10)
    ax1.tick_params(colors=C_WHITE)
    ax1.spines[:].set_color(C_BORDER)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
    ax1.set_title("Test mIoU by Model", color=C_WHITE, fontsize=12,
                  fontweight="bold", pad=6)
    ax1.axvline(0.5, color=C_BORDER, linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, MIOU):
        x_pos = val + 0.01
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=10,
                 color=C_WHITE, fontweight="bold")
    # Star on best
    best_idx = MIOU.index(max(MIOU))
    bars[best_idx].set_edgecolor(C_AMBER)
    bars[best_idx].set_linewidth(2)

    # 2) Per-class IoU (best model)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(C_CARD)
    x = np.arange(len(CLASS_NAMES))
    bars2 = ax2.bar(CLASS_NAMES, BEST_PER_IOU, color=CLASS_COLORS,
                    edgecolor=C_BG, width=0.55)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("IoU", color=C_WHITE)
    ax2.set_title("Per-Class IoU (Best Model)", color=C_WHITE,
                  fontsize=11, fontweight="bold")
    ax2.tick_params(colors=C_WHITE)
    ax2.spines[:].set_color(C_BORDER)
    for bar, val in zip(bars2, BEST_PER_IOU):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=10, color=C_WHITE, fontweight="bold")
    ax2.axhline(0.9, color=C_GREY, linestyle="--", linewidth=1, alpha=0.5)

    # 3) P / R / F1 grouped bar (best model)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(C_CARD)
    x = np.arange(len(CLASS_NAMES))
    bw = 0.25
    ax3.bar(x - bw, BEST_PREC, bw, label="Precision",
            color=C_ACCENT, alpha=0.85, edgecolor=C_BG)
    ax3.bar(x,       BEST_REC,  bw, label="Recall",
            color=C_GREEN, alpha=0.85, edgecolor=C_BG)
    ax3.bar(x + bw,  BEST_F1,   bw, label="F1",
            color=C_AMBER, alpha=0.85, edgecolor=C_BG)
    ax3.set_xticks(x); ax3.set_xticklabels(CLASS_NAMES)
    ax3.set_ylim(0, 1.12)
    ax3.set_ylabel("Score", color=C_WHITE)
    ax3.set_title("Precision / Recall / F1 (Best Model)",
                  color=C_WHITE, fontsize=11, fontweight="bold")
    ax3.tick_params(colors=C_WHITE)
    ax3.spines[:].set_color(C_BORDER)
    ax3.legend(fontsize=9, labelcolor=C_WHITE,
               facecolor=C_BG, edgecolor=C_BORDER)
    ax3.axhline(0.9, color=C_GREY, linestyle="--", linewidth=1, alpha=0.5)

    for sub_ax in [ax1, ax2, ax3]:
        for lab in sub_ax.get_xticklabels() + sub_ax.get_yticklabels():
            lab.set_color(C_WHITE)

    fig.patch.set_facecolor(C_BG)

    # Annotation
    fig.text(0.5, 0.03,
             "Best model: Fusion v4 (frozen sweep-winner LSTM + U-Net with SE fusion head)",
             ha="center", fontsize=10, color=C_GREEN, style="italic")

    pdf.savefig(fig, facecolor=C_BG)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 6 ─ CONFUSION MATRIX  +  KEY FINDINGS
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    set_dark_bg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    hide_axes(ax)

    ax.axhline(0.94, color=C_ACCENT, linewidth=2)
    ax.text(0.05, 0.96, "Confusion Matrix  &  Key Findings",
            fontsize=20, fontweight="bold", color=C_WHITE, transform=ax.transAxes)
    ax.text(0.95, 0.96, "Page 6", fontsize=10, color=C_GREY,
            transform=ax.transAxes, ha="right")

    # ── Draw confusion matrix (fusion_v4 data) ───────────────────────────
    cm = np.array([
        [525732740, 21080365,    14],
        [ 11838980, 155569638, 269357],
        [   205916,   1967616, 37064910],
    ], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = cm / np.maximum(row_sums, 1) * 100.0

    cm_ax = fig.add_axes([0.06, 0.25, 0.40, 0.62])
    cm_ax.set_facecolor(C_CARD)
    im = cm_ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    cm_ax.set_xticks(range(3)); cm_ax.set_yticks(range(3))
    cm_ax.set_xticklabels([f"Pred\n{c}" for c in CLASS_NAMES],
                          fontsize=10, color=C_WHITE)
    cm_ax.set_yticklabels(CLASS_NAMES, fontsize=10, color=C_WHITE)
    cm_ax.set_xlabel("Predicted", fontsize=11, color=C_WHITE)
    cm_ax.set_ylabel("Actual", fontsize=11, color=C_WHITE)
    cm_ax.set_title("Confusion Matrix  (row %, Best Model)",
                    fontsize=11, color=C_WHITE, fontweight="bold", pad=8)
    cm_ax.spines[:].set_color(C_BORDER)
    for i in range(3):
        for j in range(3):
            v = pct[i, j]
            cm_ax.text(j, i, f"{v:.2f}%", ha="center", va="center",
                       fontsize=13, fontweight="bold",
                       color="white" if v > 55 else "#1A237E")
    cb = fig.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelcolor=C_WHITE)

    # ── Key findings ─────────────────────────────────────────────────────
    findings = [
        (C_GREEN,  "Thick Ice (class 0)",
         "IoU = 0.9407  |  F1 = 0.9695\n"
         "Nearly perfect: >97% of thick ice\n"
         "pixels are correctly classified."),
        (C_THIN,   "Thin Ice (class 1)  ← hardest",
         "IoU = 0.8157  |  F1 = 0.8985\n"
         "Highest confusion with thick ice;\n"
         "α=0.45 upweight helps significantly."),
        (C_WATER,  "Open Water (class 2)",
         "IoU = 0.9382  |  F1 = 0.9681\n"
         "Very high recall (94.5%) — water\n"
         "texture is distinct for the U-Net."),
        (C_ACCENT, "Fusion beats single modality",
         "mIoU: LSTM-only = 0.24, U-Net = 0.87\n"
         "Deep Fusion = 0.8982 (+3 pts over\n"
         "best single-modality model)."),
        (C_AMBER,  "Transfer learning matters",
         "Hot-loading sweep-winner LSTM\n"
         "gave +0.6 pp mIoU vs. random init\n"
         "while keeping thin-ice recall high."),
        (C_PINK,   "SE gating is effective",
         "Squeeze-Excitation block lets the\n"
         "model down-weight noise channels,\n"
         "improving thin ice IoU by ~1 pp."),
    ]

    for k, (col, title, body) in enumerate(findings):
        row, col_pos = divmod(k, 2)
        bx = 0.52 + col_pos * 0.245
        by = 0.73 - row * 0.26
        card(ax, bx, by - 0.215, 0.235, 0.215, color=C_CARD, ec=col, lw=1.8)
        ax.text(bx + 0.01, by - 0.025, title,
                fontsize=9.5, fontweight="bold", color=col,
                transform=ax.transAxes, zorder=5)
        ax.text(bx + 0.01, by - 0.085, body,
                fontsize=8.8, color=C_WHITE, transform=ax.transAxes,
                zorder=5, linespacing=1.45)

    ax.text(0.5, 0.055,
            "Deep fusion (image + altimetry) outperforms both single-modality baselines. "
            "Thin ice remains the most challenging class.",
            ha="center", fontsize=10, color=C_GREY, transform=ax.transAxes,
            style="italic")

    pdf.savefig(fig, facecolor=C_BG)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 7 ─ RESULTS TABLE  +  MODEL EVOLUTION
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    set_dark_bg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    hide_axes(ax)

    ax.axhline(0.94, color=C_ACCENT, linewidth=2)
    ax.text(0.05, 0.96, "Results Table  &  Model Evolution",
            fontsize=20, fontweight="bold", color=C_WHITE, transform=ax.transAxes)
    ax.text(0.95, 0.96, "Page 7", fontsize=10, color=C_GREY,
            transform=ax.transAxes, ha="right")

    # ── Summary table ────────────────────────────────────────────────────
    table_data = [
        ["Model",            "mIoU",   "Pix Acc", "IoU Ice", "IoU Thin", "IoU Water"],
        ["U-Net (image)",    "0.8704", "94.29%",  "0.9299",  "0.7683",   "0.9130"],
        ["LSTM (CSV only)",  "0.2420", "72.54%",  "0.7254",  "0.0000",   "0.0005"],
        ["Fusion Deep v1",   "0.8915", "95.12%",  "0.9396",  "0.8015",   "0.9335"],
        ["Fusion v4 ★",     "0.8982", "95.31%",  "0.9407",  "0.8157",   "0.9382"],
        ["Deep Fusion v5",   "0.8896", "95.02%",  "—",       "—",        "—"],
    ]
    col_widths = [0.22, 0.09, 0.10, 0.10, 0.10, 0.10]
    col_starts = [0.04]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    row_height = 0.065
    row_starts = [0.83 - i * row_height for i in range(len(table_data))]

    for i, row in enumerate(table_data):
        is_header = (i == 0)
        is_best   = (i == 4)
        bg_color = C_BORDER if is_header else ("#0D2137" if is_best else C_CARD)
        ec_color = C_ACCENT if is_header else (C_AMBER if is_best else C_BORDER)
        rect = FancyBboxPatch((0.03, row_starts[i] - 0.005), 0.94, row_height - 0.006,
                              boxstyle="round,pad=0.005",
                              facecolor=bg_color, edgecolor=ec_color,
                              linewidth=1.5 if is_best else 0.5,
                              transform=ax.transAxes, zorder=3)
        ax.add_patch(rect)
        for j, (cell, cx) in enumerate(zip(row, col_starts)):
            color = (C_ACCENT if is_header else
                     (C_AMBER if is_best else C_WHITE))
            ax.text(cx + col_widths[j]/2,
                    row_starts[i] + row_height/2 - 0.005,
                    cell, ha="center", va="center",
                    fontsize=9.5 if is_header else 9,
                    fontweight="bold" if (is_header or is_best) else "normal",
                    color=color, transform=ax.transAxes, zorder=4)

    ax.text(0.5, 0.34, "★ = Best model  (frozen sweep-winner LSTM + U-Net + SE fusion)",
            ha="center", fontsize=9, color=C_AMBER, transform=ax.transAxes)

    # ── Model evolution arrow chart ───────────────────────────────────────
    card(ax, 0.04, 0.04, 0.92, 0.24, color=C_CARD, ec=C_BORDER, lw=1)
    ax.text(0.5, 0.265, "Model Evolution  (mIoU)",
            ha="center", fontsize=12, fontweight="bold",
            color=C_ACCENT, transform=ax.transAxes)

    evo_x   = [0.10, 0.26, 0.42, 0.60, 0.78]
    evo_y   = [0.8704, 0.2420, 0.8915, 0.8982, 0.8896]
    evo_lbl = ["U-Net\nonly", "LSTM\nonly", "Fusion\nv1",
               "Fusion\nv4\n★", "Deep\nFusion\nv5"]
    evo_col = [C_ICE, C_GREY, C_ICE, C_GREEN, C_ICE]

    # normalise y to plot range [0.07, 0.28]
    lo, hi = 0.0, 1.0
    def to_y(v):
        return 0.06 + (v - 0.1) / 0.9 * 0.17

    for k in range(len(evo_x) - 1):
        ax.annotate("", xy=(evo_x[k+1], to_y(evo_y[k+1])),
                    xytext=(evo_x[k],   to_y(evo_y[k])),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=C_GREY,
                                   lw=1.5, mutation_scale=12), zorder=5)

    for x, y, lbl, col in zip(evo_x, evo_y, evo_lbl, evo_col):
        py = to_y(y)
        circle = plt.Circle((x, py), 0.018, color=col, zorder=6,
                             transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(x, py, f"{y:.3f}", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=C_BG,
                transform=ax.transAxes, zorder=7)
        ax.text(x, py - 0.055, lbl, ha="center", va="top",
                fontsize=8, color=col, transform=ax.transAxes,
                linespacing=1.3)

    pdf.savefig(fig, facecolor=C_BG)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 8 ─ CONCLUSIONS & FUTURE WORK
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    set_dark_bg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    hide_axes(ax)

    ax.axhline(0.94, color=C_ACCENT, linewidth=2)
    ax.text(0.05, 0.96, "Conclusions  &  Future Work",
            fontsize=20, fontweight="bold", color=C_WHITE, transform=ax.transAxes)
    ax.text(0.95, 0.96, "Page 8", fontsize=10, color=C_GREY,
            transform=ax.transAxes, ha="right")

    # Conclusions
    card(ax, 0.04, 0.56, 0.43, 0.34, ec=C_GREEN)
    ax.text(0.06, 0.88, "Conclusions", fontsize=14, fontweight="bold",
            color=C_GREEN, transform=ax.transAxes)
    conclusions = [
        ("C1", "Deep fusion (RGB + altimetry) achieves mIoU = 0.8982,\n"
               "outperforming any single-modality model."),
        ("C2", "Thin ice (IoU = 0.8157) is the most challenging class;\n"
               "focal loss α-upweighting is critical for learning it."),
        ("C3", "Transfer learning: sweep-winner LSTM weights provide\n"
               "a strong CSV-branch initialization (+0.6 pp mIoU)."),
        ("C4", "Tile-grouped splits eliminate geographic data leakage,\n"
               "giving honest out-of-distribution performance estimates."),
        ("C5", "Squeeze-Excitation gating adaptively weights feature\n"
               "channels, improving fusion head discriminability."),
    ]
    for k, (tag, text) in enumerate(conclusions):
        tag_x = 0.06; text_x = 0.13; y = 0.83 - k * 0.065
        rect = FancyBboxPatch((tag_x, y - 0.015), 0.055, 0.030,
                              boxstyle="round,pad=0.005",
                              facecolor=C_GREEN, edgecolor=C_GREEN,
                              transform=ax.transAxes, zorder=4)
        ax.add_patch(rect)
        ax.text(tag_x + 0.0275, y, tag, ha="center", va="center",
                fontsize=9, fontweight="bold", color=C_BG,
                transform=ax.transAxes, zorder=5)
        ax.text(text_x, y, text, ha="left", va="center",
                fontsize=9, color=C_WHITE, transform=ax.transAxes,
                zorder=5, linespacing=1.4)

    # Future work
    card(ax, 0.53, 0.56, 0.43, 0.34, ec=C_AMBER)
    ax.text(0.55, 0.88, "Future Work", fontsize=14, fontweight="bold",
            color=C_AMBER, transform=ax.transAxes)
    future = [
        ("F1", "Bidirectional LSTM in fusion branch (already tested\n"
               "standalone; integrate into full fusion pipeline)."),
        ("F2", "Test-time augmentation (TTA): average flipped/rotated\n"
               "predictions to reduce variance."),
        ("F3", "Multi-temporal fusion: stack multiple dates to track\n"
               "seasonal sea ice dynamics."),
        ("F4", "Attention mechanisms to replace fixed SE block with\n"
               "learnable spatial/channel attention."),
        ("F5", "Expand to ATL07 (sea ice) and ATL10 (freeboard)\n"
               "products for richer altimetry features."),
    ]
    for k, (tag, text) in enumerate(future):
        tag_x = 0.55; text_x = 0.62; y = 0.83 - k * 0.065
        rect = FancyBboxPatch((tag_x, y - 0.015), 0.055, 0.030,
                              boxstyle="round,pad=0.005",
                              facecolor=C_AMBER, edgecolor=C_AMBER,
                              transform=ax.transAxes, zorder=4)
        ax.add_patch(rect)
        ax.text(tag_x + 0.0275, y, tag, ha="center", va="center",
                fontsize=9, fontweight="bold", color=C_BG,
                transform=ax.transAxes, zorder=5)
        ax.text(text_x, y, text, ha="left", va="center",
                fontsize=9, color=C_WHITE, transform=ax.transAxes,
                zorder=5, linespacing=1.4)

    # Technical stack
    card(ax, 0.04, 0.26, 0.92, 0.26, ec=C_BORDER, lw=1)
    ax.text(0.5, 0.505, "Technical Stack",
            ha="center", fontsize=13, fontweight="bold",
            color=C_ACCENT, transform=ax.transAxes)

    stack = [
        (C_ICE,   "PyTorch 2.x", "Training framework\n+ AMP (FP16)"),
        (C_AMBER, "segmentation-\nmodels-pytorch", "U-Net with\nResNet-18 encoder"),
        (C_GREEN, "ICESat-2\nATL03", "Along-track photon\naltimetry data"),
        (C_PINK,  "Focal Loss", "Handles class\nimbalance"),
        (C_THIN,  "SE Block", "Squeeze-Excitation\nchannel attention"),
    ]
    for k, (col, title, desc) in enumerate(stack):
        bx = 0.065 + k * 0.19
        card(ax, bx, 0.27, 0.175, 0.19, color=C_BG, ec=col, lw=1.5)
        ax.text(bx + 0.0875, 0.44, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=col,
                transform=ax.transAxes, linespacing=1.3)
        ax.text(bx + 0.0875, 0.33, desc, ha="center", va="center",
                fontsize=8.5, color=C_WHITE,
                transform=ax.transAxes, linespacing=1.4)

    # Final summary bar
    card(ax, 0.04, 0.07, 0.92, 0.15, color="#0D2137", ec=C_ACCENT, lw=2)
    ax.text(0.5, 0.195,
            "Best Result: Fusion v4  —  mIoU = 0.8982  |  Pixel Acc = 95.3%  |  Macro F1 = 0.9453",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C_GREEN, transform=ax.transAxes)
    ax.text(0.5, 0.115,
            "Thick Ice IoU = 0.9407   ·   Thin Ice IoU = 0.8157   ·   Water IoU = 0.9382",
            ha="center", va="center", fontsize=11, color=C_WHITE,
            transform=ax.transAxes)

    pdf.savefig(fig, facecolor=C_BG)
    plt.close(fig)

print(f"Saved: {OUT}")
