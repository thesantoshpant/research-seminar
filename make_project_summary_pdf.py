"""
Build project_summary.pdf -- a short, plain-language writeup of the
three models we tried (U-Net only, LSTM only, Deep Fusion) with their
benchmarks and the figures from runs/.

Run:    python make_project_summary_pdf.py
Output: project_summary.pdf  (at the repo root)
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, KeepTogether, CondPageBreak,
)
from PIL import Image as PILImage

ROOT = Path(__file__).parent
OUT  = ROOT / "project_summary.pdf"

# ---- Image paths ---------------------------------------------------------
UNET_CM     = ROOT / "archive" / "runs" / "unet_imgonly_v1" / "confmat_profstyle.png"
UNET_PREDS  = ROOT / "archive" / "runs" / "unet_imgonly_v1" / "sample_predictions.png"
LSTM_CM     = ROOT / "runs" / "lstm_winner" / "confmat.png"
FUSION_CM   = ROOT / "runs" / "fusion_winner" / "confmat.png"
FUSION_LOSS = ROOT / "runs" / "fusion_winner" / "loss_curve.png"

# ---- Styles --------------------------------------------------------------
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, leading=24,
                   spaceAfter=10, textColor=colors.HexColor("#1F4E79"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, leading=18,
                   spaceBefore=14, spaceAfter=8,
                   textColor=colors.HexColor("#2E75B6"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11,
                     leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
CAP  = ParagraphStyle("Cap", parent=styles["BodyText"], fontSize=9,
                     leading=12, alignment=TA_CENTER, textColor=colors.grey,
                     spaceAfter=12, italic=True)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=10, leading=13)

# Cells inside Tables need their own paragraph styles so HTML entities
# (e.g. &mdash;) get parsed and long labels wrap inside their cell.
TBL_CELL = ParagraphStyle("TblCell", parent=styles["BodyText"],
                          fontSize=10, leading=12, alignment=TA_LEFT,
                          spaceBefore=0, spaceAfter=0)
TBL_HEAD = ParagraphStyle("TblHead", parent=TBL_CELL,
                          fontName="Helvetica-Bold", textColor=colors.white,
                          alignment=TA_LEFT)
TBL_KEY  = ParagraphStyle("TblKey",  parent=TBL_CELL,
                          fontName="Helvetica-Bold", alignment=TA_LEFT)
TBL_VAL  = ParagraphStyle("TblVal",  parent=TBL_CELL, alignment=TA_LEFT)

# ---- Helpers -------------------------------------------------------------
def img(path: Path, width_in: float = 5.0, max_h_in: float = 6.5):
    """Return an Image with width=width_in (inches), height computed from
    the actual file's aspect ratio. Caps height at max_h_in so the image
    always fits on a page with margins."""
    if not path.exists():
        return Paragraph(f"<i>(image missing: {path.name})</i>", CAP)
    with PILImage.open(path) as p_img:
        orig_w, orig_h = p_img.size
    aspect = orig_h / orig_w   # how tall vs wide
    target_w = width_in * inch
    target_h = target_w * aspect
    if target_h > max_h_in * inch:
        scale = (max_h_in * inch) / target_h
        target_w *= scale
        target_h = max_h_in * inch
    return Image(str(path), width=target_w, height=target_h)


def metrics_table(rows):
    """rows: first row is header, rest are (label, value). Cells get
    wrapped in Paragraphs so HTML entities parse and long labels wrap."""
    wrapped = []
    for ri, row in enumerate(rows):
        new_row = []
        for ci, cell in enumerate(row):
            if ri == 0:
                style = TBL_HEAD  # header
            elif ci == 0:
                style = TBL_KEY   # leftmost column (bold)
            else:
                style = TBL_VAL
            new_row.append(Paragraph(str(cell), style))
        wrapped.append(new_row)
    tbl = Table(wrapped, colWidths=[2.8 * inch, 2.7 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E75B6")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#EDF2FA")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return tbl


def cm_table(diag_ice, diag_thin, diag_water):
    """Render a 3x3 confusion matrix as a styled Table (percentages)."""
    data = [
        ["", "Predicted thick ice", "Predicted thin ice", "Predicted water"],
        ["Actual thick ice", f"{diag_ice[0]:.2f}", f"{diag_ice[1]:.2f}", f"{diag_ice[2]:.2f}"],
        ["Actual thin ice",  f"{diag_thin[0]:.2f}", f"{diag_thin[1]:.2f}", f"{diag_thin[2]:.2f}"],
        ["Actual water",     f"{diag_water[0]:.2f}", f"{diag_water[1]:.2f}", f"{diag_water[2]:.2f}"],
    ]
    tbl = Table(data, colWidths=[1.6 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E75B6")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#D9E2F3")),
        ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#9DC3E6")),
        ("BACKGROUND", (2, 2), (2, 2), colors.HexColor("#9DC3E6")),
        ("BACKGROUND", (3, 3), (3, 3), colors.HexColor("#9DC3E6")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    return tbl


# ---- Build content -------------------------------------------------------
story = []


def add_table(rows):
    story.append(metrics_table(rows))
    story.append(Spacer(1, 0.14 * inch))


def add_fig(path, caption_text, width_in=4.5, max_h_in=5.5, heading=None):
    """Append (optional heading) + image + caption + spacer, all kept
    together so they never split across pages."""
    block = []
    if heading:
        block.append(Paragraph(heading, H2))
    block.extend([
        img(path, width_in=width_in, max_h_in=max_h_in),
        Spacer(1, 0.05 * inch),
        Paragraph(caption_text, CAP),
        Spacer(1, 0.10 * inch),
    ])
    story.append(KeepTogether(block))


def add_body(text):
    story.append(Paragraph(text, BODY))


def add_h1(text):
    story.append(Paragraph(text, H1))


def add_h2(text):
    story.append(Paragraph(text, H2))


# ----- Title page -----
add_h1("Sea Ice Classification &mdash; Project Summary")
add_body(
    "A plain-language summary of the three models we built to label each "
    "pixel of a satellite image as <b>thick ice</b>, <b>thin ice</b>, or "
    "<b>water</b>. We trained on two Antarctic tiles (T02CNA, T02CNC) and "
    "tested on a third never-seen tile (T03CWT). The scores below are all "
    "measured on that held-out test tile.")
story.append(Spacer(1, 0.10 * inch))
add_body(
    "Two kinds of data are available for every region: a Sentinel-2 RGB "
    "satellite image (looks down from above, picks up color/texture), and a "
    "set of ICESat-2 photon measurements (laser altimeter, gives precise "
    "heights along narrow ground tracks). The interesting question is "
    "<i>can we use both together to do better than either alone?</i>")
story.append(Spacer(1, 0.20 * inch))

# Headline table
add_h2("Headline results (test tile T03CWT)")
add_table([
    ["Model", "Test mIoU"],
    ["U-Net (image only)",                              "0.8704"],
    ["LSTM (photon CSV only) &mdash; sweep winner",     "0.6978"],
    ["Deep Fusion (image + LSTM) &mdash; OUR BEST",     "0.9010"],
])
add_body(
    "Higher is better; 1.0 would be perfect. The image-only U-Net already "
    "does well by itself (0.87) &mdash; the satellite picture is informative. "
    "The photon-only LSTM does much worse alone (0.70) because each photon "
    "track only covers a thin line across the patch. But <b>combining them "
    "(0.90) beats both</b>, especially on the rare thin-ice and water classes.")

story.append(PageBreak())

# ----- 1. U-Net -----
add_h1("1. U-Net &mdash; image only")
add_body(
    "U-Net is a standard image-segmentation neural network. It takes the "
    "RGB satellite picture as input and produces a 3-class label for every "
    "pixel. We used a ResNet-18 encoder pre-trained on ImageNet, which "
    "gives the model a head-start: it already knows how to recognize edges, "
    "textures, and shapes before it even sees sea ice.")
add_body(
    "U-Net is good at things the image makes obvious &mdash; large patches "
    "of open water look very different from packed ice. Where it struggles "
    "is thin ice, which can look very similar to thick ice in the visible "
    "spectrum.")
story.append(Spacer(1, 0.10 * inch))

add_h2("Benchmarks")
add_table([
    ["Metric", "Value"],
    ["Test mIoU (average IoU across the 3 classes)",    "0.8704"],
    ["Pixel accuracy",                                  "0.9429"],
    ["IoU &mdash; thick ice",                           "0.9299"],
    ["IoU &mdash; thin ice",                            "0.7683"],
    ["IoU &mdash; water",                               "0.9130"],
])

add_fig(UNET_CM,
        "Each row shows what percentage of one true class got predicted as "
        "thick ice, thin ice, or water. A perfect model would have 100% on "
        "the diagonal.",
        width_in=4.8,
        heading="Confusion matrix &mdash; U-Net")

story.append(PageBreak())
add_h1("U-Net &mdash; sample predictions")
add_body(
    "Six test patches the model never saw during training. "
    "<b>Left column:</b> the satellite image the model saw. "
    "<b>Middle:</b> the human-drawn ground truth. "
    "<b>Right:</b> what U-Net predicted. "
    "Red = thick ice, blue = thin ice, green = water.")
story.append(Spacer(1, 0.10 * inch))
story.append(img(UNET_PREDS, width_in=3.6, max_h_in=7.5))

story.append(PageBreak())

# ----- 2. LSTM -----
add_h1("2. LSTM &mdash; photon CSV only")
add_body(
    "LSTM stands for &ldquo;Long Short-Term Memory&rdquo;. It is a neural "
    "network that reads a <i>sequence</i> of measurements one at a time "
    "and remembers context as it goes. Perfect for time-series or, in our "
    "case, a series of 10-meter segments along an ICESat-2 ground track.")
add_body(
    "We followed the professor&rsquo;s own LSTM recipe (her Keras notebook "
    "for the same problem):")
add_table([
    ["Setting",                "Value"],
    ["LSTM direction",         "uni-directional (forwards only)"],
    ["Hidden units",           "96"],
    ["Number of layers",       "1"],
    ["Input window",           "5 segments (center &plusmn; 2)"],
    ["Features per segment",   "8 engineered numbers"],
    ["Loss function",          "Focal loss, &alpha;=[0.05, 0.45, 0.60], &gamma;=2.0"],
    ["Optimizer",              "Adam, learning rate 8.886e-4"],
])
add_body(
    "We didn&rsquo;t guess these numbers &mdash; we ran a 21-config sweep "
    "that varied the alpha vector, gamma, hidden size, learning rate, "
    "dropout, window length and random seed one axis at a time. The "
    "winner was <b>hidden = 96</b> &mdash; everything else stayed close to "
    "the professor&rsquo;s defaults. The 8 features are physically "
    "meaningful: mean / median / standard-deviation of corrected photon "
    "heights, photon counts, background counts, background rate, and two "
    "derived height-asymmetry features.")
story.append(Spacer(1, 0.10 * inch))

add_h2("Benchmarks")
add_table([
    ["Metric", "Value"],
    ["Test mIoU",                                       "0.6978"],
    ["Pixel accuracy",                                  "0.9594"],
    ["Macro F1",                                        "0.8080"],
    ["IoU &mdash; thick ice",                           "0.9671"],
    ["IoU &mdash; thin ice",                            "0.5427"],
    ["IoU &mdash; water",                               "0.5836"],
    ["F1 &mdash; thick ice",                            "0.9833"],
    ["F1 &mdash; thin ice",                             "0.7036"],
    ["F1 &mdash; water",                                "0.7370"],
])

add_fig(LSTM_CM,
        "Strong diagonal on thick ice (97%) because that&rsquo;s the most "
        "common class. Thin ice (76%) and water (85%) are weaker because "
        "the LSTM only sees a few photon segments per patch &mdash; it has "
        "no idea what the surrounding 128&times;128 image actually looks like.",
        width_in=4.8,
        heading="Confusion matrix &mdash; LSTM (winner)")

story.append(PageBreak())

# ----- 3. Deep Fusion (winner) -----
add_h1("3. Deep Fusion &mdash; image + LSTM &mdash; the winner")
add_body(
    "This is the model that won. It uses both data sources, and it lets "
    "the two networks compare notes inside the architecture (not just by "
    "averaging their predictions at the end).")
add_body(
    "<b>How it works (step by step):</b>")
add_body(
    "&bull; The image branch is the same U-Net from Model 1, which turns "
    "the satellite picture into a feature map.<br/>"
    "&bull; The CSV branch is the <b>LSTM winner from Model 2</b>, loaded "
    "with the weights it learned during the sweep.<br/>"
    "&bull; The LSTM&rsquo;s output is tiled across the patch and "
    "concatenated with the U-Net&rsquo;s features.<br/>"
    "&bull; A small &ldquo;attention&rdquo; layer (Squeeze-and-Excitation) "
    "learns to rescale each channel of the combined features, so the "
    "model can up- or down-weight either branch depending on what it sees.<br/>"
    "&bull; A final 3&times;3 convolution outputs the per-pixel class "
    "prediction.")
add_body(
    "<b>One trick that mattered:</b> instead of training the LSTM from "
    "scratch inside the fusion model, we loaded the sweep-winner weights "
    "and let them adapt slowly during fusion training (10&times; slower "
    "than the rest of the network). This &ldquo;two pretrained experts "
    "get glued together&rdquo; recipe gave us roughly +0.003 mIoU over "
    "freezing it and +0.006 over starting from scratch.")
story.append(Spacer(1, 0.10 * inch))

add_h2("Benchmarks")
add_table([
    ["Metric", "Value"],
    ["Test mIoU",                                       "0.9010"],
    ["Pixel accuracy",                                  "0.9530"],
    ["Macro F1",                                        "0.9468"],
    ["Macro precision",                                 "0.9460"],
    ["Macro recall",                                    "0.9482"],
    ["IoU &mdash; thick ice",                           "0.9403"],
    ["IoU &mdash; thin ice",                            "0.8138"],
    ["IoU &mdash; water",                               "0.9489"],
    ["F1 &mdash; thick ice",                            "0.9692"],
    ["F1 &mdash; thin ice",                             "0.8973"],
    ["F1 &mdash; water",                                "0.9738"],
])

story.append(PageBreak())
add_h1("Deep Fusion &mdash; visuals")

add_fig(FUSION_CM,
        "The diagonals are 96 / 92 / 96 &mdash; none below 92%. Compared "
        "to the LSTM alone (97 / 76 / 85), fusion lifted thin ice by +16 "
        "percentage points and water by +11 pp without losing thick ice.",
        width_in=4.8,
        heading="Confusion matrix &mdash; Deep Fusion (winner)")

add_fig(FUSION_LOSS,
        "Left: focal loss going down on both the training and validation "
        "sets. Middle: average IoU going up. Right: per-class IoU &mdash; "
        "thin ice (the rare class) takes longer to climb but eventually "
        "stabilizes around 0.81.",
        width_in=6.2, max_h_in=2.5,
        heading="Training curves")

story.append(PageBreak())

# ----- Side-by-side comparison -----
add_h1("Side-by-side comparison")
add_body("All three models scored on the same held-out test tile (T03CWT):")

cmp_raw = [
    ["Model", "mIoU", "Pix Acc", "IoU thick ice", "IoU thin ice", "IoU water"],
    ["U-Net (image only)",   "0.8704", "0.9429", "0.9299", "0.7683", "0.9130"],
    ["LSTM (CSV only)",      "0.6978", "0.9594", "0.9671", "0.5427", "0.5836"],
    ["Deep Fusion (winner)", "0.9010", "0.9530", "0.9403", "0.8138", "0.9489"],
]
center = ParagraphStyle("Ctr", parent=TBL_CELL, alignment=TA_CENTER)
header = ParagraphStyle("Hdr", parent=TBL_HEAD, alignment=TA_CENTER)
cmp_rows = []
for ri, row in enumerate(cmp_raw):
    new_row = []
    for ci, cell in enumerate(row):
        if ri == 0:
            new_row.append(Paragraph(cell, header))
        elif ci == 0:
            new_row.append(Paragraph(cell, TBL_KEY))
        else:
            new_row.append(Paragraph(cell, center))
    cmp_rows.append(new_row)
cmp = Table(cmp_rows, colWidths=[2.0*inch, 0.8*inch, 0.8*inch, 1.1*inch, 1.1*inch, 1.0*inch])
cmp.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E75B6")),
    ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
    ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
    ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#FFE699")),
    ("FONTSIZE",   (0, 0), (-1, -1), 10),
    ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
    ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ("GRID",       (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ("TOPPADDING",   (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
]))
story.append(cmp)
story.append(Spacer(1, 0.20 * inch))

add_h2("Key takeaways")
add_body(
    "<b>1.</b> The image-only U-Net is already a strong baseline "
    "(0.87 mIoU). It handles thick ice and water well but struggles with "
    "thin ice, because thin ice looks visually similar to thick ice in "
    "the satellite RGB.")
add_body(
    "<b>2.</b> The photon-only LSTM is weak on its own (0.70 mIoU), but "
    "the photons carry real signal about ice <i>thickness</i> that the "
    "image can&rsquo;t see. The LSTM learns to use that.")
add_body(
    "<b>3.</b> When you fuse them deeply (not just averaging predictions, "
    "but combining the internal feature maps with attention), the two "
    "modalities cover each other&rsquo;s blind spots. Result: "
    "<b>0.90 mIoU</b>, with the biggest gains on the two minority classes "
    "(thin ice +16&nbsp;pp, water +11&nbsp;pp vs LSTM alone; thin ice "
    "+5&nbsp;pp vs U-Net alone).")
add_body(
    "<b>4.</b> Loading the LSTM weights from a separate sweep and "
    "fine-tuning them slowly during fusion training was the final "
    "ingredient that pushed us past 0.90.")

story.append(Spacer(1, 0.25 * inch))
story.append(Paragraph(
    "<i>Everything in this PDF was measured on the held-out test tile "
    "T03CWT, which the model never saw during training. The training set "
    "was tiles T02CNA and T02CNC. Code, notebooks, and per-run metrics "
    "live under runs/ (winners) and archive/ (everything else we tried).</i>",
    SMALL))

# ---- Render ----
doc = SimpleDocTemplate(str(OUT), pagesize=LETTER,
                        leftMargin=0.75*inch, rightMargin=0.75*inch,
                        topMargin=0.75*inch, bottomMargin=0.75*inch,
                        title="Sea Ice Classification - Project Summary")
doc.build(story)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
