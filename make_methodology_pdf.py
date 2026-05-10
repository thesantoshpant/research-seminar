"""
Generate a shareable, plain-language methodology PDF for friends.
Output: methodology_for_friends.pdf
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    CondPageBreak,
    Table,
    TableStyle,
    KeepTogether,
    Preformatted,
    HRFlowable,
)

OUT = Path(r"C:\Users\Santosh\Desktop\0-100\Research Seminar\Project\methodology_for_friends.pdf")

# ---------------- Styles ----------------
styles = getSampleStyleSheet()

NAVY = colors.HexColor("#1F3A5F")
DARK_GRAY = colors.HexColor("#444444")
LIGHT_GRAY = colors.HexColor("#F4F4F4")
RULE_GRAY = colors.HexColor("#CCCCCC")

TITLE = ParagraphStyle(
    "Title", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=24, leading=28, alignment=TA_CENTER, textColor=NAVY, spaceAfter=6,
)
SUBTITLE = ParagraphStyle(
    "Subtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=14, leading=18, alignment=TA_CENTER, textColor=DARK_GRAY, spaceAfter=4,
)
META = ParagraphStyle(
    "Meta", parent=styles["Normal"], fontName="Helvetica",
    fontSize=11, leading=14, alignment=TA_CENTER, textColor=DARK_GRAY,
)
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=16, leading=20, textColor=NAVY, spaceBefore=16, spaceAfter=4,
    keepWithNext=1,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=2,
    keepWithNext=1,
)
BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=11, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=6,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, leftIndent=18, bulletIndent=6, spaceAfter=3,
)
CALLOUT = ParagraphStyle(
    "Callout", parent=BODY, fontName="Helvetica-Oblique",
    fontSize=10.5, leading=14, leftIndent=14, rightIndent=14,
    textColor=DARK_GRAY, spaceBefore=4, spaceAfter=8,
)
MONO = ParagraphStyle(
    "Mono", parent=styles["Code"], fontName="Courier",
    fontSize=9, leading=11.5, leftIndent=8, rightIndent=8,
    backColor=LIGHT_GRAY, borderPadding=6, spaceBefore=4, spaceAfter=8,
)


def hr(width=1.0):
    return HRFlowable(width="100%", thickness=0.6, color=RULE_GRAY,
                      spaceBefore=2, spaceAfter=8)


def section(title):
    # Force a new page if there isn't ~1.5 inches of room left, so a section
    # header is never orphaned alone at the bottom of a page.
    return [
        CondPageBreak(1.5 * inch),
        KeepTogether([Paragraph(title, H1), hr()]),
    ]


def p(text):
    return Paragraph(text, BODY)


def b(text):
    return Paragraph(text, BULLET, bulletText="•")


def callout(text):
    return Paragraph(text, CALLOUT)


def make_table(data, col_widths=None, header_bg=NAVY, header_fg=colors.white):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, 0), header_fg),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("LEADING",    (0, 0), (-1, -1), 13),
        ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE_GRAY),
        ("INNERGRID", (0, 1), (-1, -1), 0.25, RULE_GRAY),
    ]
    t.setStyle(TableStyle(style))
    return t


# ---------------- Page footer (page number) ----------------
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(DARK_GRAY)
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.drawString(0.75 * inch, 0.5 * inch, "Sea Ice Deep Fusion - Methodology")
    canvas.restoreState()


# ---------------- Build content ----------------
story = []

# Title block
story += [
    Spacer(1, 1.0 * inch),
    Paragraph("Sea Ice Classification", TITLE),
    Paragraph("Using Deep Fusion of Satellite Imagery and Laser Photon Data", TITLE),
    Spacer(1, 0.25 * inch),
    Paragraph("A Plain-Language Walkthrough of the Methodology", SUBTITLE),
    Spacer(1, 0.4 * inch),
    Paragraph("Santosh Pant", META),
    Paragraph("Research Seminar Project &middot; May 2026", META),
    Spacer(1, 0.6 * inch),
    Paragraph(
        "This document explains, in everyday language, what we are building, what data we use, "
        "how the model works, and how we measure success. No prior machine-learning experience required.",
        ParagraphStyle("Abstract", parent=BODY, alignment=TA_CENTER,
                       leftIndent=0.5 * inch, rightIndent=0.5 * inch, fontSize=10.5,
                       textColor=DARK_GRAY),
    ),
    PageBreak(),
]

# Section 1
story += section("1. The Problem We Are Solving")
story += [
    p("The Arctic and Antarctic oceans are covered with sea ice that constantly forms, melts, "
      "drifts, and breaks apart. Knowing where the ice is, how thick it is, and how it is "
      "changing matters a lot - for climate science, for shipping routes through the Arctic, "
      "and for predicting how fast the polar regions are warming."),
    p("Our goal is to build a computer model that looks at a small image of a polar region and, "
      "for every pixel in that image, decides one of three things:"),
    b("<b>Ice</b> - solid, established sea ice (shown in red on our maps)."),
    b("<b>Thin ice</b> - newly forming or refreezing ice (shown in blue)."),
    b("<b>Water</b> - open ocean (shown in green)."),
    p("This kind of pixel-by-pixel labeling is called <b>semantic segmentation</b>. The output "
      "is a colored map the same size as the input image."),
]

# Section 2
story += section("2. The Data We Have")
story += [
    p("For each labeled point in our dataset, we have three pieces of information that all "
      "describe the same spot on Earth:"),
    Spacer(1, 4),
    Paragraph("<b>1. A satellite photo (Sentinel-2)</b>", H2),
    p("Sentinel-2 is a European satellite that takes color photos of Earth from space. We took "
      "the original large photos and cut out small 128 by 128 pixel squares centered on each "
      "labeled point. Think of it as a postage-stamp-sized aerial photo of a small piece of the "
      "Arctic Ocean."),

    Paragraph("<b>2. A row of measurements from ICESat-2</b>", H2),
    p("ICESat-2 is a NASA satellite that fires a laser at the Earth's surface and counts the "
      "photons that bounce back. From these photon returns, scientists derive about 22 numbers "
      "for each measured point - things like the average height of the surface, how reflective "
      "it is, the spread of photon arrival times, and so on. These numbers come from a CSV file "
      "(basically a spreadsheet)."),
    callout("The key idea: ICESat-2 sees the surface in 1D - one measurement after another along "
            "the satellite's flight path. Sentinel-2 sees the surface in 2D - a full picture. "
            "Combining both gives the model more information than either alone."),

    Paragraph("<b>3. A ground-truth segmentation map</b>", H2),
    p("Our professor's team built a hand-coded image-processing pipeline that converts each "
      "satellite photo into a colored map: every pixel is colored red, blue, or green for ice, "
      "thin ice, or water. This is the answer key our model learns to predict."),

    Spacer(1, 6),
    p("In total, we have <b>139,335</b> such (photo, CSV row, answer-key) triples, spread "
      "across <b>3 different geographic regions</b> (called \"tiles\") on <b>2 different dates</b> "
      "in November 2019."),
]

# Section 3
story += section("3. The Goal, Stated Precisely")
story += [
    p("Given a new satellite photo and the matching CSV row, predict the colored answer-key map."),
    p("Inputs:"),
    b("128 x 128 x 3 RGB photo (one piece of the Arctic seen from space)."),
    b("A small window of 32 nearby CSV rows from the ICESat-2 satellite track (so the model "
      "sees a bit of context, not just one isolated measurement)."),
    p("Output:"),
    b("128 x 128 prediction map, with each pixel labeled as ice, thin ice, or water."),
]

story += [PageBreak()]

# Section 4
story += section("4. How We Built the Training Data")
story += [
    p("Before any machine learning, we had to prepare the data. This was actually most of the "
      "work so far. The pipeline:"),
    b("Match each of the 6 CSV files to its matching satellite photo by tile code and date."),
    b("For every labeled point in the CSV, look up its (column, row) pixel position inside "
      "the satellite photo and crop a 128 by 128 piece centered on that point."),
    b("If the point is too close to the edge of the satellite photo, fill the missing area "
      "with black so every crop ends up the same size."),
    b("Run the professor's image-processing pipeline (HSV color thresholding plus shadow and "
      "cloud removal) on each crop to produce the colored answer-key map."),
    b("Save both the original photo and the answer-key map as PNG files, paired by name."),
    p("Result: two folders of 139,335 paired images each:"),
    Preformatted(
        "  outputs/                   <- original 128x128 RGB photos\n"
        "  outputs_segmented/         <- matching 128x128 answer-key maps",
        MONO,
    ),
]

# Section 5
story += section("5. The Model - Deep Fusion")
story += [
    p("The model has two specialized parts (called \"branches\") that work in parallel, then "
      "a fusion step that combines what they learned."),

    Paragraph("<b>Branch A: U-Net for the satellite photo</b>", H2),
    p("U-Net is a famous neural network architecture designed specifically for pixel-by-pixel "
      "image labeling. The intuition: imagine a person reading a map. First they zoom out to "
      "see the big picture - where are the major shapes, where are the boundaries. Then they "
      "zoom back in to label every fine detail. U-Net does exactly this, in two halves: an "
      "<b>encoder</b> that progressively shrinks the image to capture context, and a "
      "<b>decoder</b> that progressively re-expands to produce a full-resolution map."),
    p("We use a U-Net with a ResNet-18 backbone that has been pretrained on millions of "
      "everyday photos (ImageNet). That pretraining gives us a head-start: the network "
      "already knows about edges, textures, and shapes before it ever sees a single ice photo."),

    Paragraph("<b>Branch B: Bi-LSTM for the CSV measurements</b>", H2),
    p("ICESat-2 produces measurements in a long line along its flight path. Adjacent "
      "measurements are about 10 meters apart on the ground, so they are physically related. "
      "An LSTM (Long Short-Term Memory network) is a neural network that reads sequences of "
      "numbers - like reading a sentence word by word and remembering the context as it goes."),
    p("For every labeled point, we hand the LSTM a window of 32 nearby rows (16 before, 16 "
      "after, ordered along the satellite track). The LSTM digests this sequence and outputs "
      "a single 256-number summary that captures the local along-track context. \"Bi\" in "
      "Bi-LSTM means it reads the sequence in both directions, forwards and backwards, for "
      "richer context."),

    Paragraph("<b>Fusion: combining the two branches</b>", H2),
    p("Now we have two outputs:"),
    b("From U-Net: a 16-channel feature map of size 128 x 128. Think of it as a 16-page atlas "
      "where each page highlights different aspects of the photo."),
    b("From Bi-LSTM: a 256-number summary of the CSV neighborhood."),
    p("To combine them, we \"broadcast\" the 256-number CSV summary across the entire 128 x 128 "
      "spatial grid (so every pixel gets the same CSV summary attached to it), shrink it down "
      "to 16 channels, and concatenate with the image feature map. We now have 32 channels of "
      "fused information."),
    p("A small attention module (called Squeeze-and-Excitation) lets the model automatically "
      "decide which of the 32 channels are most useful. Finally, a 1 by 1 convolution turns "
      "the 32 channels into 3 channels - one prediction per class for every pixel."),

    Paragraph("<b>Architecture diagram</b>", H2),
    Preformatted(
        "  RGB photo                 U-Net (ResNet-18 encoder + decoder)\n"
        "  (3, 128, 128)   ------>   ------------------------------------>  feat_image\n"
        "                                                                   (16, 128, 128)\n"
        "                                                                        |\n"
        "  CSV window      Linear     Bi-LSTM       center pool                  |\n"
        "  (32, 22)    -> (32,128) -> (32, 256) ->   (256,)  -----+              |\n"
        "                                                          v              v\n"
        "                                                 broadcast & concat\n"
        "                                                          |\n"
        "                                                          v\n"
        "                                            attention + 1x1 conv\n"
        "                                                          |\n"
        "                                                          v\n"
        "                                                  prediction map\n"
        "                                                  (3, 128, 128)",
        MONO,
    ),
    callout("Why \"deep fusion\"? Because we combine the two modalities at the feature level "
            "(deep inside the network) rather than at the input or at the output. The paper we "
            "are benchmarking against shows this works best."),
]

story += [PageBreak()]

# Section 6
story += section("6. How We Train the Model")
story += [
    Paragraph("<b>Splitting the data</b>", H2),
    p("We split the 139,335 samples into three sets:"),
    make_table(
        [
            ["Split", "Tiles", "Samples", "Purpose"],
            ["Train", "T02CNA + T02CNC", "~84,000", "model learns from these"],
            ["Validation", "10% held out from train", "~9,300", "tune choices, pick best epoch"],
            ["Test", "T03CWT (new tile, new date)", "~46,000", "final honest evaluation"],
        ],
        col_widths=[0.9 * inch, 2.4 * inch, 0.9 * inch, 2.5 * inch],
    ),
    Spacer(1, 6),
    p("We split <b>by region, not by row</b>. If we split randomly by row, neighboring rows "
      "(which have nearly identical photo crops) would end up in different sets, letting the "
      "model \"cheat\" by memorizing training examples. Splitting by region forces the model to "
      "generalize to genuinely new places."),

    Paragraph("<b>Loss function</b>", H2),
    p("Our class distribution is heavily skewed: about 96% of pixels are ice, 3% thin ice, "
      "and 1% water. Without correction, the model would just say \"ice\" everywhere and be 96% "
      "accurate. To prevent this, we use <b>weighted cross-entropy loss</b>, which up-weights "
      "the rare classes so the model is heavily penalized for missing them."),

    Paragraph("<b>Augmentation</b>", H2),
    p("During training, we randomly flip and rotate each photo (and its matching answer key). "
      "This teaches the model that ice looks like ice no matter the orientation - it cannot "
      "rely on \"top of image is always sky-facing\" because there is no sky here. This trick is "
      "called <b>data augmentation</b>."),

    Paragraph("<b>Training loop details</b>", H2),
    make_table(
        [
            ["Setting", "Value", "Why"],
            ["Optimizer", "AdamW", "industry-standard, robust"],
            ["Learning rate", "1e-4 with cosine decay", "starts moderate, ends fine-tuned"],
            ["Batch size", "32", "fits comfortably on a single GPU"],
            ["Epochs", "30 (with early stopping)", "stop if validation stops improving"],
            ["Mixed precision", "Yes", "about 2x faster, same accuracy"],
            ["Random seeds", "3 (42, 7, 1337)", "report mean and spread for reliability"],
        ],
        col_widths=[1.5 * inch, 2.0 * inch, 3.5 * inch],
    ),
]

# Section 7
story += section("7. How We Measure Success")
story += [
    p("Our primary metric is <b>mean Intersection over Union (mIoU)</b>. This is the standard "
      "metric for segmentation tasks - and it is what the paper we are benchmarking against uses."),
    p("For each class, we measure:"),
    b("Pixels we correctly predicted as that class (true positives)."),
    b("Pixels we predicted as that class but were not (false positives)."),
    b("Pixels that were that class but we missed (false negatives)."),
    p("IoU = correctly_predicted / (correctly_predicted + missed + over-predicted). The score "
      "is between 0 and 1, where 1 is perfect."),
    p("mIoU is just the average IoU across all 3 classes. We also report:"),
    b("<b>Per-class IoU</b> - so we can see, for example, if the model is great at ice but "
      "bad at thin ice."),
    b("<b>Pixel accuracy</b> - the simple percentage of pixels correctly labeled."),
    b("<b>Confusion matrix</b> - a 3x3 grid showing exactly which classes the model confuses "
      "with which others."),
    b("<b>Sample predictions</b> - 50 side-by-side comparisons of (input photo, true map, "
      "predicted map) so we can eyeball the results."),
]

story += [PageBreak()]

# Section 8
story += section("8. Ablation Study - Does Fusion Actually Help?")
story += [
    p("The whole point of this project is to show that combining the two modalities (photo + "
      "CSV) beats using either one alone. To prove this, we train <b>six versions</b> of the "
      "model and compare their mIoU:"),
    make_table(
        [
            ["#", "Variant", "Image branch", "CSV branch", "Fusion"],
            ["1", "Image only", "U-Net", "-", "-"],
            ["2", "CSV only", "-", "Bi-LSTM", "-"],
            ["3", "Early fusion", "U-Net", "-", "concat at input"],
            ["4", "Late fusion", "U-Net", "Bi-LSTM", "average predictions"],
            ["5", "Deep fusion (concat)", "U-Net", "Bi-LSTM", "feature-level concat"],
            ["6", "Deep fusion (attention)", "U-Net", "Bi-LSTM", "feature-level + attention"],
        ],
        col_widths=[0.3 * inch, 1.7 * inch, 1.0 * inch, 1.0 * inch, 2.0 * inch],
    ),
    Spacer(1, 6),
    p("If everything works as expected, variants 5 and 6 (deep fusion) will outperform "
      "1-4. This mirrors the comparison done in the benchmark paper and gives us a clean story: "
      "<i>combining modalities deep inside the network is the best way to fuse them</i>."),
]

# Section 9
story += section("9. Comparison with the Benchmark Paper")
story += [
    p("Our work is directly inspired by:"),
    callout("Zhao, L., et al. (2023). \"Deep-Learning-Based Sea Ice Classification With Sentinel-1 "
            "and AMSR-2 Data.\" IEEE Journal of Selected Topics in Applied Earth Observations and "
            "Remote Sensing, vol. 16, pp. 5514-5525."),
    p("Their setup is similar but not identical to ours:"),
    make_table(
        [
            ["Aspect", "Their paper", "Our project"],
            ["Main image", "Sentinel-1 SAR (radar, 2 channels)", "Sentinel-2 (optical RGB)"],
            ["Aux modality", "AMSR-2 (microwave, 14-channel grid)", "ICESat-2 (laser photons, CSV)"],
            ["Aux shape", "2D dense raster", "1D sparse along-track sequence"],
            ["Aux encoder", "CNN", "Bi-LSTM"],
            ["Number of classes", "12 (11 ice types + land)", "3 (ice, thin ice, water)"],
            ["Task", "Per-pixel segmentation", "Per-pixel segmentation"],
            ["Primary metric", "mIoU", "mIoU"],
        ],
        col_widths=[1.4 * inch, 2.6 * inch, 2.6 * inch],
    ),
    Spacer(1, 6),
    p("We are not trying to <i>beat</i> their numbers - the modalities and class counts are "
      "different so the numbers are not directly comparable. What we want to show is that <b>the "
      "same architectural pattern they discovered (deep feature-level fusion beats early and "
      "late fusion) also holds when the auxiliary modality is a sparse 1D track instead of a "
      "dense 2D grid</b>. That is the genuine contribution of this project."),
]

# Section 10
story += section("10. What Comes Next")
story += [
    p("After we get the baseline numbers, possible extensions include:"),
    b("Adding more tiles and more dates to test how well the model generalizes across seasons."),
    b("Replacing the LSTM with a Transformer for the CSV branch (Transformers often do better "
      "on long sequences)."),
    b("Adding Sentinel-1 SAR as a third modality, so we have radar + optical + photons."),
    b("Trying self-supervised pretraining on millions of unlabeled satellite photos to "
      "learn better visual features before any fine-tuning."),
    b("Producing high-resolution sea-ice maps for a full polar region by tiling and stitching "
      "predictions together."),
]

# Closing
story += [
    Spacer(1, 0.3 * inch),
    hr(),
    Paragraph(
        "Questions or feedback? This document is a living draft - corrections welcome.",
        ParagraphStyle("Closing", parent=BODY, alignment=TA_CENTER,
                       fontSize=10, textColor=DARK_GRAY),
    ),
]

# ---------------- Build the PDF ----------------
doc = SimpleDocTemplate(
    str(OUT),
    pagesize=LETTER,
    leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    topMargin=0.75 * inch, bottomMargin=0.85 * inch,
    title="Sea Ice Classification - Methodology",
    author="Santosh Pant",
)
doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"PDF written: {OUT}")
print(f"Size: {OUT.stat().st_size / 1024:.1f} KB")
