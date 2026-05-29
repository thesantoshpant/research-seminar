from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus.flowables import Flowable

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

# ── colour palette (minimal, formal) ──────────────────────────────────────────
BLACK  = colors.HexColor('#111111')
GREY   = colors.HexColor('#555555')
LGREY  = colors.HexColor('#999999')
RULE   = colors.HexColor('#CCCCCC')
HLROW  = colors.HexColor('#F5F5F5')
HLCOL  = colors.HexColor('#EEF3FA')
ACCENT = colors.HexColor('#1A3A6B')   # dark navy, used only for heading rules
WHITE  = colors.white

def build_styles():
    base = getSampleStyleSheet()

    def ps(name, **kw):
        defaults = dict(fontName='Helvetica', fontSize=10, leading=14,
                        textColor=BLACK, spaceAfter=0, spaceBefore=0)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    return {
        'title'    : ps('title',    fontName='Helvetica-Bold', fontSize=22,
                        leading=28, textColor=ACCENT, alignment=TA_CENTER,
                        spaceAfter=4),
        'subtitle' : ps('subtitle', fontSize=12, leading=16, textColor=GREY,
                        alignment=TA_CENTER, spaceAfter=2),
        'meta'     : ps('meta',     fontSize=9,  leading=12, textColor=LGREY,
                        alignment=TA_CENTER),
        'h1'       : ps('h1',       fontName='Helvetica-Bold', fontSize=14,
                        leading=18, textColor=ACCENT, spaceBefore=18,
                        spaceAfter=4),
        'h2'       : ps('h2',       fontName='Helvetica-Bold', fontSize=11,
                        leading=14, textColor=BLACK, spaceBefore=10,
                        spaceAfter=3),
        'h3'       : ps('h3',       fontName='Helvetica-BoldOblique', fontSize=10,
                        leading=13, textColor=GREY, spaceBefore=6, spaceAfter=2),
        'body'     : ps('body',     fontSize=10, leading=14, alignment=TA_JUSTIFY,
                        spaceAfter=5),
        'bullet'   : ps('bullet',   fontSize=10, leading=14, leftIndent=14,
                        firstLineIndent=-10, spaceAfter=3),
        'caption'  : ps('caption',  fontSize=8,  leading=11, textColor=LGREY,
                        alignment=TA_CENTER, spaceAfter=6),
        'small'    : ps('small',    fontSize=8,  leading=11, textColor=GREY),
        'note'     : ps('note',     fontSize=9,  leading=12, textColor=GREY,
                        fontName='Helvetica-Oblique', alignment=TA_CENTER),
    }


class SectionRule(Flowable):
    """Thin coloured rule under a section heading."""
    def __init__(self, width, color=ACCENT, thickness=1):
        super().__init__()
        self.width, self.color, self.thickness = width, color, thickness
        self.height = self.thickness + 2

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.thickness, self.width, self.thickness)


def rule(S, thin=False):
    return [SectionRule(PAGE_W - 2*MARGIN, thickness=0.5 if thin else 1),
            Spacer(1, 6 if thin else 8)]


def h1(text, S):
    return [Paragraph(text, S['h1'])] + rule(S)


def h2(text, S):
    return [Spacer(1, 4), Paragraph(text, S['h2']),
            SectionRule(PAGE_W - 2*MARGIN, color=RULE, thickness=0.5),
            Spacer(1, 4)]


def bullet(items, S, label='•'):
    out = []
    for item in items:
        out.append(Paragraph(f'<b>{label}</b>&nbsp;&nbsp;{item}', S['bullet']))
    return out


def numbered(items, S):
    out = []
    for i, item in enumerate(items, 1):
        out.append(Paragraph(f'<b>{i}.</b>&nbsp;&nbsp;{item}', S['bullet']))
    return out


def kv_table(rows, S, col_widths=None):
    """Two-column key/value table, no lines."""
    usable = PAGE_W - 2*MARGIN
    col_widths = col_widths or [usable*0.38, usable*0.62]
    data = [[Paragraph(f'<b>{k}</b>', S['small']),
             Paragraph(v, S['small'])] for k, v in rows]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('VALIGN',     (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    return t


# ── page-number footer ────────────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(LGREY)
    total = doc.page   # approximation; reset below
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN*0.55,
                           f'Page {doc.page}')
    canvas.drawString(MARGIN, MARGIN*0.55,
                      'Sea Ice Deep Fusion — Workflow Overview')
    canvas.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
def build_story(S):
    story = []
    usable = PAGE_W - 2*MARGIN

    # ── PAGE 1: TITLE ─────────────────────────────────────────────────────────
    story += [
        Spacer(1, 3*cm),
        Paragraph('Sea Ice Deep Fusion', S['title']),
        Paragraph('Workflow Overview', S['subtitle']),
        Spacer(1, 0.3*cm),
        HRFlowable(width=usable, thickness=1, color=ACCENT, spaceAfter=12),
        Paragraph(
            'Combining satellite imagery with ICESat-2 laser altimetry to map '
            'Arctic sea ice: thick ice, thin ice, and open water',
            S['body']
        ),
        Spacer(1, 0.6*cm),
        Paragraph('Research Seminar &nbsp;·&nbsp; Sea Ice Analysis', S['meta']),
        Spacer(1, 1.5*cm),
    ]

    # key-metrics summary table
    metrics = [
        ['Metric', 'Value', 'Description'],
        ['Test mIoU',        '0.8982', 'Mean intersection over union on held-out tile T03CWT'],
        ['Pixel Accuracy',   '95.3%',  'Fraction of correctly classified pixels'],
        ['Macro F1',         '0.9453', 'Unweighted average F1 across all three classes'],
    ]
    mt = Table(metrics, colWidths=[usable*0.25, usable*0.2, usable*0.55])
    mt.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('LEADING',     (0,0), (-1,-1), 14),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [mt, Spacer(1, 0.8*cm)]

    # class legend
    story += h2('Target Classes', S)
    cls_data = [
        ['Class', 'Label', 'Description'],
        ['1', 'Thick Ice', 'Multi-year or first-year consolidated ice'],
        ['2', 'Thin Ice / Melt Pond', 'Newly formed or flooded surface ice'],
        ['3', 'Open Water', 'Ice-free ocean surface'],
    ]
    ct = Table(cls_data, colWidths=[usable*0.08, usable*0.25, usable*0.67])
    ct.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [ct, Spacer(1, 0.3*cm)]

    story += [
        Paragraph(
            '3-class segmentation &nbsp;·&nbsp; tile-grouped train/test split '
            '&nbsp;·&nbsp; focal loss',
            S['note']
        ),
        PageBreak(),
    ]

    # ── PAGE 2: PROBLEM & DATA ────────────────────────────────────────────────
    story += h1('1. Problem Statement and Data', S)

    story += h2('1.1  Why Multimodal Fusion Is Required', S)
    story += bullet([
        'RGB imagery alone cannot reliably distinguish thin ice from thick ice; '
        'both appear nearly identical in visible wavelengths.',
        'Melt ponds introduce additional spectral confusion within the visible signal.',
        'Thin ice is rare in the dataset yet critically important for climate research.',
        'The test tile (T03CWT) represents a geographically distinct region with no '
        'spatial overlap with the training tiles, ensuring an honest generalisation test.',
    ], S)

    story += h2('1.2  Proposed Approach', S)
    story += bullet([
        'Satellite RGB patches (128×128 px) provide spatial texture; '
        'a U-Net architecture processes this modality.',
        'ICESat-2 ATL03 height and roughness features provide altimetric context; '
        'an LSTM processes the along-track sequence.',
        'Feature sets from both branches are merged prior to the final classification layer.',
        'A ResNet-18 encoder initialised from ImageNet weights is used for the image branch.',
        'A Squeeze-and-Excitation (SE) block re-weights channels after the fusion step.',
    ], S)

    story += h2('1.3  Data Sources', S)

    data_tbl = [
        ['Source', 'Description'],
        ['Satellite patches',
         '128×128 px RGB crops. Training tiles: T02CNA, T02CNC. '
         'Test tile: T03CWT (held out entirely). Normalised with ImageNet mean/std. '
         'Augmented with random horizontal flip and 90° rotations.'],
        ['ICESat-2 ATL03',
         'Along-track photon data at ~10 m spacing. Eight features per segment: '
         'h_cor_mean, h_diff, rel_height_min_elev, height_sd, '
         'pcnth_mean, pcnt_mean, bcnt_mean, brate_mean. '
         'Processed as a 5-segment sliding window.'],
        ['Label masks',
         'RGB-coded segmentation maps. Red (255, 0, 0) = thick ice; '
         'Blue (0, 0, 255) = thin ice / melt pond; '
         'Green (0, 255, 0) = open water.'],
    ]
    dt = Table(data_tbl, colWidths=[usable*0.22, usable*0.78])
    dt.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('LEADING',     (0,0), (-1,-1), 13),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('VALIGN',      (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',  (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [dt, Spacer(1, 0.5*cm)]

    story += h2('1.4  Train / Validation / Test Split', S)
    split_tbl = [
        ['Split', 'Tiles',                'Proportion', 'Notes'],
        ['Train',      'T02CNA + T02CNC',  '~90%',       'Primary training data'],
        ['Validation', 'T02CNA + T02CNC',  '~10%',       'Random held-out subset of train tiles'],
        ['Test',       'T03CWT only',       '—',
         'Completely unseen; geographically distinct from training region'],
    ]
    st = Table(split_tbl, colWidths=[usable*0.12, usable*0.26, usable*0.14, usable*0.48])
    st.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('LEADING',     (0,0), (-1,-1), 13),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('VALIGN',      (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [st, Spacer(1, 0.3*cm),
              Paragraph(
                  'Keeping T03CWT entirely out of training ensures that test metrics '
                  'reflect genuine generalisation rather than spatial memorisation.',
                  S['note']
              ),
              PageBreak()]

    # ── PAGE 3: MODEL ARCHITECTURE ────────────────────────────────────────────
    story += h1('2. Model Architecture', S)
    story += [
        Paragraph(
            'The model consists of two parallel branches whose representations are '
            'concatenated and passed through a shared fusion head.',
            S['body']
        ),
    ]

    story += h2('2.1  Image Branch (U-Net)', S)
    story += bullet([
        'Input: RGB patch of shape B × 3 × 128 × 128.',
        'Encoder: ResNet-18 pretrained on ImageNet. Backbone weights are loaded '
        'and fine-tuned at the standard learning rate.',
        'Decoder: U-Net-style bilinear upsampling to restore full spatial resolution.',
        'Output: B × 16 × 128 × 128 feature map.',
    ], S)

    story += h2('2.2  CSV Branch (LSTM)', S)
    story += bullet([
        'Input: sliding window of 5 ATL03 segments, each with 8 features — '
        'shape B × 5 × 8.',
        'Sequence model: unidirectional single-layer LSTM, hidden size 96, tanh activation.',
        'The final hidden state h_T (shape B × 96) is extracted.',
        'Two dense layers (96 → 16 → 16) with ELU activation and dropout '
        'project the hidden state to a 16-dimensional vector.',
        'The vector is tiled and expanded to match the spatial dimensions: '
        'B × 16 × 128 × 128.',
    ], S)

    story += h2('2.3  Fusion Head', S)
    story += bullet([
        'Concatenation of both branch outputs along the channel axis: '
        'B × 32 × 128 × 128.',
        'Squeeze-and-Excitation (SE) block: global average pooling, two FC layers, '
        'sigmoid gating — re-weights channels adaptively.',
        '3×3 convolution with BatchNorm, ReLU, and dropout.',
        '1×1 convolution producing 3-class logits: B × 3 × 128 × 128.',
        'Output: per-pixel class prediction B × 3 × 128 × 128.',
    ], S)

    story += h2('2.4  Training Configuration', S)
    cfg_rows = [
        ('Loss function',    'Focal loss; α = [0.05, 0.45, 0.60], γ = 2.0 '
                             '(thin ice class up-weighted via α = 0.45)'),
        ('Optimiser',        'Adam'),
        ('Learning rate',    '1 × 10⁻⁴ for newly initialised layers; '
                             '1 × 10⁻⁵ for pre-loaded LSTM weights'),
        ('LR schedule',      'Cosine annealing'),
        ('Precision',        'FP16 automatic mixed precision (AMP)'),
        ('Early stopping',   'After 8 consecutive epochs without validation mIoU improvement'),
        ('LSTM initialisation',
         'Weights from the best LSTM sweep run (hidden = 96) are loaded into '
         'the CSV branch. In Fusion v4 the LSTM weights are frozen; '
         'in Deep Fusion v5 they are fine-tuned at 0.1× the base LR.'),
    ]
    story += [kv_table(cfg_rows, S), PageBreak()]

    # ── PAGE 4: TRAINING PROCEDURE ────────────────────────────────────────────
    story += h1('3. Training Procedure', S)
    story += [
        Paragraph(
            'Training follows a six-step pipeline. The LSTM branch is first '
            'optimised independently via a hyperparameter sweep, and the winning '
            'weights are then incorporated into the full fusion model.',
            S['body']
        ),
    ]

    steps = [
        ('Match images to CSV rows',
         'Each 128×128 patch is linked to an ATL03 row via tile identifier, '
         'beam identifier, and row_idx stored in the manifest file. '
         'Eight-feature arrays are built for every CSV segment.'),
        ('Normalise and cache CSV features',
         'CSV features are z-scored using mean and standard deviation computed '
         'from training tiles only — test statistics never influence normalisation. '
         'Normalised arrays are saved as .npy files to avoid recomputation each epoch. '
         'Image normalisation uses ImageNet statistics and is applied on-the-fly. '
         'The training set receives random horizontal flips and 90° rotations.'),
        ('Identify optimal LSTM hyperparameters',
         'Prior to constructing the fusion model, a sweep is conducted over '
         'hidden size {32, 64, 96}, learning rate, dropout, focal loss parameters '
         '(γ, α), and window length. '
         'Best configuration: hidden = 96, lr ≈ 1 × 10⁻³, '
         'α = [0.05, 0.45, 0.60], γ = 2.0, seq_len = 5.'),
        ('Assemble the fusion model',
         'A U-Net with a ResNet-18 encoder (ImageNet weights) produces a '
         '16-channel feature map at full resolution. The sweep-winner LSTM '
         'weights are loaded into the CSV branch. The 96-dimensional hidden '
         'state is tiled to match the 128×128 spatial size before concatenation.'),
        ('Train with differentiated learning rates',
         'Focal loss with α = 0.45 on the thin ice class substantially reduces '
         'the tendency to ignore the rare class. Loaded (pretrained) layers train '
         'at LR = 1 × 10⁻⁵; freshly initialised layers train at LR = 1 × 10⁻⁴. '
         'Early stopping triggers after 8 epochs with no validation mIoU gain.'),
        ('Evaluate on held-out tile',
         'The checkpoint with the highest validation mIoU (best.pt) is loaded '
         'and applied to T03CWT — a tile the model has never seen. '
         'Per-class IoU, precision, recall, F1, and macro averages are reported. '
         'The confusion matrix is row-normalised to percentages.'),
    ]

    for i, (title, desc) in enumerate(steps, 1):
        story += [
            Paragraph(f'<b>Step {i}: {title}</b>', S['h3']),
            Paragraph(desc, S['body']),
        ]

    story += [
        Spacer(1, 0.3*cm),
        Paragraph(
            'Training tiles: T02CNA + T02CNC. &nbsp; Test tile: T03CWT. &nbsp; '
            'No geographic overlap between the two sets — the reported numbers '
            'are honest out-of-distribution estimates.',
            S['note']
        ),
        PageBreak(),
    ]

    # ── PAGE 5: MODEL COMPARISON ──────────────────────────────────────────────
    story += h1('4. Model Comparison', S)
    story += [
        Paragraph(
            'Five model variants are evaluated on the held-out test tile T03CWT. '
            'Results are summarised below.',
            S['body']
        ),
    ]

    story += h2('4.1  Test mIoU — All Models', S)
    cmp_tbl = [
        ['Model', 'mIoU', 'Pixel Acc.', 'IoU Thick', 'IoU Thin', 'IoU Water'],
        ['U-Net (image only)',      '0.8704', '94.29%', '0.9299', '0.7683', '0.9130'],
        ['LSTM (CSV only)',         '0.2420', '72.54%', '0.7254', '0.0000', '0.0005'],
        ['Fusion Deep v1',         '0.8915', '95.12%', '0.9396', '0.8015', '0.9335'],
        ['Fusion v4 ★ (best)',     '0.8982', '95.31%', '0.9407', '0.8157', '0.9382'],
        ['Deep Fusion v5',         '0.8896', '95.02%', '—',      '—',      '—'],
    ]
    cw = [usable*0.24, usable*0.1, usable*0.12,
          usable*0.14, usable*0.14, usable*0.14]
    ct2 = Table(cmp_tbl, colWidths=cw)
    best_row = 4  # Fusion v4
    ct2.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('LEADING',     (0,0), (-1,-1), 13),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('BACKGROUND',  (0,best_row), (-1,best_row), HLCOL),
        ('FONTNAME',    (0,best_row), (-1,best_row), 'Helvetica-Bold'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [ct2, Spacer(1, 0.3*cm),
              Paragraph(
                  '★ Best overall: Fusion v4 — frozen sweep-winner LSTM + U-Net + SE fusion head.',
                  S['note']
              ),
              Spacer(1, 0.5*cm)]

    story += h2('4.2  Per-class IoU — Best Model (Fusion v4)', S)
    pc_tbl = [
        ['Class', 'IoU', 'Precision', 'Recall', 'F1'],
        ['Thick Ice', '0.9407', '~0.97', '~0.96', '0.9695'],
        ['Thin Ice',  '0.8157', '~0.86', '~0.93', '0.8985'],
        ['Water',     '0.9382', '~0.99', '~0.95', '0.9681'],
    ]
    cw2 = [usable*0.2, usable*0.16, usable*0.16, usable*0.16, usable*0.16]
    pt = Table(pc_tbl, colWidths=cw2)
    pt.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('LEADING',     (0,0), (-1,-1), 13),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [pt, PageBreak()]

    # ── PAGE 6: RESULTS BREAKDOWN ─────────────────────────────────────────────
    story += h1('5. Results Breakdown', S)
    story += [
        Paragraph(
            'Thin ice is the primary performance bottleneck. The following sections '
            'analyse each class and the key design decisions that contributed to '
            'the final results.',
            S['body']
        ),
    ]

    story += h2('5.1  Confusion Matrix (Row-Normalised, Best Model)', S)
    conf_tbl = [
        ['',           'Pred: Thick Ice', 'Pred: Thin Ice', 'Pred: Water'],
        ['Thick Ice',  '96.14%',          '3.86%',          '0.00%'],
        ['Thin Ice',   '7.06%',           '92.78%',         '0.16%'],
        ['Water',      '0.52%',           '5.01%',          '94.46%'],
    ]
    cw3 = [usable*0.2, usable*0.26, usable*0.26, usable*0.26]
    conf = Table(conf_tbl, colWidths=cw3)
    conf.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,0), (0,-1),  'Helvetica-Bold'),
        ('FONTNAME',    (1,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('LEADING',     (0,0), (-1,-1), 13),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('BACKGROUND',  (0,1), (0,-1),  ACCENT),
        ('TEXTCOLOR',   (0,1), (0,-1),  WHITE),
        # diagonal (correct predictions)
        ('BACKGROUND',  (1,1), (1,1),   colors.HexColor('#C8E6C9')),
        ('BACKGROUND',  (2,2), (2,2),   colors.HexColor('#C8E6C9')),
        ('BACKGROUND',  (3,3), (3,3),   colors.HexColor('#C8E6C9')),
        ('ROWBACKGROUNDS', (1,1), (-1,-1), [WHITE, HLROW]),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',  (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [conf, Spacer(1, 0.5*cm)]

    story += h2('5.2  Per-class Analysis', S)
    analyses = [
        ('Thick ice — nearly clean',
         'IoU 0.9407 / F1 0.9695. Only ~3% of thick ice pixels are mislabelled, '
         'making it the easiest class for the model.'),
        ('Thin ice — primary bottleneck',
         'IoU 0.8157 / F1 0.8985. Most confusion is with thick ice. The focal loss '
         'weight α = 0.45 substantially mitigates the class imbalance.'),
        ('Open water — texture discriminates well',
         'IoU 0.9382 / F1 0.9681. Recall of 94.5% indicates that U-Net captures '
         'water surface texture effectively.'),
    ]
    for title, desc in analyses:
        story += [Paragraph(f'<b>{title}:</b> {desc}', S['bullet'])]

    story += [Spacer(1, 0.4*cm)]
    story += h2('5.3  Key Findings', S)
    findings = [
        ('<b>Fusion outperforms single-modality baselines.</b> '
         'LSTM alone achieves mIoU 0.24; U-Net alone achieves 0.87; '
         'fusion reaches 0.8982 — approximately 3 percentage points above the '
         'best single-modality model.'),
        ('<b>Pretrained LSTM weights provided free gains.</b> '
         'Loading the sweep-winner weights into the fusion model yielded +0.6 pp mIoU '
         'over random initialisation with no additional training time.'),
        ('<b>The SE block contributes consistent marginal improvements.</b> '
         'Re-weighting channels after concatenation improved thin ice IoU by '
         'approximately 1 percentage point while adding negligible parameters.'),
        ('<b>Tile-grouped splits are essential for honest evaluation.</b> '
         'Random splits would inflate scores due to spatial correlation across tiles.'),
    ]
    for f in findings:
        story += [Paragraph(f'• &nbsp;{f}', S['bullet'])]

    story += [PageBreak()]

    # ── PAGE 7: NUMBERS TABLE ─────────────────────────────────────────────────
    story += h1('6. Quantitative Results Summary', S)
    story += [
        Paragraph(
            'The table below consolidates all reported metrics across model iterations.',
            S['body']
        ),
        Spacer(1, 0.3*cm),
    ]

    full_tbl = [
        ['Model', 'mIoU', 'Pixel Acc.', 'IoU Thick', 'IoU Thin', 'IoU Water'],
        ['U-Net (image only)',   '0.8704', '94.29%', '0.9299', '0.7683', '0.9130'],
        ['LSTM (CSV only)',      '0.2420', '72.54%', '0.7254', '0.0000', '0.0005'],
        ['Fusion Deep v1',      '0.8915', '95.12%', '0.9396', '0.8015', '0.9335'],
        ['Fusion v4 ★',         '0.8982', '95.31%', '0.9407', '0.8157', '0.9382'],
        ['Deep Fusion v5',      '0.8896', '95.02%', '—',      '—',      '—'],
    ]
    cw4 = [usable*0.26, usable*0.11, usable*0.13,
           usable*0.14, usable*0.14, usable*0.14]
    ft = Table(full_tbl, colWidths=cw4)
    ft.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('LEADING',     (0,0), (-1,-1), 14),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('BACKGROUND',  (0,4), (-1,4),  HLCOL),
        ('FONTNAME',    (0,4), (-1,4),  'Helvetica-Bold'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',  (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [ft, Spacer(1, 0.3*cm),
              Paragraph(
                  '★ Best overall (Fusion v4): frozen sweep-winner LSTM + U-Net + SE fusion head. '
                  'Per-class IoU for Deep Fusion v5 was not recorded in the final report.',
                  S['note']
              ),
              Spacer(1, 0.8*cm)]

    story += h2('6.1  mIoU Progression Across Iterations', S)
    prog_tbl = [
        ['Iteration', 'Model',          'mIoU',  'Notes'],
        ['1',  'U-Net only',      '0.870',  'Image modality baseline'],
        ['2',  'LSTM only',       '0.242',  'CSV modality baseline; severely limited alone'],
        ['3',  'Fusion Deep v1',  '0.891',  'First fusion attempt'],
        ['4',  'Fusion v4 ★',    '0.898',  'Frozen LSTM weights; best result'],
        ['5',  'Deep Fusion v5',  '0.890',  'Fine-tuned LSTM; slight regression'],
    ]
    cw5 = [usable*0.1, usable*0.24, usable*0.12, usable*0.54]
    pt2 = Table(prog_tbl, colWidths=cw5)
    pt2.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('LEADING',     (0,0), (-1,-1), 13),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('BACKGROUND',  (0,4), (-1,4),  HLCOL),
        ('FONTNAME',    (0,4), (-1,4),  'Helvetica-Bold'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',       (2,0), (2,-1), 'CENTER'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [pt2, PageBreak()]

    # ── PAGE 8: TAKEAWAYS & FUTURE WORK ──────────────────────────────────────
    story += h1('7. Conclusions and Future Work', S)

    story += h2('7.1  Key Conclusions', S)
    conclusions = [
        ('<b>C1 — Multimodal fusion is effective.</b> '
         'The best fusion model achieves mIoU 0.8982, compared to 0.87 for '
         'U-Net alone and 0.24 for LSTM alone.'),
        ('<b>C2 — Thin ice remains the weak point.</b> '
         'IoU 0.82 for thin ice, despite focal loss up-weighting (α = 0.45). '
         'The class imbalance problem is partially addressed but not fully resolved.'),
        ('<b>C3 — LSTM pretraining provided cost-free improvement.</b> '
         'Loading sweep-winner weights yielded approximately +0.6 pp mIoU '
         'with no additional training budget.'),
        ('<b>C4 — Tile-grouped splits are critical for honest evaluation.</b> '
         'Spatially random splits would inflate scores due to cross-tile correlation.'),
        ('<b>C5 — The SE block is a small but consistent contributor.</b> '
         'The block adds almost no parameters but produces measurable improvements '
         'in thin ice IoU.'),
    ]
    for c in conclusions:
        story += [Paragraph(f'• &nbsp;{c}', S['bullet'])]

    story += [Spacer(1, 0.4*cm)]
    story += h2('7.2  Future Work', S)
    future = [
        ('<b>F1 — Bi-directional LSTM.</b> A BiLSTM variant was tested standalone '
         'and showed improvement; integration into the fusion model is pending.'),
        ('<b>F2 — Test-time augmentation.</b> Averaging predictions over flipped and '
         'rotated inputs could reduce output variance without retraining.'),
        ('<b>F3 — Multi-date temporal stacking.</b> Incorporating imagery from '
         'multiple acquisition dates may improve performance near melt season transitions.'),
        ('<b>F4 — Attention mechanisms.</b> Replacing the fixed SE block with a '
         'full spatial attention layer may provide greater flexibility for edge cases.'),
        ('<b>F5 — Extended altimetry products.</b> Exploring ATL07 and ATL10 '
         'products would provide additional signal types beyond ATL03.'),
    ]
    for f in future:
        story += [Paragraph(f'• &nbsp;{f}', S['bullet'])]

    story += [Spacer(1, 0.6*cm)]
    story += h2('7.3  Technology Stack', S)
    stack_tbl = [
        ['Component',          'Details'],
        ['Deep learning framework', 'PyTorch 2.x with FP16 AMP training'],
        ['Segmentation library',   'segmentation-models-pytorch (smp) / U-Net'],
        ['Image encoder',          'ResNet-18, pretrained on ImageNet'],
        ['Altimetry data',         'ICESat-2 ATL03, along-track laser altimetry'],
        ['Loss function',          'Focal loss for class-imbalance correction'],
        ['Attention module',       'Squeeze-and-Excitation (SE) block for '
                                   'channel re-weighting'],
    ]
    st2 = Table(stack_tbl, colWidths=[usable*0.32, usable*0.68])
    st2.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('LEADING',     (0,0), (-1,-1), 13),
        ('TEXTCOLOR',   (0,0), (-1,0),  WHITE),
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, HLROW]),
        ('VALIGN',      (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, RULE),
    ]))
    story += [st2, Spacer(1, 0.8*cm)]

    # final summary bar
    story += [
        HRFlowable(width=usable, thickness=1, color=ACCENT, spaceAfter=10),
        Paragraph(
            '<b>Best model (Fusion v4):</b> &nbsp; mIoU 0.8982 &nbsp;·&nbsp; '
            'Pixel accuracy 95.3% &nbsp;·&nbsp; Macro F1 0.9453',
            ParagraphStyle('sumbar', fontName='Helvetica-Bold', fontSize=10,
                           leading=14, textColor=ACCENT, alignment=TA_CENTER)
        ),
        Paragraph(
            'Thick ice IoU 0.9407 &nbsp;·&nbsp; Thin ice IoU 0.8157 '
            '&nbsp;·&nbsp; Water IoU 0.9382',
            S['note']
        ),
        HRFlowable(width=usable, thickness=1, color=ACCENT, spaceBefore=10),
    ]

    return story


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    out_path = 'sea_ice_deep_fusion_workflow.pdf'
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN * 1.4,
        title='Sea Ice Deep Fusion — Workflow Overview',
        author='Research Seminar',
    )
    S = build_styles()
    story = build_story(S)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f'Written: {out_path}')
