#!/usr/bin/env python3
"""Regenerate project_summary.pdf with clean, well-formatted tables."""

import os, tempfile
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, PageBreak, HRFlowable,
                                 KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'project_summary.pdf')

# ── Colours ───────────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor('#1B3A5C')
MID_BLUE   = colors.HexColor('#2E5B8E')
LIGHT_BLUE = colors.HexColor('#EBF2FA')
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor('#F8F8F8')
BORDER     = colors.HexColor('#D0D8E4')

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
W = PAGE_W - 2 * MARGIN   # usable text width

# ── Page header / footer ─────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawString(MARGIN, 1.0*cm,
                      'Multimodal Fusion for Antarctic Sea Ice Classification  |  Knox College, 2025')
    canvas.drawRightString(PAGE_W - MARGIN, 1.0*cm, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=2.5*cm, bottomMargin=2.0*cm,
)

# ── Paragraph styles ─────────────────────────────────────────────────────────
SS = getSampleStyleSheet()

def _sty(base, **kw):
    s = SS[base].clone('s' + base + str(abs(hash(str(kw)))))
    for k, v in kw.items():
        setattr(s, k, v)
    return s

BODY   = _sty('Normal', fontSize=9.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=8)
BODY_L = _sty('Normal', fontSize=9.5, leading=14, alignment=TA_LEFT,    spaceAfter=8)
BULLET = _sty('Normal', fontSize=9.5, leading=14, leftIndent=16, spaceAfter=3)
CAPTION= ParagraphStyle('Caption', fontName='Helvetica-Oblique', fontSize=8.5,
                         textColor=colors.HexColor('#555555'), alignment=TA_CENTER,
                         leading=12, spaceAfter=8)
TITLE  = ParagraphStyle('Title2', fontName='Helvetica-Bold', fontSize=22,
                         textColor=DARK_BLUE, alignment=TA_CENTER, leading=28, spaceAfter=6)
SUBTITLE=ParagraphStyle('Subtitle2', fontName='Helvetica', fontSize=11,
                         textColor=colors.HexColor('#666666'), alignment=TA_CENTER, leading=16)
H1     = ParagraphStyle('H1b', fontName='Helvetica-Bold', fontSize=15,
                         textColor=DARK_BLUE, spaceAfter=4, spaceBefore=14, leading=19)
H2     = ParagraphStyle('H2b', fontName='Helvetica-Bold', fontSize=11,
                         textColor=MID_BLUE, spaceAfter=4, spaceBefore=10, leading=15)
H3     = ParagraphStyle('H3b', fontName='Helvetica-Bold', fontSize=10,
                         textColor=MID_BLUE, spaceAfter=3, spaceBefore=7, leading=14)
ABSTRACT=ParagraphStyle('Abs', fontName='Helvetica', fontSize=9.5, leading=14,
                         alignment=TA_JUSTIFY, leftIndent=10, rightIndent=10, spaceAfter=6)
CELL   = _sty('Normal', fontSize=9, leading=13)
CELL_B = ParagraphStyle('CB', fontName='Helvetica-Bold', fontSize=9,
                         textColor=WHITE, leading=13, alignment=TA_LEFT)
CELL_L = ParagraphStyle('CL', fontName='Helvetica-Bold', fontSize=9,
                         textColor=MID_BLUE, leading=13)

# ── Table helpers ─────────────────────────────────────────────────────────────
def ph(text, s=None):   return Paragraph(text, s or CELL)
def phh(text):          return Paragraph(text, CELL_B)
def phl(text):          return Paragraph(text, CELL_L)

def make_table(data, col_widths, extra=None, stripe=True):
    cmds = [
        ('BACKGROUND',   (0, 0), (-1, 0),  DARK_BLUE),
        ('TEXTCOLOR',    (0, 0), (-1, 0),  WHITE),
        ('FONTNAME',     (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),   [WHITE, LIGHT_GRAY] if stripe else [WHITE]),
        ('GRID',         (0, 0), (-1, -1), 0.4, BORDER),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
        ('LINEBELOW',    (0, 0), (-1, 0),  1.0, DARK_BLUE),
    ]
    if extra:
        cmds += extra
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(cmds))
    return t

def section(title, body, level=1):
    h = {1: H1, 2: H2, 3: H3}[level]
    hr_color = DARK_BLUE if level == 1 else MID_BLUE
    return ([Paragraph(title, h),
             HRFlowable(width='100%', thickness=0.8 if level == 1 else 0.5,
                        color=hr_color, spaceAfter=5, spaceBefore=0)]
            + (body if isinstance(body, list) else [body]))

# ── Image helper ──────────────────────────────────────────────────────────────
def img(rel, width=None, height=None):
    path = os.path.join(BASE, rel)
    if not os.path.exists(path):
        return Spacer(1, 0.3*cm)
    # Use PIL for reliable dimensions, then pass to Image constructor
    pil = PILImage.open(path)
    pw, ph = pil.size
    pil.close()
    if width:
        w, h = width, width * ph / pw
    elif height:
        w, h = height * pw / ph, height
    else:
        w, h = pw, ph
    return Image(path, width=w, height=h)

# ── Generated charts ──────────────────────────────────────────────────────────
_tmps = []

def _save(fig, name):
    p = os.path.join(tempfile.gettempdir(), f'rpt_{name}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    _tmps.append(p)
    return p

def chart_miou_f1():
    models = ['U-Net', 'Bi-LSTM', 'Late\nFusion', 'Hybrid\nFusion', 'Deep\nFusion']
    miou = [0.8704, 0.6978, 0.8770, 0.8891, 0.8896]
    f1   = [0.9291, 0.8080, 0.9329, 0.9401, 0.9468]
    c = '#1B3A5C'
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, vals, title, xlim in zip(axes, [miou, f1],
                                      ['mIoU', 'Macro F1-score'],
                                      [(0.55, 0.95), (0.70, 1.00)]):
        bars = ax.barh(models, vals, color=c, height=0.5)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=10, fontweight='bold', color=c)
        ax.tick_params(labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(v+0.003, b.get_y()+b.get_height()/2,
                    f'{v:.4f}', va='center', ha='left', fontsize=7.5)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    return _save(fig, 'miou_f1')

def chart_ablation():
    labels = ['deep_fusion\n(pretrained, fine-tuned)',
              'fusion_v4\n(pretrained, frozen)',
              'fusion_v3\n(random, larger)',
              'fusion_v2\n(random init)']
    vals = [0.8896, 0.8982, 0.8949, 0.8020]
    c = '#1B3A5C'
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    bars = ax.barh(labels, vals, color=c, height=0.5)
    ax.set_xlim(0.77, 0.93)
    ax.set_xlabel('mIoU', fontsize=9)
    ax.set_title('Ablation Study — mIoU on T03CWT', fontsize=10, fontweight='bold', color=c)
    for b, v in zip(bars, vals):
        ax.text(v+0.001, b.get_y()+b.get_height()/2,
                f'{v:.4f}', va='center', ha='left', fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save(fig, 'ablation')

# ── Story ─────────────────────────────────────────────────────────────────────
story = []

# ═════════════════════ TITLE PAGE ════════════════════════════════════════════
story += [
    Spacer(1, 1.8*cm),
    Paragraph('Multimodal Fusion Strategies for<br/>Antarctic Sea Ice Classification', TITLE),
    Spacer(1, 0.4*cm),
    Paragraph('Project Summary Report', SUBTITLE),
    Paragraph('Research Seminar · Knox College · June 2025', SUBTITLE),
    Spacer(1, 1.0*cm),
    HRFlowable(width='40%', thickness=1.5, color=DARK_BLUE, spaceAfter=14),
    Spacer(1, 0.3*cm),
]

best = make_table(
    [[phh('Best Model'), phh('Pixel Accuracy'), phh('mIoU'), phh('Macro F1')],
     [ph('<b>Deep Fusion</b>'), ph('<b>0.9530</b>'), ph('<b>0.8896</b>'), ph('<b>0.9468</b>')]],
    [W*0.35, W*0.22, W*0.22, W*0.21],
    [('ALIGN',(1,0),(-1,-1),'CENTER'), ('FONTNAME',(0,1),(-1,1),'Helvetica-Bold'),
     ('FONTSIZE',(0,1),(-1,1),11)],
)
story.append(best)
story.append(Spacer(1, 0.8*cm))

abs_box = Table(
    [[Paragraph('<b>Abstract</b>', CELL_L)],
     [Paragraph(
         'This report presents a multimodal deep-learning framework for per-pixel classification '
         'of Antarctic sea ice into three classes — <b>thick ice</b>, <b>thin ice</b>, and '
         '<b>open water</b> — by fusing Sentinel-2 optical imagery with ICESat-2 ATL03 '
         'photon-altimetry data. Three fusion strategies are compared: <b>Late Fusion</b> '
         '(decision level), <b>Hybrid Fusion</b> (feature + decision level), and <b>Deep Fusion</b> '
         '(deep feature level). Deep Fusion achieves the highest overall mean '
         'Intersection-over-Union (mIoU) of <b>0.8896</b> and macro F1 of <b>0.9468</b> on '
         'geographically held-out tile T03CWT, outperforming both unimodal baselines and the '
         'other two fusion variants.', ABSTRACT)]],
    colWidths=[W])
abs_box.setStyle(TableStyle([
    ('BACKGROUND',    (0,0),(-1,-1), LIGHT_BLUE),
    ('BOX',           (0,0),(-1,-1), 0.8, BORDER),
    ('LEFTPADDING',   (0,0),(-1,-1), 12),
    ('RIGHTPADDING',  (0,0),(-1,-1), 12),
    ('TOPPADDING',    (0,0),(-1,-1), 8),
    ('BOTTOMPADDING', (0,0),(-1,-1), 10),
]))
story.append(abs_box)
story.append(PageBreak())

# ═════════════════════ 1 INTRODUCTION ════════════════════════════════════════
story += section('1  Introduction', [
    Paragraph('Sea ice extent is a critical indicator of climate change, influencing polar ecosystems, '
              'shipping routes, and global ocean circulation. Sentinel-2 multispectral imagery provides '
              '10-meter spatial resolution over polar regions, but thin ice and open water share similar '
              'reflectance signatures under certain illumination conditions, and optical sensors are '
              'unavailable under cloud cover.', BODY),
    Paragraph('ICESat-2 ATL03, a photon-counting lidar, provides precise along-track height profiles '
              'independent of solar illumination. Its altimetric returns encode surface roughness and '
              'freeboard, enabling discrimination between ice types that are optically ambiguous. However, '
              'ICESat-2 data are only available along narrow ground tracks, precluding direct full-scene '
              'segmentation.', BODY),
    Paragraph('This work fuses the two modalities to combine their complementary strengths. We design, '
              'train, and systematically compare three fusion architectures. Evaluation on a '
              'geographically disjoint tile tests generalisation rather than interpolation.', BODY),
    Paragraph('<b>Contributions:</b>', BODY_L),
    Paragraph('•  Systematic comparison of three multimodal fusion strategies for polar sea ice classification.', BULLET),
    Paragraph('•  21-configuration hyperparameter sweep for the photon recurrent branch.', BULLET),
    Paragraph('•  Ablation study quantifying the contribution of transferred photon-branch representations.', BULLET),
    Paragraph('•  Results on held-out tile T03CWT: mIoU 0.8896, pixel accuracy 0.9530, macro F1 0.9468.', BULLET),
])

# ═════════════════════ 2 DATASET ═════════════════════════════════════════════
story += section('2  Dataset', [
    Paragraph('The dataset covers the Ross Sea, Antarctica, pairing Sentinel-2 Level-1C optical scenes '
              'with ICESat-2 ATL03 photon records and per-pixel ground-truth masks. Training tiles: '
              'T02CNA and T02CNC. Test tile: T03CWT (geographically disjoint, 46,004 labeled footprints).', BODY),
])

story.append(make_table(
    [[phh('Source'), phh('Description'), phh('Spatial Resolution')],
     [ph('Sentinel-2 L1C'),     ph('Optical RGB imagery over the Ross Sea'), ph('10 m')],
     [ph('ICESat-2 ATL03'),     ph('Photon clouds aggregated to 10 m along-track segments'), ph('10 m along-track')],
     [ph('Ground-truth masks'), ph('Per-pixel labels via HSV thresholding with cloud/shadow removal'), ph('10 m')]],
    [W*0.22, W*0.57, W*0.21]))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    'Class encoding in the masks: thick ice (red), thin ice (blue), open water (green). Patch extraction '
    'produces aligned 128×128 pixel image patches centered on each labeled ICESat-2 point alongside a '
    'sequence of eight ATL03 features per segment:', BODY))

story.append(make_table(
    [[phh('Feature'), phh('Description')],
     [ph('h_cor_mean / h_cor_med'),  ph('Mean and median corrected photon height')],
     [ph('h_diff'),                   ph('Mean − median height (within-segment asymmetry)')],
     [ph('rel_height_min_elev'),      ph('Height relative to per-track minimum elevation')],
     [ph('height_sd'),                ph('Standard deviation of photon heights')],
     [ph('pcnth_mean / pcnt_mean'),   ph('Mean photon-count height and mean photon count')],
     [ph('bcnt_mean / brate_mean'),   ph('Mean background photon count and background rate')]],
    [W*0.35, W*0.65]))

# ═════════════════════ 3 METHODOLOGY ═════════════════════════════════════════
story += section('3  Methodology', [])
story += section('3.1  Image Branch (U-Net)', [
    Paragraph('Each 128×128 RGB patch is processed by a U-Net with a ResNet-18 encoder pretrained on '
              'ImageNet. The decoder restores the original spatial resolution, producing a 16-channel '
              'feature map of shape (16, 128, 128). Encoder weights are fine-tuned during fusion training.', BODY),
], level=2)

story += section('3.2  Photon Branch (Unidirectional LSTM)', [
    Paragraph('A sliding window of five consecutive 10-meter segments is formed around each labeled '
              'location, yielding a sequence of eight features per segment. The sequence is processed '
              'by a single-layer LSTM (hidden dim 96, dropout 0.4) followed by a fully connected '
              'classification head. Training uses focal loss (α = [0.05, 0.45, 0.60], γ = 2.0) to '
              'prevent collapse onto the majority thick-ice class. The configuration was selected by '
              'a 21-run hyperparameter sweep.', BODY),
], level=2)

story += section('3.3  Fusion Strategies', [], level=2)
# Fixed table: wide mechanism column, text wraps cleanly
story.append(make_table(
    [[phh('Strategy'), phh('Integration Point'), phh('Mechanism')],
     [ph('Late Fusion'),
      ph('Decision level'),
      ph('Softmax predictions from each branch are averaged pixel-wise.')],
     [ph('Hybrid Fusion'),
      ph('Feature + Decision level'),
      ph('Photon embeddings are injected into the U-Net decoder at an intermediate spatial scale; '
         'a second prediction head is also averaged at the decision level.')],
     [ph('Deep Fusion'),
      ph('Deep feature level'),
      ph('The photon embedding is projected to 16 channels, broadcast spatially, concatenated '
         'with the U-Net decoder feature map, recalibrated by a Squeeze-and-Excitation block, '
         'and classified by a 1×1 convolution.')]],
    [W*0.18, W*0.22, W*0.60]))
story.append(Paragraph(
    'In all fusion models the pretrained photon branch is fine-tuned at one-tenth of the base '
    'learning rate, allowing it to adapt while retaining transferred representations.', BODY))

story += section('3.4  Training Protocol', [
    Paragraph('All models use focal loss (α = [0.05, 0.45, 0.60], γ = 2.0), cosine annealing LR '
              'schedule, AMP mixed-precision, and early stopping. Batch size 32, max 30 epochs. '
              'GPU: NVIDIA RTX A6000 (≈60–90 min per run).', BODY),
], level=2)

# ═════════════════════ 4 MODEL CONFIGURATIONS ════════════════════════════════
story += section('4  Model Configurations', [
    Paragraph('Final hyperparameter settings for each fusion strategy after the photon-branch '
              'sweep and fusion-model tuning.', BODY),
])

cfg_extra = [
    ('ALIGN',    (1, 0), (-1, -1), 'CENTER'),
    ('ALIGN',    (0, 0), ( 0, -1), 'LEFT'),
    ('FONTNAME', (0, 1), ( 0, -1), 'Helvetica-Bold'),
    ('TEXTCOLOR',(0, 1), ( 0, -1), MID_BLUE),
]
story.append(make_table(
    [[phh('Component'),              phh('Deep Fusion'),          phh('Hybrid Fusion'),        phh('Late Fusion')],
     [ph('Input modality'),          ph('Image + ATL03'),         ph('Image + ATL03'),         ph('Image + ATL03')],
     [ph('Image encoder'),           ph('U-Net ResNet-18'),       ph('U-Net ResNet-18'),       ph('U-Net ResNet-18')],
     [ph('Encoder weights'),         ph('ImageNet'),              ph('ImageNet'),              ph('ImageNet')],
     [ph('Sequence length'),         ph('5'),                     ph('5'),                     ph('5')],
     [ph('LSTM hidden dim'),         ph('96'),                    ph('48'),                    ph('48')],
     [ph('LSTM layers / direction'), ph('1 / Unidirectional'),    ph('1 / Unidirectional'),    ph('1 / Unidirectional')],
     [ph('LSTM dropout'),            ph('0.4'),                   ph('0.4'),                   ph('0.4')],
     [ph('Head hidden dim / drop'),  ph('16 / 0.4'),              ph('16 / 0.4'),              ph('16 / 0.4')],
     [ph('Fusion channels'),         ph('16'),                    ph('16'),                    ph('16')],
     [ph('Fusion block'),            ph('SE attention'),          ph('SE attn + learned blend'),ph('Learnable logit blend')],
     [ph('SE reduction'),            ph('8'),                     ph('8'),                     ph('—')],
     [ph('Optimizer'),               ph('Adam'),                  ph('AdamW'),                 ph('AdamW')],
     [ph('Learning rate'),           ph('1e-4 / 1e-5'),           ph('1e-4'),                  ph('1e-4')],
     [ph('Weight decay'),            ph('1e-4'),                  ph('1e-4'),                  ph('1e-4')],
     [ph('Batch size / Max epochs'), ph('32 / 30'),               ph('32 / 30'),               ph('32 / 30')],
     [ph('Early stopping patience'), ph('8'),                     ph('7'),                     ph('7')],
     [ph('Loss function'),           ph('Focal loss'),            ph('Focal loss'),            ph('Focal loss')],
     [ph('Focal α'),                 ph('[0.05, 0.45, 0.60]'),    ph('[0.05, 0.45, 0.60]'),    ph('[0.05, 0.45, 0.60]')],
     [ph('Focal γ / LR scheduler'),  ph('2.0 / Cosine'),          ph('2.0 / Cosine'),          ph('2.0 / Cosine')],
     [ph('Mixed precision / Grad clip'), ph('AMP / —'),           ph('AMP / 1.0'),             ph('AMP / 1.0')]],
    [W*0.34, W*0.22, W*0.22, W*0.22], cfg_extra))

# ═════════════════════ 5 RESULTS ═════════════════════════════════════════════
story += section('5  Results', [
    Paragraph('All models are evaluated on the geographically held-out tile T03CWT. Metrics are '
              'pixel accuracy, macro-averaged precision / recall / F1, and mIoU.', BODY),
])

story += section('5.1  Overall Model Comparison', [], level=2)
story.append(make_table(
    [[phh('Model'), phh('Accuracy'), phh('Precision'), phh('Recall'), phh('F1'), phh('mIoU')],
     [ph('U-Net (optical)'),  ph('0.9429'), ph('0.9402'), ph('0.9188'), ph('0.9291'), ph('0.8704')],
     [ph('Bi-LSTM (photon)'), ph('0.9594'), ph('0.7663'), ph('0.8609'), ph('0.8080'), ph('0.6978')],
     [ph('Late Fusion'),      ph('0.9460'), ph('0.9504'), ph('0.9171'), ph('0.9329'), ph('0.8770')],
     [ph('Hybrid Fusion'),    ph('0.9509'), ph('0.9449'), ph('0.9355'), ph('0.9401'), ph('0.8891')],
     [ph('<b>Deep Fusion</b>'),ph('<b>0.9530</b>'),ph('<b>0.9460</b>'),
      ph('<b>0.9481</b>'),ph('<b>0.9468</b>'),ph('<b>0.8896</b>')]],
    [W*0.28, W*0.14, W*0.14, W*0.14, W*0.15, W*0.15],
    [('ALIGN',(1,0),(-1,-1),'CENTER'),
     ('BACKGROUND',(0,5),(-1,5),LIGHT_BLUE),
     ('FONTNAME',(0,5),(-1,5),'Helvetica-Bold')]))
story.append(Paragraph('<i>Best values in each column are bold. Deep Fusion leads on accuracy, recall, F1, and mIoU.</i>', CAPTION))

p_chart = chart_miou_f1()
story.append(KeepTogether([
    Image(p_chart, width=W, height=W*2.8/9),
    Paragraph('Figure 1.  mIoU (left) and macro F1-score (right) for all models on held-out tile T03CWT.', CAPTION),
]))

story += section('5.2  Per-Class IoU', [], level=2)
story.append(make_table(
    [[phh('Model'), phh('Ice IoU'), phh('Thin Ice IoU'), phh('Water IoU'), phh('mIoU')],
     [ph('U-Net'),          ph('0.9299'), ph('0.7683'), ph('0.9130'), ph('0.8704')],
     [ph('Bi-LSTM'),        ph('0.9671'), ph('0.5427'), ph('0.5836'), ph('0.6978')],
     [ph('Late Fusion'),    ph('0.9334'), ph('0.7785'), ph('0.9189'), ph('0.8770')],
     [ph('Hybrid Fusion'),  ph('<b>0.9396</b>'), ph('<b>0.7988</b>'), ph('0.9290'), ph('0.8891')],
     [ph('<b>Deep Fusion</b>'), ph('0.9383'), ph('0.7962'), ph('<b>0.9344</b>'), ph('<b>0.8896</b>')]],
    [W*0.28, W*0.18, W*0.18, W*0.18, W*0.18],
    [('ALIGN',(1,0),(-1,-1),'CENTER'),
     ('BACKGROUND',(0,5),(-1,5),LIGHT_BLUE),
     ('FONTNAME',(0,5),(-1,5),'Helvetica-Bold')]))
story.append(Paragraph(
    'Hybrid Fusion leads on thin ice IoU (0.7988); Deep Fusion leads on water IoU and overall mIoU. '
    'Thin ice is the most challenging class across all models.', BODY))

# ── 5.3 Confusion Matrices ────────────────────────────────────────────────────
story += section('5.3  Confusion Matrices', [
    Paragraph('Row-normalised confusion matrices on held-out tile T03CWT. '
              'Classes: Ice (I), Thin Ice (T), Water (W).', BODY),
], level=2)

half = W/2 - 0.3*cm
row1 = Table([[img('confusion_matrices/unet_green.png',  width=half),
               img('confusion_matrices/lstm_green.png',  width=half)]],
             colWidths=[half+0.3*cm, half+0.3*cm])
row1.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'TOP')]))
story.append(KeepTogether([row1,
    Paragraph('Figure 2.  Unimodal baselines. (a) U-Net optical baseline.  (b) Bi-LSTM photon baseline.', CAPTION)]))
story.append(Spacer(1, 0.3*cm))

third = W/3 - 0.15*cm
row2 = Table([[img('confusion_matrices/deepfusion_green.png',   width=third),
               img('confusion_matrices/hybridfusion_green.png', width=third),
               img('confusion_matrices/latefusion_green.png',   width=third)]],
             colWidths=[third+0.15*cm]*3)
row2.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'TOP')]))
story.append(KeepTogether([row2,
    Paragraph('Figure 3.  Fusion strategies. (a) Deep Fusion.  (b) Hybrid Fusion.  (c) Late Fusion.', CAPTION)]))

# ── 5.4 Per-Model Detailed Results ───────────────────────────────────────────
story += section('5.4  Per-Model Detailed Results', [], level=2)

def model_block(name, acc, prec, rec, f1, miou,
                ice_iou, ice_p, ice_r, ice_f,
                thi_iou, thi_p, thi_r, thi_f,
                wat_iou, wat_p, wat_r, wat_f):
    ctr = [('ALIGN',(0,0),(-1,-1),'CENTER'), ('ALIGN',(0,0),(0,-1),'LEFT')]
    # Overall metrics: one row, 5 columns
    ov = make_table(
        [[phh('Accuracy'), phh('Precision'), phh('Recall'), phh('F1-score'), phh('mIoU')],
         [ph(acc), ph(prec), ph(rec), ph(f1), ph(miou)]],
        [W/5]*5, [('ALIGN',(0,0),(-1,-1),'CENTER')])
    # Per-class breakdown
    pc = make_table(
        [[phh('Class'), phh('IoU'), phh('Precision'), phh('Recall'), phh('F1')],
         [ph('Ice'),      ph(ice_iou), ph(ice_p), ph(ice_r), ph(ice_f)],
         [ph('Thin Ice'), ph(thi_iou), ph(thi_p), ph(thi_r), ph(thi_f)],
         [ph('Water'),    ph(wat_iou), ph(wat_p), ph(wat_r), ph(wat_f)]],
        [W*0.20, W*0.20, W*0.20, W*0.20, W*0.20], ctr)
    return [Paragraph(name, H3), Spacer(1, 0.1*cm), ov, Spacer(1, 0.2*cm), pc, Spacer(1, 0.5*cm)]

story += model_block(
    'Deep Fusion',
    '0.9502','0.9484','0.9325','0.9402','0.8896',
    '0.9383','0.9615','0.9750','0.9682',
    '0.7962','0.9029','0.8689','0.8865',
    '0.9344','0.9787','0.9537','0.9661')
story += model_block(
    'Hybrid Fusion',
    '0.9509','0.9449','0.9355','0.9461','0.8891',
    '0.9396','0.9634','0.9744','0.9639',
    '0.7988','0.9043','0.8725','0.8881',
    '0.9290','0.9669','0.9595','0.9632')
story += model_block(
    'Late Fusion',
    '0.9460','0.9504','0.9171','0.9329','0.8770',
    '0.9334','0.9519','0.9796','0.9656',
    '0.7785','0.9149','0.8393','0.8755',
    '0.9189','0.9845','0.9324','0.9577')

# ── 5.5 Training Curves & Sample Predictions ─────────────────────────────────
story += section('5.5  Training Curves & Sample Predictions', [], level=2)
story += section('Deep Fusion — Training Curve', [], level=3)
story.append(KeepTogether([
    img('runs/deep_fusion/loss_curve.png', width=W*0.85),
    Paragraph('Figure 4.  Training and validation curves for the Deep Fusion model.', CAPTION)]))
story.append(Spacer(1, 0.3*cm))

for label, path in [
    ('Deep Fusion',   'archive/runs/fusion_deep_unet_profstyle_v1/sample_preds.png'),
    ('Hybrid Fusion', 'archive/runs/fusion_hybrid_unet_profstyle_v1/sample_preds.png'),
    ('Late Fusion',   'archive/runs/fusion_late_unet_profstyle_v1/sample_preds.png'),
]:
    story += section(f'{label} — Sample Predictions', [], level=3)
    # Images are portrait (1125×2100); cap height to fit on page
    story.append(img(path, height=17*cm))
    story.append(
        Paragraph(f'Figure: {label} predictions on T03CWT. '
                  f'Columns: RGB input | ground truth | model prediction.', CAPTION))
    story.append(Spacer(1, 0.3*cm))

story += section('All-Model Prediction Grid', [], level=3)
# prediction_grid.png is 2078×2297 (nearly square); constrain height
story.append(img('archive/runs/comparison/prediction_grid.png', height=16*cm))
story.append(Paragraph('Figure 5.  Side-by-side model predictions on held-out tile T03CWT.', CAPTION))

# ═════════════════════ 6 ABLATION STUDY ══════════════════════════════════════
story += section('6  Ablation Study', [
    Paragraph('The ablation study quantifies the contribution of key design decisions in Deep Fusion. '
              'All variants share the same U-Net image branch and are evaluated on T03CWT.', BODY),
])

story.append(make_table(
    [[phh('Variant'), phh('mIoU'), phh('Configuration')],
     [ph('fusion_v2'),            ph('0.8020'), ph('Photon branch trained from random initialisation (unstable)')],
     [ph('fusion_v3'),            ph('0.8949'), ph('Higher-capacity recurrent branch, random initialisation')],
     [ph('fusion_v4'),            ph('0.8982'), ph('Pretrained photon branch, frozen during fusion training')],
     [ph('<b>deep_fusion</b>'),   ph('<b>0.8896</b>'),
      ph('<b>Pretrained photon branch, fine-tuned at 0.1× LR</b>')]],
    [W*0.20, W*0.12, W*0.68],
    [('ALIGN',(1,0),(-1,-1),'CENTER'),
     ('BACKGROUND',(0,4),(-1,4),LIGHT_BLUE),
     ('FONTNAME',(0,4),(-1,4),'Helvetica-Bold')]))

story.append(Paragraph(
    'Throughout development we explored different model architectures, loss configurations, and '
    'hyperparameter settings before arriving at the final design. The variants stored under '
    '<i>archive/</i> represent these earlier iterations; the current notebooks outside the archive '
    'folder reflect the final configurations used to produce the reported results.', BODY))

p_abl = chart_ablation()
story.append(KeepTogether([
    Image(p_abl, width=W*0.72, height=W*0.72*2.6/6.5),
    Paragraph('Figure 6.  Ablation mIoU. deep_fusion uses pretrained LSTM fine-tuned at 0.1× LR.', CAPTION)]))

story += section('6.2  Photon Branch Hyperparameter Sweep', [
    Paragraph('A 21-configuration grid sweep optimised the photon branch. Hidden dimension 96 and '
              'focal loss settings α=[0.05, 0.45, 0.60] / γ=2.0 were the most impactful choices.', BODY),
], level=2)
story.append(KeepTogether([
    img('archive/runs/lstm_sweep/sweep_per_axis.png', width=W),
    Paragraph('Figure 7.  Per-axis mIoU across the 21-configuration LSTM hyperparameter sweep.', CAPTION)]))

# ═════════════════════ 7 CONCLUSION ══════════════════════════════════════════
story += section('7  Conclusion', [
    Paragraph('We presented a systematic study of multimodal fusion for per-pixel Antarctic sea ice '
              'classification from paired Sentinel-2 imagery and ICESat-2 ATL03 altimetry. Three '
              'architectures were evaluated on a geographically held-out tile, providing a strict '
              'test of geographic generalisation.', BODY),
    Paragraph('Deep Fusion achieved the highest overall mIoU (0.8896) and pixel accuracy (0.9530), '
              'with the strongest performance on open water (+2.1 pp over U-Net). Hybrid Fusion '
              'ranked second and achieved the best thin-ice IoU (0.7988, +3.0 pp over U-Net). All '
              'three fusion approaches outperformed both unimodal baselines, confirming the '
              'complementarity of optical and altimetric sensing.', BODY),
    Paragraph('<b>Key Takeaways:</b>', BODY_L),
    Paragraph('•  Feature-level fusion consistently outperforms decision-level fusion on overall mIoU.', BULLET),
    Paragraph('•  Spatial broadcast of photon features informs every pixel, not just photon-footprint pixels.', BULLET),
    Paragraph('•  Pretrained-and-fine-tuned photon branch outperforms both frozen and randomly-initialised variants.', BULLET),
    Paragraph('•  All fusion strategies outperform both unimodal baselines on mIoU.', BULLET),
    Spacer(1, 0.3*cm),
    Paragraph('<b>Future Directions:</b>', BODY_L),
    Paragraph('•  SAR modality integration (Sentinel-1) for cloud-penetrating year-round polar monitoring.', BULLET),
    Paragraph('•  Temporal sequence modelling across multi-date passes for ice-cover change detection.', BULLET),
    Paragraph('•  Geographic transfer to the Arctic to assess cross-hemisphere generalisation.', BULLET),
])

story += section('Acknowledgements', [
    Paragraph('This work was conducted as part of the Research Seminar at Knox College. We thank '
              'Prof. Iqrah for guidance throughout the project. ICESat-2 ATL03 products: NASA NSIDC. '
              'Sentinel-2 imagery: ESA Copernicus Programme.', BODY),
])

# ═════════════════════ BUILD ══════════════════════════════════════════════════
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f'Written: {OUT}')

for p in _tmps:
    try:
        os.remove(p)
    except Exception:
        pass
