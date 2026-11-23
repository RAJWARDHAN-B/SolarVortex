"""Build the v9 paper presentation locally.

Run with Python packages: python-pptx, Pillow, matplotlib, pymupdf.
The v9 manuscript is authoritative where the exploratory code differs.
"""

import argparse
import io
import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "Raj_Paper/IEEE-conference-template-062824/v9"
NAVY, BG, TEAL, SLATE = "1B2A4A", "F8FAFC", "0D9488", "1E293B"
MUTED, BORDER, PALE, WHITE = "64748B", "E2E8F0", "E6F5F2", "FFFFFF"
FONT = "Arial"
W, H = 13.333333, 7.5
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
TEXT_REGIONS = []
FIGURES_USED = set()


def rgb(value):
    return RGBColor.from_string(value)


def text(slide, value, x, y, w, h, size=22, color=SLATE, bold=False,
         align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text, p.alignment = line, align
        p.font.name, p.font.size = FONT, Pt(size)
        p.font.bold, p.font.color.rgb = bold, rgb(color)
        p.space_before = Pt(0)
        p.space_after = Pt(3)
    TEXT_REGIONS.append((len(prs.slides), x, y, w, h, value))
    return shape


def box(slide, x, y, w, h, fill=WHITE, border=None, radius=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if border:
        shape.line.color.rgb = rgb(border)
    else:
        shape.line.fill.background()
    if radius:
        shape.adjustments[0] = 0.08
    return shape


def slide(title, section, source):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    box(s, 0, 0, W, 0.10, TEAL, radius=False)
    text(s, section.upper(), 0.55, 0.30, 12.2, 0.30, 11, TEAL, True)
    text(s, title, 0.55, 0.80, 12.2, 0.65, 30, NAVY, True)
    box(s, 0.55, 1.60, 12.2, 0.018, BORDER, radius=False)
    text(s, source, 0.55, 7.03, 11.25, 0.22, 9, MUTED)
    text(s, f"{len(prs.slides):02d} / 16", 12.04, 7.02, 0.73, 0.25,
         10, MUTED, align=PP_ALIGN.RIGHT)
    return s


def bullets(s, items, x=0.65, y=2.05, w=5.65, step=0.80, size=23):
    assert len(items) <= 5
    for i, item in enumerate(items):
        assert len(item.split()) <= 12, item
        box(s, x, y + i * step + 0.11, 0.08, 0.08, TEAL, radius=False)
        text(s, item, x + 0.24, y + i * step, w - 0.24, step - 0.12, size)


def callout(s, heading, body, x, y, w, h=1.05):
    box(s, x, y, w, h, PALE)
    text(s, heading, x + 0.18, y + 0.10, w - 0.36, 0.34, 20, TEAL, True)
    text(s, body, x + 0.18, y + 0.50, w - 0.36, h - 0.58, 16)


def picture(s, name, x, y, w, h, crop=None):
    path = PAPER / name
    assert path.exists(), path
    FIGURES_USED.add(name)
    if path.suffix.lower() in {".pdf", ".eps"}:
        if path.suffix.lower() == ".eps":
            raise ValueError("Convert EPS with an installed local renderer first.")
        import fitz
        with fitz.open(path) as doc:
            data = doc[0].get_pixmap(dpi=300).tobytes("png")
        image = Image.open(io.BytesIO(data))
    else:
        image = Image.open(path)
    if crop:
        iw, ih = image.size
        image = image.crop(tuple(round(v * d) for v, d in
                                 zip(crop, (iw, ih, iw, ih))))
    iw, ih = image.size
    scale = min(w / iw, h / ih)
    ww, hh = iw * scale, ih * scale
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    s.shapes.add_picture(stream, Inches(x + (w - ww) / 2),
                         Inches(y + (h - hh) / 2), Inches(ww), Inches(hh))


def equation(s, value, x, y, w, h, size=24):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, value, ha="center", va="center", fontsize=size,
             color="#" + NAVY)
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=300, transparent=True)
    plt.close(fig)
    stream.seek(0)
    s.shapes.add_picture(stream, Inches(x), Inches(y), Inches(w), Inches(h))


def table(s, headers, rows, x, y, widths, row_h=0.51, size=17, highlight=()):
    shape = s.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y),
                              Inches(sum(widths)), Inches(row_h * (len(rows) + 1)))
    t = shape.table
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    for r, values in enumerate([headers] + rows):
        t.rows[r].height = Inches(row_h)
        for c, value in enumerate(values):
            cell = t.cell(r, c)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(
                NAVY if r == 0 else PALE if r - 1 in highlight
                else WHITE if r % 2 else BG)
            for p in cell.text_frame.paragraphs:
                p.font.name, p.font.size = FONT, Pt(size)
                p.font.bold = r == 0 or r - 1 in highlight
                p.font.color.rgb = rgb(WHITE if r == 0 else SLATE)
                p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
    return shape


def notes(s, content, sources):
    s.notes_slide.notes_text_frame.text = (
        content.strip() + "\n\nSource provenance: " + sources)


STAGE1 = [
    ["YOLOv8n-OBB", "3.1M", "222", "0.9950", "0.9909"],
    ["YOLOv8s-OBB", "11.4M", "216", "0.9950", "0.9932"],
    ["YOLOv8m-OBB", "26.4M", "148", "0.9950", "0.9947"],
    ["YOLOv8l-OBB", "44.5M", "118", "0.9949", "0.9919"],
    ["YOLOv8x-OBB", "69.5M", "86", "0.9950", "0.9921"],
]
STAGE2 = [
    ["DenseNet-121", "7.0", ".974 +/- .002", ".960 +/- .003", ".954 +/- .003",
     ".967 +/- .005", ".989 +/- .000", ".941 +/- .005", "3"],
    ["Swin-S", "48.8", ".973", ".958", ".939", ".977", ".988", ".939", "1"],
    ["ViT-B/16", "85.8", ".972", ".957", ".948", ".967", ".988", ".937", "1"],
    ["ConvNeXt-T", "27.8", ".970 +/- .001", ".953 +/- .001", ".936 +/- .001",
     ".971 +/- .001", ".988 +/- .001", ".931 +/- .002", "3"],
    ["MobileNet-V3L", "3.0", ".967", ".949", ".935", ".964", ".983", ".925", "1"],
    ["EffNet-B4", "17.6", ".966", ".947", ".937", ".958", ".991", ".922", "1"],
    ["ResNet-101", "42.5", ".966", ".947", ".937", ".958", ".982", ".922", "1"],
]
PERCLASS = [
    ["DenseNet-121", ".985", ".984", ".967", ".954", "15.0", "21.3"],
    ["Swin-S", ".971", ".989", ".977", ".939", "10", "28"],
    ["ViT-B/16", ".975", ".984", ".967", ".948", "15", "24"],
    ["ConvNeXt-T", ".969", ".987", ".971", ".936", "12.7", "29.7"],
    ["MobileNet-V3L", ".969", ".983", ".964", ".935", "16", "30"],
    ["EffNet-B4", ".970", ".980", ".958", ".937", "19", "29"],
    ["ResNet-101", ".970", ".980", ".958", ".937", "19", "29"],
]


def build():
    manuscript = (PAPER / "main.tex").read_text()
    active = "\n".join(line.split("%")[0] for line in manuscript.splitlines())
    referenced = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", active))
    title_match = re.search(r"\\title\{(.*?)\}", active, re.S)
    if title_match is None:
        raise ValueError("Paper title is missing.")
    title = " ".join(title_match.group(1).split())
    prs.core_properties.title = title
    prs.core_properties.subject = "PV cell localization and binary EL defect triage"
    prs.core_properties.author = "Authors not specified in submitted manuscript"
    prs.core_properties.keywords = "PV, EL, YOLOv8-OBB, binary triage, v9"

    s = slide("Cell localization meets defect triage", "Research presentation",
              "Submitted v9 manuscript | IEEE iSPEC 2026 template; venue status unverified")
    text(s, title, 0.65, 2.05, 8.20, 2.13, 33, NAVY, True)
    text(s, "Authors and affiliation not specified in submitted preamble",
         0.68, 4.57, 8.02, 0.46, 17, MUTED)
    text(s, "iSPEC 2026 manuscript context  |  23 November 2026",
         0.68, 5.22, 8.05, 0.40, 17)
    box(s, 9.29, 2.06, 3.36, 3.70, NAVY)
    text(s, "01", 9.60, 2.35, 2.68, 0.54, 34, "5EEAD4", True)
    text(s, "LOCALIZE\nOriented cell boxes", 9.60, 3.00, 2.70, 0.81, 20, WHITE, True)
    text(s, "02", 9.60, 4.06, 2.68, 0.54, 34, "5EEAD4", True)
    text(s, "TRIAGE\nOK or B-grade", 9.60, 4.70, 2.70, 0.81, 20, WHITE, True)
    notes(s, """Introduce the paper using its exact title. The supplied v9 LaTeX
author block is entirely commented out and contains placeholder names and a
placeholder institution block; neither authorship nor affiliation is confirmed.
The file header identifies an IEEE iSPEC 2026 conference-paper template, not a
verified acceptance or presentation venue. The displayed date is the deck date.
The study combines oriented module-to-cell localization with independently
evaluated binary cell-quality triage. We will separate measured stage-level
performance from the proposed integrated workflow throughout this talk.""",
          "v9/main.tex, title/preamble and abstract.")

    s = slide("Why inspect EL images at module scale?", "Motivation",
              "Source: v9 Introduction | EL imaging principle")
    bullets(s, ["Subsurface damage threatens PV quality and long-term reliability.",
                "EL reveals cracks, inactive regions, and gridline failures.",
                "Real inspection starts with modules, not prepared cell crops.",
                "Reliable extraction enables scalable cell-level quality decisions."],
            w=6.0, step=0.88)
    picture(s, "fig_yolo_detections.jpg", 7.08, 1.98, 5.36, 3.65,
            crop=(0.75, 0.0, 1.0, 0.25))
    text(s, "Actual EL example from the paper's prediction montage",
         7.08, 5.79, 5.36, 0.40, 14, MUTED)
    notes(s, """EL imaging forward-biases a photovoltaic module. Radiative carrier
recombination emits near-infrared light around 1150 nm; a camera records it in a
darkened environment. Interrupted current transport and non-radiative losses
produce dark signatures that can indicate cracks, inactive regions, or damaged
gridlines. This is an emission-based inspection modality, unlike thermal
contrast measurements. The important deployment gap is that acquisition yields
full or partial modules, while much prior classification research assumes
pre-cropped cells. The image is a crop of the top-right example in the paper's
actual localization montage, not a separately labelled OK/B-grade case.
Transition to why classical extraction and narrow classifier benchmarks do not
fully solve that input-to-decision problem.""",
          "v9/main.tex, Introduction; v9/fig_yolo_detections.jpg (top-right crop).")

    s = slide("Existing methods leave an extraction gap", "Gap analysis",
              "Source: v9 Related Work | Classical extraction failure summary")
    table(s, ["Approach", "Observed limitation"], [
        ["Projection / morphology", "Module-format and orientation sensitivity"],
        ["Hough / DBSCAN", "Image-specific thresholds and clustering"],
        ["Contour extraction", "Cell boundaries confused with dark regions"],
        ["Pre-cropped classifiers", "Module-to-cell step assumed complete"],
    ], 0.65, 2.05, [3.32, 5.90], row_h=0.72, size=20)
    callout(s, "5-10%", "Skip rate reported\nfor batch extractor pp4",
            10.03, 2.08, 2.55, 1.72)
    callout(s, "Count instability", "208 versus 182 cells\nin nominally similar modules",
            10.03, 4.10, 2.55, 1.72)
    notes(s, """The paper reviews eleven cell-extraction explorations and four
full-module explorations. Four recurring weaknesses are threshold brittleness,
module-format sensitivity, contrast variability, and intolerance to camera tilt.
The batch extractor pp4 reports a 5-10 percent skip rate. The fullmodulev3
exploration reports inconsistent cell counts, 208 versus 182, for nominally
similar formats. These are exploratory observations, not aggregate detection
scores or a matched quantitative baseline. The strongest classical method did
not produce annotation-compatible oriented boxes and required manual checking,
so it was not placed in the mAP table. The corresponding classical notebook pp4
does implement perspective rectification with getPerspectiveTransform and
warpPerspective. Its existence does not establish broad generalization.
Transition from these failure modes to the paper's four contributions.""",
          "v9/main.tex, classical failure table; CellExtraction/CellExtractOpenCV/"
          "preprocessing4.ipynb, cell 3; CellExtraction/FullModuleOpenCV/fullmodulev3.ipynb.")

    s = slide("Four contributions, one practical workflow", "Contributions",
              "Source: v9 Introduction | Contributions")
    cards = [
        ("01", "Characterize extraction failures",
         "Classical methods expose format, contrast, and perspective brittleness."),
        ("02", "Learn oriented cell localization",
         "Five YOLOv8-OBB scales deliver near-saturated localization accuracy."),
        ("03", "Benchmark seven triage backbones",
         "Common Optuna budgets compare CNN and Transformer families."),
        ("04", "Analyze deployment trade-offs",
         "False negatives, seed coverage, throughput, and model size matter."),
    ]
    for i, (num, heading, body) in enumerate(cards):
        x, y = 0.65 + (i % 2) * 6.18, 2.00 + (i // 2) * 2.20
        box(s, x, y, 5.85, 1.91, WHITE, BORDER)
        text(s, num, x + 0.22, y + 0.19, 0.57, 0.46, 27, TEAL, True)
        text(s, heading, x + 0.98, y + 0.22, 4.55, 0.51, 21, NAVY, True)
        text(s, body, x + 0.25, y + 0.91, 5.33, 0.73, 21)
    notes(s, """The first contribution is empirical characterization of why
classical module-to-cell extraction is fragile. The second is an oriented
detector and rectification formulation that removes fixed axis-aligned grid
assumptions. The third is a seven-backbone binary EL benchmark with a common
hyperparameter-search budget. The fourth is deployment-oriented interpretation:
we care about localization fidelity and speed, but also missed defective cells
and model footprint. These are the paper's stated contributions, not claims of
new backbone architectures. Evaluation uses two non-overlapping datasets, so
this contribution set should not be read as a measured end-to-end accuracy.
Next, visualize how the two stages would connect operationally.""",
          "v9/main.tex, Contributions, Dataset, and Conclusion.")

    s = slide("Two stages, evaluated on independent datasets", "System overview",
              "Source: v9 Methodology | Dataset separation is explicit")
    nodes = [
        ("Module EL", "Grayscale input"),
        ("YOLOv8-OBB", "Cell localization"),
        ("Rectify", "Upright cell crops"),
        ("Backbone + head", "Cell-level scoring"),
        ("Quality triage", "OK / B-grade"),
    ]
    for i, (head, sub) in enumerate(nodes):
        x = 0.65 + i * 2.48
        box(s, x, 2.32, 2.17, 1.42, NAVY if i in (1, 3) else WHITE, BORDER)
        color = WHITE if i in (1, 3) else NAVY
        text(s, head, x + 0.12, 2.55, 1.93, 0.44, 18, color, True,
             PP_ALIGN.CENTER)
        text(s, sub, x + 0.12, 3.14, 1.93, 0.32, 13, color,
             align=PP_ALIGN.CENTER)
        if i < 4:
            shape = s.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x + 2.22),
                                      Inches(2.89), Inches(0.20), Inches(0.30))
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb(TEAL)
            shape.line.fill.background()
    callout(s, "Stage 1 evidence", "4,392 module images; localization only",
            0.65, 4.34, 5.90, 1.17)
    callout(s, "Stage 2 evidence", "9,392 independent cell crops; binary triage only",
            6.84, 4.34, 5.83, 1.17)
    text(s, "Proposed integration; common-test-set end-to-end evaluation remains future work.",
         0.72, 6.12, 11.9, 0.55, 20, NAVY, True, PP_ALIGN.CENTER)
    notes(s, """Walk left to right through the proposed operational pipeline:
module-level EL acquisition, oriented cell localization, conversion of box
corners to a perspective transform, canonical cell crops, and a backbone-based
binary classifier. Do not multiply the reported stage scores or infer an
end-to-end quality figure. The paper explicitly excludes the Stage 1 module
images and their extracted cells from Stage 2 partitions. Stage 2 is evaluated
on independent, already assembled cell-level data. A common independently
labelled module test set is a future requirement. The OBB notebook implements
the five-scale training and validation/test benchmark; the classical crop
notebook contains the OpenCV warp primitive. The supplied exploratory classifier
uses ImageFolder data loading rather than a demonstrated live detector-to-head
connection. Next, explain the geometric representation in Stage 1.""",
          "v9/main.tex, Dataset and Eqs. 1-2; CellExtraction/YOLO/YOLOv8-OBB-aug/"
          "cell_extract-yolov8-obb.ipynb; BinaryDetection/Optuna/solar_cell_binary_benchmark.py.")

    s = slide("Stage 1: oriented boxes preserve cell geometry", "Methodology / localization",
              "Source: v9 Stage 1 | Oriented-box and rectification equations")
    bullets(s, ["Predict center, size, and orientation for each cell.",
                "Convert box parameters into four corner coordinates.",
                "Warp corners into canonical upright cell crops.",
                "Avoid excess background in tilted, axis-aligned boxes."],
            w=6.05, step=0.83, size=22)
    box(s, 7.09, 2.05, 5.53, 3.19, WHITE, BORDER)
    equation(s, r"$\mathbf{d}=(c_x,c_y,w,h,\theta)$",
             7.30, 2.31, 5.09, 0.84, 27)
    equation(s, r"$\{p_i\}_{i=1}^{4}\ \longrightarrow\ M_{\mathrm{persp}}$",
             7.30, 3.24, 5.09, 0.78, 26)
    equation(s, r"$\hat I_{\mathrm{cell}}=\mathrm{warpPerspective}(I,M)$",
             7.30, 4.17, 5.09, 0.74, 20)
    callout(s, "Geometry motivation", "At 10-degree tilt: about 34% larger axis-aligned box area.",
            0.65, 5.75, 11.97, 0.96)
    notes(s, """The oriented representation contains center coordinates, positive
width and height, and an orientation angle. The paper describes theta in a
clockwise degree convention over [-90,90); an integration must reconcile that
notation with the detector API's actual angle convention. Four corners define
the input quadrilateral for cv2.getPerspectiveTransform, and warpPerspective
produces a canonical crop. For the paper's illustrative 144-by-144 cell at
10 degrees, an enclosing upright box is approximately 167-by-167, about a
34 percent area increase. This is a geometric illustration, not a measured
axis-aligned-versus-OBB ablation. Extra background and inter-cell overlap motivate
OBBs, but the paper does not quantify a resulting classifier accuracy gain.
The benchmark notebook trains and evaluates YOLOv8 OBB scales; the OpenCV
extraction notebook demonstrates the perspective-transform primitive.
Transition from crop geometry to the classifier and tuning protocol.""",
          "v9/main.tex, Stage 1 and Eqs. 1-2; preprocessing4.ipynb, cell 3;"
          " cell_extract-yolov8-obb.ipynb, training/validation/test calls.")

    s = slide("Stage 2: binary scoring with a shared search budget", "Methodology / triage",
              "Source: v9 Stage 2 | Paper protocol; code differences documented in notes")
    bullets(s, ["Fine-tune seven ImageNet-pretrained backbone families.",
                "Paper head: pooled features, dropout, and one logit.",
                "Paper loss: class-weighted binary cross-entropy with logits.",
                "Select hyperparameters using validation AUC."],
            w=6.12, step=0.82, size=22)
    box(s, 7.00, 2.07, 5.65, 2.23, WHITE, BORDER)
    equation(s, r"$p(\mathrm{B\!-\!grade}\mid I)=\sigma(z_\phi(I))$",
             7.17, 2.35, 5.28, 0.84, 25)
    equation(s, r"$\phi^*=\arg\max_{\phi}\ \mathrm{AUC}_{\mathrm{val}}(\phi)$",
             7.17, 3.29, 5.28, 0.77, 24)
    callout(s, "30 trials per backbone", "15 epochs/trial; up to 40 final-training epochs",
            7.00, 4.60, 5.65, 1.15)
    text(s, "Early stopping: patience 10 on validation AUC (paper).",
         0.73, 6.19, 11.92, 0.42, 18, MUTED)
    notes(s, """Describe the methodology as reported by the manuscript: a
pretrained feature extractor, pooling, dropout, a single-logit linear head,
weighted BCEWithLogits loss, and inference-time sigmoid. The paper allocates
30 TPE Optuna trials per backbone, up to 15 epochs each, then up to 40 epochs
for final training. Its objective and early-stopping monitor are validation AUC,
with patience 10. The paper's search space is learning rate 1e-5 to 1e-2,
weight decay 1e-6 to 1e-2, Adam/AdamW/SGD with momentum, cosine/step/plateau
scheduling, dropout 0.01-0.5, batches 16/32/64, and none/partial/full freezing.
Important reproducibility caveat: the supplied .py notebook-style export and
BinaryDetectionOptuna notebook instead use two class logits, softmax, cross
entropy with label smoothing, macro-F1 trial selection, and patience 8. Their
batch choices are 16/32, schedulers include onecycle, and dropout starts at 0.
The source export contains notebook shell syntax and was inspected, not run.
These differences are not silently reconciled, and no rerun is claimed.
The equations depict the paper protocol, not verified identical executable code.
Next, establish the two datasets and controlled evaluation setup.""",
          "v9/main.tex, Stage 2 and Optuna search-space table;"
          " BinaryDetection/Optuna/solar_cell_binary_benchmark.py:100-145, 567-727;"
          " BinaryDetection/Optuna/BinaryDetectionOptuna.ipynb, cells 1/6/7.")

    s = slide("Datasets and experimental setup", "Evaluation design",
              "Source: v9 Dataset | Experimental Setup")
    text(s, "Stage 1 / modules", 0.65, 1.97, 5.72, 0.42, 23, NAVY, True)
    table(s, ["Split", "Images"], [["Train", "3,909"], ["Validation", "322"],
                                  ["Test", "161"]], 0.65, 2.57, [3.15, 2.55],
          row_h=0.54, size=21)
    text(s, "4,392 images | ARTS, SDLE, BMRK\nFull and partial modules; 640 x 640 input",
         0.70, 5.02, 5.59, 0.79, 19)
    text(s, "Stage 2 / independent cells", 6.84, 1.97, 5.80, 0.42, 23, NAVY, True)
    table(s, ["Split", "Total", "OK", "B-grade"],
          [["Train", "6,575", "4,428", "2,147"],
           ["Validation", "1,408", "954", "454"],
           ["Test", "1,409", "949", "460"]],
          6.84, 2.57, [1.86, 1.31, 1.14, 1.48], row_h=0.54, size=19)
    text(s, "9,392 cells | ELPV, Pratt benchmark, internal EL\nModule-grouped splits; test: 32.6% B-grade",
         6.89, 5.02, 5.70, 0.79, 19)
    text(s, "RTX PRO 4500 Blackwell  |  Threadripper 7960X  |  192 GB RAM",
         0.70, 6.33, 11.94, 0.36, 17, TEAL, True, PP_ALIGN.CENTER)
    notes(s, """Stage 1 uses 4,392 JPEG module-level images from the ARTS, SDLE,
and BMRK series of the Pratt benchmark. Full modules predominantly use
144-cell formats; partial views often contain one or two complete cells plus
neighbour fragments. A domain expert annotated oriented boxes with Roboflow,
with cross-verification. The 3,909/322/161 split is approximately 89/7/4 percent.
Auto-orientation, grayscale conversion, adaptive equalization, and stretched
640-by-640 resizing are reported. Training-only augmentation uses flips,
rotation +/-11 degrees, shear +/-8 degrees, brightness +/-15 percent, small
blur and noise, with three augmented variants per training image.
Stage 2 independently combines ELPV, additional Pratt cells, and internal EL
images. Expert-verified binary labels yield 6,575/1,408/1,409 samples, totaling
9,392. Test support is 949 OK and 460 B-grade. The paper describes module-level
grouping to avoid spatial leakage and excludes Stage 1 derived cells from Stage 2.
The manuscript reports median blur, CLAHE, crop rectification, and normalization;
the exploratory classifier export instead shows ImageNet-normalized RGB
torchvision preprocessing. The actual supplied loader reads existing train/val/
test directories and does not itself prove how group splits were constructed.
Hardware is RTX PRO 4500, Threadripper 7960X, 192 GB RAM, Ubuntu 24.04.2.
Next, clarify baseline comparability and which metrics are decision-relevant.""",
          "v9/main.tex, Dataset and Experimental Setup; benchmark export:163-220.")

    s = slide("Baselines and metrics: score the defective class", "Evaluation protocol",
              "Source: v9 Related Work | Stage 2 corrected B-grade-positive scoring")
    text(s, "Compared under the paper protocol", 0.65, 2.02, 5.9, 0.44, 22, NAVY, True)
    bullets(s, ["Localization: YOLOv8-OBB nano, small, medium, large, extra-large.",
                "CNNs: DenseNet, ConvNeXt, MobileNet, EfficientNet, ResNet.",
                "Transformers: Swin-S and ViT-B/16.",
                "Prior ELPV scores are contextual, not matched baselines."],
            y=2.78, w=5.84, step=0.78, size=21)
    table(s, ["Metric", "What it tests"], [
        ["mAP@.5 / .5:.95", "Detection and tighter localization"],
        ["B-grade F1 / recall", "Defect triage and missed defects"],
        ["PR-AUC / ROC-AUC", "Ranking across decision thresholds"],
        ["MCC / accuracy", "Overall classification agreement"],
    ], 6.86, 2.12, [2.30, 3.42], row_h=0.69, size=18)
    callout(s, "Positive class = B-grade", "False negative: defective cell predicted OK",
            6.86, 5.87, 5.72, 0.94)
    notes(s, """There are five learned localization scales and seven classifier
backbones. Classical extraction is a qualitative comparator rather than a
matched mAP baseline. Historical ELPV SVM and early CNN accuracies are only
context because dataset composition and split protocol differ; no improvement
margin over those results is claimed here. For detection, mAP@0.5 measures AP
at IoU 0.5, while mAP@0.5:0.95 averages stricter overlap thresholds. For triage,
the positive class is explicitly B-grade. B-grade recall measures the fraction
of defective cells detected; precision measures how many defect flags are
correct; F1 balances those two. A false negative lets a defective cell pass as
OK; a false positive flags an OK cell. PR-AUC and ROC-AUC are threshold-independent
ranking measures. MCC is calculated at a chosen operating threshold, so it is
not threshold-independent even though one related-work sentence suggests it is.
The exploratory ImageFolder mapping is Bgrade=0 and ok=1, and old plots may
reference the opposite positive class. Use the corrected v9 tables, not legacy
classification/confusion plots, for primary ranking. Multi-seed numbers apply
only to DenseNet and ConvNeXt. Transition to exact localization results.""",
          "v9/main.tex, Related Work and Results tables;"
          " benchmark export, CLASS_NAMES and evaluate/compute_full_metrics.")

    s = slide("Localization saturates at the smallest model scale", "Results / Stage 1",
              "Source: v9 Stage 1 results table | 161-image test set")
    table(s, ["Model", "Params", "FPS", "mAP@.5", "mAP@.5:.95"],
          STAGE1, 0.65, 2.14, [2.32, 1.02, 0.84, 1.20, 1.52],
          row_h=0.66, size=19, highlight=(0, 2))
    callout(s, "222 FPS", "Nano: 3.1M parameters; about 4.5 ms/image",
            8.02, 2.17, 4.60, 1.19)
    picture(s, "fig_yolo_detections.jpg", 8.02, 3.68, 4.60, 2.45,
            crop=(0.0, 0.0, 0.50, 0.25))
    text(s, "Actual validation detections; excerpt of paper montage",
         8.03, 6.30, 4.57, 0.39, 13, MUTED)
    notes(s, """Read the native table as an exact transcription of the v9
localization table. Nano: 3.1M parameters, 222 FPS, .9950 mAP@.5 and .9909
mAP@.5:.95. Small: 11.4M, 216, .9950, .9932. Medium: 26.4M, 148, .9950,
.9947. Large: 44.5M, 118, .9949, .9919. Extra-large: 69.5M, 86, .9950,
.9921. The stricter metric differentiates scales more meaningfully, with
medium best, while nano provides the fastest measured detector stage.
The scoped paper_table.csv corroborates the ranking and unrounded FPS, but
reports .995 for the large model; preserve the manuscript's .9949 rather than
silently correcting it. Some prose rounds all five variants to .9950.
The paper reports perfect recall at the selected detection operating point;
this does not establish perfect generalization or perfect triage. The 222 FPS
number is Stage 1 only on the stated GPU, not full-system throughput.
The montage excerpt contains validation predictions, separate from the numeric
test evaluation. Next, examine corrected classification results.""",
          "v9/main.tex, Stage 1 results; v9/fig_yolo_detections.jpg (top-left half-row);"
          " CellExtraction/YOLO/YOLOv8-OBB-aug/OBB_Benchmark/paper_table.csv.")

    s = slide("DenseNet-121 leads validated B-grade triage", "Results / Stage 2",
              "Source: v9 corrected Stage 2 table | 1,409 independent test cells")
    shown = []
    for row in STAGE2:
        shown.append([row[0], row[2], row[3], row[4], row[8]])
    table(s, ["Model", "Accuracy", "B-F1", "B-recall", "Seeds"], shown,
          0.65, 2.13, [2.00, 1.64, 1.64, 1.64, 0.68], row_h=0.48,
          size=15, highlight=(0,))
    text(s, "+/- denotes standard deviation over three seeds, not confidence intervals.",
         0.69, 6.18, 7.49, 0.43, 14, MUTED)
    picture(s, "fig_stage2_roc.png", 8.56, 2.07, 4.10, 3.59)
    text(s, "Paper ROC plot: one representative seed per model",
         8.58, 5.81, 4.06, 0.55, 14, MUTED)
    notes(s, """The native table uses B-grade-positive scoring from the corrected
v9 benchmark. DenseNet-121 and ConvNeXt-Tiny report mean +/- standard deviation
over seeds 42, 123, and 2026; all other backbones have only seed 42. DenseNet
has accuracy .974 +/-.002, B-F1 .960 +/-.003, and recall .954 +/-.003.
Swin-S and ViT-B/16 are competitive but preliminary; they are not multi-seed
validated winners. The paper reports an independent-samples t-test between
DenseNet and ConvNeXt, t=-3.08, p=.037, with only three seeds each; interpret
that small-sample evidence cautiously. Standard deviations are not confidence
intervals. The actual ROC plot uses a representative seed, specifically the
median-AUC seed for the two three-seed models, so its AUC values are not
three-seed means or PR-AUC values. EfficientNet's representative ROC-AUC is
.994; several models show .992. The caption's superlative is slightly imprecise
because the plot shows multiple ties at .992. Full exact main-table values,
in order Model / parameters M / accuracy / B-F1 / B-recall / B-precision /
PR-AUC / MCC / seeds, follow:\n""" +
          "\n".join(" | ".join(row) for row in STAGE2) +
          "\n\nNext, interpret component sensitivity without inventing an ablation.",
          "v9/main.tex, Stage 2 results table and ROC caption; v9/fig_stage2_roc.png.")

    s = slide("Component analysis: tuning and capacity trade-offs", "Component analysis",
              "Source: v9 Optuna importance figure | No controlled removal ablation reported")
    bullets(s, ["Weight decay dominates this ConvNeXt importance plot.",
                "Freeze strategy and batch size also influence tuning.",
                "Larger detectors improve tight-IoU accuracy only marginally.",
                "Importance scores are not causal component-removal effects."],
            w=5.45, step=0.87, size=22)
    picture(s, "fig_optuna_importance.png", 6.57, 2.00, 6.12, 3.82)
    text(s, "Actual paper figure: broader-search hyperparameter importance",
         6.62, 5.99, 6.00, 0.49, 15, MUTED)
    notes(s, """The manuscript does not provide a controlled remove-one-component
ablation with an accuracy delta. This slide therefore presents component
analysis, not invented ablation outcomes. The actual referenced importance plot
shows approximate weight-decay importance .28, freeze strategy .19, batch size
.16, dropout .11, learning rate .10, model key .09, optional attention .05,
and optimizer/scheduler roughly .01 each. The image comes from a broader
exploratory search, as its caption acknowledges. These fANOVA importance scores
describe variation within that search and do not prove a component's causal
performance improvement. Do not generalize this single plot to every backbone.
The paper's later discussion ranks learning rate and weight decay above
dropout, but the displayed ConvNeXt chart itself gives dropout .11 and LR .10;
the slide follows the observed figure. The five-scale Stage 1 comparison is
an independent capacity analysis: near-saturated mAP@.5, a modest strict-IoU
improvement at medium size, and a throughput cost. Next, inspect what the
actual qualitative figure can and cannot establish.""",
          "v9/main.tex, Optuna importance caption and Stage 1 table;"
          " v9/fig_optuna_importance.png.")

    s = slide("Qualitative localization: diverse EL appearances", "Qualitative analysis",
              "Source: v9 localization figure | Case excerpts, not classifier explanations")
    cases = [
        ((0.0, 0.0, 0.25, 0.25), "Adjacent cells / dark texture"),
        ((0.75, 0.0, 1.0, 0.25), "Single-cell partial view"),
        ((0.50, 0.25, 0.75, 0.50), "Smooth illumination / low contrast"),
    ]
    for i, (crop, label) in enumerate(cases):
        x = 0.65 + i * 4.10
        box(s, x, 2.09, 3.83, 3.81, WHITE, BORDER)
        picture(s, "fig_yolo_detections.jpg", x + 0.12, 2.23, 3.59, 3.17, crop)
        text(s, label, x + 0.12, 5.51, 3.59, 0.28, 14, NAVY, True,
             PP_ALIGN.CENTER)
    callout(s, "Evidence boundary", "Paper examples show localization; no per-case triage ground truth is supplied.",
            0.65, 6.02, 12.03, 0.90)
    notes(s, """These three images are spatial crops of the actual validation
prediction montage referenced by the v9 manuscript. They show an adjacent-cell
example with dark texture, a partial-module view containing a complete cell,
and a smoother, lower-contrast cell example. Preserve detector boxes and
confidence labels exactly as embedded; those confidence values are cell
localization scores, not defect probabilities. The montage visually supports
alignment with cell boundaries, but the selected excerpts do not by themselves
demonstrate strong rotation robustness or generalization to every module format.
There are no supplied expert case labels, per-cell classifier predictions,
or saliency maps for these examples, so do not label them as classifier successes
or failures. The file named classical_failures_placeholder is not referenced
by the active v9 text and is not used as verified classical-failure evidence.
Transition from illustrative localization to the practical quantitative trade-offs.""",
          "v9/main.tex, localization caption; v9/fig_yolo_detections.jpg,"
          " grid cells row 1 col 1, row 1 col 4, row 2 col 3.")

    s = slide("Practical selection: misses, footprint, and throughput", "Key takeaways",
              "Source: v9 per-class results | Parameter-F1 figure | Deployment discussion")
    picture(s, "fig_stage2_paramsf1.png", 0.65, 1.94, 6.60, 4.80)
    text(s, "Choose for the operating constraint", 7.64, 2.10, 5.01, 0.44,
         22, NAVY, True)
    bullets(s, ["DenseNet: strongest validated recall; 21.3 mean false negatives.",
                "Swin-S: fewer false alarms; single-seed evidence only.",
                "MobileNet: 3.0M parameters with B-F1 of 0.949.",
                "Nano OBB: 222 FPS for localization, not end-to-end."],
            x=7.64, y=2.89, w=5.0, step=0.83, size=21)
    notes(s, """The actual parameter-versus-B-grade-F1 figure shows that more
parameters do not consistently improve performance. DenseNet-121 offers .960
B-F1 at 7.0M parameters, while MobileNet offers .949 at 3.0M. Relative to
ResNet's 42.5M, MobileNet is about fourteen times smaller; this is a parameter
comparison, not a measured energy or latency advantage. At the reported
operating point, DenseNet has 21.3 mean missed B-grade cells out of 460,
with 15.0 false positives. Swin has 28 misses but only 10 false positives;
its .977 B-grade precision favors fewer false alarms. Which is preferable
depends on the cost of misses versus unnecessary review. All exact per-class
table values follow, ordered Model / OK precision / OK recall / BG precision /
BG recall / FP / FN. Fractional counts are means over three seeds, not
fractional physical cells:\n""" + "\n".join(" | ".join(row) for row in PERCLASS) +
          """\n\nStage 2 inference latency was not measured, so no integrated
throughput or edge-readiness number is claimed. The paper's parameter
terminology is not fully reconciled with freezing choices; use the published
parameter figures as reported model footprints. Next, state what validation
is still needed before production deployment.""",
          "v9/main.tex, per-class table and Deployment discussion;"
          " v9/fig_stage2_paramsf1.png.")

    s = slide("Limitations define the next validation steps", "Limitations / future work",
              "Source: v9 Limitations | Discussion | Conclusion")
    table(s, ["Current evidence boundary", "Next validation step"], [
        ["Controlled laboratory EL only", "Field-domain testing and adaptation"],
        ["Five classifiers: one seed only", "Multi-seed confirmation across all backbones"],
        ["Separate stage evaluation", "Common labelled end-to-end module test"],
        ["Triage latency unmeasured", "Edge latency benchmark on Jetson Orin"],
        ["Threshold / new formats untested", "Cost-sensitive sweep; wider module coverage"],
    ], 0.65, 2.06, [5.27, 6.74], row_h=0.65, size=20)
    text(s, "Reported augmentation is not a substitute for real-world validation.",
         0.75, 6.30, 11.84, 0.43, 21, NAVY, True, PP_ALIGN.CENTER)
    notes(s, """The highest-priority limitation is domain gap: all evaluation
uses controlled lab EL, while field acquisition can add stronger blur, noise,
compression, background illumination, and current variability. Moderate
augmentation resembles some of these effects but is not field-domain evidence,
and no domain-adaptation method was evaluated. Five classifiers remain
single-seed; their competitive rankings need repeated-run confirmation.
Stage 1 and Stage 2 are tested separately, so true end-to-end quality and
error propagation require a common independently labelled module test set.
Stage 2 and complete-pipeline latency need direct measurement, including
the proposed Jetson Orin edge target. A threshold sweep should explicitly
optimize the miss-versus-false-alarm cost; no selected deployment operating point
beyond the default threshold is established. Half-cut, shingled, and emerging
multi-busbar formats were not validated. The paper also proposes multi-class
type/severity triage with relabelled data. Finally, the inspected exploratory
code differs from the manuscript in loss, head, selection objective, and some
search choices; reproduction requires reconciling those artifacts first.
Close with three measured headline numbers, preserving their stage boundaries.""",
          "v9/main.tex, Domain gap, Uneven seed coverage, Format generalization,"
          " Threshold tuning, End-to-end latency, and Future work.")

    s = slide("Thank you", "Discussion / Q&A",
              "Source: v9 Stage 1 and Stage 2 results | Independent stage evaluations")
    text(s, "Questions & discussion", 0.65, 2.03, 12.03, 0.67, 38, NAVY, True,
         PP_ALIGN.CENTER)
    text(s, "Reliable localization. Defect-aware triage. Transparent evidence.",
         0.65, 3.02, 12.03, 0.50, 23, MUTED, align=PP_ALIGN.CENTER)
    metrics = [
        ("0.9950", "Nano localization mAP@0.5", "161-image test set"),
        ("222 FPS", "Nano localization throughput", "RTX PRO 4500; Stage 1 only"),
        ("96.0 +/- 0.3%", "DenseNet-121 B-grade F1", "Three seeds; mean +/- SD"),
    ]
    for i, (value, label, context) in enumerate(metrics):
        x = 0.65 + i * 4.10
        box(s, x, 4.09, 3.83, 2.13, NAVY)
        text(s, value, x + 0.13, 4.40, 3.57, 0.55,
             29 if i == 2 else 34, "5EEAD4", True, PP_ALIGN.CENTER)
        text(s, label, x + 0.13, 5.23, 3.57, 0.34, 17, WHITE, True,
             PP_ALIGN.CENTER)
        text(s, context, x + 0.13, 5.75, 3.57, 0.30, 13, "CBD5E1",
             align=PP_ALIGN.CENTER)
    notes(s, """Thank the audience and invite questions about dataset grouping,
OBB geometry, the search budget, B-grade-positive evaluation, missed-cell costs,
and reproducibility. The three takeaway numbers are deliberately scoped:
YOLOv8n-OBB achieves .9950 mAP@0.5 on the 161-image localization test set;
its reported GPU detector-stage throughput is 222 FPS; DenseNet-121 achieves
96.0 +/-0.3 percent B-grade F1 across three seeds on the independent cell test
set. Its companion accuracy is 97.4 +/-0.2 percent and B-grade recall is
95.4 +/-0.3 percent. Do not turn these into an end-to-end guarantee, field
performance claim, or classifier latency figure. A credible next step is
common-test-set pipeline evaluation and complete seed coverage. If asked about
metadata, authorship and affiliation were omitted because the submitted author
block is commented out and contains placeholders. If asked about implementation,
acknowledge the documented mismatch between exploratory two-logit macro-F1 code
and the single-logit AUC methodology reported by the manuscript.""",
          "v9/main.tex, abstract, results, limitations, and conclusion.")
    assert len(prs.slides) == 16
    assert FIGURES_USED <= referenced, (FIGURES_USED, referenced)


def validate(path):
    reopened = Presentation(path)
    assert len(reopened.slides) == 16
    assert abs(reopened.slide_width / reopened.slide_height - 16 / 9) < 1e-6
    all_text = []
    images = 0
    for i, s in enumerate(reopened.slides, 1):
        assert s.has_notes_slide
        note = s.notes_slide.notes_text_frame.text
        assert len(note.split()) >= 80, (i, "insufficient speaker notes")
        all_text.append(note)
        for shape in s.shapes:
            assert shape.left >= 0 and shape.top >= 0, (i, shape.name)
            assert shape.left + shape.width <= reopened.slide_width + 10
            assert shape.top + shape.height <= reopened.slide_height + 10
            if shape.has_text_frame:
                all_text.append(shape.text)
            if shape.has_table:
                all_text.extend(c.text for row in shape.table.rows for c in row.cells)
            if shape.shape_type == 13:
                with Image.open(io.BytesIO(shape.image.blob)) as image:
                    image.verify()
                images += 1
    assert not re.search(r"segment|unet|dice|iou.mask", "\n".join(all_text), re.I)
    for i, a in enumerate(TEXT_REGIONS):
        for b in TEXT_REGIONS[i + 1:]:
            if a[0] != b[0]:
                continue
            overlap_w = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            overlap_h = min(a[2] + a[4], b[2] + b[4]) - max(a[2], b[2])
            assert overlap_w <= 0.005 or overlap_h <= 0.005, (
                "Textboxes overlap", a[0], a[-1], b[-1])
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        names = set(z.namelist())
        for name in names:
            if not name.endswith(".rels"):
                continue
            base = posixpath.dirname(posixpath.dirname(name))
            for rel in ET.fromstring(z.read(name)):
                if rel.attrib.get("TargetMode") == "External":
                    continue
                target = rel.attrib["Target"]
                resolved = target.lstrip("/") if target.startswith("/") else (
                    posixpath.normpath(posixpath.join(base, target)))
                assert resolved in names, (name, resolved)
    return {"slides": 16, "aspect_ratio": "16:9", "embedded_images": images,
            "notes": 16, "package_relationships": "valid",
            "textboxes": "in bounds; no intersecting textboxes",
            "paper_figures": sorted(FIGURES_USED),
            "quantitative_source": str(PAPER / "main.tex")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "presentation.pptx")
    args = parser.parse_args()
    build()
    prs.save(args.output)
    report = validate(args.output)
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
