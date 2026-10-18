"""Assemble the English poster as a .docx: page 1 poster, page 2 supplementary."""
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path(r"C:\MOV\metal_defect_vlm\output")
FIGS = OUT / "poster_figs"
NAVY = RGBColor(0x17, 0x36, 0x5D)
GREY = RGBColor(0x44, 0x44, 0x44)

doc = Document()
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Cm(55.8), Cm(39.5)
sec.left_margin = sec.right_margin = Cm(1.0)
sec.top_margin = sec.bottom_margin = Cm(0.7)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(8)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")


def shade(cell, hexcolor):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcpr.append(shd)


def no_borders(table):
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    table._tbl.tblPr.append(borders)


def para(host, text, size=7.6, bold=False, color=None, align=None, space_after=2,
         italic=False, first=False):
    p = host.paragraphs[0] if first else host.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    return p


def heading(cell, text, size=10):
    p = cell.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1.5)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = NAVY
    return p


def bullets(cell, items, size=7.2):
    for it in items:
        p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.left_indent = Cm(0.3)
        run = p.add_run("- " + it)
        run.font.size = Pt(size)


def fig(cell, name, width_cm, caption=None, cap_size=6.4):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(str(FIGS / name), width=Cm(width_cm))
    if caption:
        cp = cell.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        cp.paragraph_format.space_after = Pt(2.5)
        run = cp.add_run(caption)
        run.font.size = Pt(cap_size)
        run.font.italic = True
        run.font.color.rgb = GREY


def small_table(cell, rows, widths=None, size=6.6, header_fill="17365D"):
    t = cell.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(val))
            run.font.size = Pt(size)
            if i == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(c, header_fill)
    if widths:
        for j, w in enumerate(widths):
            for i in range(len(rows)):
                t.cell(i, j).width = Cm(w)
    return t


# ================= PAGE 1 : TITLE =================
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_after = Pt(2)
r = tp.add_run("Comparative Evaluation of Vision-Language Models for Metal Surface Defect Classification")
r.font.size = Pt(24)
r.font.bold = True
r.font.color.rgb = NAVY

sp = doc.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sp.paragraph_format.space_after = Pt(3)
r = sp.add_run("Expert Prompting and 4-bit QLoRA Adaptation of Three Architecture-Distinct VLMs on the GC10-DET Dataset")
r.font.size = Pt(12)
r.font.italic = True
r.font.color.rgb = GREY

ap = doc.add_paragraph()
ap.alignment = WD_ALIGN_PARAGRAPH.CENTER
ap.paragraph_format.space_after = Pt(1)
r = ap.add_run("Shixiang Wang, Yongjing Wang, Richard Fox")
r.font.size = Pt(11.5)
r.font.bold = True

aff = doc.add_paragraph()
aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
aff.paragraph_format.space_after = Pt(4)
r = aff.add_run("School of Electronic, Electrical and Systems Engineering (EESE), University of Birmingham")
r.font.size = Pt(9)
r.font.color.rgb = GREY

rq = doc.add_table(rows=1, cols=1)
no_borders(rq)
c = rq.cell(0, 0)
shade(c, "DCE6F1")
p = c.paragraphs[0]
p.paragraph_format.space_after = Pt(0)
run = p.add_run("Research question:  ")
run.font.bold = True
run.font.size = Pt(9)
run = p.add_run("Under a common dataset split, Expert Prompting strategy and 4-bit QLoRA framework, how do different "
                "open-source VLM architecture families compare in predictive performance, class-specific robustness and "
                "practical deployment feasibility?")
run.font.size = Pt(9)

doc.add_paragraph().paragraph_format.space_after = Pt(1)
fig(doc, "fig2_pipeline.png", 53.4,
    "Figure 1. Experimental workflow. All three VLMs were adapted with 4-bit QLoRA in LLaMA-Factory on a common stratified "
    "split and evaluated on the identical 240-image test set; each model retained its native vision processor.", cap_size=7)

# ================= PAGE 1 : FIVE COLUMNS =================
cols = doc.add_table(rows=1, cols=5)
no_borders(cols)
cols.alignment = WD_TABLE_ALIGNMENT.CENTER
W = 10.6
for j in range(5):
    cols.cell(0, j).width = Cm(W)
    cols.columns[j].width = Cm(W)
c1, c2, c3, c4, c5 = [cols.cell(0, j) for j in range(5)]
for c in (c1, c2, c3, c4, c5):
    c.paragraphs[0].paragraph_format.space_after = Pt(0)

# ---------- Column 1 ----------
heading(c1, "1. Background")
para(c1, "Automated metal-surface inspection is essential for industrial quality control, yet conventional vision models "
         "need task-specific architectures for a fixed class set. Vision-language models (VLMs) offer a unified framework "
         "for image understanding and instruction-based classification, but general benchmarks do not reflect specialised "
         "industrial imagery: metal defects show low contrast, subtle texture change and strong inter-class similarity, "
         "combined with substantial class imbalance.", size=7.6)
para(c1, "Three open-source VLMs were selected along the architectural spectrum rather than by popularity: Qwen2.5-VL-7B "
         "(dynamic-resolution ViT with M-RoPE cross-modal alignment), Gemma3-4B (lightweight SigLIP-based multimodal route) "
         "and InternVL3-8B (native multimodal pre-training with a separate encoder and dynamic high-resolution tiling).",
     size=7.6)

heading(c1, "2. Objectives")
bullets(c1, [
    "Adapt three representative VLMs to ten-class defect classification with 4-bit QLoRA.",
    "Compare overall and class-wise performance on a common held-out test set.",
    "Identify shared and model-specific directional misclassification patterns.",
    "Examine deployment constraints across a consumer workstation and an HPC platform.",
    "Assess whether a smaller VLM offers a competitive accuracy-resource trade-off."])

heading(c1, "3. Dataset")
para(c1, "The experimental dataset contains 2,312 grayscale GC10-DET steel-surface images in ten defect categories, divided "
         "by stratified sampling (seed 42) into 1,845 training, 227 validation and 240 test images (approx. 8:1:1). The "
         "240-image test set is identical for all models.", size=7.6)
fig(c1, "fig1_class_distribution.png", 10.3,
    "Figure 2. Class distribution of the experimental dataset and the held-out test set. Silk spot accounts for 28.2% of the "
    "dataset while rolled pit accounts for 1.3%; macro-averaged metrics are therefore reported alongside accuracy.")
small_table(c1, [
    ["Class", "Data", "Test"], ["Punching hole", "219", "23"], ["Welding line", "273", "28"],
    ["Crescent gap", "226", "24"], ["Water spot", "289", "30"], ["Oil spot", "204", "21"],
    ["Silk spot", "651", "66"], ["Inclusion", "216", "23"], ["Rolled pit", "31", "4"],
    ["Crease", "53", "6"], ["Waist folding", "150", "15"], ["Total", "2,312", "240"]],
    widths=[5.0, 2.6, 2.2], size=6.5)

# ---------- Column 2 ----------
heading(c2, "4. Architecture-Driven Model Selection")
small_table(c2, [
    ["Model", "Architecture rationale"],
    ["Qwen2.5-VL-7B", "Dynamic-resolution ViT + M-RoPE; fine-grained grounding route"],
    ["Gemma3-4B", "SigLIP encoder, compressed visual tokens; lightweight edge route"],
    ["InternVL3-8B", "Native multimodal pre-training; separate encoder + dynamic tiling"],
    ["LLaVA-OneVision-7B", "CLIP-ViT + projector baseline; excluded (Section 10)"]],
    widths=[3.2, 7.0], size=6.6)
para(c2, "The comparison covers three distinct VLM design strategies under one task, one split, one prompting strategy and "
         "one fine-tuning framework.", size=7.2, italic=True)

heading(c2, "5. Methodology")
para(c2, "Image preprocessing. A common reference pipeline converted each image to grayscale, enhanced local contrast with "
         "CLAHE (clip limit 2.0, 8x8 tile grid), applied light Gaussian denoising (3x3, sigma 0.8), converted to RGB and "
         "resized to 512x512; each model then applied its native vision processor, so the final visual-token representation "
         "remained model-specific.", size=7.6)
para(c2, "Expert Prompting (EP). Each model acted as a senior metal-surface inspection expert, received per-class "
         "morphological descriptions and a three-step analysis scaffold, and was constrained to strict JSON output:",
     size=7.6)
pt = c2.add_table(rows=1, cols=1)
no_borders(pt)
pc = pt.cell(0, 0)
shade(pc, "F2F2F2")
para(pc, 'You are a senior metal-surface quality-inspection expert. Domain knowledge: ten defect classes with morphology '
         'descriptions (e.g. punching_hole: round/oval holes; welding_line: linear weld traces; inclusion: embedded foreign '
         'particles). Steps: (1) inspect for anomalous regions; (2) match the ten class descriptions; (3) determine defect '
         'type and severity. Output strictly JSON only: {"defect_type": <class>, "severity": <mild|moderate|severe>, '
         '"description": <brief morphology>}.', size=6.6, italic=True, first=True)
para(c2, "Training. 4-bit NF4 QLoRA in LLaMA-Factory for all three VLMs: LoRA rank 16, alpha 32, dropout 0.05 on "
         "q/k/v/o/gate/up/down projections; 3 epochs; LR 1e-4, cosine schedule, 20-step warm-up; batch 1 x accumulation 8; "
         "cutoff 1024; bf16 + gradient checkpointing; seed 42. Vision tower and projector frozen for InternVL3-8B. Adapters "
         "were merged into base weights before evaluation.", size=7.6)
para(c2, "Evaluation. Labels were parsed from the JSON defect_type field and scored on the common test set with accuracy, "
         "Macro-F1, Weighted-F1 and per-class precision, recall and F1.", size=7.6)

heading(c2, "6. Why Are the Confusion Matrices Asymmetric?")
at = c2.add_table(rows=1, cols=1)
no_borders(at)
ac = at.cell(0, 0)
shade(ac, "FCE4D6")
para(ac, "Entries Cij and Cji are two different class-conditional errors and are not theoretically equal. Unequal class "
         "support magnifies asymmetry in raw counts, while directional visual ambiguity determines the dominant error flow: "
         "an ambiguous inclusion may resemble a water or oil spot, whereas a distinctive spot rarely resembles an inclusion "
         "to the same extent. Row normalisation shows recall; column normalisation shows precision.", size=7.2, first=True)

# ---------- Column 3 ----------
heading(c3, "7. Overall Predictive Performance")
fig(c3, "fig3_overall.png", 10.2,
    "Figure 3. Overall performance of the three QLoRA-adapted VLMs (vertical axis truncated at 80%). Gemma3-4B achieved the "
    "highest accuracy and Weighted-F1; InternVL3-8B achieved the highest Macro-F1.")
small_table(c3, [
    ["Model", "Acc.", "Ma-P", "Ma-R", "Ma-F1", "W-F1", "Err."],
    ["Qwen2.5-VL-7B", "86.67", "86.19", "83.05", "84.14", "86.63", "32"],
    ["Gemma3-4B", "87.08", "85.99", "84.00", "83.59", "86.75", "31"],
    ["InternVL3-8B", "86.67", "85.57", "84.44", "84.80", "86.52", "32"]],
    widths=[3.2, 1.3, 1.2, 1.2, 1.3, 1.3, 0.9], size=6.5)
kt = c3.add_table(rows=1, cols=1)
no_borders(kt)
kc = kt.cell(0, 0)
shade(kc, "DCE6F1")
para(kc, "Key result: accuracy is closely matched (86.67-87.08%; gap 0.41 pp) yet accuracy and Macro-F1 rankings disagree, "
         "so aggregate accuracy alone cannot rank models on this imbalanced test set.", size=7.4, first=True)

heading(c3, "8. Class-Wise Recall")
fig(c3, "fig4_recall_heatmap.png", 8.2,
    "Figure 4. Class-wise recall. Gemma3-4B led on punching hole, crescent gap and water spot; Qwen2.5-VL-7B reached perfect "
    "recall on silk spot; InternVL3-8B classified all four rolled-pit samples correctly. Inclusion recall was 69.57% for all "
    "three models; rolled pit (n=4) and crease (n=6) should be read cautiously.")

# ---------- Column 4 ----------
heading(c4, "9. Per-Class F1-Score")
fig(c4, "fig5_f1_heatmap.png", 8.6,
    "Figure 5. Per-class F1-score. No model dominated every class: Qwen2.5-VL-7B led on oil spot, silk spot, crease and waist "
    "folding; Gemma3-4B on punching hole and inclusion; InternVL3-8B on crescent gap, water spot and rolled pit.")

heading(c4, "10. Error Analysis")
fig(c4, "fig6_misclass.png", 5.6,
    "Figure 6. Dominant directional misclassifications per model (true > predicted).")
para(c4, "Two recurrent confusion groups appear: punching hole vs welding line (shared elongated, edge-dominated structure) "
         "and inclusion vs spot-like defects (small, low-contrast local features). Error directions differ between models, "
         "confirming that the asymmetry is directional rather than a symmetric swap. Row-normalised confusion matrices for "
         "all three models are given in Figure S2.", size=7.4)

# ---------- Column 5 ----------
heading(c5, "11. Deployment Feasibility")
small_table(c5, [
    ["Model", "Environment", "Latency", "Observation"],
    ["Qwen2.5-VL-7B (7B)", "Local RTX 5070 Ti", "1.690 s/img", "Locally executable"],
    ["Gemma3-4B (4B)", "Local RTX 5070 Ti", "2.529 s/img", "Locally executable"],
    ["InternVL3-8B (8B)", "BlueBEAR A100", "62.723 s/img", "HPC needed here"]],
    widths=[3.4, 2.8, 1.9, 2.4], size=6.4)
para(c5, "Latency is environment-specific and not directly comparable (different hardware, token counts, I/O and decoding). "
         "InternVL3-8B failed to load on the local Windows workstation: 4-bit loading hit a bitsandbytes compatibility error "
         "during weight materialisation and BF16 loading of the 16 GB weights exceeded the Windows page-file commit limit "
         "(OS error 1455); 4-bit QLoRA then completed normally on BlueBEAR Linux/A100. This reflects platform compatibility "
         "plus resource needs, not an inherent HPC requirement.", size=7.2)
para(c5, "Gemma3-4B achieved the highest accuracy with the smallest parameter count on a consumer GPU: a favourable "
         "accuracy-resource profile in the present setup.", size=7.4, bold=True)

heading(c5, "12. Excluded Baseline and Scope Limitation")
para(c5, "LLaVA-OneVision-7B was planned as a fourth family (CLIP-ViT + projector + AnyRes multi-tile) but excluded after "
         "persistent AnyRes image-token alignment failures and dependency conflicts that could not be rebuilt under the "
         "BlueBEAR storage quota. The study therefore has no LLaVA-family baseline and does not test AnyRes multi-tile "
         "processing on small defects; this is an environment-integration limitation, not evidence of poor LLaVA ability.",
     size=7.4)

heading(c5, "13. Relation to Prior Industrial VLM Research")
para(c5, "Expert Prompting with role assignment, domain knowledge and structured output previously raised Qwen2.5-VL-7B "
         "accuracy by 12.7 points on the six-class NEU benchmark [8]; our EP schema extends that design to ten GC10-DET "
         "classes. Gemma3-4B achieved the highest aggregate accuracy without dominating every class, qualitatively "
         "consistent with the moderate-but-stable Gemma-3 behaviour reported in LNG-tank LVLM inspection research [9]; "
         "different datasets and protocols preclude direct benchmark comparison.", size=7.4)

heading(c5, "14. Key Findings")
small_table(c5, [
    ["87.08%", "Highest accuracy - Gemma3-4B"],
    ["84.80%", "Highest Macro-F1 - InternVL3-8B"],
    ["86.19%", "Highest Macro precision - Qwen2.5-VL-7B"],
    ["0.41 pp", "Maximum accuracy gap between models"],
    ["69.57%", "Inclusion recall, all three models"]],
    widths=[2.2, 8.2], size=7.0)

heading(c5, "15. Conclusions")
bullets(c5, [
    "Three architecture-distinct VLMs reached closely matched overall performance under one EP + 4-bit QLoRA protocol.",
    "Gemma3-4B: highest accuracy (87.08%) and Weighted-F1 (86.75%) with the smallest parameter count.",
    "InternVL3-8B: highest Macro-F1 (84.80%) and Macro recall (84.44%), the most balanced class-wise profile.",
    "Similar accuracy concealed large class-specific differences; recall, F1 and directional errors must be read jointly.",
    "Inclusion is a shared weakness; punching hole and welding line form the main reciprocal confusion group.",
    "Deployment observations are platform-specific and must not be read as a controlled speed ranking."], size=7.2)
para(c5, "References, limitations, future work and supplementary tables/figures: page 2.", size=6.8, italic=True)

# ================= PAGE 2 : SUPPLEMENTARY =================
doc.add_page_break()
h = doc.add_paragraph()
h.paragraph_format.space_after = Pt(4)
r = h.add_run("Supplementary Material - Comparative Evaluation of VLMs for Metal Surface Defect Classification")
r.font.size = Pt(15)
r.font.bold = True
r.font.color.rgb = NAVY

s = doc.add_table(rows=1, cols=2)
no_borders(s)
s.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, w in enumerate([27.0, 26.4]):
    s.cell(0, j).width = Cm(w)
    s.columns[j].width = Cm(w)
sl, sr = s.cell(0, 0), s.cell(0, 1)
sl.paragraphs[0].paragraph_format.space_after = Pt(0)
sr.paragraphs[0].paragraph_format.space_after = Pt(0)

fig(sl, "fig8_examples.png", 26.5,
    "Figure S1. Preprocessed GC10-DET examples (CLAHE-enhanced, 512 x 512). Visually similar spot-like defects (water spot, "
    "oil spot, inclusion) form the dominant confusion group.", cap_size=7)
fig(sl, "fig7_cm_normalized.png", 26.5,
    "Figure S2. Row-normalised confusion matrices (%) of the three models on the common 240-image test set.", cap_size=7)

heading(sr, "Table S1. Per-class recall / F1 (%) and dataset counts", size=10)
small_table(sr, [
    ["Class", "Data", "Test", "R-Q", "R-G", "R-I", "F-Q", "F-G", "F-I"],
    ["Punching hole", "219", "23", "73.9", "95.7", "78.3", "81.0", "86.3", "76.6"],
    ["Welding line", "273", "28", "78.6", "57.1", "71.4", "75.9", "72.7", "75.5"],
    ["Crescent gap", "226", "24", "95.8", "100", "95.8", "88.5", "87.3", "90.2"],
    ["Water spot", "289", "30", "86.7", "96.7", "93.3", "83.9", "86.6", "87.5"],
    ["Oil spot", "204", "21", "81.0", "85.7", "85.7", "87.2", "83.7", "83.7"],
    ["Silk spot", "651", "66", "100", "97.0", "97.0", "98.5", "98.5", "97.7"],
    ["Inclusion", "216", "23", "69.6", "69.6", "69.6", "71.1", "78.1", "74.4"],
    ["Rolled pit", "31", "4", "75.0", "75.0", "100", "85.7", "85.7", "100"],
    ["Crease", "53", "6", "83.3", "83.3", "66.7", "76.9", "71.4", "72.7"],
    ["Waist folding", "150", "15", "86.7", "80.0", "86.7", "92.9", "85.7", "89.7"],
], widths=[3.6, 1.8, 1.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], size=6.4)
para(sr, "R = recall, F = F1-score; Q = Qwen2.5-VL-7B, G = Gemma3-4B, I = InternVL3-8B.", size=6.6, italic=True)

heading(sr, "Limitations", size=10)
bullets(sr, [
    "Test set limited to 240 images; rolled pit (n=4) and crease (n=6) metrics are unstable.",
    "No ablation of CLAHE/denoising or of the EP schema; native processors yield different visual tokens.",
    "Latency measured on heterogeneous platforms; no confidence scores, hence no ROC/PR curves or paired significance "
    "tests (only Gemma per-sample predictions were retained).",
    "LLaVA-OneVision-7B could not be included (Section 12)."], size=7.2)

heading(sr, "Future Work", size=10)
bullets(sr, [
    "Identical-hardware re-benchmarking with recorded peak memory, throughput and timing boundaries.",
    "Repeated seeds with uncertainty intervals; sample-level prediction logs for McNemar and bootstrap tests.",
    "Controlled ablations of preprocessing and Expert Prompting; minority-class test expansion.",
    "Ensembles exploiting complementary class strengths; extension to localisation and segmentation.",
    "Public release of the project code, prompt templates and evaluation scripts on GitHub to support reproducibility."],
    size=7.2)

heading(sr, "References", size=10)
refs = [
    "[1] X. Lv, F. Duan, J.-J. Jiang, X. Fu and L. Gan, Deep metallic surface defect detection: The new benchmark and detection network (GC10-DET), Sensors, vol. 20, no. 6, art. 1562, 2020.",
    "[2] T. Dettmers, A. Pagnoni, A. Holtzman and L. Zettlemoyer, QLoRA: Efficient finetuning of quantized LLMs, NeurIPS 36, 2023.",
    "[3] Y. Zheng et al., LLaMA-Factory: Unified efficient fine-tuning of 100+ language models, ACL 2024 System Demonstrations.",
    "[4] S. Bai et al., Qwen2.5-VL technical report, arXiv:2502.13923, 2025.",
    "[5] Gemma Team, Gemma 3 technical report, arXiv:2503.19786, 2025.",
    "[6] J. Zhu et al., InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models, arXiv:2504.10479, 2025.",
    "[7] B. Li et al., LLaVA-OneVision: Easy visual task transfer, arXiv:2408.03326, 2024.",
    "[8] A. H. Aroua and F. Abbas, Enhanced prompting for vision language models in industrial surface defect detection: An empirical analysis, 2025 Int. Conf. on Artificial Intelligence and Innovative Applications (AIIA), doi:10.1109/AIIA68273.2025.11383671.",
    "[9] H. V. Nguyen, H. Park, N. Yoo and J. Yang, Resource-efficient adaptation of large vision-language models for multimodal defect inspection in LNG tank manufacturing, Int. J. Adv. Manuf. Technol., vol. 144, pp. 5135-5152, 2026, doi:10.1007/s00170-026-17818-y.",
]
for rtext in refs:
    p = sr.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run(rtext)
    run.font.size = Pt(6.6)

doc.save(str(OUT / "Metal_Defect_VLM_Poster_revised.docx"))
print("saved", OUT / "Metal_Defect_VLM_Poster_revised.docx")
