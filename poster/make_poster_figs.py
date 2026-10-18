"""Generate all figures for the A1-portrait poster.
Each figure's figsize equals its PRINT size in inches on the poster, so the
font sizes below are the effective printed sizes (readable at A1)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

OUT = Path(r"C:\MOV\metal_defect_vlm\output\poster_figs")
OUT.mkdir(parents=True, exist_ok=True)
PROJ = Path(r"C:\MOV\metal_defect_vlm")
CM2IN = 1 / 2.54

CLASSES = ["punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot",
           "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"]
LABELS = ["Punching hole", "Welding line", "Crescent gap", "Water spot", "Oil spot",
          "Silk spot", "Inclusion", "Rolled pit", "Crease", "Waist folding"]
MODELS = ["Qwen2.5-VL-7B", "Gemma3-4B", "InternVL3-8B"]
COLORS = {"Qwen2.5-VL-7B": "#377EB8", "Gemma3-4B": "#F28E2B", "InternVL3-8B": "#59A14F"}
NAVY = "#17365D"

FULL = [219, 273, 226, 289, 204, 651, 216, 31, 53, 150]
TEST = [23, 28, 24, 30, 21, 66, 23, 4, 6, 15]

RECALL = np.array([
    [73.91, 95.65, 78.26], [78.57, 57.14, 71.43], [95.83, 100.00, 95.83],
    [86.67, 96.67, 93.33], [80.95, 85.71, 85.71], [100.00, 96.97, 96.97],
    [69.57, 69.57, 69.57], [75.00, 75.00, 100.00], [83.33, 83.33, 66.67],
    [86.67, 80.00, 86.67]])
F1 = np.array([
    [80.95, 86.27, 76.60], [75.86, 72.73, 75.47], [88.46, 87.27, 90.20],
    [83.87, 86.57, 87.50], [87.18, 83.72, 83.72], [98.51, 98.46, 97.71],
    [71.11, 78.05, 74.42], [85.71, 85.71, 100.00], [76.92, 71.43, 72.73],
    [92.86, 85.71, 89.66]])

CM = {
    "Qwen2.5-VL-7B": [
        [17, 6, 0, 0, 0, 0, 0, 0, 0, 0], [2, 22, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 23, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 26, 0, 1, 2, 0, 1, 0],
        [0, 0, 0, 1, 17, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 66, 0, 0, 0, 0],
        [0, 0, 0, 5, 1, 1, 16, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 3, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 5, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 13]],
    "Gemma3-4B": [
        [22, 0, 0, 0, 1, 0, 0, 0, 0, 0], [6, 16, 6, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 24, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 29, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 18, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 64, 0, 0, 0, 1],
        [0, 0, 0, 6, 1, 0, 16, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 3, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 5, 0], [0, 0, 1, 0, 0, 0, 0, 0, 2, 12]],
    "InternVL3-8B": [
        [18, 4, 1, 0, 0, 0, 0, 0, 0, 0], [5, 20, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 23, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 28, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 1, 18, 0, 2, 0, 0, 0], [1, 0, 0, 0, 1, 64, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 1, 16, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 4, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 13]],
}

ERRORS = {
    "Qwen2.5-VL-7B": [("Punching hole > Welding line", 6), ("Inclusion > Water spot", 5),
                      ("Welding line > Crescent gap", 4), ("Oil spot > Inclusion", 3)],
    "Gemma3-4B": [("Welding line > Punching hole", 6), ("Welding line > Crescent gap", 6),
                  ("Inclusion > Water spot", 6), ("Oil spot > Inclusion", 2)],
    "InternVL3-8B": [("Welding line > Punching hole", 5), ("Punching hole > Welding line", 4),
                     ("Inclusion > Water spot", 3), ("Inclusion > Oil spot", 3)],
}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#888888",
                     "axes.labelcolor": "#222222", "text.color": "#222222",
                     "xtick.color": "#444444", "ytick.color": "#444444"})

# ---------- Figure 1: class distribution (print width 18.6 cm) ----------
fig, axes = plt.subplots(1, 2, figsize=(18.6 * CM2IN, 8.2 * CM2IN))
y = np.arange(len(CLASSES))[::-1]
for ax, vals, title in [(axes[0], FULL, "(a) Experimental dataset (n = 2,312)"),
                        (axes[1], TEST, "(b) Test set (n = 240)")]:
    ax.barh(y, vals, color=NAVY, alpha=0.85, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(LABELS, fontsize=12.5)
    for yy, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yy, f"{v}", va="center", fontsize=11.5)
    ax.set_title(title, fontsize=13.5, color=NAVY, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.tick_params(axis="x", labelsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig1_class_distribution.png", dpi=220)
plt.close(fig)

# ---------- Figure 2: horizontal pipeline banner (print width 51.4 cm) ----------
CANVAS_H_IN = 6.4 * CM2IN
UNIT_PT = CANVAS_H_IN * 72 / 12  # pt per y-unit
fig, ax = plt.subplots(figsize=(51.4 * CM2IN, CANVAS_H_IN))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, 100)
ax.set_ylim(0, 12)
ax.axis("off")
CANVAS_W_PT = 51.4 * 72 / 2.54

def hbox(x, y, w, h, text, fc="#FFFFFF", fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                                fc=fc, ec=NAVY, lw=1.4))
    lines = text.split("\n")
    n = len(lines)
    avail_pt = w / 100 * CANVAS_W_PT * 0.90
    for i, ln in enumerate(lines):
        bold = i == 0
        fs_ln = min(fs if bold else fs - 1.0, avail_pt / (len(ln) * 0.66))
        dy = fs_ln * 1.5 / UNIT_PT
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2 + (n / 2 - 0.5 - i) * dy, ln, ha="center",
                va="center", fontsize=fs_ln, fontweight=weight, color="#1a1a1a")

def harrow(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=18,
                                 color=NAVY, lw=1.8))

hbox(0.5, 3.4, 12.5, 5.2, "GC10-DET Dataset\n2,312 grayscale images\n10 defect classes")
harrow(13.2, 15.0, 6.0)
hbox(15.2, 3.4, 12.5, 5.2, "Stratified Split ~8:1:1\nTrain 1,845 | Val 227\nTest 240 (common)")
harrow(27.9, 29.7, 6.0)
hbox(29.9, 3.4, 14.0, 5.2, "Common Preprocessing\nCLAHE > denoise > RGB > 512x512\n+ native vision processor", fs=11.8)
harrow(44.1, 45.9, 6.0)
hbox(46.1, 3.4, 10.5, 5.2, "Expert Prompting\nconstrained JSON\noutput schema", fs=12.2)
harrow(56.8, 58.6, 6.0)
hbox(58.8, 8.2, 15.5, 3.2, "Qwen2.5-VL-7B | 4-bit QLoRA\nLocal RTX 5070 Ti", fc="#EAF1F8", fs=12)
hbox(58.8, 4.4, 15.5, 3.2, "Gemma3-4B | 4-bit QLoRA\nLocal RTX 5070 Ti", fc="#FDF0E3", fs=12)
hbox(58.8, 0.6, 15.5, 3.2, "InternVL3-8B | 4-bit QLoRA\nBlueBEAR A100", fc="#EBF3EA", fs=12)
harrow(74.5, 76.3, 6.0)
hbox(76.5, 3.4, 11.0, 5.2, "Adapter Merging\nInference + label\nparsing", fs=12.2)
harrow(87.7, 89.5, 6.0)
hbox(89.7, 3.4, 9.8, 5.2, "240-Image Eval\nmetrics + error\nanalysis", fc="#DCE6F1", fs=12.2)
fig.savefig(OUT / "fig2_pipeline.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 3: overall performance (print width 18.6 cm) ----------
metrics = ["Accuracy", "Macro-F1", "Weighted-F1"]
vals = {"Qwen2.5-VL-7B": [86.67, 84.14, 86.63],
        "Gemma3-4B": [87.08, 83.59, 86.75],
        "InternVL3-8B": [86.67, 84.80, 86.52]}
fig, ax = plt.subplots(figsize=(18.0 * CM2IN, 9.6 * CM2IN))
x = np.arange(len(metrics))
w = 0.26
for i, m in enumerate(MODELS):
    b = ax.bar(x + (i - 1) * w, vals[m], w, label=m, color=COLORS[m], edgecolor="white")
    ax.bar_label(b, fmt="%.2f", fontsize=12, padding=2)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13)
ax.set_ylim(80, 89)
ax.set_ylabel("Percentage (%)", fontsize=12.5)
ax.set_title("Overall Performance on the Common\n240-Image Test Set (y-axis truncated at 80%)",
             fontsize=13.5, color=NAVY, fontweight="bold")
ax.legend(fontsize=12, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.grid(axis="y", ls=":", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=11.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_overall.png", dpi=220)
plt.close(fig)

# ---------- Figures 4/5: heatmaps (print width 11.6 cm = canvas) ----------
def heatmap(data, title, fname):
    fig, ax = plt.subplots(figsize=(11.6 * CM2IN, 8.4 * CM2IN))
    im = ax.imshow(data, cmap="Blues", vmin=50, vmax=100, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Qwen2.5-VL-7B", "Gemma3-4B", "InternVL3-8B"], fontsize=11.5, rotation=18)
    ax.set_yticks(range(10))
    ax.set_yticklabels(LABELS, fontsize=12.5)
    best = data.argmax(axis=1)
    for i in range(10):
        for j in range(3):
            weight = "bold" if j == best[i] else "normal"
            color = "white" if data[i, j] > 88 else "#1a1a1a"
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                    fontsize=11.5, fontweight=weight, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("%", fontsize=11.5)
    cb.ax.tick_params(labelsize=10.5)
    ax.set_title(title, fontsize=13.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=220)
    plt.close(fig)

heatmap(RECALL, "Class-wise Recall (%)", "fig4_recall_heatmap.png")
heatmap(F1, "Per-class F1-score (%)", "fig5_f1_heatmap.png")

# ---------- Figure 6: dominant misclassifications (print width 16.6 cm = canvas) ----------
fig, axes = plt.subplots(1, 3, figsize=(16.6 * CM2IN, 6.4 * CM2IN), sharex=True)
for ax, m in zip(axes, MODELS):
    errs = ERRORS[m]
    names = [e[0].replace(" > ", "\n\u2192 ") for e in errs][::-1]
    counts = [e[1] for e in errs][::-1]
    ax.barh(range(len(counts)), counts, color=COLORS[m], alpha=0.9, height=0.62)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(names, fontsize=11.5)
    for i, v in enumerate(counts):
        ax.text(v + 0.12, i, str(v), va="center", fontsize=12)
    ax.set_title(m, fontsize=13, color=COLORS[m], fontweight="bold")
    ax.set_xlim(0, 7.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.tick_params(axis="x", labelsize=11)
axes[1].set_xlabel("Misclassified samples (true class \u2192 predicted class)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig6_misclass.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 7: row-normalized confusion matrices (print width 18.6 cm) ----------
fig, axes = plt.subplots(1, 3, figsize=(18.6 * CM2IN, 7.0 * CM2IN))
for ax, m in zip(axes, MODELS):
    cm = np.array(CM[m], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10))
    ax.set_xticklabels([c[:6] for c in CLASSES], fontsize=9, rotation=60, ha="right")
    ax.set_yticks(range(10))
    ax.set_yticklabels([c[:6] for c in CLASSES], fontsize=9)
    for i in range(10):
        for j in range(10):
            if cmn[i, j] >= 4:
                ax.text(j, i, f"{cmn[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if cmn[i, j] > 55 else "#1a1a1a")
    ax.set_title(m, fontsize=12, color=COLORS[m], fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10.5)
axes[0].set_ylabel("True", fontsize=10.5)
fig.tight_layout()
fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label="Row-normalised (%)")
fig.suptitle("Row-Normalised Confusion Matrices (%)", fontsize=13,
             color=NAVY, fontweight="bold", y=1.02)
fig.savefig(OUT / "fig7_cm_normalized.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 8: example images (print width 18.6 cm) ----------
EXAMPLES = [
    ("punching_hole", "Punching\nhole", r"data\processed\punching_hole\img_02_425613600_00582.jpg"),
    ("welding_line", "Welding\nline", r"data\processed\welding_line\img_02_3402617200_00001.jpg"),
    ("water_spot", "Water\nspot", r"data\processed\water_spot\img_01_425006200_00175.jpg"),
    ("oil_spot", "Oil spot", r"data\processed\oil_spot\img_02_425507000_00871.jpg"),
    ("inclusion", "Inclusion", r"data\processed\inclusion\img_02_425502300_00338.jpg"),
    ("rolled_pit", "Rolled\npit", r"data\processed\rolled_pit\img_02_4402329000_00001.jpg"),
]
fig, axes = plt.subplots(1, 6, figsize=(18.6 * CM2IN, 4.6 * CM2IN))
for ax, (_, name, rel) in zip(axes, EXAMPLES):
    img = Image.open(PROJ / rel)
    ax.imshow(img, cmap="gray")
    ax.set_title(name, fontsize=11.5, color=NAVY, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999999")
fig.tight_layout()
fig.savefig(OUT / "fig8_examples.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("figures written to", OUT)
for p in sorted(OUT.glob("*.png")):
    print(p.name, p.stat().st_size)
