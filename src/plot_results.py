"""
plot_results.py — 论文级可视化
功能：
  1. 读取 benchmark_results.json
  2. 生成对比柱状图（Accuracy / F1）
  3. 生成混淆矩阵热力图（每个模型）
  4. 生成 EP vs Basic 消融对比图
  5. 生成雷达图（多维度对比）
  6. 导出 LaTeX 表格（可直接粘贴到论文）

依赖：matplotlib, seaborn, numpy, pandas
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────
# 全局设置
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_FILE = PROJECT_ROOT / "results" / "benchmark_results.json"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_CLASSES = [
    "punching_hole", "welding_line", "crescent_gap",
    "water_spot", "oil_spot", "silk_spot", "inclusion",
    "rolled_pit", "crease", "waist_folding"
]

# 论文级配色（ColorBrewer Set2）
COLORS = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"]

# 英文字体
rcParams['figure.dpi'] = 200
rcParams['savefig.bbox'] = 'tight'

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def load_results():
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return json.load(f)

def get_model_short_name(full_name: str) -> str:
    mapping = {
        "Qwen2.5-VL-7B":       "Qwen2.5-VL",
        "InternVL3-8B":         "InternVL3",
        "LLaVA-OneVision-7B":   "LLaVA-OV",
        "Gemma3-4B":            "Gemma3",
    }
    return mapping.get(full_name, full_name)

def get_class_label(name: str) -> str:
    """将下划线命名转为论文友好的英文短标签"""
    labels = {
        "punching_hole": "Punching hole",
        "welding_line":  "Welding line",
        "crescent_gap":  "Crescent gap",
        "water_spot":    "Water spot",
        "oil_spot":      "Oil spot",
        "silk_spot":     "Silk spot",
        "inclusion":     "Inclusion",
        "rolled_pit":    "Rolled pit",
        "crease":        "Crease",
        "waist_folding":"Waist folding",
    }
    return labels.get(name, name.replace("_", " ").title())

# ─────────────────────────────────────────────
# 图1: Accuracy / F1 分组柱状图
# ─────────────────────────────────────────────
def plot_bar_comparison(results: list):
    """EP 提示词下各模型 Accuracy + F1(macro) 对比"""
    ep_results = [r for r in results if r["prompt_type"] == "EP"]
    ep_results.sort(key=lambda x: x["accuracy"], reverse=True)

    models = [get_model_short_name(r["model"]) for r in ep_results]
    accs   = [r["accuracy"] * 100 for r in ep_results]
    f1s    = [r["f1_macro"] * 100 for r in ep_results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color=COLORS[0], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 (macro)', color=COLORS[1], edgecolor='white', linewidth=0.5)

    # 数值标注
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('VLM Performance on Metal Surface Defect Detection (700 images, 9 classes)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = OUTPUT_DIR / "bar_accuracy_f1.pdf"
    fig.savefig(out)
    fig.savefig(OUTPUT_DIR / "bar_accuracy_f1.png")
    plt.close(fig)
    print(f"  [✓] {out}")

# ─────────────────────────────────────────────
# 图2: 混淆矩阵热力图
# ─────────────────────────────────────────────
def plot_confusion_matrix(results: list):
    """为每个 EP 模型画混淆矩阵"""
    ep_results = [r for r in results if r["prompt_type"] == "EP"]

    for r in ep_results:
        cm = np.array(r["confusion_matrix"])
        model_name = get_model_short_name(r["model"])

        # 归一化为百分比
        with np.errstate(all='ignore'):
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        cm_pct = np.nan_to_num(cm_pct)

        fig, ax = plt.subplots(figsize=(9, 7.5))
        labels = [get_class_label(c) for c in DEFECT_CLASSES]

        sns.heatmap(cm_pct, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=labels, yticklabels=labels,
                    vmin=0, vmax=100, linewidths=0.5,
                    cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                    ax=ax, annot_kws={'size': 8})

        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('Ground Truth', fontsize=11)
        ax.set_title(f'Confusion Matrix — {model_name} (Enhanced Prompt)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        fig.tight_layout()

        out = OUTPUT_DIR / f"confusion_{r['model'].replace('-', '_').replace('.', '')}.pdf"
        fig.savefig(out)
        fig.savefig(out.with_suffix('.png'))
        plt.close(fig)
        print(f"  [✓] {out}")

# ─────────────────────────────────────────────
# 图3: EP vs Basic 消融图
# ─────────────────────────────────────────────
def plot_ablation(results: list):
    """Qwen 和 LLaVA 的 EP vs Basic 对比"""
    ep = {r["model"]: r for r in results if r["prompt_type"] == "EP"}
    ba = {r["model"]: r for r in results if r["prompt_type"] == "Basic"}

    common_models = sorted(set(ep.keys()) & set(ba.keys()))

    if not common_models:
        print("  [跳过] 无消融数据")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(common_models))
    width = 0.3

    ep_accs = [ep[m]["accuracy"]*100 for m in common_models]
    ba_accs = [ba[m]["accuracy"]*100 for m in common_models]
    ep_f1s  = [ep[m]["f1_macro"]*100 for m in common_models]
    ba_f1s  = [ba[m]["f1_macro"]*100 for m in common_models]

    # Accuracy 子图
    ax.bar(x - width/2, ep_accs, width, label='EP Accuracy', color=COLORS[0])
    ax.bar(x + width/2, ba_accs, width, label='Basic Accuracy', color=COLORS[1])

    # 标注 Δ
    for i, (e, b) in enumerate(zip(ep_accs, ba_accs)):
        delta = e - b
        color = 'green' if delta > 0 else 'red'
        ax.annotate(f'Δ{delta:+.1f}', xy=(i, max(e, b) + 1),
                    ha='center', fontsize=9, color=color, fontweight='bold')

    short = [get_model_short_name(m) for m in common_models]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Enhanced Prompting (EP) Ablation Study', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "ablation_ep_vs_basic.pdf"
    fig.savefig(out)
    fig.savefig(OUTPUT_DIR / "ablation_ep_vs_basic.png")
    plt.close(fig)
    print(f"  [✓] {out}")

# ─────────────────────────────────────────────
# 图4: 雷达图（多维度对比）
# ─────────────────────────────────────────────
def plot_radar(results: list):
    """Accuracy / F1 / Precision / Recall / Latency(倒数) 雷达图"""
    ep_results = [r for r in results if r["prompt_type"] == "EP"]
    ep_results.sort(key=lambda x: x["accuracy"], reverse=True)

    metrics = ['Accuracy', 'F1 (macro)', 'Precision', 'Recall', '1/Latency']
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, r in enumerate(ep_results):
        vals = [
            r["accuracy"] * 100,
            r["f1_macro"] * 100,
            r["precision"] * 100,
            r["recall"] * 100,
            min(100, 1000 / max(r["avg_latency_s"], 0.01)),  # 归一化延迟
        ]
        vals += vals[:1]
        name = get_model_short_name(r["model"])
        ax.plot(angles, vals, 'o-', linewidth=2, label=name, color=COLORS[i % len(COLORS)])
        ax.fill(angles, vals, alpha=0.08, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('Multi-Dimensional VLM Comparison', fontsize=12, fontweight='bold', pad=20)

    fig.tight_layout()
    out = OUTPUT_DIR / "radar_comparison.pdf"
    fig.savefig(out)
    fig.savefig(OUTPUT_DIR / "radar_comparison.png")
    plt.close(fig)
    print(f"  [✓] {out}")

# ─────────────────────────────────────────────
# 图5: 各类别 F1 热力图
# ─────────────────────────────────────────────
def plot_per_class_heatmap(results: list):
    """每个模型在 10 个类别上的 F1 热力图"""
    ep_results = [r for r in results if r["prompt_type"] == "EP"]
    ep_results.sort(key=lambda x: x["accuracy"], reverse=True)

    models = [get_model_short_name(r["model"]) for r in ep_results]
    data = np.zeros((len(models), len(DEFECT_CLASSES)))

    for i, r in enumerate(ep_results):
        for j, cls in enumerate(DEFECT_CLASSES):
            cls_data = r["per_class"].get(cls, {})
            f1 = cls_data.get("f1-score", 0)
            data[i, j] = f1 * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    labels_x = [get_class_label(c) for c in DEFECT_CLASSES]
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=labels_x, yticklabels=models,
                vmin=0, vmax=100, linewidths=0.5,
                cbar_kws={'label': 'F1 Score (%)', 'shrink': 0.8},
                ax=ax, annot_kws={'size': 8})
    ax.set_title('Per-Class F1 Score (%) — EP Prompt', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()

    out = OUTPUT_DIR / "per_class_f1_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(OUTPUT_DIR / "per_class_f1_heatmap.png")
    plt.close(fig)
    print(f"  [✓] {out}")

# ─────────────────────────────────────────────
# LaTeX 表格导出
# ─────────────────────────────────────────────
def export_latex(results: list):
    """导出可直接粘贴到论文的 LaTeX 表格"""
    ep_results = [r for r in results if r["prompt_type"] == "EP"]
    ep_results.sort(key=lambda x: x["accuracy"], reverse=True)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{VLM Performance Comparison on Metal Surface Defect Detection (240 images, 10 classes, QLoRA fine-tuning)}")
    lines.append("  \\label{tab:vlm_comparison}")
    lines.append("  \\begin{tabular}{lccccc}")
    lines.append("    \\toprule")
    lines.append("    Model & Accuracy & F1 (macro) & Precision & Recall & Latency (s/img) \\\\")
    lines.append("    \\midrule")

    for r in ep_results:
        name = get_model_short_name(r["model"])
        lines.append(
            f"    {name} & {r['accuracy']:.4f} & {r['f1_macro']:.4f} & "
            f"{r['precision']:.4f} & {r['recall']:.4f} & {r['avg_latency_s']:.3f} \\\\"
        )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    out = OUTPUT_DIR / "table_latex.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [✓] {out}")

# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def main():
    if not RESULTS_FILE.exists():
        print(f"[错误] 未找到 {RESULTS_FILE}")
        print("请先运行 benchmark.py")
        return

    results = load_results()
    print(f"加载 {len(results)} 条评测结果\n")

    print("[1/5] 生成 Accuracy/F1 柱状图...")
    plot_bar_comparison(results)

    print("[2/5] 生成混淆矩阵热力图...")
    plot_confusion_matrix(results)

    print("[3/5] 生成 EP vs Basic 消融图...")
    plot_ablation(results)

    print("[4/5] 生成雷达图...")
    plot_radar(results)

    print("[5/5] 生成各类别 F1 热力图...")
    plot_per_class_heatmap(results)

    print("\n导出 LaTeX 表格...")
    export_latex(results)

    print(f"\n{'='*60}")
    print(f"  所有图表已保存到: {OUTPUT_DIR}/")
    print(f"{'='*60}")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
