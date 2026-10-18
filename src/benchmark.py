"""
benchmark.py — BlueBEAR 适配版
====================
改动说明（相对原始 Windows 版本）：
  1. 路径全部改为 BlueBEAR RDS 项目盘（PROJECT_ROOT / 模型权重路径）。
  2. 仅默认启用已合并完成的 InternVL3-8B（final_model_internvl3）。
     其余模型保留注册表结构，merged_path 留空/不存在即自动跳过，方便后续补上。
  3. 推理设备改为当前节点的 GPU（cuda），并逐个模型评测以释放显存。
  4. 保留原架构：每模型独立 loader / inferer，避免 chat 接口差异。
  5. 主指标仍为 Macro-F1（消除长尾偏见，对齐 2025 NEU 横评）。
"""
import os
import json
import time
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无显示器环境必须，生成 png 即可
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
PROJECT_ROOT  = Path("/rds/projects/w/wangyz-atari/metal_defect_vlm")
TEST_JSON     = PROJECT_ROOT / "data" / "llamafactory" / "metal_defect_test.json"
EP_PROMPT_FILE = PROJECT_ROOT / "data" / "ep_prompt.txt"
RESULTS_DIR    = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_CLASSES = [
    "punching_hole", "welding_line", "crescent_gap",
    "water_spot", "oil_spot", "silk_spot",
    "inclusion", "rolled_pit", "crease", "waist_folding",
]

# ★ 模型注册表：merged_path 存在才评测；不存在自动跳过
MODELS = [
    {
        "name": "internvl3_8b",
        "display": "InternVL3-8B",
        "merged_path": PROJECT_ROOT / "final_model_internvl3",
    },
    # 待补充：把 merged_path 指向对应模型目录即可启用
    # {
    #     "name": "qwen2_5_vl_7b",
    #     "display": "Qwen2.5-VL-7B",
    #     "merged_path": PROJECT_ROOT / "final_model_qwen2_5_vl_7b",
    # },
    # {
    #     "name": "llava_onevision_7b",
    #     "display": "LLaVA-OneVision-7B",
    #     "merged_path": PROJECT_ROOT / "final_model_llava_onevision_7b",
    # },
    # {
    #     "name": "gemma3_4b",
    #     "display": "Gemma3-4B",
    #     "merged_path": PROJECT_ROOT / "final_model_gemma3_4b",
    # },
]

# ─────────────────────────────────────────────
# 提示词
# ─────────────────────────────────────────────
def load_ep_prompt():
    if EP_PROMPT_FILE.exists():
        return EP_PROMPT_FILE.read_text(encoding="utf-8")
    return "请判断图像中的金属表面缺陷类型。"

BASIC_PROMPT = """请仔细观察这张金属表面图像，判断它属于以下哪种缺陷类型：
punching_hole、welding_line、crescent_gap、water_spot、oil_spot、
silk_spot、inclusion、rolled_pit、crease、waist_folding。
只输出缺陷类型的英文名称。"""

# ─────────────────────────────────────────────
# 模型加载（懒加载，逐个跑避免显存冲突）
# ─────────────────────────────────────────────
def _clean_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_internvl(model_path):
    _clean_gpu()
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    ).eval()
    return model, tokenizer

LOADERS = {
    "internvl3_8b": load_internvl,
}

# ─────────────────────────────────────────────
# 推理函数
# ─────────────────────────────────────────────
def infer_internvl(model, tokenizer, img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        resp = model.chat(
            tokenizer, image, prompt,
            generation_config={"max_new_tokens": 128, "do_sample": False}
        )
    return resp.strip()

INFERERS = {
    "internvl3_8b": infer_internvl,
}

# ─────────────────────────────────────────────
# 解析预测
# ─────────────────────────────────────────────
def parse_defect_type(text):
    if not text:
        return "unknown"
    t = text.lower()
    for cls in DEFECT_CLASSES:
        if cls in t:
            return cls
    cn_map = {
        "冲孔": "punching_hole", "焊线": "welding_line", "月牙缺口": "crescent_gap",
        "水斑": "water_spot", "油斑": "oil_spot", "丝状斑": "silk_spot",
        "夹杂物": "inclusion", "轧制麻点": "rolled_pit", "折痕": "crease", "腰折": "waist_folding",
    }
    for cn, en in cn_map.items():
        if cn in text:
            return en
    return "unknown"

# ─────────────────────────────────────────────
# 评测单个模型
# ─────────────────────────────────────────────
def evaluate_model(mcfg, test_data, prompt, prompt_type):
    name, display = mcfg["name"], mcfg["display"]
    merged = mcfg["merged_path"]
    if not merged.exists():
        print(f"  [跳过] {display}: 未找到 {merged}")
        return None

    print(f"\n{'='*55}\n  评测: {display}  [prompt={prompt_type}]\n{'='*55}")

    model, proc = LOADERS[name](str(merged))
    inferer = INFERERS[name]

    y_true, y_pred = [], []
    latencies, preds_detail = [], []

    for item in tqdm(test_data, desc=display):
        img_rel = item["images"][0]
        img_path = PROJECT_ROOT / img_rel if not os.path.isabs(img_rel) else Path(img_rel)
        assistant_text = item["messages"][1]["content"]
        true_label = parse_defect_type(assistant_text)

        t0 = time.time()
        try:
            raw = inferer(model, proc, str(img_path), prompt)
        except Exception as e:
            print(f"\n  [err] {img_path.name}: {e}")
            raw = ""
        lat = time.time() - t0

        pred = parse_defect_type(raw)
        y_true.append(true_label)
        y_pred.append(pred)
        latencies.append(lat)
        preds_detail.append({
            "image": str(img_path), "true": true_label,
            "pred": pred, "raw": raw, "lat_s": round(lat, 3)
        })

    del model, proc
    _clean_gpu()

    acc  = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, labels=DEFECT_CLASSES, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, labels=DEFECT_CLASSES, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, labels=DEFECT_CLASSES, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=DEFECT_CLASSES, average="weighted", zero_division=0)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    cls_report = classification_report(y_true, y_pred, labels=DEFECT_CLASSES, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=DEFECT_CLASSES)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=DEFECT_CLASSES, yticklabels=DEFECT_CLASSES, ax=ax)
    ax.set_title(f"Confusion Matrix — {display} ({prompt_type})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"cm_{name}_{prompt_type}.png"
    plt.savefig(cm_path, dpi=150); plt.close()

    print(f"\n  Acc={acc:.4f}  F1_macro={f1_m:.4f}★  F1_wgt={f1_w:.4f}  Lat={avg_lat:.3f}s")

    return {
        "model": display, "prompt_type": prompt_type,
        "accuracy": round(acc, 4), "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "avg_latency_s": round(avg_lat, 3),
        "total_samples": len(y_true),
        "per_class": {c: cls_report.get(c, {}) for c in DEFECT_CLASSES},
        "confusion_matrix": cm.tolist(),
        "predictions": preds_detail,
    }

# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def log(msg): print(f"\033[92m[✓]\033[0m {msg}")

def main():
    if not TEST_JSON.exists():
        print(f"[错误] 未找到 {TEST_JSON}，请先准备测试集 JSON。")
        return
    with open(TEST_JSON, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 张")

    ep_prompt = load_ep_prompt()
    all_results = []

    enabled = [m for m in MODELS if m["merged_path"].exists()]
    if not enabled:
        print("\n[错误] 没有任何模型的 merged 权重存在，请先合并模型！")
        print("已检查的路径：")
        for m in MODELS:
            print(f"  {m['display']}: {m['merged_path']}")
        return

    print("\n" + "="*60); print("  Part 1: Enhanced Prompting 评测"); print("="*60)
    for m in enabled:
        r = evaluate_model(m, test_data, ep_prompt, "EP")
        if r: all_results.append(r)

    jpath = RESULTS_DIR / "benchmark_results.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"详细结果 → {jpath}")

    print("\n" + "="*75 + "\n  对比汇总（按 Macro-F1 排序）\n" + "="*75)
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model"], "Prompt": r["prompt_type"],
            "Accuracy": r["accuracy"], "F1★": r["f1_macro"],
            "F1(wgt)": r["f1_weighted"], "Prec": r["precision"],
            "Rec": r["recall"], "Lat(s)": r["avg_latency_s"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Prompt", "F1★"], ascending=[True, False])
    print(df.to_string(index=False))
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    log(f"CSV → {csv_path}")

    if all_results:
        ep_results = [r for r in all_results if r["prompt_type"] == "EP"]
        if ep_results:
            best = max(ep_results, key=lambda x: x["f1_macro"])
            print(f"\n🏆 最佳 (EP, Macro-F1): {best['model']}  F1={best['f1_macro']:.4f}")
        cls_avg = {}
        for cls in DEFECT_CLASSES:
            f1s = [r["per_class"].get(cls, {}).get("f1-score", 0) for r in ep_results]
            cls_avg[cls] = sum(f1s)/len(f1s) if f1s else 0
        if cls_avg:
            hardest = min(cls_avg, key=cls_avg.get)
            print(f"🎯 最难识别缺陷: {hardest} (平均 F1={cls_avg[hardest]:.4f})")

if __name__ == "__main__":
    main()
