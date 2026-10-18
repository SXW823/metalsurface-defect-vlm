"""
benchmark.py — 版本1
功能：
  1. 加载 4 个微调后的模型（合并权重）
  2. 在 test set 上推理
  3. 计算 Accuracy / F1 / Per-class 指标
  4. 输出对比 CSV + JSON
  5. 额外评测 EP vs 普通提示词的差异（复现论文现象）

对齐论文指标：
  - 2025 NEU 横评：Accuracy + F1
  - 2026 LNG 罐缺陷：mAP@50 + F1 + BLEU

benchmark.py — 版本2（GC10-DET 10 类完整版 + 论文对齐）
保留原架构：每个模型独立 loader / inferer，避免 chat 接口差异
修复：
  - DEFECT_CLASSES 补回 inclusion（10 类）
  - parse_defect_type 加 inclusion 中文映射
  - 主指标改为 Macro-F1（消除长尾偏见，对齐 2025 NEU 横评）
  - 保留 EP vs Basic 消融（复现 Qwen 受益 / LLaVA 退化现象）
  - 输出 CSV + JSON + 混淆矩阵图

论文对齐：
  2025 NEU 横评 → Accuracy + Macro-F1 + EP 消融
  2026 LNG 罐缺陷 → F1 + Per-class + 混淆矩阵
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
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.resolve()
TEST_JSON     = PROJECT_ROOT / "data" / "llamafactory" / "metal_defect_test.json"
EP_PROMPT_FILE = PROJECT_ROOT / "data" / "ep_prompt.txt"
OUTPUT_ROOT    = PROJECT_ROOT / "output"
RESULTS_DIR    = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ★ 关键修复：10 类完整（含 inclusion）
DEFECT_CLASSES = [
    "punching_hole", "welding_line", "crescent_gap",
    "water_spot", "oil_spot", "silk_spot",
    "inclusion", "rolled_pit", "crease", "waist_folding",
]

# 模型注册表（训练后每个模型 merge_and_unload 保存到此路径）
MODELS = [
    #{
    #    "name": "qwen2_5_vl_7b",
    #    "display": "Qwen2.5-VL-7B",
    #    "merged_path": Path("D:/models/qwen2_5vl_7b_merged"),
    #},
    {
        "name": "internvl3_8b",
        "display": "InternVL3-8B",
        "merged_path": Path("E:/models/internvl3_8b_merged"),
    },
    {
        "name": "llava_onevision_7b",
        "display": "LLaVA-OneVision-7B",
        "merged_path": Path("E:/models/llava_onevision_7b_merged"),
    },
    {
        "name": "gemma3_4b",
        "display": "Gemma3-4B",
        "merged_path": Path("E:/models/gemma3_4b_merged"),
    },
]

# ─────────────────────────────────────────────
# 提示词
# ─────────────────────────────────────────────
def load_ep_prompt():
    if EP_PROMPT_FILE.exists():
        return EP_PROMPT_FILE.read_text(encoding="utf-8")
    return "请判断图像中的金属表面缺陷类型。"

# ★ 10 类 Basic 提示词（用于消融对比）
BASIC_PROMPT = """请仔细观察这张金属表面图像，判断它属于以下哪种缺陷类型：
punching_hole、welding_line、crescent_gap、water_spot、oil_spot、
silk_spot、inclusion、rolled_pit、crease、waist_folding。
只输出缺陷类型的英文名称。"""

# ─────────────────────────────────────────────
# 模型加载（懒加载，逐个跑避免显存冲突）
# ─────────────────────────────────────────────
def load_qwen_model(model_path: str):
    import gc
    gc.collect()
    torch.cuda.empty_cache()  # ← 加这两行，确保加载前显存是干净的
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    )
    model.eval()
    return model, processor

def load_internvl(model_path):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    ).eval()
    return model, tokenizer

def load_llava(model_path):
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    return model, processor

def load_gemma(model_path):
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    ).eval()
    return model, processor

LOADERS = {
    "qwen2_5_vl_7b":      load_qwen_model,
    "internvl3_8b":       load_internvl,
    "llava_onevision_7b": load_llava,
    "gemma3_4b":          load_gemma,
}

# ─────────────────────────────────────────────
# 推理函数（统一接口）
# ─────────────────────────────────────────────
def infer_qwen(model, processor, img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

def infer_internvl(model, tokenizer, img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        resp = model.chat(tokenizer, image, prompt,
                          generation_config={"max_new_tokens": 128, "do_sample": False})
    return resp.strip()

def infer_llava(model, processor, img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

def infer_gemma(model, processor, img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

INFERERS = {
    "qwen2_5_vl_7b":      infer_qwen,
    "internvl3_8b":       infer_internvl,
    "llava_onevision_7b": infer_llava,
    "gemma3_4b":          infer_gemma,
}

# ─────────────────────────────────────────────
# 解析预测
# ─────────────────────────────────────────────
def parse_defect_type(text):
    """从模型输出解析缺陷类型，支持中英文 + 兜底"""
    if not text: return "unknown"
    t = text.lower()
    for cls in DEFECT_CLASSES:
        if cls in t:
            return cls
    # 中文兜底
    cn_map = {
        "冲孔": "punching_hole", "焊线": "welding_line", "月牙缺口": "crescent_gap",
        "水斑": "water_spot", "油斑": "oil_spot", "丝状斑": "silk_spot",
        "夹杂物": "inclusion", "轧制麻点": "rolled_pit", "折痕": "crease", "腰折": "waist_folding",
    }
    for cn, en in cn_map.items():
        if cn in text: return en
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
        img_path = PROJECT_ROOT / item["images"][0]  # rel_path → 绝对路径
        # 真实标签从 assistant 回复 JSON 解析
        assistant_text = item["messages"][1]["content"]
        true_label = parse_defect_type(assistant_text)

        t0 = time.time()
        try:
            raw = inferer(model, proc, str(img_path), prompt)
        except Exception as e:
            print(f"\n  [err] {img_path.name}: {e}"); raw = ""
        lat = time.time() - t0

        pred = parse_defect_type(raw)
        y_true.append(true_label)
        y_pred.append(pred)
        latencies.append(lat)
        preds_detail.append({"image": str(img_path), "true": true_label,
                             "pred": pred, "raw": raw, "lat_s": round(lat, 3)})

    del model, proc
    torch.cuda.empty_cache()

    # ★ 指标：Macro-F1 为主（消除长尾偏见，对齐 2025 NEU 论文）
    acc   = accuracy_score(y_true, y_pred)
    f1_m  = f1_score(y_true, y_pred, labels=DEFECT_CLASSES, average="macro", zero_division=0)
    f1_w  = f1_score(y_true, y_pred, labels=DEFECT_CLASSES, average="weighted", zero_division=0)
    prec  = precision_score(y_true, y_pred, labels=DEFECT_CLASSES, average="weighted", zero_division=0)
    rec   = recall_score(y_true, y_pred, labels=DEFECT_CLASSES, average="weighted", zero_division=0)
    avg_lat = sum(latencies) / len(latencies)

    cls_report = classification_report(y_true, y_pred, labels=DEFECT_CLASSES, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=DEFECT_CLASSES)

    # 保存混淆矩阵图
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
        "accuracy": round(acc,4), "f1_macro": round(f1_m,4),
        "f1_weighted": round(f1_w,4), "precision": round(prec,4),
        "recall": round(rec,4), "avg_latency_s": round(avg_lat,3),
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
        print(f"[错误] 未找到 {TEST_JSON}，请先运行 prepare_data.py")
        return
    with open(TEST_JSON, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 张")

    ep_prompt = load_ep_prompt()
    all_results = []
    if not any(m["merged_path"].exists() for m in MODELS):
        print("\n[错误] 没有任何模型的 merged 权重存在，请先训练并合并模型！")
        print("已检查的路径：")
        for m in MODELS:
            print(f"  {m['display']}: {m['merged_path']}")
        return

    # Part 1: EP 提示词（全 4 个模型）
    print("\n" + "="*60); print("  Part 1: Enhanced Prompting 评测"); print("="*60)
    for m in MODELS:
        r = evaluate_model(m, test_data, ep_prompt, "EP")
        if r: all_results.append(r)

    # Part 2: Basic 消融（仅 Qwen + LLaVA，复现论文现象）
    #ablation_names = {"qwen2_5_vl_7b", "llava_onevision_7b"}
    #abl_models = [m for m in MODELS if m["name"] in ablation_names]
    #print("\n" + "="*60); print("  Part 2: Basic Prompt 消融"); print("="*60)
    #for m in abl_models:
     #   r = evaluate_model(m, test_data, BASIC_PROMPT, "Basic")
     #   if r: all_results.append(r)

    # 保存 JSON
    jpath = RESULTS_DIR / "benchmark_results.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"详细结果 → {jpath}")

    # 汇总表
    print("\n" + "="*75 + "\n  对比汇总（按 Macro-F1 排序）\n" + "="*75)
    rows = []
    for r in all_results:
        rows.append({"Model": r["model"], "Prompt": r["prompt_type"],
                     "Accuracy": r["accuracy"], "F1★": r["f1_macro"],
                     "F1(wgt)": r["f1_weighted"], "Prec": r["precision"],
                     "Rec": r["recall"], "Lat(s)": r["avg_latency_s"]})
    df = pd.DataFrame(rows)
    df = df.sort_values(["Prompt", "F1★"], ascending=[True, False])
    print(df.to_string(index=False))
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    log(f"CSV → {csv_path}")

    # EP 消融分析
    ep_r  = {r["model"]: r for r in all_results if r["prompt_type"] == "EP"}
    ba_r  = {r["model"]: r for r in all_results if r["prompt_type"] == "Basic"}
    print("\n" + "="*60 + "\n  EP vs Basic 消融分析\n" + "="*60)
    for mn in ba_r:
        if mn in ep_r:
            d = ep_r[mn]["accuracy"] - ba_r[mn]["accuracy"]
            arrow = "↑" if d > 0 else "↓"
            print(f"  {mn:25s}  EP={ep_r[mn]['accuracy']:.4f}  Basic={ba_r[mn]['accuracy']:.4f}  Δ={arrow}{abs(d):.4f}")

    # ★ 关键发现
    if rows:
        best = max([r for r in all_results if r["prompt_type"]=="EP"], key=lambda x: x["f1_macro"])
        print(f"\n🏆 最佳 (EP, Macro-F1): {best['model']}  F1={best['f1_macro']:.4f}")

    # 找出最难的缺陷类别（跨模型平均 F1 最低）
    cls_avg = {}
    ep_results = [r for r in all_results if r["prompt_type"] == "EP"]
    for cls in DEFECT_CLASSES:
        f1s = [r["per_class"].get(cls, {}).get("f1-score", 0) for r in ep_results]
        cls_avg[cls] = sum(f1s)/len(f1s) if f1s else 0
    hardest = min(cls_avg, key=cls_avg.get)
    print(f"🎯 最难识别缺陷: {hardest} (平均 F1={cls_avg[hardest]:.4f})")

if __name__ == "__main__":
    main()