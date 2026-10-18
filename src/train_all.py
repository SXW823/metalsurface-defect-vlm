"""
train_all.py — 统一训练入口
功能：依次对 4 个 VLM 执行 QLoRA 微调
模型：
  1. Qwen2.5-VL-7B-Instruct
  2. InternVL3-8B
  3. LLaVA-OneVision-7B
  4. Gemma3-4B

所有模型使用相同超参 + 相同 EP 数据集，保证公平对比。
5070Ti 16GB 优化：4-bit NF4 + Gradient Checkpointing + Flash Attention 2
"""

import torch       # 清显存
import gc          # 同上
import os
import sys
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.resolve()
LF_ROOT = Path(r"C:\Users\SHIRUI\LLaMA-Factory\Llama-Factory")  # LLaMA-Factory 安装路径
DATA_DIR       = PROJECT_ROOT / "data" / "llamafactory"
OUTPUT_ROOT = Path("E:/models/loras")
LOG_ROOT       = PROJECT_ROOT / "logs"

NUM_EPOCHS     = 3
BATCH_SIZE     = 1          # 5070Ti 16GB 安全值
GRAD_ACCUM     = 8          # 有效 batch = 8
LR             = 1e-4
LORA_RANK      = 16
LORA_ALPHA     = 32
MAX_PIXELS     = 401408      # ≈ 632×632，控制显存
MAX_LENGTH     = 1024
SEED           = 42

# ─────────────────────────────────────────────
# 模型注册表
# ─────────────────────────────────────────────
MODELS = [#Qwen already trained
    #{
        #"name": "qwen2_5_vl_7b",
        #"display": "Qwen2.5-VL-7B",
        #"model_path": "C:/models/Qwen2.5-VL-7B-Instruct",
        #"lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        #"template": "qwen2_vl",
    #},
    {
        "name": "internvl3_8b",
        "display": "InternVL3-8B",
        "model_path": "E:/models/InternVL3-8B-hf",
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "template": "intern_vl",
    },
    #{
    #    "name": "llava_onevision_7b",
    #    "display": "LLaVA-OneVision-7B",
    #    "model_path": "E:/models/LLaVA-OneVision-7B",
    #    "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    #    "template": "llava_next",
    #},
    #{
    #    "name": "gemma3_4b",
    #    "display": "Gemma3-4B",
    #    "model_path": "E:/models/Gemma3-4B/models/google--gemma-3-4b-it/snapshots/master",
    #    "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    #    "template": "gemma3",
    #},
]

# ─────────────────────────────────────────────
# 生成 LLaMA-Factory YAML 配置
# ─────────────────────────────────────────────
def build_yaml(model_cfg: dict) -> Path:
    """为每个模型生成独立的 YAML 配置文件"""
    out_dir = OUTPUT_ROOT / model_cfg["name"]
    log_dir = LOG_ROOT / model_cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    # ===== 先判断是不是 InternVL3 =====
    is_internvl = model_cfg["name"] == "internvl3_8b"

    cfg = {
        # 基础
        "stage": "sft",
        "do_train": True,
        "do_eval": True,
        "model_name_or_path": model_cfg["model_path"],
        "template": model_cfg["template"],
        "finetuning_type": "lora",

        # LoRA
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": 0.05,
    "lora_target": model_cfg["lora_target"],

    "low_cpu_mem_usage": is_internvl,      # 顺序构建 param，避免一次性在 CPU 拼完整副本
    "preprocessing_num_workers": 0 if is_internvl else 4,   # 注意：这里改成 0，不是 2
    # ★ InternVL3 第一档 fallback：先保留 4bit，但降像素 + 冻结视觉塔
    # 如果这档还崩，就把下面 3 行 4bit 注释掉、改用 bf16（第二档见下文）
    #"quantization_bit": 4 if is_internvl else 4,
    #"quantization_type": "nf4",
    #"double_quantization": True,

    # ★ InternVL3 专用稳定性参数（官方/社区 SFT 实战里都有这些）
    "freeze_vision_tower": is_internvl,            # InternVL ViT 24层、448 native，冻结很常规[5](@ref)
    "freeze_multi_modal_projector": is_internvl,    # MLP1 也冻，只训 LLM 的 LoRA[5,8](@ref)

    # 数据
    "dataset_dir": str(DATA_DIR),
    "dataset": "metal_defect_train",
    "val_size": 0.1,
    "cutoff_len": MAX_LENGTH,
    # ★ InternVL3 原生视觉是 448×448 + max_num tiles；你原来 MAX_PIXELS 若是 896k/1M 都偏大
    "image_max_pixels": 589824 if is_internvl else MAX_PIXELS,   # 589824=768×768，稳妥；再崩就改 262144[8](@ref)
    "image_min_pixels": 784,
    "preprocessing_num_workers": 2 if is_internvl else 4,        # Windows 多 worker + bnb 易踩内存越界[4](@ref)
    "media_dir": str(PROJECT_ROOT),

        # 输出
        "output_dir": str(out_dir),
        "logging_dir": str(log_dir),

        # 保存（steps 模式，确保 checkpoint 一定落盘）
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 3,
        "save_only_model": False,

        # 训练参数
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LR,
        "num_train_epochs": NUM_EPOCHS,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 20,
        "bf16": True,
        "gradient_checkpointing": True,

        # 评测
        "eval_strategy": "steps",
        "eval_steps": 200,                              # ★ 改成 200，别太频繁
        "report_to": "none",

        # 其他
        "seed": SEED,
        "trust_remote_code": True,
        "ddp_timeout": 180000000,
    }

    yaml_path = PROJECT_ROOT / f"configs/{model_cfg['name']}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return yaml_path


# ─────────────────────────────────────────────
# 运行训练
# ─────────────────────────────────────────────
def train_model(model_cfg: dict) -> bool:
    """用 LLaMA-Factory CLI 训练单个模型"""
    # ↓↓↓ 加这段：已合并则整段跳过 ↓↓↓
    merge_dir = Path("E:/models") / f"{model_cfg['name']}_merged"
    if merge_dir.exists() and any(merge_dir.glob("*.safetensors")):
        print(f"[跳过] {model_cfg['display']} 已存在: {merge_dir}")
        return True
    # ↑↑↑ 加这段 ↑↑↑

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ↑↑↑ 确保每次训练前显存干净 ↑↑↑
    yaml_path = build_yaml(model_cfg)
    display = model_cfg["display"]

    print(f"\n{'='*60}")
    print(f"  训练: {display}")
    print(f"  配置: {yaml_path}")
    print(f"  输出: E:/models/loras/{model_cfg['name']}/")
    print(f"{'='*60}")

    # 复制数据集到 LLaMA-Factory 的 data 目录
    lf_data = LF_ROOT / "data"
    lf_data.mkdir(parents=True, exist_ok=True)

    # 复制 dataset_info.json 和 metal_defect_*.json
    for f in DATA_DIR.glob("*.json"):
        shutil.copy2(f, lf_data / f.name)


    # 构建命令
    cmd = f'cd /d "{LF_ROOT}" && llamafactory-cli train "{yaml_path}"'

    # 记录日志
    log_file = LOG_ROOT / model_cfg["name"] / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] 开始训练...")

    # 实时打印 + 写日志
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        proc.wait()

    elapsed = datetime.now() - start
    if proc.returncode == 0:
        print(f"\n[✓] {display} 训练完成，耗时 {elapsed}")
        # 保存训练记录
        record = {
            "model": display,
            "start": start.isoformat(),
            "end": datetime.now().isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "status": "success",
            "output_dir": str(OUTPUT_ROOT / model_cfg["name"]),
        }
        with open(LOG_ROOT / model_cfg["name"] / "record.json", "w") as f:
            json.dump(record, f, indent=2)
        return True
    else:
        print(f"\n[✗] {display} 训练失败 (exit {proc.returncode})")
        print(f"    查看日志: {log_file}")
        return False


# ─────────────────────────────────────────────
# 合并 LoRA 权重（推理前需要）
# ─────────────────────────────────────────────
def merge_lora(model_cfg: dict) -> bool:
    """将 LoRA 权重合并到基座模型，输出可独立加载的模型"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    display = model_cfg["display"]
    lora_dir = OUTPUT_ROOT / model_cfg["name"]
    merge_dir = Path("E:/models") / f"{model_cfg['name']}_merged"

    # 跳过已合并的
    if merge_dir.exists() and any(merge_dir.glob("*.safetensors")):
        print(f"[跳过] {display} 已合并: {merge_dir}")
        return True

    # ★ 新增：如果根目录没有 adapter_config.json，找最新的 checkpoint
    actual_lora_dir = lora_dir
    if not (lora_dir / "adapter_config.json").exists():
        checkpoints = sorted(lora_dir.glob("checkpoint-*"))
        if checkpoints:
            actual_lora_dir = checkpoints[-1]  # 最新的
            print(f"[信息] 找到 {len(checkpoints)} 个 checkpoint，使用最新的: {actual_lora_dir.name}")
        else:
            print(f"[错误] 在 {lora_dir} 中既找不到 adapter_config.json 也找不到 checkpoint-*")
            return False

    print(f"\n[合并] {display} → {merge_dir}")

    export_yaml = PROJECT_ROOT / f"configs/export_{model_cfg['name']}.yaml"
    export_cfg = {
        "model_name_or_path": model_cfg["model_path"],
        "adapter_name_or_path": str(actual_lora_dir),   # ★ 用 actual_lora_dir
        "template": model_cfg["template"],
        "finetuning_type": "lora",
        "trust_remote_code": True,
        "export_dir": str(merge_dir),
        "export_size": 5,
        "export_device": "cpu",
    }
    with open(export_yaml, "w") as f:
        yaml.dump(export_cfg, f, default_flow_style=False, sort_keys=False)

    cmd = f'cd /d "{LF_ROOT}" && llamafactory-cli export "{export_yaml}"'

    log_file = LOG_ROOT / model_cfg["name"] / "merge.log"
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        proc.wait()

    if proc.returncode == 0:
        print(f"[✓] {display} 合并完成 → {merge_dir}")
        return True
    else:
        print(f"[✗] {display} 合并失败")
        return False


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("  金属缺陷 VLM 对比实验 — 训练阶段")
    print("="*60)
    print(f"  模型数量: {len(MODELS)}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  有效批大小: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  量化: 4-bit NF4 QLoRA")
    print(f"  显卡: RTX 5070Ti 16GB")
    print("="*60)

    results = {}
    for i, m in enumerate(MODELS, 1):
        print(f"\n>>> 进度: {i}/{len(MODELS)}  {m['display']}")
        ok = train_model(m)
        results[m["display"]] = "✓ success" if ok else "✗ failed"

        # 训练完立即合并权重
        if ok:
            merge_ok = merge_lora(m)
            results[f"{m['display']} (merge)"] = "✓ success" if merge_ok else "✗ failed"

    # 打印汇总
    print("\n" + "="*60)
    print("  训练汇总")
    print("="*60)
    for k, v in results.items():
        icon = "✓" if "success" in v else "✗"
        color = "\033[92m" if "success" in v else "\033[91m"
        print(f"  {color}{icon}\033[0m {k:30s} {v}")
    print("="*60)

    # 保存汇总
    with open(LOG_ROOT / "train_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n下一步: python benchmark.py")


if __name__ == "__main__":
    main()
