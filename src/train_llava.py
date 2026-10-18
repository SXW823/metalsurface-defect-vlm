"""
train_llava.py -- BlueBEAR 训练入口（仅 LLaVA-OneVision-7B QLoRA）
【最终修复版】双 Patch + 禁用 Torch Compile，彻底解决特征不匹配和编译检查报错
"""
import torch
# ==== 下面是修复 PyTorch 版本冲突的补丁 ====
if hasattr(torch, 'compilable_check'):
    # 保存原来的函数
    original_check = torch.compilable_check
    # 定义一个新的兼容函数，自动补上缺失的 msg 参数
    def patched_check(*args, **kwargs):
        if 'msg' not in kwargs:
            kwargs['msg'] = "compilable check failed"
        return original_check(*args, **kwargs)
    # 替换回去
    torch.compilable_check = patched_check
# ==========================================
import os
# 最优先禁用 torch compile，避免新版 transformers 的 torch_compilable_check 报错
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import gc
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# 核心修复：双 Patch 强制单 Tile，同时绕过编译检查
# ============================================================
try:
    # 导入 LLaVA-OV 的图像处理器和核心模型
    from transformers.models.llava_onevision.image_processing_llava_onevision import LlavaOneVisionImageProcessor
    from transformers.models.llava_onevision.modeling_llava_onevision import LlavaOnevisionForConditionalGeneration

    # ---------- Patch 1：强制图像处理器不切分（单 Tile） ----------
    _original_image_processor_preprocess = LlavaOneVisionImageProcessor.preprocess

    def _patched_image_processor_preprocess(self, images, **kwargs):
        """强制所有图像不切分，只返回单 Tile 结果，和预处理后的 336x336 图像匹配"""
        # 强制关闭动态图像大小，锁死最大尺寸
        kwargs["dynamic_image_size"] = False
        kwargs["max_image_size"] = 336
        processed = _original_image_processor_preprocess(images, **kwargs)
        
        # 兜底处理：如果意外生成了多 Tile，只保留第一个
        if "pixel_values" in processed and processed["pixel_values"].dim() == 5:
            if processed["pixel_values"].shape[1] > 1:
                processed["pixel_values"] = processed["pixel_values"][:, :1, ...]
        return processed

    LlavaOneVisionImageProcessor.preprocess = _patched_image_processor_preprocess

    # ---------- Patch 2：强制模型认为所有图像只有 1 个 Tile ----------
    _original_get_placeholder_mask = LlallaOnevisionForConditionalGeneration._get_placeholder_mask

    def _patched_get_placeholder_mask(self, images, image_sizes, vision_feature_select_strategy):
        """返回单 Tile 占位符掩码，和图像处理器的输出严格匹配"""
        batch_size = len(images)
        # 每张图仅返回 1 个 tile，和 <image>  token 数量一一对应
        num_tiles = [[1] for _ in range(batch_size)]
        return num_tiles, images, image_sizes

    LlavaOnevisionForConditionalGeneration._get_placeholder_mask = _patched_get_placeholder_mask

    print("[Patch Success] LLaVA-OneVision 已强制单 Tile 模式，已禁用 Torch Compile 避免检查冲突")
except ImportError as e:
    print(f"[Warning] 未找到 LLaVA-OneVision 相关模块，Patch 跳过: {e}")
except Exception as e:
    print(f"[Error] Patch 应用失败: {e}")

# ============================================================
# 全局配置（BlueBEAR 路径，保持你的原有配置不变）
# ============================================================
PROJECT_ROOT = Path("/rds/homes/s/sxw1316/metal_defect_vlm").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "llamafactory"
OUTPUT_ROOT = Path("/rds/homes/s/sxw1316/metal_defect_vlm/loras")
LOG_ROOT = PROJECT_ROOT / "logs"

NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 1e-4
LORA_RANK = 16
LORA_ALPHA = 32
MAX_PIXELS = 112896  # 336 * 336，和预处理尺寸严格一致
MAX_LENGTH = 1024
SEED = 42

MODELS = [
    {
        "name": "llava_onevision_7b",
        "display": "LLaVA-OneVision-7B",
        "model_path": "/rds/projects/w/wangyz-atari/metal_defect_vlm/models/LLaVA-OneVision-7B",
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "template": "llava_next",
    },
]

def build_yaml(model_cfg: dict) -> Path:
    out_dir = OUTPUT_ROOT / model_cfg["name"]
    log_dir = LOG_ROOT / model_cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "stage": "sft", "do_train": True, "do_eval": True,
        "model_name_or_path": model_cfg["model_path"],
        "template": model_cfg["template"],
        "finetuning_type": "lora",
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.05, "lora_target": model_cfg["lora_target"],
        "quantization_bit": 4, "quantization_type": "nf4", "double_quantization": True,
        "dataset_dir": str(DATA_DIR), "dataset": "metal_defect_train", "val_size": 0.1,
        "cutoff_len": MAX_LENGTH,
        "image_max_pixels": MAX_PIXELS,
        "image_min_pixels": MAX_PIXELS,  # 锁死尺寸，和 Patch 逻辑匹配
        "preprocessing_num_workers": 2, "media_dir": str(PROJECT_ROOT),
        "output_dir": str(out_dir), "logging_dir": str(log_dir),
        "save_strategy": "steps", "save_steps": 200, "save_total_limit": 3, "save_only_model": False,
        "per_device_train_batch_size": BATCH_SIZE, "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM, "learning_rate": LR,
        "num_train_epochs": NUM_EPOCHS, "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine", "warmup_steps": 20,
        "bf16": True, "gradient_checkpointing": True,
        "eval_strategy": "steps", "eval_steps": 200, "report_to": "none",
        "seed": SEED, "trust_remote_code": True, "ddp_timeout": 180000000,
    }
    yaml_path = PROJECT_ROOT / f"configs/{model_cfg['name']}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return yaml_path

def train_model(model_cfg: dict) -> bool:
    merge_dir = Path("/rds/homes/s/sxw1316/metal_defect_vlm/models") / f"{model_cfg['name']}_merged"
    if merge_dir.exists() and any(merge_dir.glob("*.safetensors")):
        print(f"[skip] {model_cfg['display']} already merged: {merge_dir}")
        return True

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    yaml_path = build_yaml(model_cfg)
    display = model_cfg["display"]
    print(f"\n{'='*60}\n  Train: {display}\n  Config: {yaml_path}\n  Output: {OUTPUT_ROOT / model_cfg['name']}\n{'='*60}")

    for f in DATA_DIR.glob("*.json"):
        print(f"  [data] {f.name} ({f.stat().st_size//1024} KB)")

    cli = "llamafactory-cli"
    cmd = f'{cli} train "{yaml_path}"'
    log_file = LOG_ROOT / model_cfg["name"] / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] start training...")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end=""); lf.write(line)
        proc.wait()

    elapsed = datetime.now() - start
    if proc.returncode == 0:
        print(f"\n[ok] {display} training done, elapsed {elapsed}")
        json.dump({"model": display, "status": "success", "elapsed_seconds": elapsed.total_seconds()},
                  open(LOG_ROOT / model_cfg["name"] / "record.json", "w"), indent=2)
        return True
    print(f"\n[fail] {display} training failed (exit {proc.returncode})"); return False

def merge_lora(model_cfg: dict) -> bool:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    display = model_cfg["display"]
    lora_dir = OUTPUT_ROOT / model_cfg["name"]
    merge_dir = Path("/rds/homes/s/sxw1316/metal_defect_vlm/models") / f"{model_cfg['name']}_merged"
    if merge_dir.exists() and any(merge_dir.glob("*.safetensors")):
        print(f"[skip] {display} already merged: {merge_dir}"); return True

    actual_lora_dir = lora_dir
    if not (lora_dir / "adapter_config.json").exists():
        cps = sorted(lora_dir.glob("checkpoint-*"))
        if cps: actual_lora_dir = cps[-1]; print(f"[info] use latest checkpoint: {actual_lora_dir.name}")
        else: print(f"[error] no adapter_config.json or checkpoint-* in {lora_dir}"); return False

    print(f"\n[merge] {display} -> {merge_dir}")
    export_yaml = PROJECT_ROOT / f"configs/export_{model_cfg['name']}.yaml"
    yaml.dump({"model_name_or_path": model_cfg["model_path"],
               "adapter_name_or_path": str(actual_lora_dir),
               "template": model_cfg["template"], "finetuning_type": "lora",
               "trust_remote_code": True, "export_dir": str(merge_dir),
               "export_size": 5, "export_device": "cpu"},
              open(export_yaml, "w"), default_flow_style=False, sort_keys=False)

    cli = "llamafactory-cli"
    cmd = f'{cli} export "{export_yaml}"'
    log_file = LOG_ROOT / model_cfg["name"] / "merge.log"
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end=""); lf.write(line)
        proc.wait()
    return proc.returncode == 0

def main():
    print("="*60 + "\n  Metal Defect VLM -- Train (LLaVA-OneVision-7B only)\n" + "="*60)
    results = {}
    for i, m in enumerate(MODELS, 1):
        print(f"\n>>> {i}/{len(MODELS)}  {m['display']}")
        ok = train_model(m); results[m["display"]] = "ok" if ok else "failed"
        if ok: results[f"{m['display']} (merge)"] = "ok" if merge_lora(m) else "failed"
    print("\n" + "="*60 + "\n  Summary\n" + "="*60)
    for k, v in results.items(): print(f"  [{'ok' if v=='ok' else 'fail'}] {k}")
    print("="*60)

if __name__ == "__main__":
    main()
