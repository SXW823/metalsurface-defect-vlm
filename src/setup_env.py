"""
setup_env.py — 一键环境配置脚本
功能：检测硬件 → 安装 CUDA 版 PyTorch → 安装 LLaMA-Factory → 安装评测依赖
运行：python setup_env.py
"""

import subprocess
import sys
import os
import platform
import json
import time

# ─────────────────────────────────────────────
# 颜色输出
# ─────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    END    = "\033[0m"

def log(msg):  print(f"{C.GREEN}[✓]{C.END} {msg}")
def warn(msg): print(f"{C.YELLOW}[!]{C.END} {msg}")
def err(msg):  print(f"{C.RED}[✗]{C.END} {msg}", file=sys.stderr)
def info(msg): print(f"{C.CYAN}[*]{C.END} {msg}")
def step(msg): print(f"\n{C.BOLD}{'='*60}\n  {msg}\n{'='*60}{C.END}")

def run(cmd, check=True, capture=False):
    """执行 shell 命令"""
    info(f"执行: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and res.returncode != 0:
        err(f"命令失败 (exit {res.returncode}): {cmd}")
        if res.stderr: print(res.stderr)
        sys.exit(1)
    return res

# ─────────────────────────────────────────────
# Step 1: 硬件检测
# ─────────────────────────────────────────────
step("Step 1/6  硬件检测")

gpu_info = run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", capture=True)
info(f"GPU: {gpu_info.stdout.strip()}")

# 检查 5070Ti（16GB）
if "5070" in gpu_info.stdout or "16" in gpu_info.stdout.split(",")[1].strip():
    log("检测到 RTX 5070Ti 16GB，QLoRA 7B-8B 模型可正常运行")
else:
    warn("未检测到 5070Ti 16GB，若显存不足请降低 image_max_pixels")

cuda_version = run("nvcc --version 2>/dev/null | tail -1", capture=True)
if cuda_version.stdout.strip():
    log(f"CUDA: {cuda_version.stdout.strip()}")
else:
    warn("未检测到 nvcc，请确认 CUDA Toolkit 已安装")

# ─────────────────────────────────────────────
# Step 2: 创建 conda 环境
# ─────────────────────────────────────────────
step("Step 2/6  创建 Conda 环境")

env_name = "vlm_defect"
res = run(f"conda env list | grep -w {env_name}", check=False, capture=True)
if env_name in res.stdout:
    warn(f"环境 '{env_name}' 已存在，跳过创建")
else:
    run(f"conda create -n {env_name} python=3.11 -y")
    log(f"环境 '{env_name}' 创建完成")

info(f"请运行:  conda activate {env_name}")
info(f"然后重新执行:  python setup_env.py  (以在该环境中继续安装)")

# 检查是否在目标环境中
current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
if current_env != env_name:
    warn(f"当前环境: {current_env}（非 {env_name}）")
    warn("请先执行:  conda activate vlm_defect")
    warn("然后重新运行:  python setup_env.py")
    # 仍然继续安装到当前环境，不退出

# ─────────────────────────────────────────────
# Step 3: 安装 PyTorch (CUDA 12.4)
# ─────────────────────────────────────────────
step("Step 3/6  安装 PyTorch (CUDA 12.4)")

torch_check = run("python -c \"import torch; print(torch.__version__)\"", check=False, capture=True)
if "2." in torch_check.stdout:
    log(f"PyTorch 已安装: {torch_check.stdout.strip()}")
else:
    run("pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124")
    log("PyTorch 2.5.1 + CUDA 12.4 安装完成")

# ─────────────────────────────────────────────
# Step 4: 安装 LLaMA-Factory（VLM 微调核心）
# ─────────────────────────────────────────────
step("Step 4/6  安装 LLaMA-Factory")

lf_check = run("pip list 2>/dev/null | grep -i llamafactory", check=False, capture=True)
if lf_check.stdout.strip():
    log("LLaMA-Factory 已安装")
else:
    run("git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory")
    run("cd /tmp/LLaMA-Factory && pip install -e \".[torch,metrics]\"")
    log("LLaMA-Factory 安装完成")

# ─────────────────────────────────────────────
# Step 5: 安装量化与加速依赖
# ─────────────────────────────────────────────
step("Step 5/6  安装量化与加速依赖")

deps = [
    "bitsandbytes>=0.45.0",       # 4-bit 量化
    "peft>=0.13.0",                # LoRA
    "accelerate>=1.0.0",
    "transformers>=4.46.0",
    "flash-attn==2.7.2",           # Flash Attention 2（5070Ti 支持）
    "pillow>=10.0",
    "opencv-python-headless>=4.8", # 灰度图预处理
    "scikit-learn>=1.3",           # 评测指标
    "pandas>=2.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "tqdm>=4.66",
    "pyyaml>=6.0",
    "safetensors>=0.4",
    "huggingface_hub>=0.25",
]

for dep in deps:
    name = dep.split(">=")[0].split("==")[0]
    check = run(f"pip list 2>/dev/null | grep -i ^{name}$", check=False, capture=True)
    if check.stdout.strip():
        log(f"{name} 已安装")
    else:
        info(f"安装 {dep} ...")
        r = run(f"pip install {dep}", check=False, capture=True)
        if r.returncode != 0:
            warn(f"{dep} 安装失败，尝试 --no-build-isolation")
            run(f"pip install {dep} --no-build-isolation", check=False)

log("所有依赖安装完成")

# ─────────────────────────────────────────────
# Step 6: 生成环境信息报告
# ─────────────────────────────────────────────
step("Step 6/6  环境验证报告")

report = {}
report["platform"] = platform.platform()
report["python"] = sys.version.split()[0]

try:
    import torch
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        report["torch_cuda_version"] = torch.version.cuda
except: pass

try:
    import transformers
    report["transformers"] = transformers.__version__
except: pass

try:
    import peft
    report["peft"] = peft.__version__
except: pass

try:
    import bitsandbytes
    report["bitsandbytes"] = bitsandbytes.__version__
except: pass

# 写入报告
with open("env_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("  环境配置完成！")
print("="*60)
for k, v in report.items():
    print(f"  {k:20s}: {v}")
print("="*60)

# 关键提醒
print(f"\n{C.BOLD}下一步：{C.END}")
print(f"  1. 将 700 张灰度图放到  raw_images/  下（按 9 个类别分子文件夹）")
print(f"  2. 运行:  python prepare_data.py")
print(f"  3. 运行:  python train_all.py")
print(f"  4. 运行:  python benchmark.py")
print(f"  5. 运行:  python plot_results.py")
