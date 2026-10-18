import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from modelscope import snapshot_download

# ModelScope 上 Gemma3-4B 的模型 ID
model_dir = snapshot_download(
    'google/gemma-3-4b-it',
    cache_dir='E:/models/Gemma3-4B'
)
print(f"✓ 下载完成: {model_dir}")