import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
BASE = "/rds/projects/w/wangyz-atari/metal_defect_vlm"
MODEL_PATH = f"{BASE}/models/InternVL3-8B-hf"
ADAPTER_PATH = f"{BASE}/loras/internvl3_8b"
OUTPUT_PATH = f"{BASE}/final_model_internvl3"
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
MODEL_PATH,
torch_dtype=torch.bfloat16,
device_map="cpu",
trust_remote_code=True,
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print("Merging...")
merged_model = model.merge_and_unload()
print("Saving merged model...")
merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor.save_pretrained(OUTPUT_PATH)
print()
print("############################")
print("合并成功！")
print(f"模型保存在: {OUTPUT_PATH}")
print("############################")
print()
