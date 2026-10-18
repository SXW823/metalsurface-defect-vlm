import os, gc, shutil
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

cands = [
    "/home/sxw1316/Desktop/metal_defect_vlm/models/InternVL3-8B-hf",
    "/rds/projects/w/wangyz-atari/metal_defect_vlm/models/InternVL3-8B-hf",
]
BASE_MODEL_PATH = None
for c in cands:
    if os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")):
        BASE_MODEL_PATH = c
        break
if BASE_MODEL_PATH is None:
    raise SystemExit("!!! base model not found in either Desktop or rds, copy InternVL3-8B-hf to rds first")

LORA_ADAPTER_PATH = "/rds/projects/w/wangyz-atari/metal_defect_vlm/loras/internvl3_8b"
OUTPUT_PATH = "/rds/projects/w/wangyz-atari/metal_defect_vlm/final_model_internvl3"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

print("using base model:", BASE_MODEL_PATH)

base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
peft = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
peft.eval()

del base
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("merging...")
merged = peft.merge_and_unload()

if os.path.isdir(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
merged.save_pretrained(OUTPUT_PATH, safe_serialization=True)

proc = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
proc.save_pretrained(OUTPUT_PATH)

print("DONE: merged model saved to", OUTPUT_PATH)
