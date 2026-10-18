from llamafactory.train.tuner import export_model
import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ['EXPORT_DIR'] = "/rds/projects/w/wangyz-atari/metal_defect_vlm/final_model_internvl3"
args = {
"model_name_or_path": "/rds/projects/w/wangyz-atari/metal_defect_vlm/models/InternVL3-8B-hf",
"adapter_name_or_path": "/rds/projects/w/wangyz-atari/metal_defect_vlm/loras/internvl3_8b",
"template": "intern_vl",
"finetuning_type": "lora",
"export_dir": "/rds/projects/w/wangyz-atari/metal_defect_vlm/final_model_internvl3",
}
export_model(args)
print()
print("############################")
print("恭喜！模型合并成功！")
print(f"请前往文件夹查看: {args['export_dir']}")
print("############################")
print()
