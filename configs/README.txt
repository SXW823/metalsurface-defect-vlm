train_all.py 自动生成的 LLaMA-Factory YAML 配置文件：

configs/
├── qwen2_5_vl_7b.yaml
├── internvl3_8b.yaml
├── llava_onevision_7b.yaml
└── gemma3_4b.yaml

每个 YAML 包含完整的 QLoRA 训练超参，可手动修改后单独训练：
  llamafactory-cli train configs/qwen2_5_vl_7b.yaml
