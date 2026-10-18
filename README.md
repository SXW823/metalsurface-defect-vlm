# Metal Surface Defect Detection with Vision Language Models

A comparative study of QLoRA fine-tuned Vision Language Models (VLMs) for metal surface defect classification on the NEU standard dataset.

## Overview

This project evaluates four mainstream VLM architectures on a metal surface defect classification task using 700 grayscale images across 9 defect categories. All models are fine-tuned with QLoRA (4-bit NF4 quantization + LoRA rank 16) and evaluated with an Enhanced Prompting (EP) framework that incorporates domain knowledge, role assignment, and structured output constraints.

### Defect Categories (NEU Standard)

| Category | Description | Category | Description |
|---|---|---|---|
| punching_hole | Punching hole | water_spot | Water spot |
| welding_line | Welding line | oil_spot | Oil spot |
| crescent_gap | Crescent gap | silk_spot | Silk spot |
| rolled_pit | Rolled pit | crease | Crease |
| waist_folding | Waist folding | | |

### Models Compared

| Model | Parameters | Key Characteristics |
|---|---|---|
| **Qwen2.5-VL-7B** | 7B | Strongest in NEU cross-evaluation; largest EP benefit |
| **InternVL3-8B** | 8B | Strong industrial multimodal understanding; NVIDIA reference model |
| **LLaVA-OneVision-7B** | 7B | Baseline comparison; may degrade under EP |
| **Gemma3-4B** | 4B | Lightweight comparison; lowest VRAM usage |

## Results

| Model | Prompt | Accuracy | F1 (macro) | Precision | Recall | Latency (s/img) |
|---|---|---|---|---|---|---|
| Gemma3-4B | EP | 87.08% | 83.59% | 88.91% | 87.08% | 2.53 |
| Qwen2.5-VL-7B | EP | 86.67% | 84.14% | 87.13% | 86.67% | 1.63 |

> Evaluation performed on 240 test images across 10 classes. Full per-class results are available in `results/benchmark_results.json`.

## Repository Structure

```
metal-defect-vlm/
├── README.md                        # This file
├── .gitignore
├── requirements.txt
├── LICENSE
│
├── src/                             # Source code
│   ├── setup_env.py                 # One-click environment setup (hardware detection + dependencies)
│   ├── prepare_data.py              # Data preparation (grayscale → 3-channel + CLAHE + split)
│   ├── train_all.py                 # Unified training entry (trains all 4 models sequentially)
│   ├── train_llava.py              # LLaVA-OneVision specific training script
│   ├── benchmark.py                 # Unified evaluation (Accuracy/F1/confusion matrix)
│   ├── benchmark_old_windows_backup.py  # Earlier Windows-compatible benchmark version
│   ├── plot_results.py              # Publication-quality visualizations
│   ├── fix_json.py                  # JSON data repair utility
│   ├── merge.py                     # LoRA adapter merge (generic)
│   ├── merge_final.py               # LoRA adapter merge (InternVL3 specific)
│   ├── merge_peft.py                # PEFT-based merge utility
│   ├── save_gemma.py                # Gemma3 model save utility
│   └── show_results.py              # Quick results display
│
├── configs/                         # Training YAML configurations
│   ├── internvl3_8b.yaml            # InternVL3-8B QLoRA config
│   ├── llava_onevision_7b.yaml      # LLaVA-OneVision-7B QLoRA config
│   └── README.md                    # Config documentation
│
├── data/                            # Dataset and prompts
│   ├── ep_prompt.txt                # Enhanced Prompting (EP) template
│   ├── llamafactory/                # LLaMA-Factory format dataset
│   │   ├── dataset_info.json
│   │   ├── metal_defect_train.json
│   │   ├── metal_defect_val.json
│   │   └── metal_defect_test.json
│   └── splits/                      # Train/val/test split indices
│       ├── train.json
│       ├── val.json
│       ├── test.json
│       └── few_shot/               # 1/5/10-shot splits
│
├── results/                         # Benchmark results and figures
│   ├── benchmark_results.json       # Full evaluation data
│   ├── benchmark_summary.csv        # Summary table
│   ├── cm_gemma3_4b_EP.png          # Confusion matrix - Gemma3
│   ├── cm_internvl3_8b_EP.png       # Confusion matrix - InternVL3
│   ├── cm_qwen2_5_vl_7b_EP.png      # Confusion matrix - Qwen2.5-VL
│   └── figures/                     # Publication-quality figures
│       ├── bar_accuracy_f1.pdf/.png
│       ├── confusion_Qwen25_VL_7B.pdf/.png
│       ├── per_class_f1_heatmap.pdf/.png
│       ├── radar_comparison.pdf/.png
│       └── table_latex.txt           # LaTeX table (ready for paper)
│
└── poster/                          # Academic poster
    ├── Metal_Defect_VLM_Poster.pdf   # Final poster (A1 format)
    ├── build_poster_a1.py            # A1 poster builder
    ├── build_poster_docx.py          # DOCX poster builder
    ├── make_poster_figs.py          # Poster figure generator
    └── render_crop.py               # Image crop utility
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n vlm_defect python=3.11 -y
conda activate vlm_defect
python src/setup_env.py
```

This script automatically detects your GPU, installs the correct CUDA-compatible PyTorch, LLaMA-Factory, and all evaluation dependencies.

### 2. Prepare Data

Place 700 grayscale images into `raw_images/` subdirectories organized by defect class, then:

```bash
python src/prepare_data.py
```

This automatically performs:
- Grayscale → 3-channel conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement
- 8:1:1 train/val/test split
- LLaMA-Factory dataset generation

### 3. Train Models

```bash
# Train all 4 models sequentially
python src/train_all.py

# Or train a specific model (e.g., LLaVA)
python src/train_llava.py
```

Training uses QLoRA with:
- 4-bit NF4 quantization
- LoRA rank 16, alpha 32
- Flash Attention 2
- Gradient checkpointing
- Batch size 1 + gradient accumulation 8

### 4. Merge LoRA Adapters

```bash
# Merge InternVL3 adapter
python src/merge_final.py

# Merge other models
python src/merge.py
```

### 5. Evaluate

```bash
python src/benchmark.py
```

Loads merged models, runs inference on the test set, and computes Accuracy/F1/per-class metrics. Results are saved as CSV + JSON.

### 6. Generate Publication Figures

```bash
python src/plot_results.py
```

Generates 5 figure types + a LaTeX table, all saved to `results/figures/`.

## Key Design Decisions

### CLAHE Enhancement for Grayscale Images
Metal surface illumination is inherently uneven in industrial settings. CLAHE significantly enhances defect region contrast without the geometric distortion caused by spatial augmentation methods like Mosaic.

### Enhanced Prompting (EP) Framework
Based on the finding from the 2025 NEU cross-evaluation paper: the EP framework (role assignment + domain knowledge + reasoning scaffolding + structured output constraints) improved Qwen2.5-VL-7B accuracy from 76.5% to 89.2% (+12.7%), while LLaVA showed degradation. This experiment uniformly applies EP templates for fair comparison and reproducibility.

### VRAM Optimization (16GB GPUs)
- **4-bit NF4 quantization**: 4x weight compression
- **Gradient checkpointing**: Trade compute for memory
- **Flash Attention 2**: Reduced attention memory footprint
- **image_max_pixels=401408** (~632x632): Sufficient for metal defect images
- **Batch size=1 + gradient accumulation 8**: Simulates batch_size=8

## Evaluation Protocol

- **Full fine-tune**: 560 train / 70 val / 70 test
- **Few-shot**: 1/5/10-shot (aligned with 2026 industrial defect evaluation standards)
- **EP vs Basic ablation**: Validates prompting impact across architectures
- **Primary metric**: Macro-F1 (eliminates long-tail bias, aligned with 2025 NEU evaluation)

## Hardware

- **Local**: NVIDIA RTX 5070Ti (16GB VRAM)
- **HPC**: BlueBEAR cluster (Birmingham University) with RDS storage

## Dependencies

See `requirements.txt` for the full dependency list. Key packages:
- PyTorch (CUDA)
- transformers
- peft
- LLaMA-Factory
- scikit-learn
- matplotlib / seaborn
- pandas

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{metal_defect_vlm,
  title={Metal Surface Defect Detection with Vision Language Models},
  year={2026},
  note={QLoRA fine-tuning comparison of Qwen2.5-VL, InternVL3, LLaVA-OneVision, and Gemma3}
}
```

## License

This project is licensed under the MIT License - see `LICENSE` for details.
