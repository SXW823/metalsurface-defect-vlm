"""
prepare_data.py — 数据准备脚本（GC10-DET 10类完整版）
功能：
  1. 扫描 raw_images/ 下 10 个类别子文件夹中的灰度图
  2. 将灰度图转为 3 通道（VLM 需要 RGB 输入）并预处理
  3. 按 8:1:1 划分 train/val/test
  4. 生成 LLaMA-Factory 兼容的 JSON 数据集
  5. 生成少样本 1/5/10-shot split

缺陷类别（GC10-DET 标准 10 类）：
  punching_hole, welding_line, crescent_gap, water_spot, oil_spot,
  silk_spot, inclusion, rolled_pit, crease, waist_folding
"""

import os
import json
import random
from pathlib import Path
import cv2
import numpy as np

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

# ★ 关键修改：DEFECT_CLASSES 现在是完整的 10 类
DEFECT_CLASSES = [
    "punching_hole",  # 冲孔
    "welding_line",   # 焊线
    "crescent_gap",   # 月牙缺口
    "water_spot",     # 水斑
    "oil_spot",       # 油斑
    "silk_spot",      # 丝斑
    "inclusion",      # 夹杂物 ← 第10类，之前漏掉的
    "rolled_pit",     # 轧坑
    "crease",         # 折痕
    "waist_folding",  # 腰折
]

# 类别中文映射（用于 EP 提示词和训练回复）
CLASS_CN = {
    "punching_hole":  "冲孔",
    "welding_line":   "焊线",
    "crescent_gap":   "月牙缺口",
    "water_spot":     "水斑",
    "oil_spot":       "油斑",
    "silk_spot":      "丝状斑",
    "inclusion":      "夹杂物",
    "rolled_pit":     "轧制麻点",
    "crease":         "折痕",
    "waist_folding":  "腰折"
}

# 路径配置
RAW_DIR       = Path("raw_images")        # 原始灰度图（按类别分子文件夹）
PROCESSED_DIR = Path("data/processed")         # 预处理后的 3 通道图
OUTPUT_DIR    = Path("data/splits")            # 划分索引 JSON
LF_DATA_DIR   = Path("data/llamafactory")      # LLaMA-Factory 数据集

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
SEED        = 42

# 图像预处理参数
TARGET_SIZE = (512, 512)    # VLM 输入分辨率（灰度图不大，512 足够）
CLIP_LIMIT  = 2.0           # CLAHE 对比度限制
TILE_GRID   = (8, 8)        # CLAHE 网格大小

# ─────────────────────────────────────────────
# Step 1: 灰度图 → 3 通道 + CLAHE 增强
# ─────────────────────────────────────────────
def preprocess_grayscale(img_path: Path, out_path: Path):
    """
    灰度图预处理流水线（针对金属表面缺陷优化）：
    1. 读取灰度图
    2. CLAHE 自适应直方图均衡化（增强缺陷对比度，不改变几何结构）
    3. 轻微高斯去噪（保留缺陷边缘）
    4. 转为 3 通道（VLM 模型需要 RGB 输入）
    5. Resize 到 512x512 并保存为 JPEG
    """
    # 1. 读取灰度图
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"  [警告] 无法读取: {img_path}，跳过")
        return False

    # 2. CLAHE 增强（金属表面光照不均，CLAHE 效果显著）
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
    img_enhanced = clahe.apply(img_gray)

    # 3. 轻微高斯去噪（保留缺陷边缘）
    img_denoised = cv2.GaussianBlur(img_enhanced, (3, 3), 0.8)

    # 4. 转 3 通道（VLM 必须）
    img_rgb = cv2.cvtColor(img_denoised, cv2.COLOR_GRAY2RGB)

    # 5. Resize 到目标尺寸
    img_rgb = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def find_images(folder: Path):
    """递归查找文件夹中所有图像文件"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([f for f in folder.rglob("*") if f.suffix.lower() in exts])


# ─────────────────────────────────────────────
# Step 2: 数据划分（分层抽样，保证各类比例一致）
# ─────────────────────────────────────────────
def log(msg):
    """简单的日志输出函数"""
    print(f"\033[92m[✓]\033[0m {msg}")


def split_dataset(all_files_by_class):
    """
    按 8:1:1 分层抽样划分
    关键：random.shuffle 后按固定比例切分，保证每个类别在 train/val/test 中占比相同
    """
    random.seed(SEED)
    splits = {"train": [], "val": [], "test": []}

    for cls, files in all_files_by_class.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        # 前 80% 训练，接下来 10% 验证，最后 10% 测试
        splits["train"].extend([(f, cls) for f in files[:n_train]])
        splits["val"].extend([(f, cls) for f in files[n_train:n_train+n_val]])
        splits["test"].extend([(f, cls) for f in files[n_train+n_val:]])

    for split, items in splits.items():
        random.shuffle(items)
        print(f"  {split:5s}: {len(items)} 张")

    return splits


def create_few_shot_splits(all_files_by_class, shots=[1, 5, 10]):
    """
    生成 1/5/10-shot 少样本划分
    每类随机取 k 张作为训练，剩余作为测试
    这是 2026 年工业缺陷检测论文的标准 few-shot 评测协议
    """
    random.seed(SEED)
    few_shot_dir = OUTPUT_DIR / "few_shot"
    few_shot_dir.mkdir(parents=True, exist_ok=True)

    for k in shots:
        train_items, test_items = [], []
        for cls, files in all_files_by_class.items():
            # 每类随机取 k 张（如果该类样本不足 k 张，则全取）
            sampled = random.sample(files, min(k, len(files)))
            train_items.extend([(f, cls) for f in sampled])
            # test = 剩余样本
            remaining = [f for f in files if f not in sampled]
            test_items.extend([(f, cls) for f in remaining])

        split_data = {"train": train_items, "test": test_items}
        for split_name, items in split_data.items():
            out_path = few_shot_dir / f"{k}shot_{split_name}.json"
            with open(out_path, "w") as f:
                json.dump([{"image": str(p), "label": c} for p, c in items], f, indent=2)
        print(f"  {k}-shot: train={len(train_items)}, test={len(test_items)}")


# ─────────────────────────────────────────────
# Step 3: 构建 EP 提示词（Enhanced Prompting）
# ─────────────────────────────────────────────
def build_ep_prompt():
    """
    构建增强提示词（Enhanced Prompting）模板
    参考 2025 NEU 钢材缺陷 VLM 横评论文的 EP 框架：
    角色分配 + 领域知识 + 推理脚手架 + 结构化输出
    """
    # ★ 关键修改：class_list 现在列举全部 10 类
    class_list = "、".join([f"{c}({CLASS_CN[c]})" for c in DEFECT_CLASSES])
    return f"""你是一位资深的金属表面质检专家。

【领域知识】
金属表面常见缺陷共 10 类：{class_list}。
- punching_hole(冲孔)：材料上的孔洞，通常为圆形或椭圆形
- welding_line(焊线)：焊接过程中产生的线条状痕迹
- crescent_gap(月牙缺口)：边缘呈月牙形的缺口缺陷
- water_spot(水斑)：水渍残留形成的斑痕
- oil_spot(油斑)：油污残留形成的暗色斑痕
- silk_spot(丝状斑)：细丝状的表面斑纹
- inclusion(夹杂物)：材料中嵌入的异质颗粒或夹杂物
- rolled_pit(轧制麻点)：轧制过程中产生的小凹坑群
- crease(折痕)：材料弯曲折叠留下的痕迹
- waist_folding(腰折)：腰部区域的折叠变形

【分析步骤】
1. 观察图像整体，判断金属表面是否存在异常区域
2. 若存在异常，对比上述 10 类缺陷的视觉特征进行匹配
3. 确定缺陷类型，并评估严重程度（轻微/中等/严重）

【输出格式】
严格输出 JSON，不要输出其他内容：
{{"defect_type": "类别英文名", "severity": "轻微/中等/严重", "description": "简要描述缺陷位置和形态"}}"""


def build_assistant_response(cls_name: str):
    """根据类别生成训练用的 assistant 回复"""
    # ★ 关键修改：severity_map 和 desc_map 都补齐 10 类
    severity_map = {
        "punching_hole":  "中等",
        "welding_line":   "轻微",
        "crescent_gap":   "中等",
        "water_spot":     "轻微",
        "oil_spot":       "轻微",
        "silk_spot":      "轻微",
        "inclusion":      "中等",    # ← 新增
        "rolled_pit":     "中等",
        "crease":         "严重",
        "waist_folding":  "严重",
    }
    desc_map = {
        "punching_hole":  "金属表面存在孔洞状缺陷",
        "welding_line":   "表面可见焊接线条痕迹",
        "crescent_gap":   "边缘区域有月牙形缺口",
        "water_spot":     "表面有水渍残留斑痕",
        "oil_spot":       "表面有油污暗色斑痕",
        "silk_spot":      "表面有丝状纹理斑纹",
        "inclusion":      "表面嵌有异质夹杂物",    # ← 新增
        "rolled_pit":     "表面有密集小凹坑",
        "crease":         "材料表面有明显折痕",
        "waist_folding":  "腰部区域有折叠变形",
    }
    sev = severity_map.get(cls_name, "中等")
    desc = desc_map.get(cls_name, "金属表面存在缺陷")
    return json.dumps({"defect_type": cls_name, "severity": sev, "description": desc},
                      ensure_ascii=False)


# ─────────────────────────────────────────────
# Step 4: 生成 LLaMA-Factory 格式 JSON
# ─────────────────────────────────────────────
def generate_lf_dataset(splits, ep_prompt: str):
    """
    生成 LLaMA-Factory 多模态数据集 JSON
    格式参考 LLaMA-Factory mllm_demo.json（ShareGPT 格式）
    关键：每条数据中 <image> 标记数量必须等于 images 列表长度
    """
    LF_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 生成 dataset_info.json（LLaMA-Factory 数据集注册文件）
    dataset_info = {
        "metal_defect_train": {
            "file_name": "metal_defect_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
        },
        "metal_defect_val": {
            "file_name": "metal_defect_val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
        },
        "metal_defect_test": {
            "file_name": "metal_defect_test.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
        }
    }

    with open(LF_DATA_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    log("生成 dataset_info.json")

    # 2. 生成各 split 的数据
    for split_name, items in splits.items():
        records = []
        for img_path, cls_name in items:
            # 图像路径：相对路径（相对于 LLaMA-Factory data 目录）
            # 格式：../data/processed/<类名>/<文件名>.jpg
            rel_path = f"data/processed/{cls_name}/{img_path.stem}.jpg"
            record = {
                "images": [rel_path],  # ★ 一张图对应一个 <image> 标记
                "messages": [
                    {"role": "user", "content": f"<image>{ep_prompt}"},
                    {"role": "assistant", "content": build_assistant_response(cls_name)}
                ]
            }
            records.append(record)

        out_file = LF_DATA_DIR / f"metal_defect_{split_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        log(f"生成 {out_file}  ({len(records)} 条)")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("  金属缺陷数据准备流水线 (GC10-DET 10类)")
    print("="*60)

    # 检查原始数据目录
    if not RAW_DIR.exists():
        print(f"\n[错误] 未找到 {RAW_DIR}/ 目录")
        print(f"请将 700 张灰度图按 10 个类别放入子文件夹：")
        for c in DEFECT_CLASSES:
            print(f"  {RAW_DIR}/{c}/  ← 放入该类图片")
        return

    # Step 1: 扫描原始灰度图
    print("\n[Step 1] 扫描原始灰度图...")
    all_files_by_class = {}
    total = 0
    for cls in DEFECT_CLASSES:
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            print(f"  [警告] {cls_dir} 不存在，跳过")
            continue
        imgs = find_images(cls_dir)
        if not imgs:
            print(f"  [警告] {cls_dir} 中没有找到图像文件")
            continue
        all_files_by_class[cls] = imgs
        total += len(imgs)
        print(f"  {cls:20s}: {len(imgs)} 张")

    if total == 0:
        print("\n[错误] 未找到任何图像，请检查 raw_images/ 目录结构")
        return

    print(f"\n  总计: {total} 张图像，{len(all_files_by_class)} 个类别")

    # Step 2: 预处理（灰度→3通道+CLAHE）
    print("\n[Step 2] 预处理灰度图（CLAHE 增强 + 转 RGB）...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_paths = {}
    for cls, imgs in all_files_by_class.items():
        out_cls_dir = PROCESSED_DIR / cls
        out_cls_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        success_count = 0
        for img_path in imgs:
            out_path = out_cls_dir / (img_path.stem + ".jpg")
            if preprocess_grayscale(img_path, out_path):
                paths.append(out_path)
                success_count += 1
        processed_paths[cls] = paths
        print(f"  {cls:20s}: 成功处理 {success_count}/{len(imgs)} 张")
    log(f"预处理完成 → {PROCESSED_DIR}/")

    # Step 3: 划分 train/val/test（分层抽样 8:1:1）
    print("\n[Step 3] 按 8:1:1 划分数据集...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = split_dataset(processed_paths)

    # 保存划分索引
    for split_name, items in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump([{"image": str(p), "label": c} for p, c in items], f, indent=2)
        log(f"保存 {out_path}")

    # Step 4: 生成 few-shot splits（1/5/10-shot）
    print("\n[Step 4] 生成 1/5/10-shot 少样本划分...")
    create_few_shot_splits(processed_paths, shots=[1, 5, 10])

    # Step 5: 生成 LLaMA-Factory 数据集
    print("\n[Step 5] 生成 LLaMA-Factory 格式数据集...")
    ep_prompt = build_ep_prompt()
    generate_lf_dataset(splits, ep_prompt)

    # 保存 EP 提示词（评测时复用）
    with open("data/ep_prompt.txt", "w", encoding="utf-8") as f:
        f.write(ep_prompt)
    log("保存 EP 提示词 → data/ep_prompt.txt")

    # 统计报告
    print("\n" + "="*60)
    print("  数据准备完成！")
    print("="*60)
    print(f"  总图像数: {total}")
    for split_name in ["train", "val", "test"]:
        items = splits[split_name]
        print(f"  {split_name:5s}: {len(items)} 张")
    print(f"\n  输出目录:")
    print(f"    {PROCESSED_DIR}/    — 预处理后的 3 通道图")
    print(f"    {OUTPUT_DIR}/       — 划分索引 JSON")
    print(f"    {LF_DATA_DIR}/      — LLaMA-Factory 数据集")
    print(f"\n  下一步: python train_all.py")


if __name__ == "__main__":
    main()
