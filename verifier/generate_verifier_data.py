import os
import json
import argparse
import random
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm


# # 解决模块导入路径问题
# # 这样做，Python就能找到位于根目录下的 kbvqa_modular_cot.py
# try:
#     # 获取本脚本所在的目录
#     # e.g., /path/to/project/scripts
#     script_dir = Path(__file__).resolve().parent
    
#     # 获取上一级目录，我们假设这是项目根目录
#     # e.g., /path/to/project
#     project_root = script_dir.parent
    
#     # 检查目标文件是否真的存在于我们猜测的根目录下
#     target_file = project_root / 'kbvqa_modular_cot.py'
    
#     if target_file.exists():
#         # 如果文件存在，就将项目根目录添加到Python搜索路径的最前面，以获得最高优先级
#         sys.path.insert(0, str(project_root))
#         print(f"  [调试信息] 成功找到目标文件: {target_file}")
#         print(f"  [调试信息] 已将项目根目录 {project_root} 添加到搜索路径。")
#     else:
#         # 如果文件不存在，给出非常明确的错误提示
#         print("【严重错误】: 无法自动定位 'kbvqa_modular_cot.py'。")
#         print(f"  - 脚本当前目录: {script_dir}")
#         print(f"  - 猜测的项目根目录: {project_root}")
#         print(f"  - 尝试寻找的文件路径: {target_file}")
#         print("  - 请确认您的文件结构是否正确，或手动修改本脚本中的路径设置。")
#         exit()

# except NameError:
#     print("  [调试信息] 在交互式环境中，请确保您的工作目录是项目根目录。")

# 确保能从您的主脚本中导入视觉处理模块
try:
    from kbvqa_modular_cot import HFExtractorAndGrounder
except ImportError:
    print("【严重错误】: 无法从 'kbvqa_modular_cot.py' 导入 HFExtractorAndGrounder。")
    print("请确保本脚本和 'kbvqa_modular_cot.py' 在同一个目录下。")
    exit()

def main(args):
    print("--- 步骤1: 初始化视觉处理模块 (HFExtractorAndGrounder) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_pipeline = HFExtractorAndGrounder(device=device)

    print(f"--- 步骤2: 加载A-OKVQA数据源自: {args.aokvqa_json_path} ---")
    with open(args.aokvqa_json_path, 'r', encoding='utf-8') as f:
        aokvqa_data = json.load(f)

    if args.limit > 0:
        aokvqa_data = aokvqa_data[:args.limit]
    
    print(f"将处理 {len(aokvqa_data)} 个样本。")

    # 创建用于存放图片切片的目录
    crops_output_dir = Path(args.crops_output_dir)
    crops_output_dir.mkdir(exist_ok=True)
    print(f"所有图片区域切片将被保存到: {crops_output_dir}")

    print("\n--- 步骤3: 开始生成Verifier训练数据 ---")
    
    successful_samples = 0
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for sample in tqdm(aokvqa_data, desc="处理A-OKVQA样本"):
            try:
                question_id = sample.get('question_id', 'N/A')
                image_id_str = str(sample['image_id']).zfill(12)
                image_filename = f"{image_id_str}.jpg"
                image_path = Path(args.coco_images_dir) / image_filename

                if not image_path.exists():
                    tqdm.write(f"【警告】: 找不到图片 {image_path}，跳过样本 {question_id}。")
                    continue
                    
                pil_image = Image.open(image_path).convert("RGB")
                
                # 使用您的视觉模块提取所有区域和描述
                regions = hf_pipeline.detect_and_caption(pil_image)
                
                # 过滤掉没有有效描述的区域
                # valid_regions = [r for r in regions if r.get('caption', '').strip()]
                # 过滤掉没有有效描述的区域
                valid_regions = []
                for r in regions:
                    caption = r.get('caption', '').strip()
                    crop_image = r.get('crop')
                    
                    if not caption or crop_image is None:
                        continue

                    # --- [新增] 质量控制/过滤规则 ---
                    width, height = crop_image.size
                    # 规则1: 过滤掉尺寸过小的图片
                    if width < args.min_crop_size or height < args.min_crop_size:
                        continue
                    
                    # 规则2: 过滤掉长宽比过于极端的图片 (例如，一条细线)
                    if width / height > args.max_aspect_ratio or height / width > args.max_aspect_ratio:
                        continue
                    
                    # 规则3: 过滤掉常见的无意义描述 (可以根据需要扩展这个列表)
                    meaningless_captions = ['a blurry photo', 'a close up of a logo', 'a sign on a building']
                    if any(mc in caption.lower() for mc in meaningless_captions):
                        continue
                    # --- [新增] 过滤结束 ---
                    
                    valid_regions.append(r)


                if len(valid_regions) < 2:
                    # 如果有效区域少于2个，无法进行负采样，跳过
                    tqdm.write(f"【信息】: 图片 {image_path} 的有效区域少于2个，跳过。")
                    continue

                # --- 核心的数据合成逻辑 ---
                for i, region_true in enumerate(valid_regions):
                    # 1. 生成正样本 (Positive Sample)
                    true_caption = region_true.get('caption').strip()
                    
                    # 保存图片切片
                    crop_filename = f"{question_id}_region_{i}_true.jpg"
                    crop_path = crops_output_dir / crop_filename
                    region_true['crop'].save(crop_path)

                    positive_sample = {
                        "image_path": str(crop_path.resolve()), # 保存绝对路径
                        "hypothesis": true_caption,
                        "label": 1 # 1 代表 "yes"
                    }
                    out_f.write(json.dumps(positive_sample, ensure_ascii=False) + "\n")
                    successful_samples += 1

                    # 2. 生成负样本 (Negative Sample)
                    # 从其他区域中随机选择一个作为“假命题”
                    other_regions = valid_regions[:i] + valid_regions[i+1:]
                    region_false = random.choice(other_regions)
                    false_caption = region_false.get('caption').strip()
                    
                    # 负样本使用与正样本相同的图片切片，但配上错误的描述
                    negative_sample = {
                        "image_path": str(crop_path.resolve()),
                        "hypothesis": false_caption,
                        "label": 0 # 0 代表 "no"
                    }
                    out_f.write(json.dumps(negative_sample, ensure_ascii=False) + "\n")
                    successful_samples += 1

            except Exception as e:
                tqdm.write(f"【严重错误】: 处理样本 {question_id} 时发生错误: {e}")
                continue

    print(f"\n--- 步骤4: 处理完成 ---")
    print(f"  [总结] 总共生成并保存了 {successful_samples} 条Verifier训练样本。")
    print(f"  [结果] 训练数据索引文件已保存到: {args.output_file}")
    print(f"  [结果] 对应的图片切片已保存到: {args.crops_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为VLM Verifier自动合成训练数据。")
    parser.add_argument("--aokvqa_json_path", type=str, required=True, help="A-OKVQA的JSON文件路径。")
    parser.add_argument("--coco_images_dir", type=str, required=True, help="COCO图片数据集的目录路径。")
    parser.add_argument("--coco_split", type=str, default="train2017", help="COCO数据集的子集名称 (如 train2017, val2017)。")
    parser.add_argument("--output_file", type=str, default="verifier_train_vlm.jsonl", help="输出的JSONL训练数据索引文件。")
    parser.add_argument("--crops_output_dir", type=str, default="./verifier_image_crops", help="保存所有图片区域切片的目录。")
    parser.add_argument("--limit", type=int, default=0, help="限制处理的样本数量，用于测试 (0表示不限制)。")
    parser.add_argument("--min_crop_size", type=int, default=32, help="图片切片的最小边长 (像素)。小于此值的将被丢弃。")
    parser.add_argument("--max_aspect_ratio", type=float, default=5.0, help="图片切片的最大长宽比。超过此值的将被丢弃。")
    args = parser.parse_args()
    main(args)