import os
import json
from pathlib import Path
import argparse

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def main(args):
    print("--- 步骤1: 加载微调后的模型和处理器 ---")

    # 加载基础模型
    print(f"  [信息] 正在从 {args.base_model_path} 加载基础模型...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True
    )

    # 加载LoRA adapter
    print(f"  [信息] 正在从 {args.adapter_path} 加载LoRA适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=torch.float16
    ).to("cuda")
    model.eval()

    # 加载处理器（包括tokenizer和image processor）
    processor = AutoProcessor.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        local_files_only=True
    )

    print("\n--- 步骤2: 加载测试数据集 ---")
    with open(args.test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]
    print(f"  找到 {len(test_data)} 条测试样本。")

    print("\n--- 步骤3: 开始微调后模型推理和评估 ---")
    predictions = []
    ground_truths = []

    for item in tqdm(test_data, desc="正在评估(微调后)"):
        image_path = item['image_path']
        hypothesis = item['hypothesis']
        true_label = item['label']

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告: 找不到图片 {image_path}, 跳过...")
            continue

        # 构建提示词，使用与训练数据一致的格式
        prompt = f"判断假设是否与图像内容一致。回答 Yes 或 No。\n{hypothesis}"

        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        try:
            with torch.no_grad():
                # 使用处理器准备输入
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")

                # 生成回答
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                answer_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].lower().strip()

            # 解析答案 - 检查 yes/no 或 是/否
            if "yes" in answer_text or "是" in answer_text:
                predicted_label = 1
            elif "no" in answer_text or "否" in answer_text:
                predicted_label = 0
            else:
                # 如果答案不明确，根据第一个字符判断
                if answer_text.startswith('y') or answer_text.startswith('是'):
                    predicted_label = 1
                else:
                    predicted_label = 0

            predictions.append(predicted_label)
            ground_truths.append(true_label)

            # 显示详细的调试信息（仅前几个样本）
            if len(predictions) <= 5:
                print(f"样本 {len(predictions)}: 图片={image_path.split('/')[-1]}, 假设=\"{hypothesis[:30]}...\", 预测=\"{answer_text}\" -> {predicted_label}, 真实={true_label}")

            # 定期显示进度
            if len(predictions) % 100 == 0:
                current_accuracy = accuracy_score(ground_truths, predictions)
                print(f"已处理 {len(predictions)} 个样本，当前准确率: {current_accuracy:.4f}")

        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            continue

    print("\n--- 步骤4: 计算并显示评估结果 ---")

    if not ground_truths:
        print("未能处理任何测试样本，无法计算指标。")
        return

    # 计算各种指标
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, average='binary', zero_division=0)
    recall = recall_score(ground_truths, predictions, average='binary', zero_division=0)
    f1 = f1_score(ground_truths, predictions, average='binary', zero_division=0)

    # 混淆矩阵
    cm = confusion_matrix(ground_truths, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print("\n" + "="*50)
    print("         微调后(Fine-tuned)评估报告")
    print("="*50)
    print(f"  基础模型: {args.base_model_path}")
    print(f"  适配器路径: {args.adapter_path}")
    print(f"  测试文件: {args.test_file}")
    print("-" * 50)
    print(f"  总准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数 (F1-Score): {f1:.4f}")
    print(f"\n  混淆矩阵:")
    print(f"    真负例(TN): {tn}, 假正例(FP): {fp}")
    print(f"    假负例(FN): {fn}, 真正例(TP): {tp}")
    print(f"\n  处理样本数: {len(predictions)}/{len(test_data)}")
    print("="*50)

    # 保存结果
    results = {
        "model_type": "finetuned",
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "test_file": args.test_file,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "total_samples": len(test_data),
        "processed_samples": len(predictions)
    }

    output_file = "finetuned_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估【LoRA微调后】的Qwen2.5-VL模型性能。")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="基础模型的本地路径。")
    parser.add_argument("--adapter_path", type=str, default="/home/xuyan/workspace/LLaMA-Factory/kbvqaCoT/qwen_vl_verifier_adapter", help="LoRA适配器的路径。")
    parser.add_argument("--test_file", type=str, default="verifier_test_split.jsonl", help="用于评估的.jsonl测试文件。")
    args = parser.parse_args()
    main(args)
