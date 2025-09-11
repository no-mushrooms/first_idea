import os
import json
from pathlib import Path
import argparse

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def main(args):
    print("--- 步骤1: 加载未经微调的基础模型和处理器 ---")
    
    # 加载模型
    print(f"  [信息] 正在从 {args.model_path} 加载原始模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda")
    model.eval()

    # 修改：加载 tokenizer 而不是 processor
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        local_files_only=True
    )

    print("\n--- 步骤2: 加载测试数据集 ---")
    with open(args.test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]
    print(f"  找到 {len(test_data)} 条测试样本。")

    print("\n--- 步骤3: 开始零样本推理和评估 ---")
    predictions = []
    ground_truths = []

    for item in tqdm(test_data, desc="正在评估(零样本)"):
        image_path = item['image_path']
        hypothesis = item['hypothesis']
        true_label = item['label'] 

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告: 找不到图片 {image_path}, 跳过...")
            continue

        # 修改：使用 Moondream2 的正确调用方式
        prompt = f"Question: Is the following statement true based on the image? Statement: {hypothesis}. Answer with 'yes' or 'no'."
        
        try:
            with torch.no_grad():
                # 使用 Moondream2 的 answer_question 方法
                answer_text = model.answer_question(
                    image, 
                    prompt, 
                    tokenizer,
                    max_new_tokens=10  # 限制生成长度
                ).lower()
            
            # 解析答案
            predicted_label = 1 if "yes" in answer_text else 0
            
            predictions.append(predicted_label)
            ground_truths.append(true_label)
            
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
    
    print("\n" + "="*40)
    print("         零样本(Zero-Shot)评估报告")
    print("="*40)
    print(f"  总准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数 (F1-Score): {f1:.4f}")
    print(f"\n  混淆矩阵:")
    print(f"    真负例(TN): {tn}, 假正例(FP): {fp}")
    print(f"    假负例(FN): {fn}, 真正例(TP): {tp}")
    print(f"\n  处理样本数: {len(predictions)}/{len(test_data)}")
    print("="*40)

    # 保存结果
    results = {
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
    
    output_file = "zeroshot_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估【未经微调】的VLM Verifier的零样本性能。")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="基础模型的本地路径。")
    parser.add_argument("--test_file", type=str, default="verifier_test_split.jsonl", help="用于评估的.jsonl测试文件。")
    args = parser.parse_args()
    main(args)