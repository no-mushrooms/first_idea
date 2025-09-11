import os
import json
from pathlib import Path
import argparse

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def main(args):
    print("--- 步骤1: 加载微调后的模型和处理器 ---")
    base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        local_files_only=True
    )
    
    # 加载LoRA适配器
    print(f"  [信息] 正在从 {args.adapter_path} 加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path).to("cuda")
    model.eval()

    # 加载处理器
    processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)

    print("\n--- 步骤2: 加载测试数据集 ---")
    with open(args.test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]
    print(f"  找到 {len(test_data)} 条测试样本。")

    print("\n--- 步骤3: 开始推理和评估 ---")
    predictions = []
    ground_truths = []

    for item in tqdm(test_data, desc="正在评估"):
        image_path = item['image_path']
        hypothesis = item['hypothesis']
        true_label = item['label'] # 0 or 1

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告: 找不到测试图片 {image_path}，跳过。")
            continue

        # 构建Prompt，注意这里不包含答案！
        prompt = f"Question: Is the following statement true based on the image? Statement: {hypothesis}. Answer with 'yes' or 'no'.\n\nAnswer:"
        
        # 准备模型输入
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

        # 模型生成
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=3)
        
        # 解码并解析答案
        answer_text = processor.decode(outputs[0], skip_special_tokens=True).lower()
        predicted_label = 1 if "yes" in answer_text else 0

        predictions.append(predicted_label)
        ground_truths.append(true_label)

    print("\n--- 步骤4: 计算并显示评估结果 ---")
    
    if not ground_truths:
        print("未能处理任何测试样本，无法计算指标。")
        return

    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions)
    recall = recall_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)
    cm = confusion_matrix(ground_truths, predictions)

    print("\n" + "="*30)
    print("         评估报告")
    print("="*30)
    print(f"  总准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1分数 (F1-Score):  {f1:.4f}")
    print("\n  混淆矩阵 (Confusion Matrix):")
    print("    (行: 真实标签, 列: 预测标签)")
    print(f"      No(0) Yes(1)")
    print(f"  No(0) [[{cm[0][0]:<5} {cm[0][1]:<5}]]")
    print(f"  Yes(1) [[{cm[1][0]:<5} {cm[1][1]:<5}]]")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估微调后的VLM Verifier性能。")
    parser.add_argument("--model_path", type=str, default="vikhyatk/moondream2", help="基础模型(moondream2)的本地路径。")
    parser.add_argument("--adapter_path", type=str, default="/home/xuyan/workspace/first_idea/vlm_verifier_adapter", help="已微调的LoRA适配器路径。")
    parser.add_argument("--test_file", type=str, default="verifier_test_split.jsonl", help="用于评估的.jsonl测试文件。")
    args = parser.parse_args()
    main(args)