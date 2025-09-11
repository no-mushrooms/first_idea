import os
import json
import argparse
from tqdm import tqdm
import numpy as np

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def read_jsonl(path: str):
    """逐行读取.jsonl文件。"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main(args):
    print("--- 步骤1: 加载微调后的Planner模型 ---")
    base_model_id = "Qwen/Qwen2-7B-Instruct" # 确保与训练时一致
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map={"": 0} # 加载到主GPU
    )
    
    print(f"  [信息] 正在从 {args.adapter_path} 加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    
    print("\n--- 步骤2: 加载测试数据集 ---")
    # 注意：我们这里加载的是由convert_planner_data.py生成的对话格式文件
    # 以便能同时获取到输入(human)和标准答案(gpt)
    test_data = json.load(open(args.test_file, 'r'))
    print(f"  找到 {len(test_data)} 条测试样本。")

    print("\n--- 步骤3: 开始推理和评估 ---")
    
    # 初始化指标
    total_samples = 0
    valid_json_count = 0
    step_count_matches = 0
    step_type_correct = 0
    step_type_total = 0

    for item in tqdm(test_data, desc="正在评估Planner"):
        total_samples += 1
        
        # 1. 提取输入和标准答案
        human_prompt = item['conversations'][0]['value']
        golden_plan_str = item['conversations'][1]['value']
        
        try:
            golden_plan = json.loads(golden_plan_str)
        except json.JSONDecodeError:
            print(f"警告: 测试集中的标准答案plan无法解析，跳过样本 {item['id']}")
            continue

        # 2. 模型生成
        inputs = tokenizer(human_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 3. 解析并评估生成的plan
        try:
            # 尝试从生成文本中提取JSON部分
            json_start = generated_text.find('[')
            json_end = generated_text.rfind(']')
            if json_start != -1 and json_end != -1:
                plan_str = generated_text[json_start : json_end + 1]
                generated_plan = json.loads(plan_str)
                valid_json_count += 1

                # 评估步骤数量
                if len(generated_plan) == len(golden_plan):
                    step_count_matches += 1
                    
                    # 评估步骤类型
                    for gen_step, gold_step in zip(generated_plan, golden_plan):
                        step_type_total += 1
                        if gen_step.get('type') == gold_step.get('type'):
                            step_type_correct += 1
            else:
                # 认为没有生成有效的JSON
                pass

        except (json.JSONDecodeError, TypeError):
            # 认为没有生成有效的JSON
            pass

    print("\n--- 步骤4: 计算并显示评估结果 ---")
    
    if total_samples == 0:
        print("未能处理任何测试样本。")
        return

    # 计算最终指标
    json_validity_rate = (valid_json_count / total_samples) * 100
    step_count_match_rate = (step_count_matches / valid_json_count) * 100 if valid_json_count > 0 else 0
    step_type_accuracy = (step_type_correct / step_type_total) * 100 if step_type_total > 0 else 0

    print("\n" + "="*40)
    print("         Planner 定量评估报告")
    print("="*40)
    print(f"  总测试样本数: {total_samples}")
    print(f"  JSON有效率 (Validity Rate): {json_validity_rate:.2f}%")
    print("  --- 在JSON有效的样本中 ---")
    print(f"  步骤数量匹配率 (Step Count Match): {step_count_match_rate:.2f}%")
    print("  --- 在步骤数量匹配的样本中 ---")
    print(f"  步骤类型准确率 (Step Type Accuracy): {step_type_accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估微调后的CoT Planner性能。")
    parser.add_argument("--adapter_path", type=str, default="./qwen_planner_adapter", help="已微调的Planner LoRA适配器路径。")
    parser.add_argument("--test_file", type=str, default="data/planner_llama_factory_test.json", help="用于评估的、LLaMA-Factory格式的.json测试文件。")
    args = parser.parse_args()
    main(args)