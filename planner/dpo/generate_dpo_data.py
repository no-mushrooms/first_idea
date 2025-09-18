import os
import json
import argparse
from tqdm import tqdm

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(args):
    print("--- 步骤1: 加载SFT微调后的学生模型 (用于生成'rejected' plan) ---")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True
    )
    
    # 加载SFT阶段的LoRA适配器
    sft_model = PeftModel.from_pretrained(base_model, args.sft_adapter_path)
    sft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter_path)
    
    print("\n--- 步骤2: 加载包含'chosen' plan的SFT源数据 ---")
    sft_data = json.load(open(args.sft_source_file, 'r', encoding='utf-8'))
    print(f"  找到 {len(sft_data)} 条源数据。")

    print("\n--- 步骤3: 开始生成DPO偏好数据集 ---")
    dpo_data = []
    
    for item in tqdm(sft_data, desc="正在生成DPO数据"):
        try:
            # 1. 提取 Prompt 和 Chosen Response
            prompt = item['conversations'][0]['value']
            chosen_response = item['conversations'][1]['value']
            
            # 2. 使用SFT学生模型生成 Rejected Response
            inputs = tokenizer(prompt, return_tensors="pt").to(sft_model.device)
            with torch.no_grad():
                outputs = sft_model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=False, 
                    temperature=0.0
                )
            
            # 从生成文本中智能提取JSON部分
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            json_start = generated_text.find('[')
            json_end = generated_text.rfind(']')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                rejected_response = generated_text[json_start : json_end + 1]
            else:
                # 如果学生模型未能生成有效JSON，我们可以用一个简单的错误plan作为rejected
                rejected_response = '[{"type": "error", "hypothesis": "Failed to generate plan"}]'

            # 3. 组装成DPO格式
            dpo_sample = {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response
            }
            dpo_data.append(dpo_sample)

        except Exception as e:
            print(f"处理样本 {item.get('id')} 时出错: {e}")
            continue

    # 4. 保存DPO数据集
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)

    print(f"\n--- 步骤4: 处理完成 ---")
    print(f"  [成功] 生成了 {len(dpo_data)} 条DPO偏好数据。")
    print(f"  [结果] DPO训练数据已保存到: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为Planner生成DPO偏好数据集。")
    parser.add_argument("--model_path", type=str, required=True, help="基础语言模型(如Qwen2-7B-Instruct)的本地路径。")
    parser.add_argument("--sft_adapter_path", type=str, required=True, help="【SFT阶段】训练好的Planner LoRA适配器路径。")
    parser.add_argument("--sft_source_file", type=str, required=True, help="用于SFT训练的、LLaMA-Factory格式的.json源文件。")
    parser.add_argument("--output_file", type=str, default="data/planner_dpo_data.json", help="输出的DPO格式.json文件。")
    args = parser.parse_args()
    main(args)