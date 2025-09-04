# -*- coding: utf-8 -*-
"""
本脚本用于处理A-OKVQA数据集，并利用一个强大的“教师模型”（如OpenAI GPT-4o）
来为您的kbvqa_modular_cot.py脚本生成高质量的'plan'训练数据。
[最终调试版 - 硬崩溃模式，用于获取完整Traceback]
"""
import os
import json
import argparse
import time
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
import re
import demjson3

try:
    from kbvqa_modular_cot import HFExtractorAndGrounder
except ImportError:
    print("【错误】: 无法导入 HFExtractorAndGrounder。请确保文件在同一目录下。")
    exit()

try:
    from openai import OpenAI
except ImportError:
    print("【错误】: 无法导入 openai。请运行 'pip install openai'。")
    exit()

# 元提示模板保持不变
META_PROMPT_TEMPLATE = """You are an expert visual question answering (VQA) planner. Your task is to generate a step-by-step plan in a pure JSON array format to answer the user's question based on the provided context. The plan should be logical and break down the problem into verifiable steps.

CRITICAL INSTRUCTION: The entire output must be a valid JSON. All keys and all string values MUST be enclosed in double quotes ("). If a string value itself contains a double quote, it MUST be escaped with a backslash (e.g., "a person\"s house").

The JSON object for each step must have the following keys:
- "type": string, must be "visual" or "kb".
- "focus": object or null. For "visual" type, it must be {{"source": "detr", "idx": integer}} referring to a region from the context. It must be null for "kb" type.
- "op": string, a short verb describing the operation, like "describe_region", "verify_color", "query_knowledge".
- "hypothesis": string, a clear, self-contained statement to be verified (for visual) or queried (for kb).

CRITICAL: The entire output must be a valid JSON. All keys and all string values MUST be enclosed in double quotes ("). Do not use single quotes ('). Do not add trailing commas.

---
**FEW-SHOT EXAMPLE 1**

Context:
[0] a white frisbee on the grass
[1] a green grassy field
[2] a person's leg and shoe

Question: What material is the frisbee likely made of?

JSON Array:
[
    {{"type": "visual", "focus": {{"source": "detr", "idx": 0}}, "op": "describe_region", "hypothesis": "The object in region 0 is a frisbee"}},
    {{"type": "kb", "focus": null, "op": "query_knowledge", "hypothesis": "What material are frisbees typically made of?"}}
]
---

**YOUR TASK**

Context:
{context_string}

Question: {question_string}

JSON Array:
"""

def generate_plan_with_teacher_model(client, context_str, question_str, model_name="gpt-4o-mini"):
    """
    使用教师模型生成plan。
    [硬崩溃调试模式]: 移除了内部的try-except，让错误直接暴露。
    """
    prompt = META_PROMPT_TEMPLATE.format(
        context_string=context_str,
        question_string=question_str
    )
    
    print("\n[DEBUG] 正在向OpenAI发送Prompt...", flush=True)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw_text = response.choices[0].message.content
    
    print(f"[DEBUG] 从OpenAI收到原始回复:\n---\n{raw_text}\n---", flush=True)
    
    json_start = raw_text.find('[')
    json_end = raw_text.rfind(']')
    
    if json_start != -1 and json_end != -1:
        plan_str_raw  = raw_text[json_start : json_end + 1]
        
        print("[DEBUG] 正在尝试修复和解析JSON...", flush=True)
        # 修复单引号问题
        # plan_str_fixed = plan_str.replace("'", '"')
        
        # # [核心修复] 步骤2: 使用正则表达式自动修复缺失的逗号
        # # 这个表达式会查找 "} {", "}   {", "}\n{" 等情况，并在中间加上逗号
        # plan_str_fixed = re.sub(r'}\s*{', '}, {', plan_str_fixed)

        # 增强的错误捕获和调试打印
        try:
            # # 打印即将被解析的字符串，用于调试
            # print("\n[ULTRA DEBUG] 尝试解析以下(已自动修复的)JSON字符串:")
            # print("------------------------------------------")
            # print(plan_str_fixed)
            # print("------------------------------------------")
            
            # # 尝试解析
            # json.loads(plan_str_fixed)
            
            # # 如果成功，返回修复后的字符串
            # return plan_str_fixed
            # demjson3可以处理单引号、缺引号的键、结尾逗号等多种不规范格式
            plan_object = demjson3.decode(plan_str_raw)
            
            # 为了保证我们存入文件的是100%标准的JSON，我们用标准库重新序列化
            # 这是一个“清洗”和“标准化”的过程
            standard_json_str = json.dumps(plan_object)
            
            print("[成功] 已成功解析并标准化Plan。")
            return standard_json_str
        except demjson3.JSONDecodeError as e:
            # 如果解析失败，打印详细的错误信息和有问题的字符串
            print(f"\n【JSON解析失败】(demjson3也无法处理): {e}", flush=True)
            print(f"  错误类型: {type(e)}")
            print(f"  错误详情: {e}")
            print("  导致失败的(已自动修复的)字符串是:")
            print("  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
            print(plan_str_raw)
            print("  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            
            # 返回None表示失败
            return None
    else:
        return None

def main(args):
    # 这部分代码与之前完全相同
    print("--- 步骤1: 初始化模型 ---", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_pipeline = HFExtractorAndGrounder(device=device)
    api_key = os.getenv("OPENAI_API_KEY") or args.openai_api_key
    if not api_key:
        raise ValueError("【错误】未找到OpenAI API密钥。")
    teacher_model_client =  OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key,
        base_url="https://api.chatanywhere.tech/v1"
        # base_url="https://api.chatanywhere.org/v1"
    )
    print(f"--- 步骤2: 加载A-OKVQA数据 ---", flush=True)
    with open(args.aokvqa_json_path, 'r', encoding='utf-8') as f:
        aokvqa_data = json.load(f)
    if args.limit > 0:
        aokvqa_data = aokvqa_data[:args.limit]
    print("\n--- 步骤3: 开始生成Plan数据 ---", flush=True)
    
    successful_plans = 0
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        # --- [核心调试改动] ---
        # 我们在这里也移除了外层的try-except，让任何错误都能暴露
        for i, sample in enumerate(tqdm(aokvqa_data, desc="处理A-OKVQA样本")):
            print(f"\n{'='*20} 正在处理样本 {i+1}/{len(aokvqa_data)} (ID: {sample.get('question_id', 'N/A')}) {'='*20}", flush=True)
            
            image_id_str = str(sample['image_id']).zfill(12)
            image_filename = f"{image_id_str}.jpg"
            image_path = Path(args.coco_images_dir) / image_filename

            if not image_path.exists():
                print(f"【警告】: 找不到图片 {image_path}，跳过。", flush=True)
                continue
                
            pil_image = Image.open(image_path).convert("RGB")
            
            regions = hf_pipeline.detect_and_caption(pil_image)
            image_captions = [r.get('caption', '').strip() for r in regions if r.get('caption')]
            
            if not image_captions:
                print("【警告】: 未能提取任何物体描述，跳过。", flush=True)
                continue

            context_string = "\n".join([f"[{i}] {cap}" for i, cap in enumerate(image_captions)])
            question_string = sample['question']
            
            generated_plan_str = generate_plan_with_teacher_model(teacher_model_client, context_string, question_string, model_name=args.openai_model)
            
            if generated_plan_str:
                training_example = {"image_captions": image_captions, "question": question_string, "plan": generated_plan_str}
                out_f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                successful_plans += 1
                print("  [成功] 已将Plan写入到输出文件。", flush=True)
            else:
                print("【警告】: 教师模型未能生成有效的Plan，跳过。", flush=True)
            
            time.sleep(1)

    print(f"\n--- 步骤4: 处理完成 ---", flush=True)
    print(f"  [总结] 总共处理了 {len(aokvqa_data)} 个样本。", flush=True)
    print(f"  [总结] 成功生成并保存了 {successful_plans} 条Plan。", flush=True)
    print(f"  [结果] 生成的训练数据已保存到: {args.output_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为KB-VQA模型生成Plan训练数据。")
    parser.add_argument("--aokvqa_json_path", type=str, required=True)
    parser.add_argument("--coco_images_dir", type=str, required=True)
    parser.add_argument("--coco_split", type=str, default="train2017")
    parser.add_argument("--output_file", type=str, default="planner_train_generated.jsonl")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    main(args)
