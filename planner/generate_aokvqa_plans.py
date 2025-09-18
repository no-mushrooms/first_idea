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
import traceback

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


# -------------------- [最终优化版] 元提示 (Meta-Prompt) --------------------
META_PROMPT_TEMPLATE = """
You are an expert tutor creating a lesson plan for a student AI. Your task is to first EXPLAIN your reasoning step-by-step, and THEN generate a JSON array based on that reasoning.

**CRITICAL CONTEXT:** The student AI is a 7B model with limited knowledge. Your plan must be extremely explicit and foundational.

**YOUR TASK**
First, think about how to break down the user's question into the smallest logical steps. Write down your thought process in a <thinking> block. Consider what needs to be visually confirmed first and what requires external knowledge.

Second, based on your reasoning in the <thinking> block, generate a pure, 100% syntactically correct JSON array.

Finally, output a single JSON object with two keys: "thinking" and "plan". The "thinking" key should contain your reasoning text, and the "plan" key should contain the JSON array as a string.
---
**JSON FORMAT & AVAILABLE TOOLS (OPERATIONS)**

The JSON object for each step must use the following keys and `op` values:
- "type": "visual" or "kb".
- "focus": an object with "source" and "idx" for "visual", or null for "kb".
- "hypothesis": a clear statement to be verified or queried.
- "op": A verb for the operation. Choose from the following list of available tools:
    - **"describe_region"**: Get a general description of a focused region.
    - **"verify_existence"**: Verify if a specific object exists in a region.
    - **"verify_attribute"**: Verify a specific attribute (like color, texture, state) of an object in a region. The hypothesis should be a full statement, e.g., "The bananas in region 4 are green".
    - **"extract_text"**: Extract any text or numbers visible in a region (OCR).
    - **"query_knowledge"**: Query for external, non-visual knowledge.

---
**EXAMPLE**

Context:
[0] a bunch of yellow bananas with a '5' sticker on them
[1] a bunch of green bananas

Question: What is the number written on top of the yellow bananas?

**YOUR RESPONSE (must be a single JSON object)**

{{
    "thinking": "1. The user wants a number from the \\"yellow\\" bananas.\\n2. I see two bunches of bananas, in region 0 and region 1.\\n3. First, I need to identify which bunch is yellow.\\n    - Step 1: Check if bananas in region 0 are yellow.\\n    - Step 2: Check if bananas in region 1 are yellow.\\n4. Once I confirm region 0 contains the yellow bananas, I need to extract the number from it.\\n    - Step 3: Use the text extraction tool on region 0.\\n5. This sequence of steps (verify color, then extract text) will answer the question.",
    "plan": "[{{\\"type\\": \\"visual\\", \\"focus\\": {{\\" source\\": \\"detr\\", \\"idx\\": 0}}, \\"op\\": \\"verify_attribute\\", \\"hypothesis\\": \\"The bananas in region 0 are yellow\\"}}, {{ \\" type\\": \\"visual\\", \\"focus\\": {{\\"source\\": \\"detr\\", \\"idx\\": 1}}, \\"op\\": \\"verify_attribute\\", \\"hypothesis\\": \\"The bananas in region 1 are yellow\\"}}, {{\\"type\\": \\"visual\\", \\"focus\\": {{\\"source\\": \\"detr\\", \\"idx\\": 0}}, \\"op\\": \\"extract_text\\", \\"hypothesis\\": \\"Extract the number written on the bananas in region 0\\"}}] "
}}

---

**NOW, YOUR TURN**

Context:
{context_string}

Question: {question_string}

**YOUR RESPONSE(must be a single JSON object):**
"""

def generate_plan_with_teacher_model(client, context_str, question_str, model_name="gpt-4o-mini"):
    """
    使用教师模型生成plan，并返回包含thinking和plan的完整Python字典。
    [V4 - 终极健壮版]
    """
    prompt = META_PROMPT_TEMPLATE.format(
        context_string=context_str,
        question_string=question_str
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            # 我们暂时去掉 response_format，因为我们要自己手动解析，这样更可靠
        )
        raw_text = response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] OpenAI API调用失败: {str(e)}")
        return None

    try:
        # 核心逻辑：从返回的文本中稳健地提取出JSON对象
        # 1. 找到第一个 '{'
        json_start = raw_text.find('{')
        # 2. 找到最后一个 '}'
        json_end = raw_text.rfind('}')
        
        if json_start == -1 or json_end == -1 or json_end < json_start:
            print(f"\n【JSON提取失败】: 未能在返回文本中找到有效的JSON对象边界。")
            print("  原始回复文本:\n---\n" + raw_text + "\n---")
            return None

        # 3. 提取出可能是JSON的字符串
        json_str_candidate = raw_text[json_start : json_end + 1]

        # 4. 使用标准json库进行解析
        response_obj = json.loads(json_str_candidate)
        
        # 5. 验证返回的对象是否是我们期望的格式
        if isinstance(response_obj, dict) and 'thinking' in response_obj and 'plan' in response_obj:
            # print("[成功] 已成功提取并解析包含thinking和plan的JSON对象。")
            return response_obj # 直接返回整个Python字典
        else:
            print(f"[ERROR] 解析出的JSON对象缺少 'thinking' 或 'plan' 键。")
            return None
            
    except json.JSONDecodeError as e:
        print(f"\n【JSON解析失败】: {e}", flush=True)
        print("  原始回复文本:\n---\n" + raw_text + "\n---")
        print("  尝试解析的子串:\n---\n" + json_str_candidate + "\n---")
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
    # if args.limit > 0:
    #     aokvqa_data = aokvqa_data[:args.limit]
    full_dataset_size = len(aokvqa_data)
    print(f"  发现总共 {full_dataset_size} 个样本。")
    
    # --- [核心修改] 数据集切片逻辑 ---
    start = args.start_index
    # 如果end_index未指定或为0，则处理到数据集末尾
    end = args.end_index if args.end_index is not None and args.end_index > 0 else full_dataset_size
    
    if start >= full_dataset_size:
        print(f"【错误】: 起始索引 ({start}) 超出数据集范围 ({full_dataset_size})。无需处理。")
        return

    # 对数据集进行切片
    data_to_process = aokvqa_data[start:end]
    print(f"  [处理范围] 本次将处理从索引 {start} 到 {end-1} 的 {len(data_to_process)} 个样本。")
    
    
    
    
    print("\n--- 步骤3: 开始生成Plan数据 ---", flush=True)
    
    successful_plans = 0
    
    file_mode = 'a' if args.append else 'w'
    if file_mode == 'a' and not os.path.exists(args.output_file):
        print(f"  [信息] 追加模式已启用，但输出文件不存在。将创建新文件。")
    elif file_mode == 'a':
        print(f"  [信息] 追加模式已启用，将在现有文件 {args.output_file} 末尾写入。")
    else:
        print(f"  [警告] 写入模式将覆盖现有文件 {args.output_file}。")
    
    with open(args.output_file, file_mode, encoding='utf-8') as out_f:
        for sample in tqdm(data_to_process, desc=f"处理样本 {start}-{end-1}"):
            try:
                question_id = sample.get('question_id', 'N/A')
                image_id_str = str(sample['image_id']).zfill(12)
                image_filename = f"{image_id_str}.jpg"
                image_path = Path(args.coco_images_dir) / image_filename

                if not image_path.exists():
                    tqdm.write(f"【警告】: 找不到图片 {image_path}，跳过。")
                    continue
                
                pil_image = Image.open(image_path).convert("RGB")
                regions = hf_pipeline.detect_and_caption(pil_image)
                image_captions = [r.get('caption', '').strip() for r in regions if r.get('caption')]

                if not image_captions:
                    tqdm.write("【警告】: 未能提取有效描述，跳过。")
                    continue

                context_string = "\n".join([f"[{i}] {cap}" for i, cap in enumerate(image_captions)])
                question_string = sample['question']

                # ==========================================================
                # 【核心修改】处理新的返回数据类型
                # ==========================================================
                
                # 1. 接收返回的字典，而不是字符串
                generated_data = generate_plan_with_teacher_model(
                    teacher_model_client, 
                    context_string, 
                    question_string, 
                    model_name=args.openai_model
                )

                # 2. 检查返回的字典是否有效
                if generated_data:
                    # 3. 从字典中提取 'thinking' 和 'plan' 来构建训练样本
                    #    注意：'plan' 字段本身就是我们需要的字符串格式
                    training_example = {
                        "image_captions": image_captions,
                        "question": question_string,
                        "thinking": generated_data['thinking'],
                        "plan": generated_data['plan'] 
                    }
                    out_f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                    successful_plans += 1
                else:
                    # 如果生成失败，tqdm.write会在generate_plan_...函数内部打印错误信息
                    tqdm.write(f"【警告】样本 {question_id} 未能生成有效plan，已跳过。")
                
                # ==========================================================
                
                time.sleep(1) # 速率控制
                
            except Exception as e:
                # 【核心修改】打印完整的错误堆栈信息，而不仅仅是错误消息
                print(f"\n【严重错误】处理样本 {question_id} 时发生致命错误。")
                # traceback.format_exc() 会返回一个包含完整错误信息的字符串
                traceback_str = traceback.format_exc()
                print("--- TRACEBACK START ---")
                print(traceback_str)
                print("--- TRACEBACK END ---")

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
     # --- [核心修改] 新增的命令行参数 ---
    parser.add_argument("--start_index", type=int, default=0, help="从数据集的哪个索引开始处理 (包含)。")
    parser.add_argument("--end_index", type=int, default=0, help="处理到数据集的哪个索引结束 (不包含)。如果为0或未提供，则处理到末尾。")
    parser.add_argument("--append", action="store_true", help="如果设置，则在输出文件末尾追加，而不是覆盖。")
    args = parser.parse_args()
    main(args)
