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
import string

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
You are a professional AI planner. Your core task is to create a step-by-step execution plan for a student AI with limited internal knowledge (e.g., a 7B model) to solve user problems.

Your output must be a single, 100% syntactically correct JSON object containing two keys: "thinking" and "plan".

---

### 1. "thinking" Module Guidelines

In this module, you need to elaborate on your reasoning and analysis process step-by-step in text format. This should include:

1. **Problem Decomposition**: Analyze the core intent of the user's query and break it down into the smallest, verifiable logical units.
    
2. **Entity Recognition**: Identify the key entities, attributes, and relationships in the image or text that require attention.
    
3. **Path Planning**: Plan a clear verification path. Generally, follow the principle of "visual confirmation first, then knowledge query." Explicitly state which information needs to be verified through visual tools and which requires external knowledge bases.
    

---

### 2. "plan" Module Guidelines

In this module, you need to provide a pure and 100% syntactically correct JSON array in string format. **Fundamental Principle**: This plan is intended to "guide" the student AI on how to find the answer step-by-step. **It is strictly forbidden to include any conclusions or final answers in the plan**. You must assume you are a planner with no world knowledge. **JSON Structure of Plan Steps**: Each step in the array is a JSON object and must contain the following keys:

- `"type"`: `(string)` The type of operation, must be either `"visual"` or `"kb"`.
    
- `"focus"`: `(object)` The focus of the operation.
    
    - When `type` is `"visual"`, the value is an object like `{"source": "detector", "idx": 0}`.
        
    - When `type` is `"kb"`, the value is `null`.
        
- `"op"`: `(string)` The specific action verb, e.g., `"verify_existence"`, `"verify_attribute"`, `"query_knowledge"`.
    
- `"hypothesis"`: `(string)` The hypothesis to be verified or queried, the format of which **strictly depends on the "type"**.
    
- **Key Rule: `hypothesis` Format**
    
- This is the most important rule and must be strictly followed:
    
- **If `type` is `"visual"`**: The `hypothesis` **must** be a complete **declarative sentence** to be verified.
    
    - **Example**: "The object in region 0 is a teddy bear."
        
- **If `type` is `"kb"`**: The `hypothesis` **must** be a neutral **interrogative sentence** to be queried.
    
    - **Example**: "With what demographic are teddy bears typically associated?" It is **absolutely forbidden** to use a declarative sentence in a `"kb"` step or an interrogative sentence in a `"visual"` step.
        

---

### Core Constraints

1. **No Reliance on Internal Knowledge**: All your planning steps must be based on information that can be explicitly derived from the image (visual) or a knowledge base (kb).
    
2. **Atomic Steps**: Ensure that each step is the smallest logical unit; do not skip any reasoning links.
    
3. **Strict Formatting**: The final output must be a single JSON object containing the "thinking" and "plan" keys.
    

---

### Few-Shot Example (Demonstrating correct decomposition and questioning)

**Context**: 
[0] A teddy bear is sitting on a bed 

**Question**: What type of person might this room belong to? 

**Your Answer (Must be a single JSON object)：**
{ 
    "thinking": "1. The user wants to infer the type of person who owns the room.\\n2. The main clue in the context is a 'teddy bear' in region 0.\\n3. I cannot answer directly. I must first confirm the key object and then query its common associations.\\n4. Step 1 (visual): I must first verify that the object in region 0 is indeed a teddy bear. This grounds my reasoning in the image.\\n5. Step 2 (kb): Once I've confirmed it's a teddy bear, I need to ask a general knowledge question about what kind of people are associated with teddy bears. This must be a question, not a statement.", 
    "plan": "[{\\"type\\": \\"visual\\", \\"focus\\": {\\"source\\": \\"detr\\", \\"idx\\": 0}, \\"op\\": \\"verify_existence\\", \\"hypothesis\\": \\"The object in region 0 is a teddy bear\\"}, {\\"type\\": \\"kb\\", \\"focus\\": null, \\"op\\": \\"query_knowledge\\", \\"hypothesis\\": \\"What type of person are teddy bears typically associated with?\\"}]" 
}

---

**NOW, YOUR TURN**

Context:
${context_string}

Question: ${question_string}

**YOUR RESPONSE(must be a single JSON object):**
"""

def generate_plan_with_teacher_model(client, context_str, question_str, model_name="gpt-4o-mini"):
    """
    使用教师模型生成plan，并返回包含thinking和plan的完整Python字典。
    [V4 - 终极健壮版]
    """
    # 使用 string.Template 替代 .format()
    template = string.Template(META_PROMPT_TEMPLATE)
    prompt = template.substitute(
        context_string=context_str,
        question_string=question_str
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
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
            return response_obj # 直接返回整个Python字典
        else:
            print(f"[ERROR] 解析出的JSON对象缺少 'thinking' 或 'plan' 键。")
            return None
            
    except json.JSONDecodeError as e:
        print(f"\n【JSON解析失败】: {e}", flush=True)
        print("  原始回复文本:\n---\n" + raw_text + "\n---")
        print("  尝试解析的子串:\n---\n" + json_str_candidate + "\n---")
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
