import json
import argparse
from tqdm import tqdm


def convert_planner_data_to_llamafactory(input_file, output_file):
    """
    将新格式的planner数据转换为LLaMA-Factory微调所需的格式
    
    输入格式：
    {
        "image_captions": [...],
        "question": "...",
        "thinking": "...",
        "plan": "[{...}]"
    }
    
    输出格式：
    {
        "id": "planner_0",
        "conversations": [
            {
                "from": "human",
                "value": "Context:\n[0] ...\n[1] ...\n\nQuestion: ..."
            },
            {
                "from": "gpt", 
                "value": "JSON Array:\n[{...}]"
            }
        ]
    }
    """
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="转换数据")):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                
                # 构建context字符串
                context_parts = []
                for idx, caption in enumerate(item['image_captions']):
                    context_parts.append(f"[{idx}] {caption}")
                context_str = "\n".join(context_parts)
                
                # 构建human prompt
                human_prompt = f"Context:\n{context_str}\n\nQuestion: {item['question']}"
                
                # 构建assistant response (直接使用plan字段)
                plan_str = item['plan']
                # 确保plan是正确的JSON字符串格式
                if isinstance(plan_str, str):
                    # 验证是否为有效JSON
                    try:
                        json.loads(plan_str)  # 验证JSON有效性
                        assistant_response = f"JSON Array:\n{plan_str}"
                    except json.JSONDecodeError:
                        print(f"警告: 样本 {i} 的plan字段不是有效JSON，跳过")
                        continue
                else:
                    # 如果plan不是字符串，转换为JSON字符串
                    assistant_response = f"JSON Array:\n{json.dumps(plan_str)}"
                
                # 构建LLaMA-Factory格式
                converted_item = {
                    "id": f"planner_{i}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": assistant_response
                        }
                    ]
                }
                
                converted_data.append(converted_item)
                
            except json.JSONDecodeError as e:
                print(f"警告: 跳过第 {i} 行，JSON解析错误: {e}")
                continue
            except KeyError as e:
                print(f"警告: 跳过第 {i} 行，缺少必要字段: {e}")
                continue
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成！")
    print(f"输入样本数: {i+1}")
    print(f"成功转换: {len(converted_data)}")
    print(f"输出文件: {output_file}")
    
def read_jsonl(path: str):
    """逐行读取.jsonl文件。"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def convert_planner_data_with_thinking_as_instruction(input_file, output_file):
    """
    将thinking作为system指令的三段式对话格式转换版本
    
    格式：
    System: 角色定义 
    Human: 纯净的Context + Question  
    Assistant: thinking推理过程+纯JSON计划输出
    """
    converted_data = []
    failed_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="转换数据（包含thinking）")):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                
                # 构建context字符串
                context_parts = []
                for idx, caption in enumerate(item['image_captions']):
                    context_parts.append(f"[{idx}] {caption}")
                context_str = "\n".join(context_parts)
                
                # 获取thinking内容
                thinking_text = item.get('thinking', '')
                
                
                # 构建system prompt（角色定义 + thinking指导）
                system_prompt = """You are an AI assistant that generates structured reasoning plans.

                    Your entire response MUST be a single, syntactically correct JSON object. This object must contain exactly two keys: "Analysis" and "Plan". Do not output any text, explanations, or code formatting before or after the JSON object.

                    **1. `Analysis` Key Rules:**
                    The value must be a string that explains the step-by-step reasoning process. You must:
                    - Deconstruct the user's question into its simplest parts.
                    - Identify which information must be verified visually (`visual`).
                    - Identify which information requires external knowledge (`kb`).

                    **2. `Plan` Key Rules:**
                    The value must be a STRING containing a valid JSON array.
                    - The plan must contain steps to FIND an answer, not the answer itself. Assume you have no prior knowledge.
                    - Each step in the array is an object with four keys: `type`, `focus`, `op`, `hypothesis`.

                    **3. CRITICAL RULE: `hypothesis` Formatting**
                    The format of the `hypothesis` string is STRICTLY determined by the `type` key.
                    - If `type` is **"visual"**: The `hypothesis` MUST be a **declarative statement** to be verified.
                    - Correct Example: "The object in region 0 is a teddy bear."
                    - If `type` is **"kb"**: The `hypothesis` MUST be a **neutral question** to be queried.
                    - Correct Example: "What demographic is typically associated with teddy bears?"

                    **Do NOT mix these formats.** Do not use a question for "visual" or a statement for "kb".
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
                    **Summary of Your Task:**
                    - Receive user input (Context and Question).
                    - Output a single JSON object with "thinking" and "plan" keys.
                    - Follow all formatting rules precisely.
                """

                # 构建human prompt（纯净的输入）
                human_prompt = f"""Context:\n{context_str}\n\nQuestion: {item['question']}"""

                # 构建assistant response（纯JSON输出）
                plan_str = item['plan']
                
                # 确保plan是正确的JSON字符串格式
                if isinstance(plan_str, str):
                    # 验证是否为有效JSON
                    try:
                        json.loads(plan_str)  # 验证JSON有效性
                        assistant_response = f"JSON Array:\n{plan_str}"
                    except json.JSONDecodeError:
                        print(f"警告: 样本 {i} 的plan字段不是有效JSON，跳过")
                        continue
                else:
                    # 如果plan不是字符串，转换为JSON字符串
                    assistant_response = f"JSON Array:\n{json.dumps(plan_str)}"
                
                # 组合thinking和plan
                assistant_response = f"Analysis:\n{thinking_text}\n\nPlan:\n{plan_str}"
                
                # 构建LLaMA-Factory三段式对话格式
                converted_item = {
                    "id": f"planner_{i}",
                    "conversations": [
                        {
                            "from": "system",
                            "value": system_prompt
                        },
                        {
                            "from": "human", 
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": assistant_response
                        }
                    ]
                }
                
                converted_data.append(converted_item)
                
            except json.JSONDecodeError as e:
                print(f"警告: 跳过第 {i} 行，JSON解析错误: {e}")
                continue
            except KeyError as e:
                print(f"警告: 跳过第 {i} 行，缺少必要字段: {e}")
                continue
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成！")
    print(f"输入样本数: {i+1}")
    print(f"成功转换: {len(converted_data)}")
    print(f"输出文件: {output_file}")
    
    # 如果有失败的样本，保存详细信息
    # if failed_samples:
    #     error_file = output_file.replace('.json', '_errors.json')
    #     with open(error_file, 'w', encoding='utf-8') as f:
    #         json.dump(failed_samples, f, indent=2, ensure_ascii=False)
    #     print(f"失败样本详情保存到: {error_file}")
    
def convert_planner_data_with_thinking(input_file, output_file):
    """
    包含thinking过程的转换版本（原版本保留）
    这个版本在assistant response中也包含thinking过程
    """
    converted_data = []
    failed_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="转换数据（包含thinking）")):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                
                # 构建context字符串
                context_parts = []
                for idx, caption in enumerate(item['image_captions']):
                    context_parts.append(f"[{idx}] {caption}")
                context_str = "\n".join(context_parts)
                
                # 构建human prompt
                human_prompt = f"Context:\n{context_str}\n\nQuestion: {item['question']}"
                
                # 构建包含thinking的assistant response
                thinking_text = item.get('thinking', '')
                plan_str = item['plan']
                
                # 确保plan是正确的JSON字符串格式
                if isinstance(plan_str, str):
                    # 验证是否为有效JSON
                    try:
                        json.loads(plan_str)  # 验证JSON有效性
                        assistant_response = f"JSON Array:\n{plan_str}"
                    except json.JSONDecodeError:
                        print(f"警告: 样本 {i} 的plan字段不是有效JSON，跳过")
                        continue
                else:
                    # 如果plan不是字符串，转换为JSON字符串
                    assistant_response = f"JSON Array:\n{json.dumps(plan_str)}"
                
                # 组合thinking和plan
                assistant_response = f"Thinking:\n{thinking_text}\n\nJSON Array:\n{plan_str}"
                
                # 构建LLaMA-Factory格式
                converted_item = {
                    "id": f"planner_{i}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": assistant_response
                        }
                    ]
                }
                
                converted_data.append(converted_item)
                
            except json.JSONDecodeError as e:
                print(f"警告: 跳过第 {i} 行，JSON解析错误: {e}")
                failed_samples.append({"index": i, "error": f"Line JSON error: {e}"})
                continue
            except KeyError as e:
                print(f"警告: 跳过第 {i} 行，缺少必要字段: {e}")
                failed_samples.append({"index": i, "error": f"Missing field: {e}"})
                continue
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成！")
    print(f"输入样本数: {i+1}")
    print(f"成功转换: {len(converted_data)}")
    print(f"失败样本数: {len(failed_samples)}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转换新格式planner数据为LLaMA-Factory格式")
    parser.add_argument("--input_file", type=str,default="planner_train_split.jsonl", help="输入的jsonl文件路径")
    parser.add_argument("--output_file", type=str, default="/home/xuyan/workspace/LLaMA-Factory/data/planner_llama_factory_newPromptMini.json",help="输出的json文件路径")
    parser.add_argument("--include_thinking", action="store_true",help="是否在输出中包含thinking过程")
    parser.add_argument("--include_thinking_system", action="store_true", help="存在system指令")
    
    args = parser.parse_args()
    
    if args.include_thinking_system:
        print("thinking作为指令的转换")
        convert_planner_data_with_thinking_as_instruction(args.input_file, args.output_file)
    elif args.include_thinking:
        print("包含thinking过程的转换")
        convert_planner_data_with_thinking(args.input_file, args.output_file)
    else:
        print("不包含thinking过程的转换")
        convert_planner_data_to_llamafactory(args.input_file, args.output_file)