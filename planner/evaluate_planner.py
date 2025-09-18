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

def extract_json_from_text(text):
    """改进的JSON提取函数，支持thinking格式"""
    import re
    
    # 方法1：查找 "JSON Array:" 标记（适用于thinking格式）
    json_marker = "JSON Array:"
    marker_pos = text.find(json_marker)
    if marker_pos != -1:
        search_text = text[marker_pos + len(json_marker):]
        json_start = search_text.find('[')
        json_end = search_text.rfind(']')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                plan_str = search_text[json_start : json_end + 1]
                return json.loads(plan_str)
            except json.JSONDecodeError:
                pass
    
    # 方法2：直接尝试解析整个文本（如果是纯JSON）
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # 方法3：使用正则表达式查找JSON数组
    json_pattern = r'\[[\s\S]*?\]'
    matches = re.findall(json_pattern, text)
    for match in reversed(matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

def main(args):
    print("--- 步骤1: 加载微调后的Planner模型 ---")
    
    try:
        print(f"  [信息] 正在从本地路径 {args.model_path} 加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )
        
        print(f"  [信息] 正在从 {args.adapter_path} 加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model.eval()

        print(f"  [信息] 正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print("\n--- 步骤2: 加载测试数据集 ---")
    try:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"  找到 {len(test_data)} 条测试样本。")
    except Exception as e:
        print(f"❌ 测试数据加载失败: {e}")
        return

    print("\n--- 步骤3: 开始推理和评估 ---")
    
    total_samples = 0
    valid_json_count = 0
    step_count_matches = 0
    step_type_correct = 0
    step_type_total = 0
    
    # 详细的失败案例分类
    json_parse_failures = []      # JSON解析失败的案例
    golden_plan_failures = []     # 标准答案JSON格式错误的案例
    step_count_mismatches = []    # 步骤数量不匹配的案例
    step_type_mismatches = []     # 步骤类型不匹配的案例
    processing_errors = []        # 其他处理错误的案例
    
    # 新增：成功案例记录
    successful_cases = []         # 完全匹配的成功案例

    for item in tqdm(test_data, desc="正在评估Planner"):
        total_samples += 1
        
        try:
            human_prompt = item['conversations'][0]['value']
            golden_plan_str = item['conversations'][1]['value']
            
            # 使用改进的提取函数解析golden plan
            golden_plan = extract_json_from_text(golden_plan_str)
            if golden_plan is None:
                print(f"警告: 样本 {item.get('id', total_samples)} 的golden plan JSON格式错误")
                print(f"  原始内容: {golden_plan_str[:100]}...")
                golden_plan_failures.append({
                    "id": item.get('id', f"sample_{total_samples}"),
                    "error_type": "golden_plan_parse_error",
                    "prompt": human_prompt[:200] + "..." if len(human_prompt) > 200 else human_prompt,
                    "prompt_full": human_prompt,  # 保留完整prompt
                    "golden_plan_str": golden_plan_str,
                    "golden_plan_raw": golden_plan_str,  # 保留原始golden plan
                    "error_msg": "无法解析标准答案的JSON格式"
                })
                continue

            # 模型推理
            inputs = tokenizer(
                human_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1024, 
                    do_sample=False, 
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_text = tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            # 使用改进的JSON提取函数
            generated_plan = extract_json_from_text(generated_text)
            
            if generated_plan is not None:
                valid_json_count += 1
                
                # 检查步骤数量匹配
                if len(generated_plan) == len(golden_plan):
                    step_count_matches += 1
                    
                    # 检查步骤类型匹配
                    step_types_match = True
                    type_mismatch_details = []
                    
                    for idx, (gen_step, gold_step) in enumerate(zip(generated_plan, golden_plan)):
                        step_type_total += 1
                        if isinstance(gen_step, dict) and isinstance(gold_step, dict):
                            gen_type = gen_step.get('type')
                            gold_type = gold_step.get('type')
                            if gen_type == gold_type:
                                step_type_correct += 1
                            else:
                                step_types_match = False
                                type_mismatch_details.append({
                                    "step_index": idx,
                                    "generated_type": gen_type,
                                    "golden_type": gold_type,
                                    "generated_step": gen_step,
                                    "golden_step": gold_step
                                })
                    
                    # 如果步骤类型完全匹配，记录为成功案例
                    if step_types_match:
                        successful_cases.append({
                            "id": item.get('id', f"sample_{total_samples}"),
                            "prompt": human_prompt[:200] + "..." if len(human_prompt) > 200 else human_prompt,
                            "prompt_full": human_prompt,
                            "generated_plan": generated_plan,
                            "golden_plan": golden_plan,
                            "generated_text": generated_text,
                            "golden_plan_raw": golden_plan_str,
                            "step_count": len(generated_plan),
                            "match_status": "perfect_match"
                        })
                    else:
                        # 步骤类型不匹配
                        step_type_mismatches.append({
                            "id": item.get('id', f"sample_{total_samples}"),
                            "error_type": "step_type_mismatch",
                            "prompt": human_prompt[:200] + "..." if len(human_prompt) > 200 else human_prompt,
                            "prompt_full": human_prompt,
                            "generated_plan": generated_plan,
                            "golden_plan": golden_plan,
                            "generated_text": generated_text,
                            "golden_plan_raw": golden_plan_str,  # 保留原始golden plan
                            "mismatch_details": type_mismatch_details
                        })
                        
                else:
                    # 步骤数量不匹配
                    step_count_mismatches.append({
                        "id": item.get('id', f"sample_{total_samples}"),
                        "error_type": "step_count_mismatch",
                        "prompt": human_prompt[:200] + "..." if len(human_prompt) > 200 else human_prompt,
                        "prompt_full": human_prompt,
                        "generated_plan": generated_plan,
                        "golden_plan": golden_plan,
                        "generated_text": generated_text,
                        "golden_plan_raw": golden_plan_str,  # 保留原始golden plan
                        "generated_count": len(generated_plan),
                        "golden_count": len(golden_plan)
                    })
            else:
                # JSON解析失败
                json_parse_failures.append({
                    "id": item.get('id', f"sample_{total_samples}"),
                    "error_type": "json_parse_failure",
                    "prompt": human_prompt[:200] + "..." if len(human_prompt) > 200 else human_prompt,
                    "prompt_full": human_prompt,
                    "golden_plan": golden_plan,
                    "golden_plan_raw": golden_plan_str,  # 保留原始golden plan
                    "generated_text": generated_text,
                    "error_msg": "无法从生成文本中提取有效的JSON"
                })
                
        except Exception as e:
            print(f"警告: 处理样本 {total_samples} 时出错: {e}")
            processing_errors.append({
                "id": item.get('id', f"sample_{total_samples}"),
                "error_type": "processing_error",
                "error_msg": str(e),
                "prompt": human_prompt[:200] + "..." if 'human_prompt' in locals() else "无法获取prompt",
                "prompt_full": human_prompt if 'human_prompt' in locals() else "无法获取完整prompt",
                "golden_plan_raw": golden_plan_str if 'golden_plan_str' in locals() else "无法获取golden plan"
            })
            continue

    # 计算指标
    json_validity_rate = (valid_json_count / total_samples) * 100 if total_samples > 0 else 0
    step_count_match_rate = (step_count_matches / valid_json_count) * 100 if valid_json_count > 0 else 0
    step_type_accuracy = (step_type_correct / step_type_total) * 100 if step_type_total > 0 else 0

    # 打印报告
    print("\n" + "="*50)
    print("           Planner 定量评估报告")
    print("="*50)
    print(f"  总测试样本数: {total_samples}")
    print(f"  JSON有效率 (Validity Rate): {json_validity_rate:.2f}% ({valid_json_count}/{total_samples})")
    if valid_json_count > 0:
        print(f"  步骤数量匹配率 (Step Count Match): {step_count_match_rate:.2f}% ({step_count_matches}/{valid_json_count})")
    if step_type_total > 0:
        print(f"  步骤类型准确率 (Step Type Accuracy): {step_type_accuracy:.2f}% ({step_type_correct}/{step_type_total})")
    print(f"  完美匹配案例: {len(successful_cases)} 条")
    print("="*50)

    # 错误统计
    print(f"\n--- 错误类型统计 ---")
    print(f"  标准答案解析错误: {len(golden_plan_failures)} 条")
    print(f"  模型JSON解析失败: {len(json_parse_failures)} 条")
    print(f"  步骤数量不匹配: {len(step_count_mismatches)} 条")
    print(f"  步骤类型不匹配: {len(step_type_mismatches)} 条")
    print(f"  其他处理错误: {len(processing_errors)} 条")
    print(f"  完美匹配案例: {len(successful_cases)} 条")

    # 保存结果
    results = {
        "total_samples": total_samples,
        "valid_json_count": valid_json_count,
        "json_validity_rate": json_validity_rate,
        "step_count_matches": step_count_matches,
        "step_count_match_rate": step_count_match_rate,
        "step_type_correct": step_type_correct,
        "step_type_total": step_type_total,
        "step_type_accuracy": step_type_accuracy,
        "perfect_matches": len(successful_cases),
        "error_statistics": {
            "golden_plan_failures": len(golden_plan_failures),
            "json_parse_failures": len(json_parse_failures),
            "step_count_mismatches": len(step_count_mismatches),
            "step_type_mismatches": len(step_type_mismatches),
            "processing_errors": len(processing_errors),
            "successful_cases": len(successful_cases)
        }
    }
    
    output_file = args.test_file.replace('.json', '_evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📊 详细结果已保存到: {output_file}")

    # 保存详细的案例分析（包含成功案例）
    all_cases = {
        "successful_cases": successful_cases,
        "golden_plan_failures": golden_plan_failures,
        "json_parse_failures": json_parse_failures,
        "step_count_mismatches": step_count_mismatches,
        "step_type_mismatches": step_type_mismatches,
        "processing_errors": processing_errors,
        "summary": {
            "total_samples": total_samples,
            "successful_count": len(successful_cases),
            "failure_count": total_samples - len(successful_cases),
            "json_validity_rate": json_validity_rate,
            "step_count_match_rate": step_count_match_rate,
            "step_type_accuracy": step_type_accuracy
        }
    }
    
    failure_file = args.test_file.replace('.json', '_detailed_analysis.json')
    with open(failure_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)
    print(f"📋 详细案例分析（包含成功案例）已保存到: {failure_file}")

    # 显示详细的案例分析（在终端输出）
    print(f"\n" + "="*80)
    print("                    详细案例分析")
    print("="*80)
    
    # 0. 成功案例展示
    if successful_cases:
        print(f"\n【0. 完美匹配案例】({len(successful_cases)} 条)")
        for i, case in enumerate(successful_cases[:2]):  # 显示前2个成功案例
            print(f"\n  成功案例 {i+1} - ID: {case['id']}")
            print(f"  Prompt: {case['prompt']}")
            print(f"  生成计划: {case['generated_plan']}")
            print(f"  标准计划: {case['golden_plan']}")
            print(f"  步骤数量: {case['step_count']}")

    # 1. 标准答案解析错误
    if golden_plan_failures:
        print(f"\n【1. 标准答案解析错误】({len(golden_plan_failures)} 条)")
        for i, case in enumerate(golden_plan_failures[:2]):
            print(f"\n  案例 {i+1} - ID: {case['id']}")
            print(f"  错误: {case['error_msg']}")
            print(f"  Prompt: {case['prompt']}")
            print(f"  标准答案原文: {case['golden_plan_raw'][:150]}...")

    # 2. JSON解析失败
    if json_parse_failures:
        print(f"\n【2. 模型JSON解析失败】({len(json_parse_failures)} 条)")
        for i, case in enumerate(json_parse_failures[:2]):
            print(f"\n  案例 {i+1} - ID: {case['id']}")
            print(f"  Prompt: {case['prompt']}")
            print(f"  模型输出: {case['generated_text'][:200]}...")
            print(f"  标准答案: {case['golden_plan']}")
            print(f"  标准答案原文: {case['golden_plan_raw'][:150]}...")

    # 3. 步骤数量不匹配
    if step_count_mismatches:
        print(f"\n【3. 步骤数量不匹配】({len(step_count_mismatches)} 条)")
        for i, case in enumerate(step_count_mismatches[:2]):
            print(f"\n  案例 {i+1} - ID: {case['id']}")
            print(f"  Prompt: {case['prompt']}")
            print(f"  生成步骤数: {case['generated_count']}, 标准步骤数: {case['golden_count']}")
            print(f"  生成计划: {case['generated_plan']}")
            print(f"  标准计划: {case['golden_plan']}")
            print(f"  标准答案原文: {case['golden_plan_raw'][:150]}...")

    # 4. 步骤类型不匹配
    if step_type_mismatches:
        print(f"\n【4. 步骤类型不匹配】({len(step_type_mismatches)} 条)")
        for i, case in enumerate(step_type_mismatches[:2]):
            print(f"\n  案例 {i+1} - ID: {case['id']}")
            print(f"  Prompt: {case['prompt']}")
            print(f"  生成计划: {case['generated_plan']}")
            print(f"  标准计划: {case['golden_plan']}")
            print(f"  标准答案原文: {case['golden_plan_raw'][:150]}...")
            print(f"  不匹配详情:")
            for detail in case['mismatch_details'][:3]:  # 只显示前3个不匹配
                print(f"    步骤 {detail['step_index']}: 生成类型='{detail['generated_type']}', 标准类型='{detail['golden_type']}'")

    # 5. 其他处理错误
    if processing_errors:
        print(f"\n【5. 其他处理错误】({len(processing_errors)} 条)")
        for i, case in enumerate(processing_errors[:2]):
            print(f"\n  案例 {i+1} - ID: {case['id']}")
            print(f"  错误信息: {case['error_msg']}")
            print(f"  Prompt: {case['prompt']}")
            print(f"  标准答案原文: {case['golden_plan_raw'][:150]}...")

    print(f"\n" + "="*80)
    print(f"💾 完整的案例分析（包含成功案例和失败案例）已保存到文件中。")
    print(f"📈 完美匹配率: {len(successful_cases)/total_samples*100:.2f}% ({len(successful_cases)}/{total_samples})")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估微调后的CoT Planner性能。")
    parser.add_argument("--model_path", type=str, required=True, help="基础语言模型(如Qwen2-7B-Instruct)的本地路径。")
    parser.add_argument("--adapter_path", type=str, required=True, help="已微调的Planner LoRA适配器路径。")
    parser.add_argument("--test_file", type=str, required=True, help="用于评估的、LLaMA-Factory格式的.json测试文件。")
    args = parser.parse_args()
    main(args)