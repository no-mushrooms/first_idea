import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import time

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

def batch_inference(model, tokenizer, prompts, batch_size=4):
    """批量推理函数"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # 批量tokenize
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048,
            padding_side='left'     # 🚀 明确指定左填充
        ).to(model.device)
        
        with torch.no_grad():
            # 🚀 优化的生成配置
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512,      # 🔧 减少生成长度
                do_sample=False, 
                # temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                # 🚀 性能优化参数
                use_cache=True,          # 启用KV缓存
                num_beams=1,            # 单束搜索
                # early_stopping=True,     # JSON完成即停止
                # 🛑 停止条件
                eos_token_id=tokenizer.eos_token_id,
                # pad_token_id=tokenizer.pad_token_id,
            )
        
        # 解码结果
        input_length = inputs['input_ids'].shape[1]
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(
                output[input_length:], 
                skip_special_tokens=True
            )
            results.append(generated_text)
    
    return results

def main(args):
    print("--- 步骤1: 加载微调后的Planner模型 ---")
    
    try:
        print(f"  [信息] 正在从本地路径 {args.model_path} 加载基础模型...")
        
        # 🚀 优化的模型加载配置
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,  # 或 torch.bfloat16 如果支持
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            # 🚀 性能优化
            low_cpu_mem_usage=True,     # 减少CPU内存使用
            # torch_compile=False,        # 避免编译开销（可选）
        )
        
        print(f"  [信息] 正在从 {args.adapter_path} 加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model.eval()
        
        # 🚀 模型优化
        if hasattr(model, 'merge_and_unload'):
            print("  [优化] 合并LoRA权重以提升推理速度...")
            model = model.merge_and_unload()

        print(f"  [信息] 正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            trust_remote_code=True,
            local_files_only=True,
            padding_side='left'     # 🚀 修复：设置左填充
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

    print("\n--- 步骤3: 批量推理和评估 ---")
    
    # 🚀 预处理：提取所有prompts
    prompts = []
    valid_indices = []
    
    for idx, item in enumerate(test_data):
        try:
            human_prompt = item['conversations'][0]['value']
            prompts.append(human_prompt)
            valid_indices.append(idx)
        except Exception as e:
            print(f"警告: 样本 {idx} 格式错误: {e}")
    
    print(f"  有效样本数: {len(prompts)}")
    
    # 🚀 批量推理
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 4
    print(f"  使用批量大小: {batch_size}")
    
    start_time = time.time()
    generated_texts = batch_inference(model, tokenizer, prompts, batch_size)
    inference_time = time.time() - start_time
    
    print(f"  推理完成，耗时: {inference_time:.2f}秒")
    print(f"  平均每样本: {inference_time/len(prompts):.3f}秒")
    
    # 评估逻辑（保持不变）
    total_samples = 0
    valid_json_count = 0
    step_count_matches = 0
    step_type_correct = 0
    step_type_total = 0
    
    # 详细的失败案例分类
    json_parse_failures = []
    golden_plan_failures = []
    step_count_mismatches = []
    step_type_mismatches = []
    processing_errors = []
    successful_cases = []

    print("\n--- 步骤4: 评估结果 ---")
    
    for i, generated_text in enumerate(tqdm(generated_texts, desc="分析结果")):
        total_samples += 1
        item_idx = valid_indices[i]
        item = test_data[item_idx]
        
        try:
            human_prompt = item['conversations'][0]['value']
            golden_plan_str = item['conversations'][1]['value']
            
            # 解析golden plan
            golden_plan = extract_json_from_text(golden_plan_str)
            if golden_plan is None:
                golden_plan_failures.append({
                    "id": item.get('id', f"sample_{total_samples}"),
                    "error_type": "golden_plan_parse_error",
                    "prompt": human_prompt[:200] + "..." if len(human_prompt) > 200 else human_prompt,
                    "golden_plan_str": golden_plan_str,
                    "error_msg": "无法解析标准答案的JSON格式"
                })
                continue

            # 解析生成的plan
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
                                    "golden_type": gold_type
                                })
                    
                    if step_types_match:
                        successful_cases.append({
                            "id": item.get('id', f"sample_{total_samples}"),
                            "generated_plan": generated_plan,
                            "golden_plan": golden_plan,
                            "match_status": "perfect_match"
                        })
                    else:
                        step_type_mismatches.append({
                            "id": item.get('id', f"sample_{total_samples}"),
                            "error_type": "step_type_mismatch",
                            "generated_plan": generated_plan,
                            "golden_plan": golden_plan,
                            "mismatch_details": type_mismatch_details
                        })
                        
                else:
                    step_count_mismatches.append({
                        "id": item.get('id', f"sample_{total_samples}"),
                        "error_type": "step_count_mismatch",
                        "generated_count": len(generated_plan),
                        "golden_count": len(golden_plan)
                    })
            else:
                json_parse_failures.append({
                    "id": item.get('id', f"sample_{total_samples}"),
                    "error_type": "json_parse_failure",
                    "generated_text": generated_text[:200] + "...",
                    "error_msg": "无法从生成文本中提取有效的JSON"
                })
                
        except Exception as e:
            processing_errors.append({
                "id": item.get('id', f"sample_{total_samples}"),
                "error_type": "processing_error",
                "error_msg": str(e)
            })
            continue

    # 计算和打印结果（保持原有逻辑）
    json_validity_rate = (valid_json_count / total_samples) * 100 if total_samples > 0 else 0
    step_count_match_rate = (step_count_matches / valid_json_count) * 100 if valid_json_count > 0 else 0
    step_type_accuracy = (step_type_correct / step_type_total) * 100 if step_type_total > 0 else 0

    print("\n" + "="*50)
    print("           Planner 快速评估报告")
    print("="*50)
    print(f"  总测试样本数: {total_samples}")
    print(f"  推理总耗时: {inference_time:.2f}秒")
    print(f"  平均每样本: {inference_time/total_samples:.3f}秒")
    print(f"  JSON有效率: {json_validity_rate:.2f}% ({valid_json_count}/{total_samples})")
    if valid_json_count > 0:
        print(f"  步骤数量匹配率: {step_count_match_rate:.2f}% ({step_count_matches}/{valid_json_count})")
    if step_type_total > 0:
        print(f"  步骤类型准确率: {step_type_accuracy:.2f}% ({step_type_correct}/{step_type_total})")
    print(f"  完美匹配案例: {len(successful_cases)} 条")
    print("="*50)

    # 保存结果（简化版）
    results = {
        "total_samples": total_samples,
        "inference_time": inference_time,
        "avg_time_per_sample": inference_time / total_samples,
        "json_validity_rate": json_validity_rate,
        "step_count_match_rate": step_count_match_rate,
        "step_type_accuracy": step_type_accuracy,
        "perfect_matches": len(successful_cases)
    }
    
    output_file = args.test_file.replace('.json', '_fast_evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📊 快速评估结果已保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="快速评估微调后的CoT Planner性能。")
    parser.add_argument("--model_path", type=str, required=True, help="基础语言模型的本地路径。")
    parser.add_argument("--adapter_path", type=str, required=True, help="已微调的Planner LoRA适配器路径。")
    parser.add_argument("--test_file", type=str, required=True, help="测试文件路径。")
    parser.add_argument("--batch_size", type=int, default=4, help="批量推理大小，根据显存调整。")
    args = parser.parse_args()
    main(args)