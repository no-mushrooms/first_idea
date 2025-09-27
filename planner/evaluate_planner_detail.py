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
    
    # 方法1：查找 "Plan:" 标记后的JSON数组
    plan_marker = "Plan:"
    marker_pos = text.find(plan_marker)
    if marker_pos != -1:
        search_text = text[marker_pos + len(plan_marker):].strip()
        json_start = search_text.find('[')
        json_end = search_text.rfind(']')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                plan_str = search_text[json_start : json_end + 1]
                return json.loads(plan_str)
            except json.JSONDecodeError:
                pass
    
    # 方法2：查找 "JSON Array:" 标记（适用于thinking格式）
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
    
    # 方法3：直接尝试解析整个文本（如果是纯JSON）
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # 方法4：使用正则表达式查找JSON数组
    json_pattern = r'\[[\s\S]*?\]'
    matches = re.findall(json_pattern, text)
    for match in reversed(matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

def build_inference_prompt(conversations):
    """构建完整的推理输入prompt"""
    prompt_parts = []
    
    for conv in conversations:
        if conv["from"] == "system":
            prompt_parts.append(conv["value"])
        elif conv["from"] == "human":
            prompt_parts.append(conv["value"])
        # 注意：不包含assistant/gpt的回复，因为那是我们要生成的
    
    return "\n\n".join(prompt_parts)

def batch_inference_single(model, tokenizer, prompts):
    """推理单个批次（不包含循环）"""
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=2048,
        padding_side='left'
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=4096,  # 增加到4096，避免截断
            do_sample=True,       # 改为True，增加多样性
            temperature=0.3,      # 降低temperature，减少重复
            top_p=0.9,           # 添加top_p采样
            repetition_penalty=1.1,  # 添加重复惩罚
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            # 移除num_beams=1，使用采样而不是贪心搜索
        )
    
    # 解码结果
    input_length = inputs['input_ids'].shape[1]
    results = []
    for output in outputs:
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
        
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        print(f"  [信息] 正在从 {args.adapter_path} 加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model.eval()
        
        if hasattr(model, 'merge_and_unload'):
            print("  [优化] 合并LoRA权重以提升推理速度...")
            model = model.merge_and_unload()

        print(f"  [信息] 正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            trust_remote_code=True,
            local_files_only=True,
            padding_side='left'
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 在 main 函数的模型加载后添加：
        print(f"  [信息] 模型最大位置编码: {model.config.max_position_embeddings}")
        print(f"  [信息] 词汇表大小: {model.config.vocab_size}")
        print(f"  [信息] 模型类型: {model.config.model_type}")
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print("\n--- 步骤2: 加载测试数据集 ---")
    try:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"  找到 {len(test_data)} 条测试样本。")
        
        # 如果指定了最大样本数，则限制数据
        if hasattr(args, 'max_samples') and args.max_samples is not None:
            test_data = test_data[:args.max_samples]
            print(f"  限制处理样本数为: {len(test_data)}")
            
    except Exception as e:
        print(f"❌ 测试数据加载失败: {e}")
        return

    print("\n--- 步骤3: 批量推理和评估 ---")
    
    # 预处理：提取所有prompts
    prompts = []
    valid_indices = []
    
    print("  正在预处理数据...")
    for idx, item in enumerate(tqdm(test_data, desc="提取prompts")):
        try:
            conversations = item['conversations']
            
            # 构建完整的推理输入
            inference_prompt = build_inference_prompt(conversations)
            prompts.append(inference_prompt)
            valid_indices.append(idx)
            
            # 调试：打印前几个样本的构建过程
            if idx < 3:
                print(f"\n=== 样本 {idx} 构建过程 ===")
                print("原始conversations:")
                for i, conv in enumerate(conversations):
                    print(f"  [{i}] {conv['from']}: {conv['value'][:100]}...")
                print(f"构建的推理输入: {inference_prompt[:200]}...")
                
        except Exception as e:
            print(f"警告: 样本 {idx} 格式错误: {e}")
    
    print(f"  有效样本数: {len(prompts)}")
    
    # 批量推理
    batch_size = getattr(args, 'batch_size', 4)
    print(f"  使用批量大小: {batch_size}")
    
    start_time = time.time()
    generated_texts = []
    
    # 添加推理进度条
    with tqdm(total=len(prompts), desc="批量推理", unit="样本") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # 推理单个批次
            batch_results = batch_inference_single(model, tokenizer, batch_prompts)
            generated_texts.extend(batch_results)
            
            # 更新进度条
            pbar.update(len(batch_prompts))
    
    inference_time = time.time() - start_time
    
    print(f"  推理完成，耗时: {inference_time:.2f}秒")
    print(f"  平均每样本: {inference_time/len(prompts):.3f}秒")
    
    print("\n--- 步骤4: 详细评估结果 ---")
    
    # 初始化统计变量
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
    
    # 添加评估进度条
    for i, generated_text in enumerate(tqdm(generated_texts, desc="分析结果", unit="样本")):
        total_samples += 1
        item_idx = valid_indices[i]
        item = test_data[item_idx]
        
        # 打印每个样本的完整输出
        print("\n" + "="*100)
        print(f"【样本 #{total_samples}】ID: {item.get('id', f'sample_{total_samples}')}")
        print("="*100)
        
        try:
            conversations = item['conversations']
            
            # 构建输入prompt用于显示
            inference_prompt = build_inference_prompt(conversations)
            
            # 提取标准答案
            golden_plan_str = None
            for conv in conversations:
                if conv['from'] in ['gpt', 'assistant']:
                    golden_plan_str = conv['value']
                    break
            
            if golden_plan_str is None:
                print("❌ 无法找到标准答案")
                continue
            
            # 打印完整的对话结构
            print("📁 完整对话结构:")
            print("-" * 50)
            for j, conv in enumerate(conversations):
                print(f"[{j}] {conv['from']}: {conv['value'][:150]}{'...' if len(conv['value']) > 150 else ''}")
            print()
            
            # 打印构建的推理输入
            print("📝 构建的推理输入:")
            print("-" * 50)
            print(inference_prompt)
            print()
            
            # 打印模型完整输出
            print("🤖 模型完整输出:")
            print("-" * 50)
            print(generated_text)
            print()
            
            # 打印标准答案
            print("✅ 标准答案:")
            print("-" * 50)
            print(golden_plan_str)
            print()
            
            # 解析golden plan
            golden_plan = extract_json_from_text(golden_plan_str)
            if golden_plan is None:
                print("❌ 标准答案JSON解析失败")
                print(f"标准答案原文: {golden_plan_str[:300]}...")
                golden_plan_failures.append({
                    "id": item.get('id', f"sample_{total_samples}"),
                    "error_type": "golden_plan_parse_error",
                    "prompt": inference_prompt[:200] + "..." if len(inference_prompt) > 200 else inference_prompt,
                    "golden_plan_str": golden_plan_str,
                    "error_msg": "无法解析标准答案的JSON格式"
                })
                continue

            # 解析生成的plan
            generated_plan = extract_json_from_text(generated_text)
            
            if generated_plan is not None:
                valid_json_count += 1
                print("✅ 模型输出JSON解析成功")
                print(f"📋 解析后的JSON:")
                print(json.dumps(generated_plan, indent=2, ensure_ascii=False))
                print()
                
                print(f"📊 标准答案JSON:")
                print(json.dumps(golden_plan, indent=2, ensure_ascii=False))
                print()
                
                # 检查步骤数量匹配
                if len(generated_plan) == len(golden_plan):
                    step_count_matches += 1
                    print(f"✅ 步骤数量匹配: {len(generated_plan)} 步")
                    
                    # 检查步骤类型匹配
                    step_types_match = True
                    type_mismatch_details = []
                    
                    print("🔍 步骤类型对比:")
                    for idx, (gen_step, gold_step) in enumerate(zip(generated_plan, golden_plan)):
                        step_type_total += 1
                        if isinstance(gen_step, dict) and isinstance(gold_step, dict):
                            gen_type = gen_step.get('type')
                            gold_type = gold_step.get('type')
                            gen_op = gen_step.get('op', 'N/A')
                            gold_op = gold_step.get('op', 'N/A')
                            
                            if gen_type == gold_type:
                                step_type_correct += 1
                                print(f"  步骤 {idx+1}: ✅ type={gen_type}, op={gen_op} vs {gold_op}")
                            else:
                                step_types_match = False
                                print(f"  步骤 {idx+1}: ❌ type={gen_type} != {gold_type}, op={gen_op} vs {gold_op}")
                                type_mismatch_details.append({
                                    "step_index": idx,
                                    "generated_type": gen_type,
                                    "golden_type": gold_type,
                                    "generated_op": gen_op,
                                    "golden_op": gold_op
                                })
                        else:
                            print(f"  步骤 {idx+1}: ❌ 格式错误 - gen: {type(gen_step)}, gold: {type(gold_step)}")
                    
                    if step_types_match:
                        print("🎉 评估结果: 完美匹配!")
                        successful_cases.append({
                            "id": item.get('id', f"sample_{total_samples}"),
                            "generated_plan": generated_plan,
                            "golden_plan": golden_plan,
                            "match_status": "perfect_match"
                        })
                    else:
                        print("⚠️ 评估结果: 步骤类型不匹配")
                        step_type_mismatches.append({
                            "id": item.get('id', f"sample_{total_samples}"),
                            "error_type": "step_type_mismatch",
                            "generated_plan": generated_plan,
                            "golden_plan": golden_plan,
                            "mismatch_details": type_mismatch_details
                        })
                        
                else:
                    print(f"❌ 步骤数量不匹配: 生成 {len(generated_plan)} 步, 标准 {len(golden_plan)} 步")
                    step_count_mismatches.append({
                        "id": item.get('id', f"sample_{total_samples}"),
                        "error_type": "step_count_mismatch",
                        "generated_count": len(generated_plan),
                        "golden_count": len(golden_plan),
                        "generated_plan": generated_plan,
                        "golden_plan": golden_plan
                    })
            else:
                print("❌ 模型输出JSON解析失败")
                print(f"原始输出前500字符: {generated_text[:500]}...")
                
                # 尝试分析为什么解析失败
                if "Do not include any other text" in generated_text:
                    print("🔍 检测到重复的负面指令 - 可能的训练数据污染或过拟合")
                elif len(generated_text.strip()) == 0:
                    print("🔍 模型输出为空")
                elif "[" not in generated_text and "]" not in generated_text:
                    print("🔍 输出中没有找到JSON数组标记")
                else:
                    print("🔍 可能的JSON格式错误")
                
                json_parse_failures.append({
                    "id": item.get('id', f"sample_{total_samples}"),
                    "error_type": "json_parse_failure",
                    "generated_text": generated_text,
                    "error_msg": "无法从生成文本中提取有效的JSON"
                })
                
        except Exception as e:
            print(f"❌ 处理样本时出错: {str(e)}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            processing_errors.append({
                "id": item.get('id', f"sample_{total_samples}"),
                "error_type": "processing_error",
                "error_msg": str(e),
                "generated_text": generated_text if 'generated_text' in locals() else "N/A"
            })
            continue
        
        # 添加暂停选项（如果启用交互模式）
        if hasattr(args, 'interactive') and args.interactive:
            user_input = input("\n按回车键继续下一个样本，输入 'q' 退出，输入 's' 跳过剩余详细输出: ")
            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 's':
                args.interactive = False

    # 计算最终指标
    json_validity_rate = (valid_json_count / total_samples) * 100 if total_samples > 0 else 0
    step_count_match_rate = (step_count_matches / valid_json_count) * 100 if valid_json_count > 0 else 0
    step_type_accuracy = (step_type_correct / step_type_total) * 100 if step_type_total > 0 else 0

    print("\n" + "="*80)
    print("                    Planner 详细评估报告")
    print("="*80)
    print(f"  总测试样本数: {total_samples}")
    print(f"  推理总耗时: {inference_time:.2f}秒")
    print(f"  平均每样本: {inference_time/total_samples:.3f}秒")
    print(f"  JSON有效率: {json_validity_rate:.2f}% ({valid_json_count}/{total_samples})")
    if valid_json_count > 0:
        print(f"  步骤数量匹配率: {step_count_match_rate:.2f}% ({step_count_matches}/{valid_json_count})")
    if step_type_total > 0:
        print(f"  步骤类型准确率: {step_type_accuracy:.2f}% ({step_type_correct}/{step_type_total})")
    print(f"  完美匹配案例: {len(successful_cases)} 条")
    print("="*80)

    # 错误类型统计
    print("\n--- 错误类型统计 ---")
    print(f"  标准答案解析错误: {len(golden_plan_failures)} 条")
    print(f"  模型JSON解析失败: {len(json_parse_failures)} 条")
    print(f"  步骤数量不匹配: {len(step_count_mismatches)} 条")
    print(f"  步骤类型不匹配: {len(step_type_mismatches)} 条")
    print(f"  其他处理错误: {len(processing_errors)} 条")
    print(f"  完美匹配案例: {len(successful_cases)} 条")

    # 保存详细结果
    detailed_results = {
        "summary": {
            "total_samples": total_samples,
            "inference_time": inference_time,
            "avg_time_per_sample": inference_time / total_samples if total_samples > 0 else 0,
            "json_validity_rate": json_validity_rate,
            "step_count_match_rate": step_count_match_rate,
            "step_type_accuracy": step_type_accuracy,
            "perfect_matches": len(successful_cases)
        },
        "error_analysis": {
            "golden_plan_failures": len(golden_plan_failures),
            "json_parse_failures": len(json_parse_failures),
            "step_count_mismatches": len(step_count_mismatches),
            "step_type_mismatches": len(step_type_mismatches),
            "processing_errors": len(processing_errors)
        },
        "detailed_errors": {
            "golden_plan_failures": golden_plan_failures,
            "json_parse_failures": json_parse_failures,
            "step_count_mismatches": step_count_mismatches,
            "step_type_mismatches": step_type_mismatches,
            "processing_errors": processing_errors
        },
        "successful_cases": successful_cases
    }
    
    output_file = args.test_file.replace('.json', '_complete_evaluation_results_cleanData.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"📊 完整评估结果已保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="详细评估微调后的CoT Planner性能（修复版）。")
    parser.add_argument("--model_path", type=str, required=True, help="基础语言模型的本地路径。")
    parser.add_argument("--adapter_path", type=str, required=True, help="已微调的Planner LoRA适配器路径。")
    parser.add_argument("--test_file", type=str, required=True, help="测试文件路径。")
    parser.add_argument("--batch_size", type=int, default=4, help="批量推理大小，根据显存调整。")
    parser.add_argument("--interactive", action="store_true", help="交互模式，每个样本后暂停")
    parser.add_argument("--max_samples", type=int, default=None, help="限制处理的样本数量，用于快速测试")
    args = parser.parse_args()
    main(args)