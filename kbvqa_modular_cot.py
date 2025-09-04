"""
kbvqa_modular_cot.py - 面向可解释与可修正推理的模块化知识视觉问答系统

这是一个先进的、模块化的KB-VQA（基于知识的视觉问答）流程。
本代码是我们多次迭代和调试的最终版本，整合了所有核心功能。

核心特性:
- 视觉感知 (HFExtractorAndGrounder): 使用DETR进行物体检测，BLIP-2进行区域描述。
- 思考链规划器 (CoTPlanner): 一个经过LoRA微调的LLM（如falcon-7b），负责生成结构化的JSON行动计划。
- 视觉验证器 (VLM_VisualVerifier): 一个经过LoRA微调的轻量级VLM（如moondream2），负责验证语言假设与视觉证据是否相符。
- 知识库桥梁 (WebSearchKnowledgeBridge): 连接外部搜索引擎API，实现开放域知识查询。
- 执行与自我修正引擎 (ModularKBVQA): 系统的总指挥，负责执行计划并在验证失败时触发再规划循环。
- 答案合成 (Answer Synthesizer): 在计划成功后，调用一个强大的LLM来综合所有证据，生成最终的自然语言答案。
- 训练流程 (train_planner): 集成了accelerate库，用于高效地微调CoT Planner。

使用示例 (推理模式):
  accelerate launch kbvqa_modular_cot.py --mode infer \
    --image ./path/to/your/image.jpg \
    --question "What is happening in the image?" \
    --planner_adapter_path ./out/planner \
    --verifier_adapter_path ./vlm_verifier_adapter

使用示例 (训练Planner模式):
  accelerate launch kbvqa_modular_cot.py --mode train_planner \
    --planner_train ./data/planner_train.jsonl \
    --output_dir ./out/planner_new
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests

# Transformers, PEFT, Accelerate, and BitsAndBytes
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    DetrImageProcessor,
    DetrForObjectDetection,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from accelerate import Accelerator
import bitsandbytes as bnb

# ----------------------------- 辅助工具 -----------------------------

def read_jsonl(path: str):
    """逐行读取.jsonl文件。"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# ----------------------------- 核心模块 -----------------------------

class HFExtractorAndGrounder:
    """
    系统的“眼睛”，负责从原始图像中提取结构化的视觉上下文。
    使用DETR进行物体检测，BLIP-2进行区域描述。
    """
    def __init__(self, device: torch.device, 
                 detr_model: str = 'facebook/detr-resnet-50', 
                 blip_model: str = 'Salesforce/blip2-opt-2.7b'):
        self.device = device
        print("  [视觉模块] 正在加载DETR模型...")
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_model)
        self.detr = DetrForObjectDetection.from_pretrained(detr_model).to(device)
        self.detr.eval()

        print("  [视觉模块] 正在加载BLIP-2模型...")
        self.blip_processor = Blip2Processor.from_pretrained(blip_model)
        self.blip = Blip2ForConditionalGeneration.from_pretrained(blip_model, torch_dtype=torch.float16).to(device)
        self.blip.eval()

    def detect_and_caption(self, pil_image: Image.Image, top_k: int = 10) -> List[Dict[str, Any]]:
        """检测物体并为每个区域生成描述。"""
        inputs = self.detr_processor(images=pil_image, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.detr(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.detr_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
        
        boxes = results['boxes']
        scores = results['scores']
        
        regions = []
        for i in range(min(len(boxes), top_k)):
            box = boxes[i].cpu().int().numpy()
            score = float(scores[i].cpu())
            x1, y1, x2, y2 = box
            
            crop = pil_image.crop((x1, y1, x2, y2))
            if crop.width < 10 or crop.height < 10: continue

            # 为裁剪出的区域生成描述
            blip_inputs = self.blip_processor(images=crop, return_tensors='pt').to(self.device, torch.float16)
            with torch.no_grad():
                gids = self.blip.generate(**blip_inputs, max_new_tokens=20)
            caption = self.blip_processor.decode(gids[0], skip_special_tokens=True).strip()
            
            if caption:
                regions.append({'bbox': [x1, y1, x2, y2], 'score': score, 'crop': crop, 'caption': caption})
        
        return regions

class WebSearchKnowledgeBridge:
    """
    系统的“外部专家”，连接搜索引擎API以实现开放域知识查询。
    示例使用Serper API。
    """
    def __init__(self):
        self.api_key = os.environ.get("SERPER_API_KEY")
        if not self.api_key:
            print("【警告】: 未找到 SERPER_API_KEY 环境变量。知识库查询将不可用。")
        self.search_url = "https://google.serper.dev/search"

    def query(self, hypothesis: str, top_k: int = 3) -> List[str]:
        if not self.api_key or not hypothesis:
            return ["知识库不可用或查询为空。"]

        payload = json.dumps({"q": hypothesis})
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}

        try:
            response = requests.post(self.search_url, headers=headers, data=payload, timeout=5)
            response.raise_for_status()
            search_results = response.json()
            
            if "answerBox" in search_results and search_results["answerBox"].get("snippet"):
                return [search_results["answerBox"]["snippet"]]
            
            return [r.get('snippet', '') for r in search_results.get('organic', [])[:top_k]]
        except requests.exceptions.RequestException as e:
            print(f"【网络错误】: 调用搜索引擎API失败: {e}")
            return [f"网络查询失败: {e}"]

class VLM_VisualVerifier:
    """
    系统的“事实核查员”，一个经过微调的轻量级VLM。
    负责加载并使用微调后的moondream2 LoRA适配器。
    """
    def __init__(self, adapter_path: str, device: torch.device):
        self.device = device
        base_model_id = "vikhyatk/moondream2"
        
        print(f"  [Verifier] 正在从缓存加载基础模型: {base_model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16,
            local_files_only=True # 假设模型已在服务器缓存
        )
        
        print(f"  [Verifier] 正在加载LoRA适配器自: {adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
        self.processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
        self.model.eval()

    def verify(self, image_crop: Image.Image, hypothesis: str) -> bool:
        prompt = f"Question: Is the following statement true based on the image? Statement: {hypothesis}. Answer with 'yes' or 'no'.\n\nAnswer:"
        
        inputs = self.processor(images=image_crop, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=3)
        
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).lower()
        return "yes" in answer

class CoTPlanner:
    """
    系统的“战术大脑”，一个经过微调的LLM。
    负责加载并使用微调后的falcon-7b LoRA适配器。
    """
    def __init__(self, adapter_path: str, device: torch.device):
        self.device = device
        base_model_id = "tiiuae/falcon-7b"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        print(f"  [Planner] 正在从缓存加载量化的基础模型: {base_model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=True
        )

        print(f"  [Planner] 正在加载LoRA适配器自: {adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.model.eval()

    def generate_plan(self, question: str, region_caps: List[str]) -> Optional[List[Dict]]:
        # 构建Prompt
        parts = [f"Question: {question}", "Context: Region captions:"]
        for i, c in enumerate(region_caps):
            parts.append(f"[{i}] {c}")
        instruction = "Please output a pure JSON array of action steps. Each step is an object with 'type', 'focus', 'op', and 'hypothesis'."
        prompt = '\n'.join(parts) + '\n' + instruction + '\nJSON Array:'

        # 生成文本
        tok = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**tok, max_new_tokens=512, do_sample=False, temperature=0.0)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        
        # 提取并解析JSON
        try:
            json_start = text.find('[')
            json_end = text.rfind(']')
            if json_start != -1 and json_end != -1:
                plan_str = text[json_start : json_end + 1]
                return json.loads(plan_str)
            return None
        except json.JSONDecodeError:
            print(f"【警告】: Planner生成的plan无法被解析为JSON: {text}")
            return None

class AnswerSynthesizer:
    """
    最终的“总结者”，负责根据所有证据生成自然语言答案。
    """
    def __init__(self, model_name: str = "tiiuae/falcon-7b", device: torch.device = None):
        # 为了节省资源，可以复用Planner的模型，或者使用API
        # 这里为了演示，我们假设复用Planner
        pass

    def synthesize(self, planner: CoTPlanner, question: str, visual_facts: List[str], knowledge_snippets: List[str]) -> str:
        prompt_parts = ["Synthesize a concise answer for the question based on the following verified facts."]
        prompt_parts.append(f"\n**Original Question:**\n{question}")
        
        if visual_facts:
            prompt_parts.append("\n**Verified Visual Facts from the Image:**")
            for fact in visual_facts:
                prompt_parts.append(f"- {fact}")
        
        if knowledge_snippets:
            prompt_parts.append("\n**Retrieved Knowledge Snippets from the Web:**")
            for i, snippet in enumerate(knowledge_snippets):
                prompt_parts.append(f"- Snippet {i+1}: \"{snippet}\"")
        
        prompt_parts.append("\n**Final Answer:**")
        prompt = "\n".join(prompt_parts)

        # 调用LLM进行最终推理
        tok = planner.tokenizer(prompt, return_tensors='pt').to(planner.device)
        with torch.no_grad():
            out_ids = planner.model.generate(**tok, max_new_tokens=50, do_sample=False, temperature=0.0)
        
        # 清理输出
        full_text = planner.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        answer_start = full_text.find("Final Answer:")
        if answer_start != -1:
            return full_text[answer_start + len("Final Answer:"):].strip()
        else:
            # 如果找不到标记，返回原始生成的一部分
            return full_text.replace(prompt, "").strip()


class ModularKBVQA:
    """
    系统的“总指挥”，编排所有模块完成推理任务。
    """
    def __init__(self, planner: CoTPlanner, verifier: VLM_VisualVerifier, bridge: WebSearchKnowledgeBridge, synthesizer: AnswerSynthesizer, perception: HFExtractorAndGrounder):
        self.planner = planner
        self.verifier = verifier
        self.bridge = bridge
        self.synthesizer = synthesizer
        self.perception = perception

    def infer(self, pil_image: Image.Image, question: str, max_replans: int = 2):
        print("\n--- [阶段1] 视觉感知 ---")
        regions = self.perception.detect_and_caption(pil_image)
        if not regions:
            return {"error": "无法从图片中提取任何有效区域。"}
        
        region_caps = [r['caption'] for r in regions]
        print(f"  提取到 {len(region_caps)} 个区域描述: {region_caps}")

        # --- 自我修正循环 ---
        for attempt in range(max_replans + 1):
            print(f"\n--- [阶段2 - 尝试 {attempt+1}/{max_replans+1}] 思考链规划 ---")
            
            action_plan = self.planner.generate_plan(question, region_caps)
            
            if not action_plan:
                print("  [失败] Planner未能生成有效的行动计划。")
                if attempt == max_replans: break
                region_caps.append("Planner failed to generate a plan, need to reconsider.") # 增加失败反馈
                continue

            print("  [成功] Planner生成了行动计划:")
            for i, step in enumerate(action_plan):
                print(f"    步骤 {i+1}: {step}")
            
            # --- [阶段3] 执行与验证 ---
            print("\n--- [阶段3] 计划执行与验证 ---")
            is_plan_successful = True
            execution_results = {"visual_facts": [], "knowledge_snippets": []}

            for i, step in enumerate(action_plan):
                print(f"  正在执行步骤 {i+1}: {step['op']}")
                step_type = step.get('type')
                hypothesis = step.get('hypothesis')
                
                if step_type == 'visual':
                    focus_idx = step.get('focus', {}).get('idx')
                    if focus_idx is not None and 0 <= focus_idx < len(regions):
                        crop = regions[focus_idx]['crop']
                        is_verified = self.verifier.verify(crop, hypothesis)
                        print(f"    - Verifier对 '{hypothesis}' 的验证结果: {'通过' if is_verified else '失败'}")
                        if is_verified:
                            execution_results["visual_facts"].append(hypothesis)
                        else:
                            is_plan_successful = False
                            # 为再规划提供失败反馈
                            region_caps.append(f"Feedback: The hypothesis '{hypothesis}' was proven false by visual verification.")
                            break 
                    else:
                        is_plan_successful = False
                        region_caps.append(f"Feedback: The plan referred to an invalid region index {focus_idx}.")
                        break

                elif step_type == 'kb':
                    snippets = self.bridge.query(hypothesis)
                    print(f"    - 知识库为 '{hypothesis}' 返回了 {len(snippets)} 条信息。")
                    if snippets:
                        execution_results["knowledge_snippets"].extend(snippets)
                    # KB查询失败通常不中断计划，但可以根据需要添加逻辑

            if is_plan_successful:
                print("\n--- [成功] 所有计划步骤均已成功执行！ ---")
                
                # --- [阶段4] 答案合成 ---
                print("\n--- [阶段4] 正在合成最终答案 ---")
                final_answer = self.synthesizer.synthesize(
                    planner=self.planner, # 可以复用planner的LLM
                    question=question,
                    visual_facts=execution_results["visual_facts"],
                    knowledge_snippets=execution_results["knowledge_snippets"]
                )
                return {"final_answer": final_answer, "successful_plan": action_plan}

        return {"error": "在达到最大尝试次数后，仍未能找到成功的行动计划。"}

# ----------------------------- Planner训练逻辑 -----------------------------

def train_planner(args, accelerator: Accelerator):
    device = accelerator.device
    
    # 初始化
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quantization_config, trust_remote_code=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 数据集
    class PDataset(Dataset):
        def __init__(self, path, tokenizer):
            self.rows = list(read_jsonl(path))
            self.tokenizer = tokenizer

        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            prompt_part = f"Question: {r['question']}\nContext: Region captions:\n" + '\n'.join([f'[{i}] {c}' for i, c in enumerate(r['image_captions'])]) + '\nJSON Array:'
            target_part = r['plan'] + self.tokenizer.eos_token
            
            prompt_tokens = self.tokenizer(prompt_part, add_special_tokens=False)
            full_tokens = self.tokenizer(prompt_part + target_part, add_special_tokens=False, max_length=1024, truncation=True)
            
            input_ids = torch.LongTensor(full_tokens['input_ids'])
            labels = input_ids.clone()
            labels[:len(prompt_tokens['input_ids'])] = -100
            
            return {"input_ids": input_ids, "labels": labels}

    ds = PDataset(args.planner_train, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
    dl = DataLoader(ds, batch_size=args.per_device_train_batch_size, collate_fn=data_collator, shuffle=True)
    
    # 训练组件
    opt = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = len(dl) * args.num_train_epochs
    sched = get_scheduler('cosine', optimizer=opt, num_warmup_steps=int(0.1 * steps), num_training_steps=steps)

    model, opt, dl, sched = accelerator.prepare(model, opt, dl, sched)

    # 训练循环
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dl, disable=not accelerator.is_main_process)
        for batch in pbar:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            opt.step()
            sched.step()
            opt.zero_grad()
            
            total_loss += loss.detach().float()
            pbar.set_description(f"Epoch {epoch+1}, Loss: {total_loss/ (pbar.n + 1):.4f}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    accelerator.print(f'Planner适配器已保存到 {args.output_dir}')

# ----------------------------- 命令行接口 (CLI) -----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模块化KB-VQA系统")
    parser.add_argument('--mode', choices=['infer', 'train_planner'], required=True, help="运行模式: 'infer' (推理) 或 'train_planner' (训练Planner)。")
    
    # 推理模式参数
    parser.add_argument('--image', type=str, help="[推理模式] 输入图片的路径。")
    parser.add_argument('--question', type=str, help="[推理模式] 用户提出的问题。")
    parser.add_argument('--planner_adapter_path', type=str, help="[推理模式] 已微调的Planner LoRA适配器路径。")
    parser.add_argument('--verifier_adapter_path', type=str, help="[推理模式] 已微调的Verifier LoRA适配器路径。")
    
    # 训练模式参数
    parser.add_argument('--planner_train', type=str, help="[训练模式] Planner训练数据的.jsonl文件路径。")
    parser.add_argument('--model_name', type=str, default='tiiuae/falcon-7b', help="[训练模式] Planner的基础模型名称。")
    parser.add_argument('--output_dir', type=str, default='./out/planner', help="[训练模式] 保存微调后Planner适配器的目录。")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=1)

    args = parser.parse_args()
    
    if args.mode == 'infer':
        if not all([args.image, args.question, args.planner_adapter_path, args.verifier_adapter_path]):
            parser.error("[推理模式] 需要 --image, --question, --planner_adapter_path, 和 --verifier_adapter_path 参数。")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- 正在初始化所有模块，将在设备 {device} 上运行 ---")
        
        # 1. 初始化所有专家模块
        perception_module = HFExtractorAndGrounder(device)
        planner_module = CoTPlanner(args.planner_adapter_path, device)
        verifier_module = VLM_VisualVerifier(args.verifier_adapter_path, device)
        bridge_module = WebSearchKnowledgeBridge()
        synthesizer_module = AnswerSynthesizer()

        # 2. 组装系统
        kbvqa_system = ModularKBVQA(planner_module, verifier_module, bridge_module, synthesizer_module, perception_module)

        # 3. 执行推理
        pil_image = Image.open(args.image).convert("RGB")
        result = kbvqa_system.infer(pil_image, args.question)
        
        print("\n" + "="*50)
        print("                 最终推理结果")
        print("="*50)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'train_planner':
        if not args.planner_train:
            parser.error("[训练模式] 需要 --planner_train 参数。")
        
        print("--- 正在初始化Accelerator并开始训练Planner ---")
        accelerator = Accelerator()
        train_planner(args, accelerator)
