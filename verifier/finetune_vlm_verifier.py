import os
import json
from pathlib import Path
import argparse

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    AutoProcessor,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
)
from torch.utils.data import Dataset
from transformers import ProcessorMixin
from dataclasses import dataclass


# from transformers import MoondreamProcessor
os.environ["http_proxy"] = "http://10.109.70.178:7897"
os.environ["https_proxy"] = "http://10.109.70.178:7897"
# ----------------- 数据集类 (回归简洁) -----------------
# Dataset只负责加载原始数据
class VerifierDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except FileNotFoundError:
            print(f"警告: 找不到图片 {item['image_path']}, 将使用黑色占位图。")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        hypothesis = item['hypothesis']
        label = item['label']
        
        answer = "yes" if label == 1 else "no"
        question = f"Question: Is the following statement true based on the image? Statement: {hypothesis}. Answer with 'yes' or 'no'.\n\nAnswer:"
        
        return {"image": image, "question": question, "answer": answer}

# ----------------- 自定义数据整理器 (核心) -----------------
@dataclass
class VLMDataCollator:
    tokenizer: PreTrainedTokenizer
    model: AutoModelForCausalLM

    def __call__(self, features):
        images = [f['image'] for f in features]
        questions = [f['question'] for f in features]
        answers = [f['answer'] for f in features]

        # 1. 编码图像
        with torch.no_grad():
            image_embeds = self.model.encode_image(images)

        # 2. 编码文本并拼接
        batch_inputs_embeds = []
        batch_labels = []
        
        bos_emb = self.model.text_model.embed_tokens(
            torch.tensor([[self.tokenizer.bos_token_id]], device=self.model.device)
        )

        for i in range(len(features)):
            question_tokens = self.tokenizer(questions[i], return_tensors="pt").input_ids.to(self.model.device)
            answer_tokens = self.tokenizer(answers[i] + self.tokenizer.eos_token, return_tensors="pt").input_ids.to(self.model.device)

            question_embeds = self.model.text_model.embed_tokens(question_tokens)
            answer_embeds = self.model.text_model.embed_tokens(answer_tokens)
            
            # 3. 手动拼接嵌入向量，完美复刻官方逻辑
            inputs_embeds = torch.cat([
                bos_emb, 
                image_embeds[i].unsqueeze(0).unsqueeze(0), # 保证维度正确
                question_embeds, 
                answer_embeds
            ], dim=1)
            
            # 4. 创建Labels，屏蔽掉非答案部分
            labels = torch.cat([
                torch.full_like(bos_emb, -100),
                torch.full_like(image_embeds[i].unsqueeze(0).unsqueeze(0), -100),
                torch.full_like(question_embeds, -100),
                answer_tokens
            ], dim=1)

            batch_inputs_embeds.append(inputs_embeds)
            batch_labels.append(labels)

        # 5. 将批次内的所有样本填充到相同长度
        max_len = max(emb.shape[1] for emb in batch_inputs_embeds)
        
        padded_inputs_embeds = []
        padded_labels = []
        attention_mask = []

        for i in range(len(batch_inputs_embeds)):
            pad_len = max_len - batch_inputs_embeds[i].shape[1]
            
            padded_inputs_embeds.append(
                torch.nn.functional.pad(batch_inputs_embeds[i], (0, 0, 0, pad_len), value=0)
            )
            padded_labels.append(
                torch.nn.functional.pad(batch_labels[i], (0, pad_len), value=-100)
            )
            attention_mask.append(
                torch.cat([torch.ones_like(batch_inputs_embeds[i]), torch.zeros(1, pad_len, device=self.model.device)], dim=1)
            )

        return {
            "inputs_embeds": torch.cat(padded_inputs_embeds, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
            "labels": torch.cat(padded_labels, dim=0),
        }

# ----------------- 主函数 -----------------
def main(args):
    print("--- 步骤1: 加载基础模型和分词器 ---")
    model_id = args.model_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("  [信息] 正在以torch.float16半精度模式加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True 
    )
    
    print("--- 步骤2: 应用LoRA配置 ---")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["qkv", "proj", "fc1", "fc2"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("--- 步骤3: 准备数据集 ---")
    train_dataset = VerifierDataset(args.verifier_train_jsonl)
    data_collator = VLMDataCollator(tokenizer=tokenizer, model=model)

    print("--- 步骤4: 开始训练 ---")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    
    print("--- 步骤5: 保存模型 ---")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"微调后的LoRA适配器和分词器已保存到: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微调Moondream2作为Visual Verifier。")
    parser.add_argument("--verifier_train_jsonl", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True) # 虽然没直接用，但dataset里需要
    parser.add_argument("--model_path", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--output_dir", type=str, default="./vlm_verifier_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1) # batch size设小一点，用梯度累积
    args = parser.parse_args()
    main(args)