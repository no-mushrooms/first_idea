import os
import json
from pathlib import Path
import argparse
import time

import torch
import wandb
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from torch.utils.data import Dataset
from dataclasses import dataclass
from qwen_vl_utils import process_vision_info

# 设置代理（如果需要）
# os.environ["http_proxy"] = "http://10.109.69.32:7897"
# os.environ["https_proxy"] = "http://10.109.69.32:7897"

os.environ["http_proxy"] = "http://10.109.70.178:7897"
os.environ["https_proxy"] = "http://10.109.70.178:7897"

# ----------------- 自定义 Trainer -----------------
class VLMTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_count = 0
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        标准的损失计算 - Qwen2.5-VL 支持标准的 transformers 接口
        """
        # 直接使用模型的前向传播
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 记录到 W&B
        self.step_count += 1
        if self.step_count % 10 == 0:
            wandb.log({
                "custom/loss": loss.item(),
                "custom/step": self.step_count,
                "custom/learning_rate": self.get_lr(),
            })
        
        return (loss, outputs) if return_outputs else loss
    
    def get_lr(self):
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()[0]
        elif hasattr(self, 'optimizer') and self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        return 0

# ----------------- 数据集类 -----------------
class VerifierDataset(Dataset):
    def __init__(self, jsonl_path, image_dir=None):
        self.data = []
        self.image_dir = image_dir
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        
        # 如果提供了 image_dir，则拼接完整路径
        if self.image_dir:
            image_filename = os.path.basename(item['image_path'])
            item['image_path'] = os.path.join(self.image_dir, image_filename)
        
        return item

# ----------------- 数据整理器 -----------------
@dataclass
class VLMDataCollator:
    processor: AutoProcessor
    
    def __call__(self, features):
        # 准备批次数据
        images = []
        texts = []
        
        for item in features:
            # 加载图像
            image_path = item['image_path']
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            
            # 构建对话格式的文本
            question = item['question']
            answer = item['answer']
            
            # Qwen2.5-VL 使用对话格式
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(text)
        
        # 使用 processor 处理
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 创建标签（用于计算损失）
        batch["labels"] = batch["input_ids"].clone()
        
        # 在非助手回复的地方设置 -100（不计算损失）
        for i, text in enumerate(texts):
            # 找到助手回复的开始位置
            assistant_start = text.find("<|im_start|>assistant\n")
            if assistant_start != -1:
                # 编码到助手回复开始的部分
                prefix_tokens = self.processor.tokenizer.encode(
                    text[:assistant_start + len("<|im_start|>assistant\n")],
                    add_special_tokens=False
                )
                # 将前缀部分设置为 -100
                batch["labels"][i, :len(prefix_tokens)] = -100
        
        return batch

# ----------------- 主函数 -----------------
def main(args):
    # 计时开始
    start_time = time.time()
    print(f"=== 实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} ===")
    
    print("--- 步骤1: 加载 Qwen2.5-VL 模型和处理器 ---")
    
    # 加载模型
    print(f"  正在从 {args.model_path} 加载模型...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # Qwen2.5-VL 建议使用 bfloat16
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    
    print("--- 步骤2: 配置 LoRA ---")
    
    # LoRA 配置 - 针对 Qwen2.5-VL 的架构
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
            "gate_proj", "up_proj", "down_proj",      # MLP 层
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("--- 步骤3: 准备数据集 ---")
    train_dataset = VerifierDataset(args.verifier_train_jsonl, args.image_dir)
    print(f"训练数据集大小: {len(train_dataset)}")
    
    data_collator = VLMDataCollator(processor=processor)
    
    print("--- 步骤4: 初始化 W&B 和开始训练 ---")
    
    # 初始化 W&B
    wandb.init(
        project="qwen2.5-vl-verifier",
        name=f"vlm-verifier-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "model_path": args.model_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-5,  # Qwen2.5-VL 建议更小的学习率
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
        tags=["qwen2.5-vl", "vlm", "verifier", "lora"]
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,  # 更小的学习率
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,  # 使用 bfloat16
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to="wandb",
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=3,
    )
    
    trainer = VLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,  # 重要：传入 tokenizer
    )
    
    trainer.train()
    
    print("--- 步骤5: 保存模型 ---")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"微调后的 LoRA 适配器和处理器已保存到: {args.output_dir}")
    
    # 计算时间
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    # 记录最终统计到 W&B
    wandb.log({
        "final/total_time_seconds": total_time,
        "final/total_time_hours": total_time / 3600,
        "final/final_loss": trainer.state.log_history[-1].get('train_loss', 0) if trainer.state.log_history else 0,
        "final/total_steps": trainer.state.global_step,
    })
    
    print(f"=== 实验结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))} ===")
    print(f"=== 总耗时: {hours}小时 {minutes}分钟 {seconds}秒 ({total_time:.2f}秒) ===")
    
    # 保存时间记录
    with open(f"{args.output_dir}/training_time.txt", "w") as f:
        f.write(f"实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"实验结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒\n")
        f.write(f"总耗时(秒): {total_time:.2f}\n")
        f.write(f"W&B 项目链接: {wandb.run.url}\n")
    
    print(f"时间记录已保存到: {args.output_dir}/training_time.txt")
    print(f"W&B 训练监控链接: {wandb.run.url}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Qwen2.5-VL 进行 VLM Verifier 微调")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen2.5-VL 模型路径")
    parser.add_argument("--verifier_train_jsonl", type=str, default="verifier_train_split.jsonl", help="训练数据集JSONL文件路径")
    parser.add_argument("--image_dir", type=str, help="图像文件夹路径")  # 添加这行
    parser.add_argument("--output_dir", type=str, default="./qwen2.5-vl-verifier-lora", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    
    args = parser.parse_args()
    main(args)