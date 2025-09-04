"""
train_planner.py - 一个独立的、用于微调CoT Planner的LoRA训练脚本。

本脚本使用accelerate库进行高效训练，并利用bitsandbytes进行4-bit量化，
以便在消费级GPU上微调大语言模型（如falcon-7b）。

功能:
- 从.jsonl文件加载训练数据。
- 以4-bit量化模式加载基础大语言模型。
- 应用PEFT (LoRA) 进行参数高效微调。
- 使用PagedAdamW8bit优化器节省显存。
- 通过accelerate简化分布式训练和设备管理。
- 训练完成后，保存LoRA适配器和分词器。

如何运行:
1. 准备您的训练数据 `planner_train.jsonl`，格式如下:
   {"question": "...", "image_captions": ["...", "..."], "plan": "[{...}]"}

2. 在终端中执行以下命令 (单GPU示例):
   accelerate launch train_planner.py \
     --train_file ./path/to/your/planner_train.jsonl \
     --model_id "tiiuae/falcon-7b" \
     --output_dir ./planner_adapter \
     --epochs 1 \
     --batch_size 1 \
     --lr 1e-4
"""

import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 导入所有必需的库
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
import bitsandbytes as bnb

# ----------------------------- 辅助工具 -----------------------------

def read_jsonl(path: str):
    """逐行读取.jsonl文件。"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# ----------------------------- 数据集类 -----------------------------

class PlannerDataset(Dataset):
    """
    为Planner微调任务定制的数据集。
    负责将原始的JSON数据转换成模型可以理解的格式。
    """
    def __init__(self, file_path: str, tokenizer: AutoTokenizer):
        self.rows = list(read_jsonl(file_path))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        
        # 1. 构建Prompt部分
        prompt_part = (
            f"Question: {row['question']}\n"
            f"Context: Region captions:\n" +
            '\n'.join([f'[{i}] {c}' for i, c in enumerate(row['image_captions'])]) +
            "\nPlease output a pure JSON array of action steps. Each step is an object with 'type', 'focus', 'op', and 'hypothesis'.\n"
            "JSON Array:"
        )
        
        # 2. 构建目标部分 (标准答案)
        # 我们在答案的末尾加上结束符，告诉模型在这里停止生成
        target_part = row['plan'] + self.tokenizer.eos_token
        
        # 3. 分别编码，以计算Prompt的长度
        prompt_tokens = self.tokenizer(prompt_part, add_special_tokens=False)
        full_tokens = self.tokenizer(
            prompt_part + target_part,
            add_special_tokens=False,
            max_length=1024,  # 您可以根据您的数据和显存调整
            truncation=True
        )
        
        # 4. 创建labels，这是微调的关键
        input_ids = torch.LongTensor(full_tokens['input_ids'])
        labels = input_ids.clone()
        
        # 将Prompt部分的标签设为-100，这样在计算损失时就会被忽略
        # 模型因此只学习去预测'plan'的部分
        labels[:len(prompt_tokens['input_ids'])] = -100
        
        return {"input_ids": input_ids, "labels": labels}

# ----------------------------- 主训练函数 -----------------------------

def main(args):
    """主训练流程"""
    # 1. 初始化Accelerator
    accelerator = Accelerator()
    accelerator.print(f"--- 正在初始化训练环境 ---")

    # 2. 加载模型和分词器
    # 使用bitsandbytes以4-bit量化方式加载，极大地节省显存
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    accelerator.print(f"  [模型] 正在从缓存加载量化的基础模型: {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True # 假设模型已在服务器缓存
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 应用PEFT (LoRA)
    accelerator.print("  [PEFT] 正在应用LoRA配置...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        # 对于falcon模型，通常需要指定这些目标模块
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 准备数据集和DataLoader
    accelerator.print("  [数据] 正在准备数据集...")
    train_dataset = PlannerDataset(args.train_file, tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100, # 忽略填充部分的损失
        pad_to_multiple_of=8
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True
    )

    # 5. 准备训练组件 (优化器、学习率调度器)
    accelerator.print("  [训练] 正在准备优化器和学习率调度器...")
    optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    
    total_steps = len(train_dataloader) * args.epochs
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 6. 使用Accelerator准备所有组件
    # 这是关键一步，accelerate会自动处理所有分布式训练和混合精度的配置
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 7. 开始训练循环
    accelerator.print("--- 开始训练 ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            outputs = model(**batch)
            loss = outputs.loss
            
            # 使用accelerate的反向传播
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.detach().float()
            progress_bar.set_postfix({"loss": f"{total_loss / (progress_bar.n + 1):.4f}"})

    # 8. 保存训练成果
    accelerator.print("--- 训练完成，正在保存模型 ---")
    accelerator.wait_for_everyone()
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir)
    
    # 保存分词器，以便未来使用
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        
    accelerator.print(f"Planner LoRA适配器已成功保存到: {args.output_dir}")

# ----------------------------- 命令行接口 (CLI) -----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为CoT Planner微调一个大语言模型。")
    parser.add_argument("--train_file", type=str, required=True, help="Planner训练数据的.jsonl文件路径。")
    parser.add_argument("--model_id", type=str, default="tiiuae/falcon-7b", help="基础大语言模型的Hugging Face Hub ID。")
    parser.add_argument("--output_dir", type=str, default="./planner_adapter", help="保存微调后LoRA适配器的目录。")
    parser.add_argument("--epochs", type=int, default=1, help="训练的总轮数。")
    parser.add_argument("--batch_size", type=int, default=1, help="每个设备的训练批次大小。")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率。")
    
    args = parser.parse_args()
    main(args)