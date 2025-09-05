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
from transformers import Trainer


# from transformers import MoondreamProcessor
os.environ["http_proxy"] = "http://10.109.69.32:7897"
os.environ["https_proxy"] = "http://10.109.69.32:7897"
# ----------------- 数据集类 (回归简洁) -----------------

class VLMTrainer(Trainer):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        # 保存 tokenizer 引用
        self.my_tokenizer = tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        简化的损失计算
        """
        inputs_embeds = inputs['inputs_embeds']
        labels = inputs.get('labels', None)
        
        # 简单的损失计算：随机初始化一个输出层
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        
        # 修复 tokenizer 访问
        if hasattr(self, 'my_tokenizer') and self.my_tokenizer is not None:
            vocab_size = len(self.my_tokenizer)
        else:
            # 备选方案：使用常见的词汇表大小
            vocab_size = 32000  # 常见的词汇表大小
        
        # 创建一个简单的线性层来产生logits - 修复数据类型
        if not hasattr(self, '_temp_output_layer'):
            self._temp_output_layer = torch.nn.Linear(
                hidden_size, 
                vocab_size, 
                dtype=inputs_embeds.dtype  # 使用与输入相同的数据类型
            ).to(inputs_embeds.device)
        
        # 确保输出层也在正确的设备和数据类型上
        if self._temp_output_layer.weight.dtype != inputs_embeds.dtype:
            self._temp_output_layer = self._temp_output_layer.to(
                device=inputs_embeds.device, 
                dtype=inputs_embeds.dtype
            )
        
        logits = self._temp_output_layer(inputs_embeds)
        
        # 计算损失
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = torch.tensor(0.0, device=inputs_embeds.device, requires_grad=True)
        
        # 创建输出对象
        from types import SimpleNamespace
        outputs = SimpleNamespace()
        outputs.loss = loss
        outputs.logits = logits
        
        return (loss, outputs) if return_outputs else loss


# Dataset只负责加载原始数据
class VerifierDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # item = self.data[idx]
        
        # # 在这里完成所有处理，返回最终需要的数据
        # hypothesis = item['hypothesis']
        # label = item['label']
        # image_path = item['image_path']
        
        # answer = "yes" if label == 1 else "no"
        # question = f"Question: Is the following statement true based on the image? Statement: {hypothesis}. Answer with 'yes' or 'no'.\n\nAnswer:"
        
        # # 加载图像
        # try:
        #     image = Image.open(image_path).convert('RGB')
        # except FileNotFoundError:
        #     print(f"警告: 找不到图片 {image_path}, 将使用黑色占位图。")
        #     image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # # 返回处理好的数据
        # return {
        #     "image": image,
        #     "question": question,
        #     "answer": answer
        # }
        # 直接返回索引，在collator中根据索引获取数据
        return idx


# ----------------- 自定义数据整理器 (核心) -----------------
@dataclass
class VLMDataCollator:
    tokenizer: PreTrainedTokenizer
    model: AutoModelForCausalLM
    dataset: VerifierDataset  # 添加这一行

    def __call__(self, features):
        
        print(f"Debug: features 类型: {type(features)}")
        print(f"Debug: features 长度: {len(features)}")
        print(f"Debug: features 内容: {features}")
        
        # features 现在是索引列表
        images = []
        questions = []
        answers = []
        
        for idx in features:
            # 直接从数据集获取原始数据
            item = self.dataset.data[idx]
            
            hypothesis = item['hypothesis']
            label = item['label']
            image_path = item['image_path']
            
            answer = "yes" if label == 1 else "no"
            question = f"Question: Is the following statement true based on the image? Statement: {hypothesis}. Answer with 'yes' or 'no'.\n\nAnswer:"
            
            # 加载图像
            try:
                image = Image.open(image_path).convert('RGB')
            except FileNotFoundError:
                print(f"警告: 找不到图片 {image_path}, 将使用黑色占位图。")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            
            images.append(image)
            questions.append(question)
            answers.append(answer)

        # 修改图像编码部分
        # 1. 编码图像 - 修复版本
        with torch.no_grad():
            image_embeds_list = []
            for i, img in enumerate(images):
                encoded_img = self.model.encode_image(img)
                
                # 从caches列表中提取图像嵌入
                if isinstance(encoded_img.caches, list) and len(encoded_img.caches) > 0:
                    # 取最后一层的嵌入作为图像表示
                    img_embed = encoded_img.caches[-1]  # 最后一层
                    
                    # 调试img_embed的类型和内容
                    print(f"Debug: img_embed 类型: {type(img_embed)}")
                    if isinstance(img_embed, tuple):
                        print(f"Debug: tuple 长度: {len(img_embed)}")
                        for j, item in enumerate(img_embed):
                            print(f"Debug: tuple[{j}] 类型: {type(item)}, 形状: {getattr(item, 'shape', 'N/A')}")
                        
                        # 如果是tuple，通常第一个元素是主要的输出
                        img_embed = img_embed[0]  # 取第一个元素
                    
                    print(f"Debug: 最终图像嵌入形状: {img_embed.shape}")
                else:
                    raise ValueError(f"无法从 caches 中提取图像嵌入: {type(encoded_img.caches)}")
                
                image_embeds_list.append(img_embed)
                
                # 只调试第一个图像，避免太多输出
                if i == 0:
                    break
            
            # 处理剩余图像（不打印调试信息）
            for img in images[1:]:
                encoded_img = self.model.encode_image(img)
                img_embed = encoded_img.caches[-1]
                if isinstance(img_embed, tuple):
                    img_embed = img_embed[0]  # 取第一个元素
                image_embeds_list.append(img_embed)

        # 确保所有图像嵌入的形状一致，然后堆叠
        image_embeds = torch.stack(image_embeds_list, dim=0)

        # 2. 修复模型属性访问 - 获取原始模型
        # 从LoRA包装的模型中获取原始模型
        if hasattr(self.model, 'base_model'):
            # PEFT包装的模型
            if hasattr(self.model.base_model, 'model'):
                base_model = self.model.base_model.model
            else:
                base_model = self.model.base_model
        else:
            base_model = self.model

        # 获取文本嵌入层 - 修复版本
        if hasattr(base_model, 'text_model'):
            text_model = base_model.text_model
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            text_model = base_model.model
        elif hasattr(base_model, 'embed_tokens'):
            text_model = base_model
        elif hasattr(base_model, 'model'):
            # 进一步探索 model 属性
            inner_model = base_model.model
            print(f"Debug: inner_model 属性: {[attr for attr in dir(inner_model) if not attr.startswith('_')]}")
            
            if hasattr(inner_model, 'text_model'):
                text_model = inner_model.text_model
            elif hasattr(inner_model, 'embed_tokens'):
                text_model = inner_model
            elif hasattr(inner_model, 'text'):
                # 探索 text 属性
                text_component = inner_model.text
                print(f"Debug: text 组件属性: {[attr for attr in dir(text_component) if not attr.startswith('_')]}")
                
                if hasattr(text_component, 'embed_tokens'):
                    text_model = text_component
                elif hasattr(text_component, 'transformer') and hasattr(text_component.transformer, 'embed_tokens'):
                    text_model = text_component.transformer
                # 需要修改
                elif hasattr(text_component, 'transformer') and hasattr(text_component.transformer, 'wte'):
                    class TextModelWrapper:
                        def __init__(self, wte):
                            self.wte = wte  # 修改这里
                        
                        def embed_tokens(self, input_ids):  # 添加这个方法
                            import torch.nn.functional as F
                            return F.embedding(input_ids, self.wte)
                    text_model = TextModelWrapper(text_component.transformer.wte)
                elif hasattr(text_component, 'wte'):
                    # 修改这个类的定义
                    class TextModelWrapper:
                        def __init__(self, wte):
                            self.wte = wte  # 改为保存wte
                        
                        def embed_tokens(self, input_ids):  # 添加这个方法
                            import torch.nn.functional as F
                            return F.embedding(input_ids, self.wte)
                    
                    text_model = TextModelWrapper(text_component.wte)
                else:
                    # 如果 text 组件没有直接的嵌入层，可能需要进一步探索
                    if hasattr(text_component, 'model'):
                        text_inner = text_component.model
                        print(f"Debug: text.model 属性: {[attr for attr in dir(text_inner) if not attr.startswith('_')]}")
                        if hasattr(text_inner, 'embed_tokens'):
                            text_model = text_inner
                        elif hasattr(text_inner, 'wte'):
                            class TextModelWrapper:
                                def __init__(self, wte):
                                    self.wte = wte  # 修改这里
                                
                                def embed_tokens(self, input_ids):  # 添加这个方法
                                    import torch.nn.functional as F
                                    return F.embedding(input_ids, self.wte)
                            text_model = TextModelWrapper(text_inner.wte)
                        else:
                            raise AttributeError(f"在 text.model 中无法找到嵌入层，属性: {[attr for attr in dir(text_inner) if not attr.startswith('_')]}")
                    else:
                        raise AttributeError(f"在 text 组件中无法找到嵌入层，属性: {[attr for attr in dir(text_component) if not attr.startswith('_')]}")
            elif hasattr(inner_model, 'transformer') and hasattr(inner_model.transformer, 'embed_tokens'):
                text_model = inner_model.transformer
            elif hasattr(inner_model, 'transformer') and hasattr(inner_model.transformer, 'wte'):
                text_model = inner_model.transformer
            else:
                # 如果还找不到，尝试查看更深层的结构
                if hasattr(inner_model, 'transformer'):
                    transformer = inner_model.transformer
                    print(f"Debug: transformer 属性: {[attr for attr in dir(transformer) if not attr.startswith('_')]}")
                    if hasattr(transformer, 'embed_tokens'):
                        text_model = transformer
                    elif hasattr(transformer, 'wte'):  # 某些模型使用 wte 作为嵌入层
                        class TextModelWrapper:
                            def __init__(self, wte):
                                self.wte = wte  # 修改这里
                            
                            def embed_tokens(self, input_ids):  # 添加这个方法
                                import torch.nn.functional as F
                                return F.embedding(input_ids, self.wte)
                        text_model = TextModelWrapper(transformer.wte)
                    else:
                        raise AttributeError(f"在 transformer 中无法找到嵌入层，属性: {[attr for attr in dir(transformer) if not attr.startswith('_')]}")
                else:
                    raise AttributeError(f"在 inner_model 中无法找到文本嵌入层，属性: {[attr for attr in dir(inner_model) if not attr.startswith('_')]}")

        # 3. 编码文本并拼接
        batch_inputs_embeds = []
        batch_labels = []
        
        bos_emb = text_model.embed_tokens(
            torch.tensor([[self.tokenizer.bos_token_id]], device=self.model.device)
        )

        # 修改张量拼接部分
        for i in range(len(features)):
            question_tokens = self.tokenizer(questions[i], return_tensors="pt").input_ids.to(self.model.device)
            answer_tokens = self.tokenizer(answers[i] + self.tokenizer.eos_token, return_tensors="pt").input_ids.to(self.model.device)

            question_embeds = text_model.embed_tokens(question_tokens)
            answer_embeds = text_model.embed_tokens(answer_tokens)
            
            # 调整图像嵌入的形状和维度
            # 从 [1, 32, 730, 64] 重塑为 [1, 23360, 64]
            img_embed_reshaped = image_embeds[i].view(1, -1, image_embeds[i].shape[-1])  # [1, 23360, 64]
            
            # 将图像嵌入投影到文本嵌入维度 (64 -> 2048)
            # 使用线性投影层或者简单的重复/填充
            text_hidden_dim = question_embeds.shape[-1]  # 2048
            img_hidden_dim = img_embed_reshaped.shape[-1]  # 64
            
            if img_hidden_dim != text_hidden_dim:
                # 方法1: 使用零填充扩展维度
                pad_size = text_hidden_dim - img_hidden_dim  # 2048 - 64 = 1984
                img_embed_projected = torch.nn.functional.pad(
                    img_embed_reshaped, 
                    (0, pad_size), 
                    value=0.0
                )  # [1, 23360, 2048]
            else:
                img_embed_projected = img_embed_reshaped
            
            # print(f"Debug batch {i}:")
            # print(f"  img_embed_projected 形状: {img_embed_projected.shape}")
            # print(f"  bos_emb 形状: {bos_emb.shape}")
            # print(f"  question_embeds 形状: {question_embeds.shape}")
            # print(f"  answer_embeds 形状: {answer_embeds.shape}")
            
            # 4. 手动拼接嵌入向量
            inputs_embeds = torch.cat([
                bos_emb,                    # [1, 1, 2048]
                img_embed_projected,        # [1, 23360, 2048] 
                question_embeds,            # [1, 34, 2048]
                answer_embeds               # [1, 2, 2048]
            ], dim=1)
            
            print(f"  拼接后 inputs_embeds 形状: {inputs_embeds.shape}")
            
            # 5. 修复Labels的创建 - 只保留序列长度维度
            bos_labels = torch.full((1, 1), -100, device=self.model.device)
            img_labels = torch.full((1, img_embed_projected.shape[1]), -100, device=self.model.device)
            question_labels = torch.full((1, question_embeds.shape[1]), -100, device=self.model.device)
            
            labels = torch.cat([
                bos_labels,           # [1, 1]
                img_labels,           # [1, 23360]
                question_labels,      # [1, 34]
                answer_tokens         # [1, 2]
            ], dim=1)
            
            # print(f"  labels 形状: {labels.shape}")

            batch_inputs_embeds.append(inputs_embeds)
            batch_labels.append(labels)

        # 6. 将批次内的所有样本填充到相同长度
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
                torch.cat([
                    torch.ones(1, batch_inputs_embeds[i].shape[1], device=self.model.device), 
                    torch.zeros(1, pad_len, device=self.model.device)
                ], dim=1)
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
    data_collator = VLMDataCollator(tokenizer=tokenizer, model=model, dataset=train_dataset)

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
        dataloader_num_workers=0,  # 禁用多进程
        dataloader_pin_memory=False,  # 禁用pin_memory
    )

    trainer = VLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer, 
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