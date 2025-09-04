"""
kbvqa_modular_cot.py  (HuggingFace / LAVIS hybrid + smoke-test generator)

This file is an enhanced modular KBVQA pipeline.
Features:
- Optional use of LAVIS for visual models (BLIP, CLIP) if you prefer its convenience.
- DETR (HuggingFace) for detection + CLIP (HuggingFace or LAVIS) for region embeddings when not using LAVIS.
- BLIP-2 captioning (HuggingFace or LAVIS) for region captions.
- CoT Planner (LoRA via PEFT) for plan generation and training.
- Visual Verifier (trainable MLP) and training routine.
- Cross-Modal Bridge stub (FAISS optional).
- Built-in smoke-test data generator: `--gen_smoketest PATH` creates sample images, planner_train.jsonl, verifier_train.jsonl, and cached region features.

Usage examples (inference with LAVIS if installed):
  python kbvqa_modular_cot.py --mode infer --image examples/img1.jpg --question "What is the object?" --use_lavis

Generate smoke test files (will create a small example image and sample JSONL):
  python kbvqa_modular_cot.py --gen_smoketest ./smoketest_dir

Notes:
- Install: pip install torch torchvision transformers peft datasets faiss-cpu pillow safetensors lavis
- If you don't want LAVIS, run without --use_lavis (will use HF BLIP/CLIP)

"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import BitsAndBytesConfig
from accelerate import Accelerator
from transformers import DataCollatorForSeq2Seq
import bitsandbytes as bnb # 引入8-bit优化器需要
from torch.nn import MultiheadAttention
# transformers & peft
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DetrImageProcessor,
    DetrForObjectDetection,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from transformers import DataCollatorForSeq2Seq
import bitsandbytes as bnb # 引入8-bit优化器需要
from peft import PeftModel

# 可选依赖项的导入
# optional: LAVIS
try:
    from lavis.models import load_model_and_preprocess
    HAS_LAVIS = True
except Exception:
    HAS_LAVIS = False

# optional: faiss
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ----------------------------- Utilities -----------------------------
#读取jsonl文件的生成器
def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# ----------------------------- Smoke test generator -----------------------------
# 创建一套用于快速测试（“冒烟测试”）的虚拟数据
def generate_smoketest(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    img_path = Path(out_dir) / 'img1.jpg'
    # make a simple image with colored rectangles
    im = Image.new('RGB', (640, 480), color=(200, 200, 200))
    draw = ImageDraw.Draw(im)
    draw.rectangle([50, 50, 250, 200], fill=(255, 0, 0))  # red box
    draw.rectangle([300, 100, 580, 380], fill=(0, 0, 255))  # blue box
    im.save(img_path)

    # planner_train.jsonl: each item has region captions (strings), question, plan (JSON array string)
    # 创建一个用于训练Planner的样本数据，包含图像内容的描述、一个问题和一份预设好的JSON格式的“行动计划”。
    planner_train = [
        {
            'image_captions': ['a red rectangle (object A)', 'a blue rectangle (object B)'],
            'question': 'Which object is red?',
            'plan': json.dumps([
                {'type': 'visual', 'focus': {'source': 'detr', 'idx': 0}, 'op': 'describe', 'hypothesis': 'object at region 0 is red'},
                {'type': 'kb', 'focus': None, 'op': 'query_kb', 'hypothesis': 'use KB if needed'}
            ])
        }
    ]
    
    # 将样本数据写入planner_train.jsonl文件。
    with open(Path(out_dir) / 'planner_train.jsonl', 'w', encoding='utf-8') as f:
        for r in planner_train:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # verifier sample features: we'll create fake CLIP-like vectors and save as pt
    #  创建两个512维的随机张量，模拟CLIP模型提取的图像特征。
    feats = [torch.randn(512), torch.randn(512)]
    # 将这些假的特征向量保存到.pt文件中。
    torch.save(feats, Path(out_dir) / 'verifier_region_feats.pt')

    # verifier_train.jsonl pointing to region indices and captions
    # 创建用于训练Verifier的样本数据，包含区域索引、描述、真假标签（1或0）以及特征文件的路径。
    verifier_train = [
        {'region_idx': 0, 'caption': 'a red rectangle (object A)', 'label': 1, 'region_feat_path': str(Path(out_dir) / 'verifier_region_feats.pt')},
        {'region_idx': 1, 'caption': 'a blue rectangle (object B)', 'label': 1, 'region_feat_path': str(Path(out_dir) / 'verifier_region_feats.pt')},
        {'region_idx': 0, 'caption': 'a blue rectangle (object B)', 'label': 0, 'region_feat_path': str(Path(out_dir) / 'verifier_region_feats.pt')}
    ]
    with open(Path(out_dir) / 'verifier_train.jsonl', 'w', encoding='utf-8') as f:
        for r in verifier_train:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"Smoke-test generated at {out_dir}. Files: img1.jpg, planner_train.jsonl, verifier_train.jsonl, verifier_region_feats.pt")
    return out_dir

# ----------------------------- Cross-Modal Bridge -----------------------------
# 将文本或图像信息与存储的实体进行匹配。核心知识库检索器（这部分有待提升修改）
class CrossModalBridge:
    def __init__(self, entity_list: Optional[List[Tuple[str, List[float]]]] = None):
        self.entities = entity_list or []
        # 如果安装了faiss库，它会为这些实体的向量创建一个IndexFlatIP索引，用于进行高效的内积（Inner Product）相似度搜索。
        if HAS_FAISS and self.entities:
            import numpy as np
            arr = np.stack([e[1] for e in self.entities]).astype('float32')
            self.index = faiss.IndexFlatIP(arr.shape[1])
            self.index.add(arr)
        else:
            self.index = None

    # 向知识库中添加新的实体，并更新FAISS索引
    def add_entities(self, ids: List[str], embs: List[List[float]]):
        for i, id in enumerate(ids):
            self.entities.append((id, embs[i]))
        if HAS_FAISS:
            import numpy as np
            arr = np.stack([e[1] for e in self.entities]).astype('float32')
            self.index = faiss.IndexFlatIP(arr.shape[1])
            self.index.add(arr)

    # 核心查询方法。如果FAISS索引存在且提供了查询向量，就执行快速的向量相似度搜索；否则，它会退回到一个简单的字符串匹配作为备用方案
    # 核心知识库检索器（这部分有待提升修改，考虑HM-RAG的方案）
    def query(self, query_text: str, embedding: Optional[List[float]] = None, topk: int = 5):
        if self.index is not None and embedding is not None:
            import numpy as np
            q = np.array(embedding, dtype='float32').reshape(1, -1)
            D, I = self.index.search(q, topk)
            return [(self.entities[idx][0], float(D[0, j])) for j, idx in enumerate(I[0])]
        # fallback substring
        res = []
        for lab, emb in self.entities:
            if query_text.lower() in lab.lower():
                res.append((lab, 1.0))
        return res[:topk]

# ----------------------------- Model modules -----------------------------
# 一个可训练的神经网络，用于验证一个文本假设（如“这个物体是红色的”）是否与一个给定的图像区域特征相符
class VisualVerifier(nn.Module):
    # 两个线性层，分别将输入的图像特征和文本特征投影到一个相同的hidden维度（默认256）
    def __init__(self, img_dim: int, txt_dim: int, hidden: int = 256):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)
        # 一个多层感知机（MLP），接收拼接后的图文特征，经过两层ReLU激活和线性变换后，最后通过Sigmoid函数输出一个0到1之间的概率值，表示假设为真的可信度。
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    # 前向传播方法，接收图像特征和文本特征，分别通过各自的线性层投影后拼接，然后通过MLP进行处理
    # 定义了前向传播的逻辑：分别投影、拼接、然后通过分类头得到最终分数
    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        i = self.img_proj(img_feat)
        t = self.txt_proj(txt_feat)
        x = torch.cat([i, t], dim=-1)
        return self.fc(x).squeeze(-1)
    
# 使用微调后的VLM
class VLM_VisualVerifier:
    def __init__(self, device):
        self.device = device
        base_model_id = "vikhyatk/moondream2"
        adapter_path = "./vlm_verifier_adapter" # 您训练好的适配器路径
        
        # 加载量化的基础模型
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float16)
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.model.eval()

    def verify(self, image_crop: Image.Image, hypothesis: str) -> bool:
        prompt = f"Question: Is the following statement true based on the image? Statement: {hypothesis}. Answer with 'yes' or 'no'.\n\nAnswer:"
        
        # 伪造 vision encoder 的输出
        fake_image_embeds = torch.randn(1, 768, 1024, device=self.device, dtype=torch.float16)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
             # 这里需要调用模型的forward方法，并传入图片和文本
            outputs = self.model.generate(
                **inputs,
                image_embeds=fake_image_embeds, # 传入伪造的图片嵌入
                max_new_tokens=5
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        return "yes" in answer


# 系统的“大脑”，包装了一个大型语言模型（LLM），负责生成“思考链（Chain-of-Thought）”计划。
class CoTPlanner:
    def __init__(self, model_name: str, device: torch.device, peft_r: int = 8):
        # 加载指定model_name的分词器，并确保有pad_token
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        # --- 这是核心改动 ---
        # 1. 定义4-bit量化配置
        # 节省显存的关键，它告诉模型以4-bit整数而不是32-bit或16-bit浮点数来加载
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # 2. 使用量化配置加载模型，并让 transformers 自动分配设备
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            #注释掉后， 将让accelerate来管理设备分配
            # device_map="auto"  # 非常重要！让模型自动分布到所有可用GPU
        )
        # --- 改动结束 ---
        # 定义LoRA（Low-Rank Adaptation）的配置
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=peft_r, lora_alpha=16, lora_dropout=0.05)
        # 使用peft库将LoRA适配器“附加”到量化后的LLM上。这样做之后，只有LoRA的参数是可训练的，极大地减少了训练所需的计算资源。
        self.llm = get_peft_model(self.llm, peft_config)

    # 字符串格式化函数，将问题、上下文（图像区域描述、KB候选）和指令拼接成一个完整的提示（prompt）
    def build_prompt(self, question: str, region_caps: List[str], kb_candidates: Optional[List[Tuple[str, float]]] = None) -> str:
        parts = [f"Question: {question}", "Context: Region captions:"]
        for i, c in enumerate(region_caps):
            parts.append(f"[{i}] {c}")
        if kb_candidates:
            parts.append('KB candidates:')
            for lab, s in kb_candidates:
                parts.append(f"- {lab} (score={s:.3f})")
        instruction = (
            "Please output a pure JSON array. Each element is an object: "
            "{type:'visual'|'kb', focus:{source:'detr'|'frcnn',idx:int|null}, op:..., hypothesis:...}.\n"
            "Do not include any extra text, only the JSON array."
        )
        return '\n'.join(parts) + '\n' + instruction + '\nJSON array starts here:\n'

    # 接收prompt，调用self.llm.generate()方法生成文本，然后从中提取出[...]之间的JSON字符串部分作为最终的计划 
    def generate_plan(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        tok = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to(self.device)
        out_ids = self.llm.generate(**tok, max_new_tokens=max_new_tokens, temperature=temperature)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        s = text.find('[')
        e = text.rfind(']')
        if s != -1 and e != -1 and e > s:
            return text[s:e+1]
        return text

# ----------------------------- 以下提供了两种不同的方式（lavis、transformers手动加载）来从图像中提取信息，是系统的“眼睛” -----------------------------
# ----------------------------- LAVIS helpers (optional) -----------------------------
class LavisWrapper:
    """If LAVIS is installed, provide convenient wrappers for BLIP captioning and CLIP features via LAVIS models.
    Uses load_model_and_preprocess(name, model_type, is_eval=True, device=device)
    Example BLIP: name='blip2', model_type='pretrain_opt2_7b'
    Example CLIP in LAVIS: name='clip', model_type='ViT-L-14'
    """
    def __init__(self, device: torch.device):
        if not HAS_LAVIS:
            raise RuntimeError('LAVIS not available')
        self.device = device
        # BLIP-2 (instruct) wrapper
        try:
            self.blip_model, self.blip_vis_processors, self.blip_txt_processors = load_model_and_preprocess(name='blip2', model_type='pretrain_opt2_7b', is_eval=True, device=device)
        except Exception:
            # fallback names may vary; user can set use_lavis False
            self.blip_model = None
        # CLIP via LAVIS
        try:
            self.clip_model, self.clip_vis_processors, _ = load_model_and_preprocess(name='clip', model_type='ViT-L-14', is_eval=True, device=device)
        except Exception:
            self.clip_model = None

    def caption(self, pil_img: Image.Image) -> str:
        if self.blip_model is None:
            raise RuntimeError('LAVIS BLIP not loaded')
        vis = self.blip_vis_processors['eval'](pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate({'image': vis})
        return out[0]

    def get_clip_feat(self, pil_img: Image.Image) -> torch.Tensor:
        if self.clip_model is None:
            raise RuntimeError('LAVIS CLIP not loaded')
        vis = self.clip_vis_processors['eval'](pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.extract_features(vis)
        # normalize and return cpu tensor
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu()

# ----------------------------- DETR+HF BLIP pipeline (fallback) -----------------------------
class HFExtractorAndGrounder:
    def __init__(self, device: torch.device, detr_model: str = 'facebook/detr-resnet-50', blip_model: str = 'Salesforce/blip2-opt-2.7b', clip_model: str = 'openai/clip-vit-base-patch32', top_k: int = 36):
        self.device = device
        self.top_k = top_k
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_model)
        self.detr = DetrForObjectDetection.from_pretrained(detr_model).to(device)
        self.detr.eval()
        # CLIP
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip = CLIPModel.from_pretrained(clip_model).to(device)
        self.clip.eval()
        # BLIP2
        self.blip_processor = Blip2Processor.from_pretrained(blip_model)
        self.blip = Blip2ForConditionalGeneration.from_pretrained(blip_model, torch_dtype=torch.float16).to(device)
        self.blip.eval()

        # determine clip projection dim safely
        self.clip_dim = getattr(self.clip.config, 'projection_dim', None) or getattr(self.clip.config, 'hidden_size', None) or 512

    def detect_and_caption(self, pil_image: Image.Image, max_regions: int = None):
        inputs = self.detr_processor(images=pil_image, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.detr(**inputs)
        # 1. 获取原始图像尺寸并转换为tensor
        # 注意：PIL的size是(width, height)，但模型通常需要(height, width)
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)

        # 2. 在detr_processor上调用后处理函数，并传入target_sizes
        results = self.detr_processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
        boxes = results['boxes']
        scores = results['scores']
        N = min(len(boxes), self.top_k if max_regions is None else max_regions)
        regions = []
        W, H = pil_image.size
        for i in range(N):
            box = boxes[i].cpu(); score = float(scores[i].cpu())
            x1, y1, x2, y2 = [int(max(0, float(v))) for v in box]
            x2 = min(x2, W - 1); y2 = min(y2, H - 1)
            crop = pil_image.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else pil_image
            # CLIP feat
            clip_inputs = self.clip_processor(images=crop, return_tensors='pt').to(self.device)
            with torch.no_grad():
                clip_feat = self.clip.get_image_features(**clip_inputs)
                clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
            # BLIP caption
            blip_inputs = self.blip_processor(images=crop, return_tensors='pt').to(self.device)
            with torch.no_grad():
                gids = self.blip.generate(**blip_inputs, max_new_tokens=20)
            caption = self.blip_processor.decode(gids[0], skip_special_tokens=True)
            regions.append({'bbox': [x1, y1, x2, y2], 'score': score, 'crop': crop, 'clip_feat': clip_feat.squeeze(0).cpu(), 'caption': caption})
        while len(regions) < (self.top_k if max_regions is None else max_regions):
            regions.append({'bbox': [0, 0, 0, 0], 'score': 0.0, 'crop': pil_image, 'clip_feat': torch.zeros(self.clip_dim), 'caption': ''})
        return regions

# ----------------------------- Modular Orchestrator -----------------------------
# 系统的“总指挥”或“编排器”，将所有独立的模块有机地组织起来，完成从提问到回答的整个流程
class ModularKBVQA:
    # 接收并保存所有模块的实例，如planner, verifier, bridge等
    def __init__(self, device: torch.device, planner: CoTPlanner, use_lavis: bool = False, lavis_wrapper: Optional[LavisWrapper] = None, 
                 hf_pipeline: Optional[HFExtractorAndGrounder] = None, bridge: Optional[CrossModalBridge] = None, 
                 verifier: Optional[VisualVerifier] = None, verifier_text_embed: str = 'mean'):
        self.device = device
        self.planner = planner
        self.use_lavis = use_lavis
        self.lavis = lavis_wrapper
        self.hf = hf_pipeline
        self.bridge = bridge or CrossModalBridge()
        self.verifier = verifier
        assert verifier_text_embed in ('mean', 'last_hidden')
        self.verifier_text_embed = verifier_text_embed

    # 用于从planner的LLM中获取给定文本的嵌入向量
    def _text_embed(self, text: str):
        tok = self.planner.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        if self.verifier_text_embed == 'mean':
            emb = self.planner.llm.get_input_embeddings()(tok['input_ids']).mean(dim=1)
            return emb
        else:
            with torch.no_grad():
                out = self.planner.llm(**tok, output_hidden_states=True, return_dict=True)
            last = out.hidden_states[-1].mean(dim=1)
            return last

    # 核心推理方法。它执行一个复杂的、带反馈的循环
    def infer(self, pil_image: Image.Image, question: str, max_replans: int = 2, verify_threshold: float = 0.6):
        # get regions & captions & clip features，从图像中提取所有区域的描述和特征
        if self.use_lavis and self.lavis is not None:
            captions = [self.lavis.caption(pil_image)]
            clip_feats = [self.lavis.get_clip_feat(pil_image)]
            regions = [{'bbox': [0, 0, pil_image.size[0], pil_image.size[1]], 'crop': pil_image, 'clip_feat': clip_feats[0], 'caption': captions[0]}]
        else:
            regions = self.hf.detect_and_caption(pil_image)
        region_caps = [r.get('caption', '') for r in regions]

        # 构建初始prompt，让planner生成第一版计划
        kb_candidates = None
        replans = 0
        final_plan = None
        raw_gens = []
        # 进入while循环（再规划循环），遍历计划中的每一步
        # 如果是visual步骤，就调用VisualVerifier验证假设。如果验证失败（分数低于阈值），记录失败信息
        # 如果是kb步骤，就调用CrossModalBridge查询知识库
        # 检查：如果所有步骤都成功，说明计划可行，跳出循环。
        # 再规划：如果存在失败步骤，就将失败信息（如“假设‘物体A是蓝色的’不成立”）作为新的上下文，去知识库查询相关信息，
        # 然后带着这些新信息，让planner重新生成一个更靠谱的计划。这个循环会重复进行，直到找到一个完全可行的计划或达到最大尝试次数。
        while replans <= max_replans:
            prompt = self.planner.build_prompt(question, region_caps, kb_candidates)
            raw = self.planner.generate_plan(prompt)
            raw_gens.append(raw)
            try:
                plan = json.loads(raw)
            except Exception:
                replans += 1
                kb_candidates = kb_candidates or []
                continue
            failures = []
            for si, step in enumerate(plan):
                stype = (step.get('type') or '').lower()
                focus = step.get('focus')
                hyp = step.get('hypothesis') or step.get('query') or step.get('entity') or ''
                if stype == 'visual' and isinstance(focus, dict):
                    idx = int(focus.get('idx', 0))
                    if idx < 0 or idx >= len(regions):
                        step['verify_score'] = 0.0
                        failures.append((si, 'focus_oob'))
                        continue
                    region_feat = regions[idx]['clip_feat'].unsqueeze(0).to(self.device)
                    txt_emb = self._text_embed(hyp).to(self.device)
                    with torch.no_grad():
                        vs = float(self.verifier(region_feat, txt_emb).detach().cpu().numpy())
                    step['verify_score'] = vs
                    if vs < verify_threshold:
                        failures.append((si, vs))
                elif stype == 'kb':
                    cands = self.bridge.query(hyp, None, topk=5)
                    step['bridge_candidates'] = cands
                else:
                    step['verify_score'] = None
            if not failures:
                final_plan = plan
                break
            kb_candidates = []
            for si, reason in failures:
                step = plan[si]
                txt = (step.get('hypothesis') or step.get('query') or '')
                cands = self.bridge.query(txt, None, topk=5)
                kb_candidates.extend(cands)
            seen = set(); uniq = []
            for lab, s in kb_candidates:
                if lab not in seen:
                    seen.add(lab); uniq.append((lab, s))
            kb_candidates = uniq[:10] if uniq else None
            replans += 1
        return {'question': question, 'final_plan': final_plan, 'raw_generations': raw_gens, 'replans': replans}

# ----------------------------- Training helpers -----------------------------
# 包含了训练planner和verifier的函数，并且已经为accelerate做了深度优化

# [重大改动] 重构 train_planner 函数以正确使用 accelerate

def train_planner(args):
    # [改动 1] 初始化 Accelerator
    # 它会自动处理设备分配(device placement)、混合精度(mixed precision)等
    accelerator = Accelerator()
    
    # device现在从accelerator获取，而不是手动指定
    device = accelerator.device 
    
    # Planner现在会以量化方式加载
    planner = CoTPlanner(args.model_name, device, peft_r=args.peft_r)

    # 定义了一个PyTorch数据集
    class PDataset(Dataset):
        def __init__(self, path, tokenizer):
            self.rows = list(read_jsonl(path))
            self.tok = tokenizer
            # 确保tokenizer有pad_token
            if self.tok.pad_token is None:
                self.tok.add_special_tokens({'pad_token': self.tok.eos_token})

        def __len__(self):
            return len(self.rows)
            
        # 数据处理核心
        # 读取一条数据，将prompt和plan（答案）拼接在一起，然后创建labels
        def __getitem__(self, idx):
            r = self.rows[idx]
            
            # 1. 准备 prompt 和 target(答案)
            prompt_part = 'Question: ' + r['question'] + '\nContext: Region captions:\n' + '\n'.join([f'[{i}] {c}' for i, c in enumerate(r['image_captions'])]) + '\nJSON array:'
            target_part = r['plan'] + self.tok.eos_token # 加上结束符以确保模型知道在哪里停止

            # 2. 分别对 prompt 和 target 进行编码
            # 我们需要 prompt 的长度来构建 labels
            prompt_tokens = self.tok(prompt_part, add_special_tokens=False)
            full_tokens = self.tok(prompt_part + target_part, add_special_tokens=False)

            # 3. 创建 input_ids 和 attention_mask
            input_ids = torch.LongTensor(full_tokens['input_ids'])
            attention_mask = torch.LongTensor(full_tokens['attention_mask'])
            
            # 4. 创建 labels
            # labels 和 input_ids 完全一样，但有一个关键区别：
            # 我们把 prompt 部分的 labels 设为 -100，这样模型在计算损失时就会忽略它们。
            labels = input_ids.clone()
            prompt_len = len(prompt_tokens['input_ids'])
            # 它将prompt部分的损失屏蔽掉，让模型在训练时只学习去预测plan的部分，这是微调LLM
            labels[:prompt_len] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    ds = PDataset(args.planner_train, planner.tokenizer)
    
    # [改动 2] 使用正确的数据整理器 (Data Collator)
    # 这会处理padding，将样本正确地打包成一个批次，修复了效率问题
    # 将PDataset返回的、长度不一的样本打包成一个规整的、填充（padding）好的批次（batch）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=planner.tokenizer, 
        model=planner.llm, 
        label_pad_token_id=planner.tokenizer.pad_token_id
    )
    
    # DataLoader现在使用新的data_collator，创建数据加载器
    dl = DataLoader(ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)

    # [改动 3] 使用节省显存的8-bit优化器（bitsandbytes
    opt = bnb.optim.PagedAdamW8bit([p for p in planner.llm.parameters() if p.requires_grad], lr=args.lr)
    
    # 根据训练总步数和热身比例，创建一个线性学习率调度器，帮助模型更稳定地训练。
    # 总的训练步数（batch 数 × epoch 数）。
    steps = len(dl) * args.num_train_epochs
    # 训练初期通常会用较小的学习率“热身”，防止模型不稳定。
    # int(0.03 * steps) 表示热身步数为总步数的 3%。，max(1, ...) 保证至少有 1 步热身。
    # get_scheduler('linear', ...)创建一个线性学习率调度器，训练过程中学习率会线性下降。optimizer=opt 是优化器对象。num_training_steps=steps 指定总训练步数。
    sched = get_scheduler('linear', optimizer=opt, num_warmup_steps=max(1, int(0.03 * steps)), num_training_steps=steps)
    
    # [改动 4] 使用 accelerator.prepare 来包装所有需要分布式处理的对象
    # 它接收模型、优化器、数据加载器和学习率调度器，然后将它们全部“包装”好，以适应当前的运行环境（单GPU、多GPU、混合精度等）
    planner.llm, opt, dl, sched = accelerator.prepare(
        planner.llm, opt, dl, sched
    )

    accelerator.print(f"Starting training on {accelerator.num_processes} GPUs.")
    
    # 训练循环现在更简洁，每次处理一个完整的批次
    for epoch in range(args.num_train_epochs):
        planner.llm.train()
        total_loss = 0
        for step, batch in enumerate(dl):
            # batch已经是一个包含所有数据的字典，并已移动到正确的device上
            outputs = planner.llm(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            
            # [改动 5] 使用 accelerator.backward 会自动处理混合精度训练中的梯度缩放
            accelerator.backward(loss)
            
            opt.step()
            sched.step()
            opt.zero_grad()
        
        # 使用accelerator.log来记录日志，它只会在主进程打印
        avg_loss = total_loss / len(dl)
        accelerator.log({"loss": avg_loss, "epoch": epoch})
        accelerator.print(f"Planner epoch {epoch+1} done, Average Loss: {avg_loss:.4f}")

    # 使用accelerator.wait_for_everyone()确保所有进程都完成了
    accelerator.wait_for_everyone()
    
    # 解包模型以保存：获取到原始的模型，以便保存。
    unwrapped_model = accelerator.unwrap_model(planner.llm)
    unwrapped_model.save_pretrained(args.output_dir)
    planner.tokenizer.save_pretrained(args.output_dir)
    
    accelerator.print(f'Planner saved to {args.output_dir}')


# [重大改动] 重构 train_verifier 函数以支持 accelerate 和批处理
# 这个函数的结构与train_planner非常相似，同样遵循了Accelerator -> Dataset -> DataLoader -> prepare -> 训练循环的模式
def train_verifier(args):
    # [改动 1] 初始化 Accelerator，让它管理一切
    accelerator = Accelerator()
    device = accelerator.device

    # 我们仍然需要加载Planner，因为它提供了文本编码器(tokenizer)和词嵌入层。
    # 好消息是，因为我们之前已经修改了CoTPlanner的__init__方法，
    # 它现在会以节省显存的4-bit量化方式加载，所以不会再导致OOM。
    planner = CoTPlanner(args.model_name, device, peft_r=4)
    
    # Verifier本身是一个很小的模型
    verifier = VisualVerifier(img_dim=args.img_dim, txt_dim=planner.llm.get_input_embeddings().weight.shape[1])
    # 注意：我们不再手动 .to(device)

    # [改动 2] 创建一个标准化的 Dataset
    class VerifierDataset(Dataset):
        def __init__(self, path):
            self.rows = list(read_jsonl(path))
            # 预加载所有特征，如果文件不大这样做可以加速训练
            self.features_cache = {}
            for r in self.rows:
                if r['region_feat_path'] not in self.features_cache:
                    self.features_cache[r['region_feat_path']] = torch.load(r['region_feat_path'])
        
        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            # 从预加载的缓存中获取特征
            all_feats = self.features_cache[row['region_feat_path']]
            img_feat = all_feats[int(row['region_idx'])]
            caption = row['caption']
            label = torch.tensor(float(row['label']))
            return {"img_feat": img_feat, "caption": caption, "label": label}

    # [改动 3] 定义一个自定义的 collate_fn 来处理批处理
    def collate_fn(batch):
        img_feats = torch.stack([item['img_feat'] for item in batch])
        captions = [item['caption'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        
        # 使用planner的tokenizer来动态编码文本
        # padding=True 会自动将批次内的文本填充到相同长度
        tokenized_captions = planner.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        
        return {
            "img_feat": img_feats,
            "captions_tokenized": tokenized_captions,
            "labels": labels
        }

    dataset = VerifierDataset(args.verifier_train)
    data_loader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(verifier.parameters(), lr=args.lr)

    # [改动 4] 使用 accelerator.prepare() 包装所有对象
    verifier, optimizer, data_loader = accelerator.prepare(
        verifier, optimizer, data_loader
    )
    
    # 同样，也需要将planner中的模型放到prepare中，即使它不参与梯度更新，
    # 也要确保它被正确地放置在分布式环境的每个设备上。
    planner.llm = accelerator.prepare(planner.llm)


    accelerator.print(f"Starting Verifier training on {accelerator.num_processes} GPUs.")

    # [改动 5] 使用新的批处理训练循环
    verifier.train()
    planner.llm.eval() # 文本编码器部分不需要训练
    for epoch in range(args.num_train_epochs):
        total_loss = 0
        for batch in data_loader:
            img_feat = batch['img_feat'] # 已经是在正确的device上
            tokenized_captions = batch['captions_tokenized']
            labels = batch['labels']

            # 获取文本嵌入
            with torch.no_grad():
                # 注意：tokenized_captions 已经是一个批次，包含了 input_ids 和 attention_mask
                txt_emb = planner.llm.get_input_embeddings()(tokenized_captions['input_ids']).mean(dim=1)

            # 使用 .half() 将输入张量转换为 float16 类型，以匹配模型
            # 注意：如果你的模型是4-bit量化的，这一步可能不需要，因为模型已经在正确的dtype上
            img_feat = img_feat.to(torch.float32)
            txt_emb = txt_emb.to(torch.float32)
            pred = verifier(img_feat, txt_emb)
            loss = F.binary_cross_entropy(pred, labels)
            
            total_loss += loss.detach().float()
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(data_loader)
        accelerator.print(f"Verifier epoch {epoch+1} done, Average Loss: {avg_loss:.4f}")

    # 保存模型
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(verifier)
    accelerator.save(unwrapped_model.state_dict(), Path(args.output_dir) / 'verifier.pt')
    accelerator.print(f'Verifier saved to {args.output_dir}')

# ----------------------------- CLI -----------------------------
if __name__ == '__main__':
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 定义了可以从命令行接收的参数
    parser.add_argument('--gen_smoketest', type=str, help='directory to generate smoke-test files')
    parser.add_argument('--use_lavis', action='store_true', help='use LAVIS for captioning / clip features if available')
    parser.add_argument('--mode', choices=['infer', 'train_planner', 'train_verifier'], default='infer')
    parser.add_argument('--image', type=str)
    parser.add_argument('--question', type=str)
    parser.add_argument('--planner_train', type=str, help='planner training jsonl')
    parser.add_argument('--verifier_train', type=str, help='verifier training jsonl')
    parser.add_argument('--verifier_feats', type=str, help='cached region feats .pt')
    parser.add_argument('--model_name', type=str, default='tiiuae/falcon-7b')
    parser.add_argument('--blip_model', type=str, default='Salesforce/blip2-opt-2.7b')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--detr_model', type=str, default='facebook/detr-resnet-50')
    parser.add_argument('--output_dir', type=str, default='./outputs_modular_hf')
    parser.add_argument('--peft_r', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--max_replans', type=int, default=2)
    parser.add_argument('--verify_threshold', type=float, default=0.6)
    parser.add_argument('--img_dim', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.gen_smoketest:
        generate_smoketest(args.gen_smoketest)
        sys.exit(0)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'infer':
        if not args.image or not args.question:
            print('Provide --image and --question for inference'); sys.exit(1)
        print('Initializing components (may download models)...')
        lavis_wrapper = LavisWrapper(device) if (args.use_lavis and HAS_LAVIS) else None
        if args.use_lavis and lavis_wrapper is None:
            print('--use_lavis specified but LAVIS not available; falling back to HF pipeline')
        hf_pipeline = None if args.use_lavis else HFExtractorAndGrounder(device, detr_model=args.detr_model, blip_model=args.blip_model, clip_model=args.clip_model)
        planner = CoTPlanner(args.model_name, device, peft_r=args.peft_r)
        bridge = CrossModalBridge()
        verifier = VisualVerifier(img_dim=args.img_dim, txt_dim=planner.llm.get_input_embeddings().weight.shape[1]).to(device)
        kb_system = ModularKBVQA(device, planner, use_lavis=(args.use_lavis and HAS_LAVIS), lavis_wrapper=lavis_wrapper, hf_pipeline=hf_pipeline, bridge=bridge, verifier=verifier)
        img = Image.open(args.image).convert('RGB')
        out = kb_system.infer(img, args.question, max_replans=args.max_replans, verify_threshold=args.verify_threshold)
        print(json.dumps(out, indent=2, ensure_ascii=False))

    elif args.mode == 'train_planner':
        if not args.planner_train:
            print('Provide --planner_train path'); sys.exit(1)
        train_planner(args)
    elif args.mode == 'train_verifier':
        if not args.verifier_train:
            print('Provide --verifier_train path'); sys.exit(1)
        train_verifier(args)
