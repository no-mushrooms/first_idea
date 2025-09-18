import os
# 设置国内镜像源（必须在导入transformers之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 直接使用模型名称，HuggingFace会自动从缓存加载
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("开始下载 Qwen/Qwen3-8B...")
print(f"使用镜像: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")

try:
    print("正在下载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=True
    )
    print("✅ Tokenizer 下载完成")
    
    print("正在下载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("✅ 模型下载完成")
    
    print(f"🎉 模型已缓存到默认位置: ~/.cache/huggingface/hub/")
    
except Exception as e:
    print(f"❌ 下载失败: {e}")