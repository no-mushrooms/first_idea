# 文件名: download_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch

# proxies_dict = {
#     "http": "http://10.109.69.32:7897",
#     "https": "http://10.109.69.32:7897",
# }

# # --- 需要下载的模型 ---
model_id = "vikhyatk/moondream2"
# --- 要保存到的本地文件夹名称 ---
local_path = Path("/home/xuyan/model/moondream2-local")
# # --- 需要下载的模型 ---
# model_id = "microsoft/Phi-3-vision-128k-instruct"
# # --- 要保存到的本地文件夹名称 ---
# local_path = Path("/home/xuyan/model/phi-3-vision-local")

print(f"即将下载模型 '{model_id}' 到本地文件夹 '{local_path}'...")
local_path.mkdir(exist_ok=True)

# 1. 下载并保存模型
print("正在下载模型权重...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map="auto", # 使用device_map可以更高效地加载
    _attn_implementation="eager" # 推荐为vision模型添加
    # proxies=proxies_dict 
)
model.save_pretrained(local_path)
print("  [成功] 模型权重已保存。")

# 2. 下载并保存分词器
print("正在下载分词器...")
# tokenizer = AutoTokenizer.from_pretrained(model_id,proxies=proxies_dict)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(local_path)
print("  [成功] 分词器已保存。")

print(f"\n下载完成！所有文件都已保存在 '{local_path}' 文件夹中。")
print("下一步：请将这个完整的文件夹上传到您的服务器。")