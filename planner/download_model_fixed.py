#!/usr/bin/env python3
"""
下载 Qwen2-7B-Instruct 模型 - 使用国内镜像
"""

import os
import sys
import subprocess

def download_with_huggingface_hub():
    """使用 huggingface-hub 命令行工具下载"""
    
    # 设置环境变量
    env = os.environ.copy()
    env['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    model_name = "Qwen/Qwen2-7B-Instruct"
    local_path = "/home/xuyan/workspace/models"
    
    print(f"使用 HF_ENDPOINT={env['HF_ENDPOINT']}")
    print(f"开始下载 {model_name}")
    print(f"保存路径: {local_path}")
    
    # 确保目录存在
    os.makedirs(local_path, exist_ok=True)
    
    try:
        # 使用 huggingface-hub 下载
        cmd = [
            "huggingface-cli", "download",
            model_name,
            "--local-dir", f"{local_path}/Qwen2-7B-Instruct",
            "--local-dir-use-symlinks", "False"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 运行下载命令
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 下载成功！")
            print(result.stdout)
            return True
        else:
            print("❌ 下载失败！")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ 未找到 huggingface-cli 命令")
        print("请安装: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_with_transformers():
    """使用 transformers 库下载"""
    
    # 在导入前设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"设置 HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "Qwen/Qwen2-7B-Instruct"
    local_path = "/home/xuyan/workspace/models/Qwen2-7B-Instruct"
    
    print(f"开始下载 {model_name}")
    print(f"保存路径: {local_path}")
    
    # 确保目录存在
    os.makedirs(local_path, exist_ok=True)
    
    try:
        # 下载 tokenizer
        print("正在下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_path,
            trust_remote_code=True,
            force_download=False,
            local_files_only=False
        )
        print("✅ Tokenizer 下载完成")
        
        # 下载模型
        print("正在下载模型 (约14GB，请耐心等待)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=local_path,
            torch_dtype="auto",
            device_map="cpu",  # 避免GPU内存问题
            trust_remote_code=True,
            force_download=False,
            local_files_only=False
        )
        print("✅ 模型下载完成")
        
        print(f"\n✅ 下载成功！模型保存在: {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    print("=== Qwen2-7B-Instruct 模型下载工具 ===")
    
    # 方法1: 尝试使用 huggingface-cli
    print("\n方法1: 使用 huggingface-cli 下载...")
    if download_with_huggingface_hub():
        print("下载完成！")
        sys.exit(0)
    
    # 方法2: 使用 transformers
    print("\n方法2: 使用 transformers 库下载...")
    if download_with_transformers():
        print("下载完成！")
        sys.exit(0)
    
    print("\n❌ 所有下载方法都失败了")
    print("\n手动解决方案:")
    print("1. 在终端中设置环境变量:")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    print("2. 然后运行:")
    print("   huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ./models/Qwen2-7B-Instruct")
