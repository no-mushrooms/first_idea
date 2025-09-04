#!/bin/bash
cd "$(dirname "$0")/.."   # 切到 project 根目录
export PYTHONPATH=$(pwd)

accelerate launch --num_processes=1 --gpu_ids '5'  verifier/finetune_vlm_verifier.py \
    --verifier_train_jsonl /home/xuyan/workspace/idea/verifier_train_vlm.jsonl \
    --image_dir /home/xuyan/workspace/idea/verifier_image_crops \
    --output_dir ./vlm_verifier_adapter \
    --epochs 3 \
    --batch_size 4