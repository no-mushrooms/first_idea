#!/bin/bash
cd "$(dirname "$0")/.."   # 切到 project 根目录
export PYTHONPATH=$(pwd)

python verifier/generate_verifier_data.py \
        --aokvqa_json_path /home/xuyan/workspace/VisualCoT/dataset/aokvqa/datasets/aokvqa/aokvqa_v1p0_train.json \
        --coco_images_dir /home/xuyan/workspace/VisualCoT/dataset/aokvqa/datasets/coco17/train2017 \
        --output_file ./verifier_train_vlm.jsonl \
        --crops_output_dir ./verifier_image_crops \
        --limit 500 # 同样，建议先用小数据量测试