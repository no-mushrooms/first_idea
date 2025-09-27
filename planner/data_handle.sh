#!/bin/bash
cd "$(dirname "$0")/.."   # 切到 project 根目录
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python planner/generate_aokvqa_plans.py \
    --aokvqa_json_path /home/xuyan/workspace/VisualCoT/dataset/aokvqa/datasets/aokvqa/aokvqa_v1p0_train.json \
    --coco_images_dir /home/xuyan/workspace/VisualCoT/dataset/aokvqa/datasets/coco17/train2017 \
    --coco_split train2017 \
    --output_file /home/xuyan/workspace/first_idea/planner/planner_aokvqa_openai_thinkAndPlan_newPromptMiniTest.jsonl \
    --openai_api_key sk-XQwrLQpptEkeyHNIbf733S0RUGfhATavCxg4BXHTonsVVs2s \
    --start_index 0 \
    --end_index  3000 \
    # --append

#付费
# --openai_api_key sk-XQwrLQpptEkeyHNIbf733S0RUGfhATavCxg4BXHTonsVVs2s \

