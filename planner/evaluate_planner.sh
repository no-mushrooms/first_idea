python convert_planner_data.py --output_file planner_finetuned_test.json --input_file planner_test_split.jsonl


CUDA_VISIBLE_DEVICES=0 python evaluate_planner_speed.py \
    --model_path Qwen/Qwen3-8B \
    --adapter_path /home/xuyan/workspace/LLaMA-Factory/qwen_planner_adapter_qwen3_only_plan \
    --test_file planner_finetuned_test.json