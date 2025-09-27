# python convert_planner_data.py --output_file planner_llama_factory_thinkAndPlan_mini_test.json --input_file planner_test_split.jsonl  --include_thinking_system


# CUDA_VISIBLE_DEVICES=1 python evaluate_planner_speed.py \
#     --model_path Qwen/Qwen3-8B \
#     --adapter_path /home/xuyan/workspace/LLaMA-Factory/kbvqaCoT/save/qwen_planner_adapter_qwen3_test \
#     --test_file planner_llama_factory_thinkAndPlan_mini_test.json

# 查看所有样本的完整输出
# python evaluate_planner_detail.py \
#     --model_path Qwen/Qwen3-8B \
#     --adapter_path /home/xuyan/workspace/LLaMA-Factory/kbvqaCoT/save/qwen_planner_adapter_qwen3_test \
#     --test_file planner_llama_factory_thinkAndPlan_mini_test.json

# # 交互模式，逐个查看（可以按q退出，按s跳过详细输出）
# python evaluate_planner_detail.py 
#     --model_path Qwen/Qwen3-8B \
#     --adapter_path /home/xuyan/workspace/LLaMA-Factory/kbvqaCoT/save/qwen_planner_adapter_qwen3_test \
#     --test_file planner_llama_factory_thinkAndPlan_mini_test.json \
#     --interactive

# 只处理前5个样本进行快速测试
CUDA_VISIBLE_DEVICES=1 python evaluate_planner_detail.py \
    --model_path Qwen/Qwen3-8B \
    --adapter_path /home/xuyan/workspace/LLaMA-Factory/qwen_planner_probe_adapter_cleanData \
    --test_file planner_llama_factory_thinkAndPlan_mini_test_cleaned_20250926_074021.json \
    # --max_samples 5