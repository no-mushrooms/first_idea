CUDA_VISIBLE_DEVICES=2 python kbvqa_modular_cot.py --gen_smoketest ./smoketest



accelerate launch kbvqa_modular_cot.py --mode train_verifier --verifier_train smoketest/verifier_train.jsonl --model_name tiiuae/falcon-7b --output_dir ./out/verifier --num_train_epochs 3 --img_dim 512


python kbvqa_modular_cot.py --mode infer --image smoketest/img1.jpg --question "Which object is red?" --use_lavis
--use_lavis可选 如果你没有安装 LAVIS，省去 --use_lavis，脚本会使用 HuggingFace DETR+BLIP2+CLIP 路径（前提是已安装 transformers）。

#planner_train
accelerate launch --gpu_ids '2' --num_processes=1 kbvqa_modular_cot.py \
    --mode train_planner \
    --planner_train smoketest/planner_train.jsonl \
    --model_name tiiuae/falcon-7b \
    --output_dir ./out/planner \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --lr 2e-4 

#verifier_train
accelerate launch --gpu_ids '4' --num_processes=1 kbvqa_modular_cot.py \
    --mode train_verifier \
    --verifier_train smoketest/verifier_train.jsonl \
    --model_name tiiuae/falcon-7b \
    --output_dir ./out/verifier \
    --num_train_epochs 3 \
    --img_dim 512 \
    --per_device_train_batch_size 8 # <-- 加上这个参数，可以设为8, 16, 32等

#推理
CUDA_VISIBLE_DEVICES=4 python kbvqa_modular_cot.py --mode infer --image smoketest/img1.jpg --question "Which object is red?" --use_lavis
--use_lavis可选 如果你没有安装 LAVIS，省去 --use_lavis，脚本会使用 HuggingFace DETR+BLIP2+CLIP 路径（前提是已安装 transformers）。