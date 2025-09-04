python kbvqa_modular_cot.py \
  --mode infer \
  --image /path/to/img.jpg \
  --question "What brand is the shoe?" \
  --model_name tiiuae/falcon-7b \
  --blip_model Salesforce/blip2-opt-2.7b \
  --clip_model openai/clip-vit-base-patch32 \
  --detr_model facebook/detr-resnet-50 \
  --output_dir ./outputs_modular_hf