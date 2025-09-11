import json
from tqdm import tqdm

def convert_to_llama_factory_format(input_file, output_file):
    """
    将verifier数据转换为LLaMA-Factory VLM微调所需的格式，
    在用户消息开头添加<image>标记。
    """
    new_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="正在转换数据")):
            item = json.loads(line)
            
            # 构建标准的LLaMA-Factory对话格式
            new_item = {
                "id": f"verifier_{i}",
                "image": item["image_path"],
                "conversations": [
                    {
                        "from": "human",
                        # 在开头添加<image>标记
                        "value": f"<image>判断假设是否与图像内容一致。回答 Yes 或 No。\n{item['hypothesis']}"
                    },
                    {
                        "from": "gpt",
                        "value": "yes" if item['label'] == 1 else "no"
                    }
                ]
            }
            new_data.append(new_item)
            
    # 保存为JSON格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n转换完成！已将 {len(new_data)} 条数据保存到 {output_file}")

if __name__ == "__main__":
    input_file = "/home/xuyan/workspace/first_idea/verifier/verifier_train_split.jsonl"
    output_file = "/home/xuyan/workspace/LLaMA-Factory/data/verifier_train_llamafactory_fixed.json"
    
    convert_to_llama_factory_format(input_file, output_file)