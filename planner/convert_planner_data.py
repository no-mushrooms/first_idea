import json
import argparse
from tqdm import tqdm

def read_jsonl(path: str):
    """逐行读取.jsonl文件。"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def convert_planner_data(input_file, output_file):
    """
    将您的Planner数据转换为LLaMA-Factory所需的对话格式。
    """
    new_data = []
    for i, item in enumerate(tqdm(read_jsonl(input_file), desc="正在转换Planner数据")):
        
        # 1. 构建人类的提问部分 (Prompt)
        # 这完美复刻了您之前为Planner设计的输入格式
        context = '\n'.join([f'[{i}] {c}' for i, c in enumerate(item['image_captions'])])
        human_prompt = (
            f"Question: {item['question']}\n"
            f"Context: Region captions:\n{context}\n"
            "Please output a pure JSON array of action steps. Each step is an object with 'type', 'focus', 'op', and 'hypothesis'.\n"
            "JSON Array:"
        )
        
        # 2. 构建模型应该学会回答的部分 (Plan)
        # 注意：plan本身就是一个字符串，这正是我们想要的
        gpt_answer = item['plan']

        new_item = {
            "id": f"planner_{i}",
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "gpt",
                    "value": gpt_answer
                }
            ]
        }
        new_data.append(new_item)
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n转换完成！已将 {len(new_data)} 条数据保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="planner_train_split.jsonl",help="原始的 planner_train.jsonl 文件路径。")
    parser.add_argument("--output_file", type=str, default="/home/xuyan/workspace/LLaMA-Factory/data/planner_llama_factory.json", help="转换后的输出文件路径。")
    args = parser.parse_args()
    convert_planner_data(args.input_file, args.output_file)