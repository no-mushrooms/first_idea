import json

input_file = "/home/xuyan/workspace/idea/verifier_train_vlm.jsonl"
output_file = "verifier_train_llamafactory.json"

mapping = {0: "No", 1: "Yes"}

data = []
with open(input_file, "r") as f:
    for line in f:
        item = json.loads(line)
        entry = {
            "instruction": "判断假设是否与图像内容一致。回答 Yes 或 No。",
            "input": item["hypothesis"],
            "output": mapping[item["label"]],
            "image": item["image_path"]
        }
        data.append(entry)

with open(output_file, "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"转换完成，共 {len(data)} 条数据，保存到 {output_file}")