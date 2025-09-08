import json
from sklearn.model_selection import train_test_split # 导入专业的分割工具

# 读取你完整的数据集
full_data = [json.loads(line) for line in open('verifier_train_vlm.jsonl', 'r')]

# 使用80/20的比例分割，random_state确保每次分割结果都一样
train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)

# 将分割后的数据写回新文件
with open('verifier_train_split.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('verifier_test_split.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print(f"数据分割完成！")
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")