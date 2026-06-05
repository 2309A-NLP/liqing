# -*- coding: utf-8 -*-
import json
import random

# ===================== 【修改为你的文件路径】 =====================
# 原始数据文件路径
input_path = r"/研发\knowledge\legal\finetune_dataset.jsonl"
# 抽取后保存的新文件路径
output_path = r"/研发\knowledge\legal\finetune_dataset_20000.jsonl"

# 1. 读取所有数据
print("正在读取原始数据...")
data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

total = len(data)
print(f"原始数据总条数：{total}")

# 2. 随机抽取 10000 条
sample_num = 10000
if total < sample_num:
    print(f"警告：数据不足{sample_num}条，抽取全部{total}条")
    sampled_data = data
else:
    sampled_data = random.sample(data, sample_num)

# 3. 保存到新文件
with open(output_path, "w", encoding="utf-8") as f:
    for item in sampled_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ===================== 完成 =====================
print(f"\n🎉 抽取完成！")
print(f"✅ 抽取条数：{len(sampled_data)}")
print(f"✅ 保存路径：{output_path}")