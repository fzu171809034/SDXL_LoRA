import os
import json

dataset_dir = "Dataset"
images_dir = os.path.join(dataset_dir, "images")

# 假设你的图片名是 "1.jpg", "2.jpg", ..., "16.jpg"
# 文本文件名是 "1.txt", "2.txt", ..., "16.txt"
num_samples = 71

lines = []
for i in range(1, num_samples + 1):
    img_path = f"images/{i}.jpg"
    txt_path = os.path.join(dataset_dir, f"{i}.txt")

    # 读取txt文件内容
    with open(txt_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    line = {
        "file_name": img_path,
        "text": caption
    }
    lines.append(line)

# 将每个样本的字典序列化为一行 JSON
metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
with open(metadata_path, "w", encoding="utf-8") as f:
    for line_dict in lines:
        f.write(json.dumps(line_dict, ensure_ascii=False) + "\n")

print(f"metadata.jsonl 已生成到 {metadata_path}")
