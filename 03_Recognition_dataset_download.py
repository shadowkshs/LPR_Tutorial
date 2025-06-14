from datasets import load_dataset

import os
from tqdm import tqdm

# === 基本設定 ===
base_dir = "./LPR_recognition"
splits = ["train", "test", "validation"]
os.makedirs(base_dir, exist_ok=True)

# === 建立子資料夾 ===
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# === 檔名安全化 ===
def sanitize_filename(label):
    return "".join(c for c in label if c.isalnum() or c in "_-")


# === 主處理流程 ===
for split in splits:
    dataset = load_dataset("EZCon/taiwan-license-plate-recognition", split=split)
    print(f"處理 {split} 資料，共 {len(dataset)} 筆")

    for i in tqdm(range(len(dataset))):
        try:
            example = dataset[i]
            license_num = sanitize_filename(example["license_number"])
            image = example["image"]

            class_dir = os.path.join(base_dir, split)
            ensure_folder(class_dir)

            # 修正檔名避免衝突（加入全域 index）
            filename_prefix = f"{i:04d}_{license_num}"

            # 儲存原圖
            origin_path = os.path.join(class_dir, f"{filename_prefix}_origin.png")
            image.save(origin_path)


        except Exception as e:
            print(f"❌ Error at {split} index {i}: {e}")
