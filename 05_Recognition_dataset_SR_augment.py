from PIL import Image, ImageDraw
import os
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import io
from diffusers import LDMSuperResolutionPipeline
import torch
from pathlib import Path


import csv
from datetime import datetime

log_path = "./augmentation_log.csv"

# === 初始化 log.csv 檔案 ===
if not os.path.exists(log_path):
    with open(log_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp", "original_filename", "sr_path",
            "easy_aug_path", "hard_aug_path", "occlusion_aug_path"
        ])

def log_augmentation(filename, sr_path=None, easy_path=None, hard_path=None, occ_path=None):
    with open(log_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().isoformat(),
            filename,
            sr_path or "",
            easy_path or "",
            hard_path or "",
            occ_path or ""
        ])

# === 設定區 ===
image_dir = "./cropped_plates"
output_dir = "./cropped_plates_SR"
os.makedirs(output_dir, exist_ok=True)

# === AUG setting ===
easy_aug = A.Compose([
    A.Resize(64, 128),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.Rotate(limit=3, p=0.7),
    ToTensorV2()
])

hard_aug = A.Compose([
    A.Resize(64, 128),
    A.OneOf([
        A.MotionBlur(blur_limit=1, p=0.3),
        A.MedianBlur(blur_limit=1, p=0.3),
        A.GaussianBlur(blur_limit=(1, 1), p=0.3),
    ], p=0.3),
    A.GaussNoise(var_limit=(2.0, 10.0), mean=0, p=0.15),
    A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
    A.Perspective(scale=(0.005, 0.01), p=0.25),
    A.Affine(rotate=(-2, 2), shear=(-2, 2), translate_percent=0.01, scale=(0.98, 1.02), p=0.3),
    A.ImageCompression(quality_lower=85, quality_upper=95, p=0.2),
    A.OneOf([
        A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.03),
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
    ], p=0.2),

    ToTensorV2()
])

# === 遮擋與壓縮 ===
def apply_occlusion_safe(image: Image.Image, num_patches=1) -> Image.Image:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for _ in range(num_patches):
        max_patch_size = int(min(w, h) * 0.1)  # 限制最大遮蔽尺寸
        while True:
            x0 = random.randint(0, w - max_patch_size)
            y0 = random.randint(0, h - max_patch_size)
            patch_w = random.randint(5, max_patch_size)
            patch_h = random.randint(5, max_patch_size)
            if not (w//3 < x0 < w*2//3 and h//3 < y0 < h*2//3):  # 避免遮住中央
                break
        fill_color = tuple([random.randint(80, 150)] * 3)  # 淺灰色調遮蔽
        draw.rectangle([x0, y0, x0 + patch_w, y0 + patch_h], fill=fill_color)
    return image

def jpeg_compress(image: Image.Image, quality=40) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def pil_to_numpy(image: Image.Image):
    return np.array(image)

# === 模型設定 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

image_paths = list(Path(image_dir).glob("*.[jp][pn]g"))
print(f"共找到 {len(image_paths)} 張圖片")

# === SR ===
sr_resolution_threshold = 128  # 可自訂

for image_path in image_paths:
    try:
        image = Image.open(image_path).convert("RGB")
        filename = image_path.stem
        print(f"處理 SR: {filename}")

        # === 檢查解析度是否足夠 ===
        if min(image.size) >= sr_resolution_threshold:
            print(f"跳過 SR：原圖解析度已足夠 - {image.size}")
            continue
        else:
            print(f"執行 SR：原圖解析度 {image.size}")
            upscaled_image = pipeline(image, num_inference_steps=100, eta=1).images[0]

        save_name = f"{filename}_SR.jpg"
        save_path = os.path.join(output_dir, save_name)
        upscaled_image.save(save_path)

        log_augmentation(filename=filename, sr_path=save_path)

    except Exception as e:
        print(f"❌ SR 失敗 - {filename}: {e}")


# ==== AUG ====
image_dir = "./cropped_plates_SR"
output_dir = "./cropped_plates_aug"
os.makedirs(output_dir, exist_ok=True)

image_paths = list(Path(image_dir).glob("*.[jp][pn]g"))
print(f"共找到 {len(image_paths)} 張圖片")

# === AUG 處理 ===
for image_path in image_paths:
    try:
        image = Image.open(image_path).convert("RGB")
        filename = image_path.stem
        basename = filename.replace("_SR", "")  # 防止重複命名
        print(f"處理 AUG: {basename}")

        easy_path = hard_path = occ_path = None

        # 生成控制邏輯
        generate_easy = random.random() < 0.4
        generate_hard = random.random() < 0.15
        generate_occlusion = random.random() < 0.08

        # 輕度增強
        if generate_easy:
            easy_tensor = easy_aug(image=pil_to_numpy(image))["image"]
            easy_path = os.path.join(output_dir, f"{basename}_SR_aug_easy.png")
            to_pil_image(easy_tensor).save(easy_path)

        # 重度增強
        if generate_hard:
            hard_tensor = hard_aug(image=pil_to_numpy(image))["image"]
            hard_path = os.path.join(output_dir, f"{basename}_SR_aug_hard.png")
            to_pil_image(hard_tensor).save(hard_path)

        # 遮蔽 + 壓縮 + 增強
        if generate_occlusion:
            occ_img = apply_occlusion_safe(image.copy())
            occ_jpg = jpeg_compress(occ_img, quality=random.randint(30, 60))
            occ_tensor = hard_aug(image=pil_to_numpy(occ_jpg))["image"]
            occ_path = os.path.join(output_dir, f"{basename}_SR_aug_occ.png")
            to_pil_image(occ_tensor).save(occ_path)

        # 寫入 log
        log_augmentation(basename, sr_path=None, easy_path=easy_path, hard_path=hard_path, occ_path=occ_path)

    except Exception as e:
        print(f"❌ AUG 失敗 - {filename}: {e}")


# === 輸入資料夾 ===
source_dirs = ["./cropped_plates", "./cropped_plates_SR", "./cropped_plates_aug"]
output_base = "./cropped_train"
os.makedirs(output_base, exist_ok=True)

splits = ["train", "test", "validation"]
split_ratio = [0.8, 0.1, 0.1]  # 可調整

for split in splits:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

# === 收集所有圖片路徑 ===
all_images = []
for dir in source_dirs:
    all_images.extend(list(Path(dir).glob("*.[jp][pn]g")))  # 支援 jpg, png

print(f"總共圖片數量: {len(all_images)}")

# === 隨機打散與分割 ===
random.shuffle(all_images)
n_total = len(all_images)
n_train = int(n_total * split_ratio[0])
n_test = int(n_total * split_ratio[1])

train_imgs = all_images[:n_train]
test_imgs = all_images[n_train:n_train + n_test]
val_imgs = all_images[n_train + n_test:]

split_map = {
    "train": train_imgs,
    "test": test_imgs,
    "validation": val_imgs
}

# === 建立新 labels.json ===
import shutil
import json

new_labels = {"train": {}, "test": {}, "validation": {}}

def infer_label_from_filename(filename):
    # 例如：029-BBJ.jpg -> 029BBJ
    return filename.replace("-", "").replace("_SR", "").replace("_aug_easy","").replace("_aug_hard","").replace("_aug_occ","").replace(".jpg", "").replace(".png", "").upper()

# === 移動與標記 ===
for split, image_paths in split_map.items():
    for path in image_paths:
        filename = path.name
        label = infer_label_from_filename(filename)
        target_path = os.path.join(output_base, split, filename)
        shutil.copy(path, target_path)
        new_labels[split][filename] = label

# === 儲存標籤 ===
for split in splits:
    with open(os.path.join(output_base, f"{split}_labels.json"), "w") as f:
        json.dump(new_labels[split], f, indent=2)

print("✅ 資料整合與切分完成。")