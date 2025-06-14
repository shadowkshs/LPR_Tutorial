from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from pathlib import Path

# import torch
# import torchvision
# print("torch:", torch.__version__)
# print("torchvision:", torchvision.__version__)


# === 設定區 ===
model_path = "runs/detect/lpr_yolo11n/weights/best.pt"
image_dir = "./LPR_recognition/test"
output_crop_dir = "./cropped_plates"
conf_threshold = 0.8
os.makedirs(output_crop_dir, exist_ok=True)

# === 載入模型 ===
model = YOLO(model_path)
image_paths = list(Path(image_dir).glob("*.[jp][pn]g"))
print(f"共找到 {len(image_paths)} 張圖片")

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    results = model.predict(source=str(image_path), conf=conf_threshold)

    for result in results:
        if result.boxes is None or result.boxes.xyxy.shape[0] == 0:
            print(f"❌ No detection: {image_path.name}")
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        # === 取得最高分的 box ===
        best_idx = np.argmax(scores)
        best_box = boxes[best_idx]
        x1, y1, x2, y2 = map(int, best_box)

        cropped_plate = image.crop((x1, y1, x2, y2))

        # === 命名與輸出 ===
        filename = image_path.stem  # '0000_029-BBJ_origin'
        parts = filename.split("_")
        if len(parts) >= 2:
            raw_plate = parts[1].replace("origin", "").strip("-_")
        else:
            raw_plate = filename

        save_name = f"{raw_plate}.jpg"
        save_path = os.path.join(output_crop_dir, save_name)
        cropped_plate.save(save_path)
        print(f"✅ Saved: {save_name}")
