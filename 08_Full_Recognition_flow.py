from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from diffusers import LDMSuperResolutionPipeline
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import torch
import os, re
import numpy as np
from pathlib import Path
import cv2

# === 基本參數設定 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
conf_threshold = 0.5
sr_resolution_threshold = 512

# input_dir = "./LPR_recognition/test"
# output_dir = "./pipeline_output"

input_dir = "./real_test"
output_dir = "./real_output"
os.makedirs(output_dir, exist_ok=True)

# === 模型載入 ===
detection_model = YOLO("runs/detect/lpr_yolo11n/weights/best.pt")
sr_pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages").to(device)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
recognizer = VisionEncoderDecoderModel.from_pretrained("./trocr-plate/checkpoint-3000").to(device)
recognizer.eval()

# === 車牌辨識函式 ===
def recognize_plate(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = recognizer.generate(inputs.pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# === 傳統影像強化 ===
def enhance_image(pil_img):
    img = np.array(pil_img)

    # === 銳化處理（Sharpening）===
    kernel_sharpen = np.array([[0, -1,  0],
                               [-1,  5, -1],
                               [0, -1,  0]])
    img = cv2.filter2D(img, -1, kernel_sharpen)

    # === 自適應直方圖均衡化 (CLAHE) ===
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # === 轉回 PIL 物件 ===
    return Image.fromarray(img)

# 傾斜校正(Tilt Correction)
# 去模糊(Debluring)

'''
極端狀況
[偵測] → [裁切] → [影像增強]  → [超解析] → [去模糊] → [透視傾斜校正]  → [字符級OCR] → [AI字元推理]  → [結果置信分數報告]                                                               
                              影像處理                          |   辨識      |   推理 

一般狀況
[偵測] → [裁切] → [影像增強]  → [超解析]  → [字符級OCR]


Clear to Blur dataset
'''


# === 處理每一張圖片 ===
image_paths = list(Path(input_dir).glob("*.[jpJP][pnPN]*[gG]"))  # 支援 jpg, png, jpeg
for image_path in image_paths:
    try:
        image = Image.open(image_path).convert("RGB")
        parts = image_path.stem.split("_")
        folder_name = parts[1] if len(parts) > 2 else image_path.stem

        subdir = os.path.join(output_dir, folder_name)
        os.makedirs(subdir, exist_ok=True)

        # 儲存原始圖
        raw_path = os.path.join(subdir, "raw.jpg")
        image.save(raw_path)

        # YOLO 偵測
        preds = detection_model.predict(source=str(image_path), conf=conf_threshold)
        boxes = preds[0].boxes
        if boxes is None or boxes.xyxy.shape[0] == 0:
            with open(os.path.join(subdir, "No_Detection_LP.txt"), "w", encoding="utf-8") as f:
                f.write("❌ No detection")
            continue

        # 取信心值最高的 box
        best_idx = np.argmax(boxes.conf.cpu().numpy())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])
        cropped = image.crop((x1, y1, x2, y2))
        crop_path = os.path.join(subdir, "crop.jpg")
        cropped.save(crop_path)
        print(f"🔄 crop size = {cropped.size}")
        # SR 判斷與處理：多次放大直到最短邊 >= 128，最多進行 max_sr_iter 次
        sr_img = cropped

        # 過小時強制放大
        if min(sr_img.size) < sr_resolution_threshold:
            scale = sr_resolution_threshold / min(sr_img.size)
            new_size = (int(sr_img.width * scale), int(sr_img.height * scale))
            sr_img = sr_img.resize(new_size, Image.BICUBIC)
            print(f"🔧 Fallback Resize to: {new_size}")

        max_sr_iter = 5
        for i in range(max_sr_iter):
            if min(sr_img.size) >= sr_resolution_threshold:
                break
            sr_img = sr_pipeline(sr_img, num_inference_steps=100, eta=1).images[0]
            print(f"🔄 SR Step {i + 1}: size = {sr_img.size}")

        # 若仍未達門檻，紀錄失敗警告
        if min(sr_img.size) < sr_resolution_threshold:
            with open(os.path.join(subdir, "SR_Fail.txt"), "w", encoding="utf-8") as f:
                f.write(f"⚠️ After {max_sr_iter} SR steps, min size = {min(sr_img.size)} < {sr_resolution_threshold}")

        sr_path = os.path.join(subdir, "sr.jpg")
        sr_img = enhance_image(sr_img)
        sr_img.save(sr_path)
        # TrOCR 辨識
        result_text = recognize_plate(sr_img)
        with open(os.path.join(subdir, f"{result_text}.txt"), "w", encoding="utf-8") as f:
            f.write(result_text)

    except Exception as e:
        with open(os.path.join(subdir, "Error.txt"), "w", encoding="utf-8") as f:
            f.write(f"❌ Error: {str(e)}")

# === Score ===
import os
import re

output_dir = "./pipeline_output"
total = 0
correct = 0
char_total = 0
char_correct = 0
wrong_samples = []

# === 擷取標籤的函式（保留第一個與第二個 "_" 間的內容）===
def extract_label(folder_name):
    match = re.search(r'_(.*?)_', folder_name)
    raw_label = match.group(1) if match else folder_name
    clean_label = re.sub(r'[^A-Za-z0-9]', '', raw_label)  # 移除非英數字
    return clean_label.upper()

# === 遍歷每個資料夾進行比對 ===
for folder in os.listdir(output_dir):
    folder_path = os.path.join(output_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    label = extract_label(folder)
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and f not in ["Error.txt", "No_Detection_LP.txt"]]
    if not txt_files:
        continue

    pred_text = os.path.splitext(txt_files[0])[0].upper()

    total += 1
    if pred_text == label:
        correct += 1
    else:
        wrong_samples.append((folder, label, pred_text))

    # === 字元層級比較 ===
    max_len = max(len(label), len(pred_text))
    for i in range(max_len):
        true_char = label[i] if i < len(label) else ''
        pred_char = pred_text[i] if i < len(pred_text) else ''
        if true_char == pred_char:
            char_correct += 1
        char_total += 1

# === 顯示結果 ===
exact_acc = correct / total * 100 if total else 0
char_acc = char_correct / char_total * 100 if char_total else 0

print(f"\n📊 Exact Match Accuracy: {correct}/{total} ({exact_acc:.2f}%)")
print(f"🔠 Character-Level Accuracy: {char_correct}/{char_total} ({char_acc:.2f}%)")

if wrong_samples:
    print("\n❌ Misclassified Samples:")
    for folder, true_label, pred in wrong_samples:
        print(f" - {folder} | True: {true_label} → Pred: {pred}")


'''
Exact Match Accuracy: 236/246 (95.93%)
Character-Level Accuracy: 1581/1641 (96.34%)
'''