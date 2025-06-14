from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import re
from difflib import SequenceMatcher
from diffusers import LDMSuperResolutionPipeline

# === 基本設定 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "./trocr-plate/checkpoint-3000"  # 修改為你實際的checkpoint目錄

# === 載入 Processor 與模型 ===
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")  # 🔧 使用 base 模型的 processor
model = VisionEncoderDecoderModel.from_pretrained("./trocr-plate/checkpoint-3000").to(device)  # 🔧 使用 fine-tuned 模型權重
model.eval()

# === 推論函式 ===
def recognize_plate(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()


# === helper：提取 ground truth ===
def extract_label_from_filename(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_.*$", "", name)  # 移除第一個底線後的所有內容（包含底線）
    label = re.sub(r"[^A-Za-z0-9]", "", name)  # 僅保留英數字
    return label.upper()


# === 批次測試資料夾 ===
test_dir = "./cropped_train/test"

sr_pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages").to(device)


# === 評估準確率 ===
total = 0
correct = 0
char_total = 0
char_correct = 0

for fname in os.listdir(test_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        gt = extract_label_from_filename(fname)
        pred = recognize_plate(os.path.join(test_dir, fname)).upper()

        total += 1
        if pred == gt:
            correct += 1

        # 字元比對
        match = SequenceMatcher(None, gt, pred).get_matching_blocks()
        matched_chars = sum(m.size for m in match)
        char_correct += matched_chars
        char_total += max(len(gt), len(pred))

        print(f"{fname}: GT={gt}, Pred={pred}")



# === 顯示結果 ===
print("\n=== 評估結果 ===")
print(f"Exact Match Accuracy: {correct / total:.2%} ({correct}/{total})")
print(f"Character-Level Accuracy: {char_correct / char_total:.2%} ({char_correct}/{char_total})")

'''
=== 評估結果 ===
Exact Match Accuracy: 93.98% (281/299)
Character-Level Accuracy: 98.11% (1968/2006)
'''