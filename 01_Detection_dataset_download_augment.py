from datasets import load_dataset
from PIL import Image, ImageDraw
from io import BytesIO
import os
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# === 儲存結構 ===
base_dir = "./LPR_detection"
splits = ["train", "test", "validation"]
for split in splits:
    os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

# === 增強策略 ===
bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'])
easy_aug = A.Compose([
    A.Resize(64, 128),
    A.ColorJitter(p=0.8),
    A.Rotate(limit=3, p=0.7),
    ToTensorV2()
], bbox_params=bbox_params)

hard_aug = A.Compose([
    A.Resize(64, 128),

    A.OneOf([  # 模糊保留但機率與強度降低
        A.MotionBlur(blur_limit=3, p=0.3),
        A.MedianBlur(blur_limit=3, p=0.3),
        A.GaussianBlur(blur_limit=(1, 2), p=0.3),
    ], p=0.4),

    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),  # 雜訊限縮
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),  # 更穩定亮度對比
    A.Perspective(scale=(0.01, 0.02), p=0.3),  # 降低視角扭曲
    A.Affine(rotate=(-4, 4), shear=(-3, 3), translate_percent=0.02, p=0.4),  # 降低仿射變換範圍
    A.ImageCompression(quality_lower=75, quality_upper=95, p=0.3),  # 保持基本清晰度

    ToTensorV2()
], bbox_params=bbox_params)


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
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def write_yolo_label(filepath, bboxes):
    with open(filepath, "w") as f:
        for box in bboxes:
            f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

def find_polygons(obj):
    if isinstance(obj, (list, tuple)):
        if (len(obj) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in obj)):
            yield obj
        else:
            for sub in obj:
                yield from find_polygons(sub)

def polygon2yolo(poly):
    xs, ys = zip(*poly)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        xmax - xmin,
        ymax - ymin
    ]

def extract_yolo_boxes(raw_boxes):
    yolo = []
    for poly in find_polygons(raw_boxes):
        box = polygon2yolo(poly)
        if box[2] > 0 and box[3] > 0:
            yolo.append(box)
    return yolo

def tensor_to_pil(img):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def is_valid_yolo_box(box):
    x, y, w, h = box
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    if w * h < 0.001:
        return False
    return True

def is_valid_bbox_list(boxes):
    return all(is_valid_yolo_box(b) for b in boxes) and len(boxes) > 0

def is_meaningful_image(image_tensor):
    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.cpu().numpy()
        if img.max() - img.min() < 0.05:
            return False
    return True

# === 主處理流程 ===
for split in splits:
    dataset = load_dataset("EZCon/taiwan-license-plate-detection", split=split)
    print(f"\n📂 Split: {split}, Total samples: {len(dataset)}")

    valid_count, skip_count = 0, 0

    for i in tqdm(range(len(dataset))):
        try:
            ex = dataset[i]
            raw_polys = ex["xyxyxyxyn"]
            yolo_boxes = extract_yolo_boxes(raw_polys)
            if not is_valid_bbox_list(yolo_boxes):
                skip_count += 1
                continue

            image = Image.open(BytesIO(ex["image"]["bytes"])).convert("RGB")
            image_np = np.asarray(image)
            class_labels = [0] * len(yolo_boxes)
            image_id = f"{i + 1:05}"

            # 原圖儲存
            name = f"{split}_origin_ezcon_{image_id}"
            image.save(os.path.join(base_dir, "images", split, f"{name}.jpg"))
            write_yolo_label(os.path.join(base_dir, "labels", split, f"{name}.txt"), yolo_boxes)
            valid_count += 1

            # === 增強比例 ===
            # 調整這裡的比例參數
            generate_origin = True
            generate_easy = random.random() < 0.4  # 輕度增強比率提高
            generate_hard = random.random() < 0.15  # 重度降至15%
            generate_occlusion = random.random() < 0.08  # 遮蔽限制在8%

            # 輕度增強
            if generate_easy:
                augmented = easy_aug(image=image_np, bboxes=yolo_boxes, class_labels=class_labels)
                if is_valid_bbox_list(augmented["bboxes"]) and is_meaningful_image(augmented["image"]):
                    name = f"{split}_augmented(easy)_ezcon_{image_id}"
                    tensor_to_pil(augmented["image"]).save(os.path.join(base_dir, "images", split, f"{name}.jpg"))
                    write_yolo_label(os.path.join(base_dir, "labels", split, f"{name}.txt"), augmented["bboxes"])
                    valid_count += 1
                else:
                    skip_count += 1

            # 重度增強
            if generate_hard:
                augmented = hard_aug(image=image_np, bboxes=yolo_boxes, class_labels=class_labels)
                if is_valid_bbox_list(augmented["bboxes"]) and is_meaningful_image(augmented["image"]):
                    name = f"{split}_augmented(hard)_ezcon_{image_id}"
                    tensor_to_pil(augmented["image"]).save(os.path.join(base_dir, "images", split, f"{name}.jpg"))
                    write_yolo_label(os.path.join(base_dir, "labels", split, f"{name}.txt"), augmented["bboxes"])
                    valid_count += 1
                else:
                    skip_count += 1

            # 遮擋增強
            if generate_occlusion:
                occ_img = apply_occlusion_safe(image.copy())
                occ_jpg = jpeg_compress(occ_img, quality=random.randint(30, 60))
                occ_np = np.array(occ_jpg)
                augmented = hard_aug(image=occ_np, bboxes=yolo_boxes, class_labels=class_labels)
                if is_valid_bbox_list(augmented["bboxes"]) and is_meaningful_image(augmented["image"]):
                    name = f"{split}_augmented(occlusion)_ezcon_{image_id}"
                    tensor_to_pil(augmented["image"]).save(os.path.join(base_dir, "images", split, f"{name}.jpg"))
                    write_yolo_label(os.path.join(base_dir, "labels", split, f"{name}.txt"), augmented["bboxes"])
                    valid_count += 1
                else:
                    skip_count += 1

        except Exception as e:
            skip_count += 1
            print(f"❌ Error at {split} index {i}: {e}")

    print(f"✅ Split '{split}' summary: Valid={valid_count}, Skipped={skip_count}")


# === 主處理流程2: 處理 keremberke/license-plate-object-detection 資料集 ===
for split in splits:
    dataset = load_dataset("keremberke/license-plate-object-detection", "full", split = split)
    print(f"\n📂 Split: {split}, Total samples: {len(dataset)}")

    valid_count, skip_count = 0, 0

    for i in tqdm(range(len(dataset))):
        try:
            ex = dataset[i]
            image = ex["image"]
            width = ex["width"]
            height = ex["height"]
            image_np = np.asarray(image)

            # YOLO 格式轉換
            yolo_boxes = []
            object_info = ex["objects"]
            bboxes = object_info["bbox"]  # 這是一個 list of [x, y, w, h]
            width = ex["width"]
            height = ex["height"]

            for box in bboxes:
                x, y, w, h = box
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                box_w = w / width
                box_h = h / height
                yolo_box = [x_center, y_center, box_w, box_h]
                if is_valid_yolo_box(yolo_box):
                    yolo_boxes.append(yolo_box)

            if not is_valid_bbox_list(yolo_boxes):
                skip_count += 1
                continue

            class_labels = [0] * len(yolo_boxes)
            image_id = f"{i + 1:05d}"

            # 儲存原始圖片與標註
            name = f"{split}_origin_kerem_{image_id}"
            image.save(os.path.join(base_dir, "images", split, f"{name}.jpg"))
            write_yolo_label(os.path.join(base_dir, "labels", split, f"{name}.txt"), yolo_boxes)
            valid_count += 1

            # === 增強策略 ===
            generate_origin = True
            generate_easy = random.random() < 0.4  # 輕度增強比率提高
            generate_hard = random.random() < 0.15  # 重度降至15%
            generate_occlusion = random.random() < 0.08  # 遮蔽限制在8%

            # 輕度增強
            if generate_easy:
                augmented = easy_aug(image=image_np, bboxes=yolo_boxes, class_labels=class_labels)
                if is_valid_bbox_list(augmented["bboxes"]) and is_meaningful_image(augmented["image"]):
                    name = f"{split}_augmented(easy)_kerem_{image_id}"
                    tensor_to_pil(augmented["image"]).save(os.path.join(base_dir, "images", split, f"{name}.jpg"))
                    write_yolo_label(os.path.join(base_dir, "labels", split, f"{name}.txt"), augmented["bboxes"])
                    valid_count += 1
                else:
                    skip_count += 1

            # 重度增強
            if generate_hard:
                augmented = hard_aug(image=image_np, bboxes=yolo_boxes, class_labels=class_labels)
                if is_valid_bbox_list(augmented["bboxes"]) and is_meaningful_image(augmented["image"]):
                    name = f"{split}_augmented(hard)_kerem_{image_id}"
                    tensor_to_pil(augmented["image"]).save(os.path.join(base_dir, "images", split, f"{name}.jpg"))
                    write_yolo_label(os.path.join(base_dir, "labels", split, f"{name}.txt"), augmented["bboxes"])
                    valid_count += 1
                else:
                    skip_count += 1

            # 遮擋增強
            if generate_occlusion:
                occ_img = apply_occlusion_safe(image.copy())
                occ_jpg = jpeg_compress(occ_img, quality=random.randint(30, 60))
                occ_np = np.array(occ_jpg)
                augmented = hard_aug(image=occ_np, bboxes=yolo_boxes, class_labels=class_labels)
                if is_valid_bbox_list(augmented["bboxes"]) and is_meaningful_image(augmented["image"]):
                    name = f"{split}_augmented(occlusion)_kerem_{image_id}"
                    tensor_to_pil(augmented["image"]).save(os.path.join(base_dir, "images", split, f"{name}.jpg"))
                    write_yolo_label(os.path.join(base_dir, "labels", split, f"{name}.txt"), augmented["bboxes"])
                    valid_count += 1
                else:
                    skip_count += 1

        except Exception as e:
            skip_count += 1
            print(f"❌ Error at {split} index {i}: {e}")

    print(f"✅ Split '{split}' summary: Valid={valid_count}, Skipped={skip_count}")

