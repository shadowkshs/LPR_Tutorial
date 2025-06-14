# LPR_Tutorial 使用說明

本專案示範如何建構從車牌偵測、圖像強化到文字辨識的完整流程，並透過微調模型提升對台灣車牌的辨識效果。

---

## 📥 專案下載

```bash
git clone https://github.com/shadowkshs/LPR_Tutorial.git
```

---

## 📦 安裝套件

```bash
pip install -r requirements.txt
```

---

## 🚀 執行流程

### Step 1：下載偵測資料集並進行資料擴增

使用 EZCon/taiwan-license-plate-detection 與 keremberke/license-plate-object-detection 資料集，擴增後生成 `LPR_detection/` 資料夾並建立 YOLO 所需的 `data.yaml`。

```bash
python 01_Detection_dataset_download_augment.py
```

---

### Step 2：訓練 YOLO 偵測模型

使用 `yolo11n.pt` 模型進行微調，訓練完成後可於 `runs/detect/lpr_yolo11n/weights/` 中取得 `best.pt`。

```bash
python 02_Detection_model_train.py
```

---

### Step 3：下載辨識資料集

下載 EZCon/taiwan-license-plate-recognition 資料集，並產生 `LPR_recognition/` 資料夾。

```bash
python 03_Recognition_dataset_download.py
```

---

### Step 4：進行車牌切割

使用訓練好的 YOLO 模型對辨識資料集進行車牌偵測並裁切，產生 `cropped_plates/`。

```bash
python 04_Detection.py
```

---

### Step 5：資料擴增與超解析

對 `cropped_plates/` 中解析度過低的圖像進行增強與超解析，使用 `CompVis/ldm-super-resolution-4x-openimages` 模型，輸出結果至 `cropped_train/`。

```bash
python 05_Recognition_dataset_SR_augment.py
```

---

### Step 6：微調 TrOCR 模型

以 `microsoft/trocr-base-printed` 為基礎，針對台灣車牌字體進行微調，輸出模型存於 `trocr-plate/`。

```bash
python 06_Recognition_model_train.py
```

---

### Step 7：評估辨識效果

使用 `cropped_train/test/` 資料進行模型評估：

- **Exact Match Accuracy**: 93.98% (281/299)
- **Character-Level Accuracy**: 98.11% (1968/2006)

```bash
python 07_Recognition.py
```

---

### Step 8：整合全流程測試

使用 `real_test/` 中的圖片進行偵測與辨識，並將結果與評分輸出至 `real_output/`。請確保圖片命名格式為真實車牌（如 `123-ABC.jpg`）。

```bash
python 08_Full_Recognition_flow.py
```

---

## ✅ 完整流程涵蓋

- 車牌偵測與微調訓練（YOLO）
- 圖像超解析與資料擴增
- 光學辨識（TrOCR）與字體適應微調
- 全流程整合與實測輸出
