from pathlib import Path
import os
import json
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import evaluate
from transformers import TrainerCallback
import csv

# === 基本參數設定 ===
model_name = "microsoft/trocr-base-printed"
data_root = Path("./cropped_train")
splits = ["train", "validation", "test"]
output_dir = "./trocr-plate"

# === 載入圖片與標籤為 DatasetDict ===
dataset_splits = {}
for split in splits:
    with open(data_root / f"{split}_labels.json", "r") as f:
        label_dict = json.load(f)
    records = [{"image_path": str(data_root / split / fn), "label": lbl} for fn, lbl in label_dict.items()]
    dataset_splits[split] = Dataset.from_list(records)

raw_datasets = DatasetDict(dataset_splits)

# === 初始化 processor 和 model ===
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# === 前處理 function ===
def preprocess(examples):
    images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        examples["label"],
        padding="max_length",
        max_length=16,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": labels}

# === 對所有 splits 套用前處理 ===
processed_datasets = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets["train"].column_names)

# === 訓練參數 ===
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_steps=100,
    num_train_epochs=5,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir=f"{output_dir}/logs"
)

# === 評估指標（CER & WER） ===
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
    }

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "cer", "wer"])

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        epoch = int(state.epoch)
        loss = metrics.get("eval_loss", -1)
        cer = metrics.get("eval_cer", -1)
        wer = metrics.get("eval_wer", -1)

        with open(self.log_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, cer, wer])

# === 初始化 Trainer 並訓練 ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[MetricsLoggerCallback("./trocr-plate/training_log.csv")]
)

trainer.train()

# === 儲存模型與 processor ===
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

#==== plot train curve====
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./trocr-plate/training_log.csv")

plt.figure(figsize=(12, 6))

# Loss curve
plt.subplot(1, 3, 1)
plt.plot(df["epoch"], df["loss"], marker="o")
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# CER
plt.subplot(1, 3, 2)
plt.plot(df["epoch"], df["cer"], marker="o", color="green")
plt.title("CER")
plt.xlabel("Epoch")
plt.ylabel("Character Error Rate")

# WER
plt.subplot(1, 3, 3)
plt.plot(df["epoch"], df["wer"], marker="o", color="red")
plt.title("WER")
plt.xlabel("Epoch")
plt.ylabel("Word Error Rate")

plt.tight_layout()
plt.savefig("./trocr-plate/training_curve.png")
