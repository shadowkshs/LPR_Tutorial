from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# 訓練模型
model.train(
    data='./LPR_detection/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='lpr_yolo11n',
    augment=False,
    save_period=5,
    workers=4  # 可視環境調整
)

metrics = model.val()
print(metrics)


# Perform object detection on an image
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model

