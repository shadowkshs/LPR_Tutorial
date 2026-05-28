from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="./LPR_detection/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,        
        device=0,
        project="./runs",
        name="LPR_detection_yolo26n",
        exist_ok=True
    )


if __name__ == "__main__":
    freeze_support()
    main()
