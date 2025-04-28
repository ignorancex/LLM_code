from ultralytics import YOLO


def train_yolov8():
    model = YOLO("yolov8s.pt")

    model.train(
        epochs=50,
        batch=16,
        data="configs/yolov8/data/smartphone_sample.yaml",
        device=1,
        name="yolov8_smartphone_sample_fold_1",
        project="/mnt/Enterprise/safal/AI_assisted_microscopy_system/outputs/smartphone_sample/yolov8/fold_1/mmdetection_cysts",
        seed=42,
        lr0=0.001,
        lrf=0.001,
    )


if __name__ == "__main__":
    train_yolov8()
