from ultralytics import YOLO


def test_yolov8():
    model = YOLO(
        (
            "/mnt/Enterprise/safal/AI_assisted_microscopy_system/outputs/"
            "smartphone_sample/yolov8/fold_5/mmdetection_cysts/"
            "yolov8_smartphone_sample_fold_5/weights/best.pt"
        )
    )

    metrics = model.val(
        data="configs/yolov8/data/smartphone_test.yaml",
        device=1,
        # split="test",
        save_json=True,
    )

    print(metrics)


if __name__ == "__main__":
    test_yolov8()
