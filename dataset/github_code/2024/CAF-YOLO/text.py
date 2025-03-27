from ultralytics import YOLO

# 预测
model = YOLO("E:/ultralytics-main/runs/train18/weights/best.pt")
source = ("E:/yolov8-pytorch-master/dataset/images/BloodImage_00002.jpg")
results = model.predict(source, save=True, show_conf=True)  # predict on an image
print('ok')