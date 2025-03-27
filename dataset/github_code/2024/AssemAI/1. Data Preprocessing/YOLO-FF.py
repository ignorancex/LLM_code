import torch
from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# Train the YOLOv8 model

print("starting with the training")
model = YOLO("yolov8s.yaml")
torch.cuda.memory_summary(device=None, abbreviated=False)
model.train(data="/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/data.yaml",project="/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/yolo_output_50epoch/output",epochs=50,imgsz=512,patience=10, plots=True)
