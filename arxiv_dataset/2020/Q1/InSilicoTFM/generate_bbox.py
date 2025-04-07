import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

datasets_path = ["datasets/training", "datasets/testing", "datasets/validation"]
categories = ["RAW", "FRET", "FORCE"]


def get_main_bbox(image, threshold=100):
    _, image_th = cv2.threshold(image, threshold, 65535, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint16)
    image_closed = cv2.morphologyEx(image_th, cv2.MORPH_CLOSE, kernel)
    image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, kernel)
    image_opened = np.uint8(image_opened)
    contours, hierarchy = cv2.findContours(
        image_opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    best_bbox = None
    best_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > best_area:
            best_area = w * h
            best_bbox = (x, y, w, h)
    return best_bbox


for dataset_path in datasets_path:
    print(f"Generating bounding boxes in {dataset_path}")
    for folder in tqdm(sorted(os.listdir(dataset_path))):
        folder_path = os.path.join(dataset_path, folder)
        # It must be a folder
        if not os.path.isdir(folder_path):
            continue
        # It must end with RAW
        if not folder.endswith("RAW"):
            continue
        # Let's find the bboxes
        bbox_path = os.path.join(dataset_path, f"{folder[:-3]}_bbox.csv")
        f = open(bbox_path, "w+")
        f.write("filename,x,y,w,h\n")

        for file in sorted(os.listdir(folder_path)):
            # Reading the image
            filename = os.path.join(folder_path, file)
            image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            # Getting the main bouding box
            bbox = get_main_bbox(image)
            x, y, w, h = bbox
            f.write(f"{filename},{','.join(map(str, bbox))}\n")
        f.close()
