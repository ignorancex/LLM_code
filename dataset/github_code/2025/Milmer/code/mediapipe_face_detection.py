import cv2
import mediapipe as mp
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection
model = mp_face_detection.FaceDetection(   
        min_detection_confidence=0.7, 
        model_selection=0,
)

def process_participant_data(participant_id):
    participant_str = f's{participant_id:02}'
    input_dir = f'./photo/{participant_str}/'
    output_dir = f'./faces/{participant_str}/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = os.listdir(input_dir)
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for image_file in image_files:
        filename, ext = os.path.splitext(image_file)
        if ext.lower() not in valid_image_extensions:
            continue

        img = cv2.imread(f'{input_dir}{image_file}')
        if img is None:
            print(f"Failed to load image at '{input_dir}{image_file}'")
            continue
        else:
            print(f"'{input_dir}{image_file}'")

        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.process(img_RGB)

        annotated_image = img.copy()

        detection = results.detections[0]
        
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        bbox = [int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)]
        
        # Calculate the new coordinates of the face frame, expand the distance by 10%.
        expansion_ratio = 0.1
        bbox_expanded = [max(0, bbox[0] - bbox[2] * expansion_ratio),  # left
                    max(0, bbox[1] - bbox[3] * expansion_ratio),  # top
                    min(iw, bbox[0] + bbox[2] * (1 + expansion_ratio)),  # right
                    min(ih, bbox[1] + bbox[3] * (1 + expansion_ratio))]  # bottom

        cv2.rectangle(annotated_image, (int(bbox_expanded[0]), int(bbox_expanded[1])), (int(bbox_expanded[2]), int(bbox_expanded[3])), (255,0,0), 2)

        face_image = img[int(bbox_expanded[1]):int(bbox_expanded[3]), int(bbox_expanded[0]):int(bbox_expanded[2])]
        
        cv2.imwrite('{}{}{}'.format(output_dir, filename, ext), face_image)

for participant_id in range(1, 23):
    process_participant_data(participant_id)