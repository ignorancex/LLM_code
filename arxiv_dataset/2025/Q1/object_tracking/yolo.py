from ultralytics import YOLO
import re
import cv2
import yaml
import torch
import os

from config import sessions, trials, all_camera_names

# cpu takes too long
assert torch.cuda.is_available(), 'Cuda needed for runtime reasons'

yolo_image_width = 1280
yolo_conf_threshold = 0.2
all_camera_names_gt = ['table-side', 'table-top', 'back', 'counter-top', 'ceiling', 'front']

# download trained yolo model
model = YOLO('data/yolo_net.pt')

batch_size = 5 # depends on vram available
for session in sessions:
    for trial in trials:
        # -----------------------------------
        # find paths of videos for this trial
        # -----------------------------------
        path_prefix = f'data/{session}/{trial}/s{str(session).zfill(3)}t{str(trial).zfill(2)}.'
        for camera_name in [camera_name for camera_name in all_camera_names]:
            vid_path = path_prefix + 'video.' + camera_name + '.data.mp4'
            if not(os.path.isfile(vid_path)):
                print('did not find video file', vid_path)
                continue
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fc = 0
            ret = True
            detections_dict = {}
            print('num frames:', total_frames)
            # process in batches of batch_size
            for x in [x for x in range(0, total_frames, batch_size)]:
                batch = []
                for i in range(batch_size):
                    if (x+i)%100==0:
                        print('processing:', (x+i))
                    if not(ret and fc < total_frames):
                        break
                    ret, frame = cap.read()
                    batch.append(frame)
                    fc += 1
                if len(batch) == 0:
                    break
                # ----------------
                # yolo predictions
                # ----------------
                results = model.predict(
                    batch,
                    save=False,
                    imgsz=yolo_image_width,
                    conf=yolo_conf_threshold,
                    rect=True,
                    verbose=False,
                    device=0)
                id_to_class_name = results[0].names
                for i, result in enumerate(results):
                    detected_classes_as_int = [
                        int(d) for d in result.boxes.cls.cpu().numpy().tolist()]
                    detected_classes = [
                        id_to_class_name[d] for d in detected_classes_as_int]
                    # these classes are too similar to seperate
                    for j in range(len(detected_classes)):
                        spoons = ['utensil-spoon, salad', 'utensil-fork, salad']
                        if detected_classes[j] in ['salt', 'sugar']:
                            detected_classes[j] = 'salt/sugar'
                        elif detected_classes[j] in spoons:
                            detected_classes[j] = 'utensil, salad'
                    detections_in_image_coords = [
                        box[:4].cpu().numpy().tolist() for box in result.boxes.xywh]
                    conf = [c for c in result.boxes.conf.cpu().numpy().tolist()]
                    detections_dict[str(x+i)] = [
                        [cl, cords, conf] for cl, cords, conf in zip(
                            detected_classes, detections_in_image_coords, conf)]
            upload_path = f'{path_prefix}mocap.objecttracking.{camera_name}.yolodetections.yaml'
            with open(upload_path, 'w') as outfile:
                yaml.dump(detections_dict, outfile, default_flow_style=False, sort_keys=False)