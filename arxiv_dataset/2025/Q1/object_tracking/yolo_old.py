import lakefs_config as cfg
from util import create_boto3_client, dirname_for_upload, create_boto3_session, get_aux_config, get_aux_json

import numpy as np
from ultralytics import YOLO
import tempfile
import botocore
import awswrangler as wr
import re
import cv2
import yaml
import torch

bounds = get_aux_json('bounds')
scope = get_aux_config()

# abort if this Sensor is not selected in scope_config.json
assert('ObjectTrackingYOLO' in scope["sensors_out"])

# cpu would take too long
assert torch.cuda.is_available(), 'Cuda not available'

yolo_image_width = 1280
yolo_conf_threshold = 0.2
batch_size = 30 # depends on vram available
all_camera_names_gt = ['table-side',
                       'table-top',
                       'back',
                       'counter-top',
                       'ceiling',
                       'front',
                       'pov']

# download trained yolo model
s3 = create_boto3_client()
model_file = tempfile.NamedTemporaryFile(suffix='.pt')
s3.download_fileobj(
    Bucket=cfg.repo,
    Key=f'{cfg.download_branch}/yolo_object_tracking.pt',
    Fileobj=model_file)
model = YOLO(model_file.name)

my_session = create_boto3_session()
wr.config.s3_endpoint_url = cfg.endpoint
for session in bounds.keys():
    for trial in bounds[session].keys():
        # -----------------------------------
        # find paths of videos for this trial
        # -----------------------------------
        paths = []
        prefix = dirname_for_upload(session, trial, False)
        paths.extend(wr.s3.list_objects(f"s3://{cfg.repo}/{prefix}*",
                                        boto3_session=my_session))
        paths = [p for p in paths if re.match('^.*\.data\.mp4$', p)]
        paths = [p.replace(f's3://{cfg.repo}/', '') for p in paths]
        for path in paths:
            # --------------
            # download video
            # --------------
            camera_name = path.split('.data')[0].split('.')[-1]
            if not any([cam in path for cam in all_camera_names_gt]):
                 print(f'skipping unknown camera {camera_name}')
                 continue
            print(f'downloading session {session} trial {trial} camera {camera_name}')
            with tempfile.NamedTemporaryFile() as vid_file:
                s3.download_fileobj(Bucket=cfg.repo, Key=path, Fileobj=vid_file)
                vid_file.seek(0)
                cap = cv2.VideoCapture(vid_file.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fc = 0
                ret = True
                detections_dict = {}
                print('num frames:', total_frames)
                # process in batches of batch_size
                for x in [x for x in range(0, total_frames, batch_size)]:
                    batch = []
                    for i in range(batch_size):
                        if (x+i)%1000==0:
                            print('processing frame:', (x+i))
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
                    results = model.predict(batch,
                                            save=False,
                                            imgsz=yolo_image_width,
                                            conf=yolo_conf_threshold,
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
            prefix = dirname_for_upload(session, trial, False)
            upload_path = f'{prefix}.mocap.objecttracking.{camera_name}.yolodetections.yaml'
            dump = yaml.dump(detections_dict,
                             default_flow_style=False,
                             sort_keys=False).encode('UTF-8')
            s3.put_object(Body=bytes(dump), Bucket=cfg.repo, Key=upload_path)
            print(f'wrote {upload_path}')