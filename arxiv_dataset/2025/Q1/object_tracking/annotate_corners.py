all_camera_names = ['table-side', 'table-top', 'back', 'counter-top', 'ceiling', 'front']
corners = ['ul', 'ur', 'lr', 'll'] # how the corners of the table/counter are called
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import yaml
from config import example_annotation as pix_pos_by_cam, sessions, trials

def click_event(event, x, y, flags, params):
    global pix_pos_by_cam, cam, corner, rectangle
    if cam and rectangle and corner and event == cv2.EVENT_LBUTTONDOWN:
        pix_pos_by_cam[cam][rectangle][corner] = [x, y]
        # print(f'Position for {cam}-camera, {corner} corner of {rectangle}: {[x, y]}')
        cv2.imshow(window_name, get_annottated_img(camera_name))

color_by_obj = {'table': np.array([255,255,0], dtype=np.uint8), 'counter': np.array([0,255,255], dtype=np.uint8), 'origin': np.array([255,0,255], dtype=np.uint8)}

def get_img(session, trial, camera_name):
    path_prefix = f'data/{session}/{trial}/s{str(session).zfill(3)}t{str(trial).zfill(2)}.'
    cap = cv2.VideoCapture(path_prefix + 'video.' + camera_name + '.data.mp4')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fc = 0
    ret = True
    frames_to_avg = [] # we want avg multiple frames, so it is unlikely that points to annottate are blocked by person in front of them
    while ret and fc < total_frames:
        ret, frame = cap.read()
        if fc in [100, 300, 500, 700, 900]:
            frames_to_avg.append(frame)
        fc +=1
    cap.release()
    cv2.imwrite(f'tempimg.png', np.mean(frames_to_avg, axis=0))
    return


def get_annottated_img(camera_name):
    img = example_frame[camera_name].copy()
    for obj in pix_pos_by_cam[camera_name]:
        for corner in pix_pos_by_cam[camera_name][obj]:
            corner = pix_pos_by_cam[camera_name][obj][corner]
            if not (corner is None):
                for i in range(-10, 11):
                    try:
                        img[corner[1]+i, corner[0]+i] = color_by_obj[obj]
                        img[corner[1]+i, corner[0]-i] = color_by_obj[obj]
                    except:
                        pass
    return img

for session in sessions:
    for trial in trials:
        example_frame = {}
        cam = None
        corner = None
        rectangle = None # table, counter, origin
        camera_names_to_write = [] # only write annotations if we pressed 's'
        i = 0
        while True:
            camera_name = all_camera_names[i]
            i += 1
            if not(camera_name in pix_pos_by_cam):
                pix_pos_by_cam[camera_name] = {} # camera did not exist in previous annottation so we need to add cameraname as key
                pix_pos_by_cam[camera_name]['counter'] = {corners[0]: None, corners[1]: None, corners[2]: None, corners[3]: None}
                pix_pos_by_cam[camera_name]['table'] = {corners[0]: None, corners[1]: None, corners[2]: None, corners[3]: None}
                pix_pos_by_cam[camera_name]['origin'] = {corners[0]: None}
            try:
                get_img(session, trial, camera_name)
                example_frame[camera_name] = cv2.imread(f'tempimg.png')
                assert not(example_frame[camera_name] is None)
            except:
                cv2.destroyAllWindows()
                if i == len(all_camera_names):
                    break
                continue
            cam = camera_name
            global window_name
            window_name = f'{camera_name}: session {session}, trial {trial}'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 2500, 1400)
            img = get_annottated_img(camera_name)
            cv2.imshow(window_name, img)
            cv2.imshow(window_name, get_annottated_img(camera_name))
            cv2.setMouseCallback(window_name, click_event)
            while True:
                pressed_key = cv2.waitKey(0)
                if pressed_key == ord('c'):
                    rectangle = 'counter'
                    pix_pos_by_cam[camera_name]['counter'] = {corners[0]: None, corners[1]: None, corners[2]: None, corners[3]: None}
                    cv2.imshow(window_name, get_annottated_img(camera_name))
                if pressed_key == ord('t'):
                    rectangle = 'table'
                    pix_pos_by_cam[camera_name]['table'] = {corners[0]: None, corners[1]: None, corners[2]: None, corners[3]: None}
                    cv2.imshow(window_name, get_annottated_img(camera_name))
                if pressed_key == ord('o'):
                    pix_pos_by_cam[camera_name]['origin'] = {corners[0]: None}
                    cv2.imshow(window_name, get_annottated_img(camera_name))
                    rectangle = 'origin'
                    corner = corners[0]
                if pressed_key == ord('q'):
                    corner = corners[0]
                if pressed_key == ord('e'):
                    corner = corners[1]
                if pressed_key == ord('d'):
                    corner = corners[2]
                if pressed_key == ord('a'):
                    corner = corners[3]
                if pressed_key == ord('s'): # next image
                    camera_names_to_write.append(camera_name)
                    rectangle = None
                    corner = None
                    cv2.destroyAllWindows()
                    break
                if pressed_key == ord('x'): # this camera should not be annottated
                    #del pix_pos_by_cam[camera_name]
                    rectangle = None
                    corner = None
                    cv2.destroyAllWindows()
                    break
                if pressed_key == ord('r'): # go back one image within trial
                    i = max(-1, i-2)
                    rectangle = None
                    corner = None
                    cv2.destroyAllWindows()
                    break
            if i == len(all_camera_names):
                break
        if len(camera_names_to_write) > 0:
            pix_pos_by_cam_to_upload = {cam: pix_pos_by_cam[cam] for cam in camera_names_to_write}
            path_prefix = f'data/{session}/{trial}/s{str(session).zfill(3)}t{str(trial).zfill(2)}.'
            upload_path = f'{path_prefix}mocap.objecttracking.imageannotations.yaml'
            with open(upload_path, 'w') as outfile:
                yaml.dump(pix_pos_by_cam_to_upload, outfile, default_flow_style=False, sort_keys=False)