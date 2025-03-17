""" Save 3 frames form a RGB video, the same 3 frames from the D camera """
from utils import get_filenames
import random
import cv2
import os
import numpy as np
random.seed(1)

mode = "depth"
if mode == "rgb":
    path_to_video_dir = "/media/sealab-ws/Hard Disk/CORSMAL challenge/train/view3/rgb"
    path_to_dest = "/media/sealab-ws/Hard Disk/CORSMAL challenge/IMAGES/rgb"

    # retrieve file names
    path_to_rgb_video = get_filenames(path_to_video_dir, ".mp4")
    path_to_rgb_video.sort()
    ind = 591 # random.randint(a=0, b=len(path_to_rgb_video))

    video_cap = cv2.VideoCapture(path_to_rgb_video[ind])
    offset = 1
    counter = 0
    while video_cap.isOpened():
        ret, bgr_frame = video_cap.read()

        # if frame is correctly retrieved
        if ret:
            cv2.imwrite(os.path.join(path_to_dest, "{}.png".format(counter)), bgr_frame)
            counter += offset
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
            cv2.imshow("", bgr_frame)
            cv2.waitKey(10)
    video_cap.release()
    cv2.destroyAllWindows()
elif mode == "depth":
    path_to_depth_dir = "/media/sealab-ws/Hard Disk/CORSMAL challenge/IMAGES/patches/depth"
    path_to_depth_frames = get_filenames(path_to_depth_dir, ".png")
    path_to_depth_frames.sort()
    path_to_dest = "/media/sealab-ws/Hard Disk/CORSMAL challenge/IMAGES/patches/depth"
    MAX_DIST = 2000
    for i in range(0, len(path_to_depth_frames)):
        depth_frame = cv2.imread(path_to_depth_frames[i], -1)
        depth_frame = (depth_frame/MAX_DIST * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(path_to_dest, "{}.png".format(i)), depth_frame)
        cv2.imshow("", depth_frame)
        cv2.waitKey(10)