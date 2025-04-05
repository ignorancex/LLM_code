from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import pickle
import os

def get_all_paths():
	all_paths = []
	with open("../extract_faces_from_videos/All_videos/videonames.txt") as f:
		for line in f:
			all_paths.append(line.strip())
	return all_paths


ten_test_paths = ["50_ukZfEhLYCCE_542.mp4",
		"280_d6Oawpqklso_795.mp4",
		"83_VJOHO1h6GjI_242.mp4",
		"13_Opnpqqo--r0_580.mp4",
		"191_CT3TKsR8PpM_676.mp4",
		"91_SaboGMXREJ8_205.mp4",
		"142_PHFsg78j0tY_161.mp4",
		"154_19YrF892S3c_641.mp4",
		"102_VycNPRwt-qg_449.mp4",
		"238_P0QwXyGdAcw_448.mp4",
		"149_KN5GioP-sBA_246.mp4"]


def get_full_path(p):
	return os.path.join("../../MuSe-CAR/dataset/video",p)


def get_images(path, fps):
	capture = cv2.VideoCapture(path)
	success, image = capture.read()
	assert(success)
	multiplier = capture.get(cv2.CAP_PROP_FPS) / fps
	next_sample_frame = 1
	
	images = []
	while success:
		frame_id = int(round(capture.get(1)))
		if frame_id == int(next_sample_frame):
			next_sample_frame += multiplier
			images.append(image)
		success, image = capture.read()
	capture.release()
	return images

def extract_gocar(path_list, dst_dir = "quick_features", fps = 4):
		yolo = YOLO()
		for video_path in path_list:
			images = get_images(get_full_path(video_path), fps)
			features={}
			for i, img in enumerate(images):
				image = Image.fromarray(img)
				features[i*1000/fps]=yolo.detect_image(image)
			dst_path = os.path.join(dst_dir,video_path.split(".")[0] + ".pickle")
	
			with open(dst_path, 'wb') as handle:
				pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
