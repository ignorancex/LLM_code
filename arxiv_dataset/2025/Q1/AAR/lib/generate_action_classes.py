""" Generates action classes for frames in Action Genome """
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
from glob import glob
import csv
import pickle
import os
import argparse
import ffmpeg
import datetime
import glob


def parse_charades_csv(filename):
	labels = {}
	vids = []
	with open(filename) as f:
		reader = csv.DictReader(f)
		for row in reader:
			vid = row['id']
			actions = row['actions']
			if actions == '':
				actions = []
			else:
				actions = [a.split(' ') for a in actions.split(';')] # split individual actions then their class and start and end time
				actions = [{'class': x, 'start': float(
					y), 'end': float(z)} for x, y, z in actions]
			labels[vid] = actions
			vids.append(vid)
	return vids, labels

def cls2int(x):
	return int(x[1:])


if __name__ == '__main__':
	# Download ffmpeg_2.8.15.orig.tar.xz
	# Extract ffmpeg_2.8.15.orig.tar.xz via -tar xvf 
	# Ensure yasm is downloaded.
	# Read instructions in INSTALL.md
	# make sure ffmpeg is 2.8.15 
	# after installing ffmpeg 2.8.15, do pip install ffmpeg-python before running this script
	actions = {}

	vids, labels = parse_charades_csv('../Charades_annotations/Charades_v1_train.csv')
	test_vids, test_labels = parse_charades_csv('../Charades_annotations/Charades_v1_test.csv')

	# combine train and test sets
	vids.extend(test_vids) # total dataset no. = 9848
	labels.update(test_labels) # total dataset no. = 9848

	path_to_vids = '/mnt/nvme/dataset/ag/videos/'
	path = glob.glob('/mnt/nvme/dataset/ag/frames/*/*')

	temp_vid_name = ''
	count = 0

	for frame in path:
		vid_name = frame.split('/')[-2].split('.mp4')[0]
		frame_name = frame.split('/')[-1]
		frame_no = int(frame.split('/')[-1].split('.')[0])

		if temp_vid_name != vid_name:
			count += 1
			temp_vid_name = vid_name

			metadata = ffmpeg.probe(os.path.join(path_to_vids, temp_vid_name + '.mp4'))
			video_stream = next((stream for stream in metadata['streams'] if stream['codec_type'] == 'video'), None)

			indiv_frame_duration = float(metadata['streams'][0]['duration']) / int(metadata['streams'][0]['nb_frames'])

		current_time = frame_no * indiv_frame_duration

		# individual frame action list
		frame_action_list = []

		for action in labels[vid_name]:
			if current_time >= action['start'] and current_time <= action['end']:
				action_class = int(action['class'].split('c')[-1])

				frame_action_list.append(action_class)

			actions[os.path.join(vid_name + '.mp4', frame_name)] = list(set(frame_action_list))

	print("Total number of videos: {}".format(count))

	with open('frame_action_list.pkl', 'wb') as file:
		pickle.dump(actions, file, protocol=pickle.HIGHEST_PROTOCOL)

	with open('frame_action_list.pkl', 'rb') as file:
		actions = pickle.load(file)