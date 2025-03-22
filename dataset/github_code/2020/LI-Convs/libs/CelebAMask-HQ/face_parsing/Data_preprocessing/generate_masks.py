import os
import cv2
import numpy as np
import argparse
import seaborn as sbr

def make_folder(path):
	if not os.path.exists(os.path.join(path)):
		os.makedirs(os.path.join(path))

def visual_label(label, col_set):
	unique_classes = np.unique(label)
	unique_classes = np.sort(unique_classes)
	label_vis = np.zeros(label.shape[0:2]+(3,), dtype=np.uint8)
	for class_id in unique_classes:
		if class_id > 0:  # no need to fill background class
			label_vis[label==class_id] = np.asarray([c*255 for c in col_set[class_id]],dtype=np.uint8)
	return label_vis

#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2	 
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

DATASET_ROOT = '../../../../../dataset/CelebAMask-HQ'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default=DATASET_ROOT,
					help='The path of CelebAMask-HQ dataset')
args = parser.parse_args()

folder_base = os.path.join(args.data_root.strip(), 'CelebAMask-HQ-mask-anno')
folder_save = os.path.join(args.data_root.strip(), 'CelebAMask-HQ-mask')
mask_plots_save = os.path.join(args.data_root.strip(), 'CelebAMask-HQ-mask-plots')

img_num = 30000

make_folder(folder_save)
make_folder(mask_plots_save)

col_set = sbr.color_palette("Paired")+sbr.color_palette("Set2")

for k in range(img_num):
	folder_num = k // 2000
	im_base = np.zeros((512, 512),dtype=np.uint8)

	for idx, label in enumerate(label_list):
		filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
		if os.path.exists(filename):
			print(label, idx+1)
			im=cv2.imread(filename)
			im = im[:, :, 0]
			im_base[im != 0] = (idx + 1)

	im_plot = visual_label(im_base,col_set)

	filename_save = os.path.join(folder_save, str(k) + '.png')
	print(filename_save)
	cv2.imwrite(filename_save, im_base)

	plot_save_path = os.path.join(mask_plots_save, str(k) + '.png')
	cv2.imwrite(plot_save_path, im_plot)


