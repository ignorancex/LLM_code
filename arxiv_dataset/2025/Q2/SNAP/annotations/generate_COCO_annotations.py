# This script generates sensor bias annotations in the COCO format
# In sensor_bias dataset we take multiple images of the same scene with different camera parameters
# and illumination condition. Therefore, we only need to annotate one exemplar for each scene
# and then copy the annotations across all conditions

# run from the root directory
# python3 annotations/generate_sensor_bias_annotations.py

import json
import csv
import os
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np                                 # (pip install numpy)
import skimage.measure as measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
from tqdm import tqdm 

def create_sub_masks(mask_image):
	width, height = mask_image.size

	# Initialize a dictionary of sub-masks indexed by RGB colors
	sub_masks = {}
	for x in range(width):
		for y in range(height):
			# Get the RGB values of the pixel
			pixel = mask_image.getpixel((x,y))[:3]

			# If the pixel is not black...
			if pixel != (0, 0, 0):
				# Check to see if we've created a sub-mask...
				pixel_str = str(pixel)
				sub_mask = sub_masks.get(pixel_str)
				if sub_mask is None:
				   # Create a sub-mask (one bit per pixel) and add to the dictionary
					# Note: we add 1 pixel of padding in each direction
					# because the contours module doesn't handle cases
					# where pixels bleed to the edge of the image
					sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

				# Set the pixel value to 1 (default is 0), accounting for padding
				sub_masks[pixel_str].putpixel((x+1, y+1), 1)

	return sub_masks

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
	# Find contours (boundary lines) around each sub-mask
	# Note: there could be multiple contours if the object
	# is partially occluded. (E.g. an elephant behind a tree)
	contours = measure.find_contours(np.asarray(sub_mask), 0.5, positive_orientation='low')

	segmentations = []
	polygons = []
	for contour in contours:
		# Flip from (row, col) representation to (x, y)
		# and subtract the padding pixel
		for i in range(len(contour)):
			row, col = contour[i]
			contour[i] = (col - 1, row - 1)

		# Make a polygon and simplify it
		poly = Polygon(contour)
		poly = poly.simplify(1.0, preserve_topology=False)
		polygons.append(poly)
		segmentation = np.array(poly.exterior.coords).ravel().tolist()
		segmentations.append(segmentation)

	# Combine the polygons to calculate the bounding box and area
	multi_poly = MultiPolygon(polygons)
	x, y, max_x, max_y = multi_poly.bounds
	width = max_x - x
	height = max_y - y
	bbox = (x, y, width, height)
	area = multi_poly.area

	annotation = {
		'segmentation': [], #segmentations,
		'iscrowd': is_crowd,
		'image_id': image_id,
		'category_id': category_id,
		'id': annotation_id,
		'bbox': bbox,
		'area': area
	}

	return annotation

# setup dictionary to convert category names to COCO ids
cat2id = {}
categories = []
# get 80 categories of objects for COCO
with open('annotations/coco_categories.csv', 'r') as fid:
	reader = csv.DictReader(fid)
	for row in reader:
		cat2id[row['name']] = int(row['id'])
		row['id'] = int(row['id'])
		categories.append(row)


# setup structs for COCO descriptions

info = {'description': "sensor bias dataset", 
			'url': None, 'version': '1.0',
			'year': 2024,
			'contributor': 'Iuliia Kotseruba',
			'date_created': '2024/07/08'}
licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"}]

img_dict = {}
img_dict['license'] = 1
img_dict['id'] = None
img_dict['file_name'] = None
img_dict['height'] = 640
img_dict['width'] = 960
img_dict['flickr_url'] = ''
img_dict['coco_url'] = ''
img_dict['date_captured'] = '2024/10/22'

annot_dict = {}
annot_dict['id'] = None
annot_dict['image_id'] = None
annot_dict['category_id'] = None
annot_dict['segmentation'] = [],
annot_dict['area'] = None
annot_dict['bbox'] = None
annot_dict['iscrowd'] = 0 # is the box drawn around multiple objects



images = []
annotations = []

data_df = pd.read_excel('annotations/SNAP.xlsx')
data_df['Num_objects'] = 0

mask_dir = 'annotations/masks_SNAP/'

name2id = {}

mask_imgs = []

annot_id = 0
object_names = list(data_df['Object_name'].unique())

for obj_name in tqdm(object_names):

	temp = obj_name.split('_')
	obj_id = temp[0]
	obj_cat = ' '.join(temp[1:])

	if obj_cat == 'comic book':
		obj_cat = 'book'
	elif obj_cat == 'water bottle':
		obj_cat = 'bottle'

	category_id = cat2id[obj_cat]

	# resize the mask image to the size of the images in the dataset
	mask_img = Image.open(f'{mask_dir}/{obj_name}_10_auto.png')
	w, h = mask_img.size
	# use nearest neighbor resampling to preserve the colors of the original mask
	mask_img = mask_img.resize((int(w/2), int(h/2)), Image.Resampling.NEAREST)

	sub_masks = create_sub_masks(mask_img)

	num_objects = len(sub_masks)
	data_df.loc[data_df['Object_id'] == int(obj_id), 'Num_objects'] = num_objects

	mask_annotations = []
	for color, sub_mask in sub_masks.items():
		mask_annotations.append(create_sub_mask_annotation(sub_mask, 0, category_id, 0, 0))

	img_df = data_df[data_df['Object_id'] == int(obj_id)]

	for img_idx, row in img_df.iterrows():
		img_dict['id'] = img_idx
		img_dict['file_name'] = row['Path']
		images.append(img_dict.copy())

		for annotation in mask_annotations:
			annotation['id'] = annot_id
			annotation['image_id'] = img_idx
			annotations.append(annotation.copy())
			annot_id += 1

# make sure the number of images in the generated annotations
# is the same as the number of images in the database
assert len(images) == len(data_df)

# annotations for all images
annotations_all = {}
annotations_all['info'] = info
annotations_all['licenses'] = licenses
annotations_all['categories'] = categories
annotations_all['images'] = images
annotations_all['annotations'] = annotations

with open('annotations/sensor_bias_COCO_v5.json', 'w') as fid:
	json.dump(annotations_all, fid, ensure_ascii=False, indent=2)

data_df.to_excel('annotations/sensor_bias_data_v5.xlsx')