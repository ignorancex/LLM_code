import os
import cv2
import numpy as np
import shutil
import json
from imantics import Mask, Polygons
from PIL import Image
from glob import glob
from tqdm import tqdm
# from iteration_utilities import duplicates
# from shapely.geometry import Polygon

ROOT_PATH = './MAS3K'
SUB_PATH = './sub_MAS3K'
mask_image_folder = 'Mask_Image'
dataset_types = ['train', 'test']
image_types = ['Image', 'Mask']
wrong_shape = 'wrong_shape'
many_instances = 'many_instances'
true_instances = 'Many_Instances_True'
mask_many_instances = 'mask_many_instances'
connect_component_folder = 'connectComponent'
largest_instance = 'largest_instance'
visualize = 'visualize'

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (0, 0, 0), (255, 255, 255)]


def extract_polygons():
    for data_type in dataset_types:
        mask_folder = os.path.join(ROOT_PATH, data_type, 'Mask')
        image_folder = os.path.join(ROOT_PATH, data_type, 'Image')
        image_mask_path = os.path.join(ROOT_PATH, data_type, mask_image_folder)
        true_instances_folder = os.path.join(ROOT_PATH,
                                             data_type,
                                             true_instances)
        save_folder = os.path.join(ROOT_PATH,
                                   data_type,
                                   visualize)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        mask_img_list = glob(true_instances_folder + '/**/*')

        for mask_img_file in tqdm(mask_img_list):
            img_id = mask_img_file.split('/')[-1].split('.')[0]
            poly_type = mask_img_file.split('/')[-2]

            img_path = os.path.join(image_folder, img_id + '.jpg')
            mask_path = os.path.join(mask_folder, img_id + '.png')

            img = cv2.imread(img_path)
            np_mask = np.array(Image.open(mask_path))
            polygons = Mask(np_mask).polygons()
            poly_list = polygons.segmentation

            if poly_type == 'largest':
                poly_list = list(sorted(poly_list, key=lambda a: len(a)))
                poly_list = [poly_list[-1]]
            elif 0 < int(poly_type) <= 10:

                tmp_poly_list = [poly for poly in poly_list if len(poly) > 100]

                if len(tmp_poly_list) > 0:
                    poly_list = tmp_poly_list
                else:
                    poly_list = [poly_list[-1]]

            for i, poly in enumerate(poly_list):
                poly_np = np.array([[[int(poly[2*i]), int(poly[2*i+1])]
                                     for i in range(len(poly)//2)]]).astype(np.int32)
                img = cv2.polylines(img, poly_np, True,
                                    colors[i % len(colors)], 4)
                cv2.imwrite(os.path.join(
                    save_folder, img_id + '.jpg'), img)


def visualize_json(json_path):
    with open(json_path, 'r') as json_file:
        annotations = json.load(json_file)

    json_file.close()
    categories = [item['name'] for item in annotations['categories']]
    for img_info in tqdm(annotations['images']):
        img_id = img_info['id']
        img_path = img_info['file_name'][1:]
        img = cv2.imread(img_path)
        img_file_id = img_path.split('/')[-1]
        i = 0
        save_folder = os.path.join('./visualize', *img_path.split('/')[2:-1])
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for ann in annotations['annotations']:

            image_id = ann['image_id']
            if image_id == img_id:
                segmentation = ann['segmentation']
                label = categories[ann['category_id']]
                x1, y1, w, h = list(map(int, ann['bbox']))
                x2 = x1 + w
                y2 = y1 + h
                for poly in segmentation:
                    poly_np = np.array([[[int(poly[2*i]), int(poly[2*i+1])]
                                        for i in range(len(poly)//2)]]).astype(np.int32)
                    img = cv2.polylines(img, poly_np, True,
                                        colors[i % len(colors)], 4)

                    img = cv2.rectangle(
                        img, (x1, y1), (x2, y2), colors[i % len(colors)], 2)
                    img = cv2.putText(img, label, (x1+10, y1+25), cv2.FONT_HERSHEY_SIMPLEX,
                                      1, colors[i % len(colors)], 2, cv2.LINE_AA)
                    i += 1
        cv2.imwrite(os.path.join(save_folder, img_file_id), img)

        # break


def main():
    # visualize_json('./MERGE_OCEAN/Annotations/merge_ocean_train.json')
    # visualize_json('./MERGE_OCEAN/Annotations/merge_ocean_test.json')
    visualize_json('./MERGE_OCEAN/Annotations/augmentation.json')


if __name__ == '__main__':
    main()
