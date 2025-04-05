import cv2
import os
import json
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from imantics import Mask, Polygons
from iteration_utilities import duplicates
from shapely.geometry import Polygon
from copy import deepcopy
from os.path import join, exists

ROOT = './MERGE_OCEAN'
datasets = ['CAMO', 'COD10K', 'MAS3K']
dataset_types = ['Train', 'Test']


def merge_annotations():

    classes = []
    [classes.extend(os.listdir(os.path.join(ROOT, 'Train', dataset)))
     for dataset in datasets]
    classes = list(sorted(set(classes)))
    classes = [a.lower() for a in classes]
    categories = [{"id": idx, "name": item}
                  for idx, item in enumerate(classes)]
    map2int = dict()
    print(classes)
    for idx, item in enumerate(classes):
        map2int[item] = idx

    for dataset_type in dataset_types:
        filter_annotation = dict()
        filter_annotation['categories'] = deepcopy(categories)

        filter_images = []
        filter_anno = []
        img_count = 0
        ins_count = 0

        for dataset in datasets:
            json_path = join(ROOT, 'Annotations', 'ocean_{}_{}.json'.format(
                dataset.lower(),
                dataset_type.lower()
            ))
            image_folder = join(ROOT, dataset_type, dataset)
            list_images = glob(image_folder + '/**/*')

            with open(json_path, 'r') as json_file:
                annotations = json.load(json_file)
            json_file.close()
            print(dataset, dataset_type)
            print(len(list_images), len(annotations['images']))
            assert len(list_images) == len(annotations['images'])
            for image_info in annotations['images']:

                flag = True
                img_file = image_info['file_name'].split('/')[-1]
                w, h = image_info['width'], image_info['height']
                for item in list_images:
                    if img_file in item:
                        img_h, img_w, _ = cv2.imread(item).shape
                        if img_h == h and img_w == w:
                            flag = False
                            category = item.split('/')[-2].lower()
                            break
                if flag:
                    print('bug')
                    continue
                if category not in classes:
                    print(img_file)
                    print('bug')
                    break
                temp_image = deepcopy(image_info)
                temp_image['file_name'] = join(
                    '.' + image_folder, category, img_file)
                image_id = temp_image['id']

                for anno_info in annotations['annotations']:
                    img_id = anno_info['image_id']

                    if image_id == img_id:
                        temp_anno = deepcopy(anno_info)
                        temp_anno['image_id'] = img_count
                        temp_anno['id'] = ins_count
                        temp_anno['category_id'] = map2int[category]
                        filter_anno.append(deepcopy(temp_anno))
                        ins_count += 1

                temp_image['id'] = img_count
                filter_images.append(deepcopy(temp_image))
                img_count += 1

            filter_annotation['images'] = filter_images
            filter_annotation['annotations'] = filter_anno

            del annotations
        json_outpath = join(ROOT, 'Annotations',
                            'merge_ocean_{}.json'.format(dataset_type.lower()))

        check_len = len(glob(
            join(ROOT, dataset_type) + '/**/**/*.jpg'))
        print(check_len, len(filter_images))
        assert len(filter_images) == check_len

        with open(json_outpath, 'w') as json_out:
            json.dump(filter_annotation, json_out)
        json_out.close()


def append_annotation(source_annotations_path: dict, new_annotations_path: dict):
    with open(source_annotations_path, 'r') as json_file:
        source_annotations = json.load(json_file)
    json_file.close()

    with open(new_annotations_path, 'r') as json_file:
        new_annotations = json.load(json_file)
    json_file.close()

    assert source_annotations['categories'] == new_annotations['categories']

    num_images = len(source_annotations['images']) + \
        len(new_annotations['images'])
    last_img_id = source_annotations['images'][-1]['id'] + 1
    last_anno_id = source_annotations['annotations'][-1]['id'] + 1

    for img_anno in tqdm(new_annotations['images']):
        temp_img = deepcopy(img_anno)
        img_id = temp_img['id']
        for ann in new_annotations['annotations']:
            temp_ann = deepcopy(ann)
            image_id = temp_ann['image_id']
            if img_id == image_id:
                temp_ann['image_id'] = last_img_id
                temp_ann['id'] = last_anno_id
                last_anno_id += 1

                source_annotations['annotations'].append(deepcopy(temp_ann))

        temp_img['id'] = last_img_id
        last_img_id += 1
        source_annotations['images'].append(deepcopy(temp_img))

    assert len(source_annotations['images']) == num_images

    with open('./MERGE_OCEAN/Annotations/merge_ocean_train_augment++.json', 'w') as json_file:
        json.dump(source_annotations, json_file)
    json_file.close()


def main():
    # merge_annotations()
    append_annotation('./MERGE_OCEAN/Annotations/merge_ocean_train.json',
                      './MERGE_OCEAN/Annotations/augmentation++.json')


if __name__ == '__main__':
    main()
