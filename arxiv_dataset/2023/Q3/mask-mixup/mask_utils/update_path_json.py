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


def update_path():
    for dataset_type in dataset_types:
        for dataset in datasets:
            json_path = join(ROOT,
                             'Annotations',
                             'ocean_{}_{}.json'.format(dataset.lower(),
                                                       dataset_type.lower()))
            with open(json_path, 'r') as json_file:
                annotations = json.load(json_file)
            json_file.close()
            image_folder = join(ROOT, dataset_type, dataset)
            list_images = glob(image_folder + '/**/*')
            for image_info in annotations['images']:

                img_file = image_info['file_name'].split('/')[-1]
                for item in list_images:
                    if img_file in item:
                        category = item.split('/')[-2].lower()
                image_info['file_name'] = join(
                    '.' + image_folder, category, img_file)
            json_outpath = join(ROOT, 'Annotations',
                                'ocean_{}_{}_update.json'.format(dataset.lower(),
                                                                 dataset_type.lower()))

            with open(json_outpath, 'w') as json_out:
                json.dump(annotations, json_out)
            json_out.close()


def main():
    update_path()


if __name__ == '__main__':
    main()
