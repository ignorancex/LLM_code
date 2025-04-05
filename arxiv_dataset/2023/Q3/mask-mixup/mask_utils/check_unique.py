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


def check_unique():
    for dataset_type in dataset_types:
        json_path = join(ROOT,
                         'Annotations',
                         'merge_ocean_{}.json'.format(dataset_type.lower()))

        with open(json_path, 'r') as json_file:
            annotations = json.load(json_file)
        json_file.close()

        check_list = []
        for ann in annotations['annotations']:
            ins_id = ann['id']
            if ins_id in check_list:
                print(ins_id, ann['image_id'])
            check_list.append(ins_id)


def main():
    check_unique()


if __name__ == '__main__':
    main()
