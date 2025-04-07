"""
download food-101 dataset ~9.37 GB
curl from Kaggle: 
https://www.kaggle.com/api/v1/datasets/download/dansbecker/food-101
"""
import os
import sys
import json
import shutil
import requests
import numpy as np
from tqdm import tqdm

sys.path.append('../')
from utils.directory import load_config

# fix RNG seed
np.random.seed(10086)


def download_dataset(dest_dir):
    # download dataset
    url = "https://www.kaggle.com/api/v1/datasets/download/dansbecker/food-101"
    download_path = os.path.join(dest_dir, "food101.zip")

    if not os.path.exists(download_path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(download_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise RuntimeError("ERROR, something went wrong while downloading the food-101 dataset")

    # unzip .gz file
    unzip_path = os.path.join(dest_dir, "food-101-original")
    if not os.path.exists(unzip_path):
        shutil.unpack_archive(download_path, unzip_path)
    print(f'done, unzipped at {unzip_path}')
    return


def create_food_101_test_set(src_img_dir, dest_test_dir, test_split_path):
    """
        create pytorch image folder for from source test split
    """

    with open(test_split_path, "r") as f:
        D = json.load(f)

    keys = list(D.keys())
    keys = sorted(keys)
    

    for k, name in tqdm(enumerate(keys), total=len(keys)):
        # 'class-name/image-id\n' -> 'class-name/image-id'
        
        ls = D[name]
        dest_test_class_dir = os.path.join(dest_test_dir, name + f"@{k}")
        if not os.path.exists(dest_test_class_dir):
            os.makedirs(dest_test_class_dir)

        for fpath in ls:
            fname = fpath.split('/')[1]
            src_path = os.path.join(src_img_dir, fpath + ".jpg")
            dest_path = os.path.join(dest_test_class_dir, fname + ".jpg")
            shutil.copyfile(src_path, dest_path)    

    return 


def create_food_101_train_val_split_from_src_train(src_img_dir, dest_home_dir, src_train_split_path, num_val):
    """
        create pytorch image folder for train val split from original train split
        food-101 has 750 training img/class
        - our validation split needs to remain within (0, 750) img/class
    """

    assert 0 < num_val and num_val < 750, f"invalid number of validation examples: {num_val}"

    with open(src_train_split_path, "r") as f:
        D = json.load(f)
    
    keys = list(D.keys())
    keys = sorted(keys)

    for k, name in tqdm(enumerate(keys), total=len(keys)):
        ls = D[name]

        # train/val split per class
        np.random.shuffle(ls)
        ls_val = ls[:num_val]
        ls_train = ls[num_val:]

        # train ----
        dest_train_class_dir = os.path.join(dest_home_dir, "train", name + f"@{k}")
        if not os.path.exists(dest_train_class_dir):
            os.makedirs(dest_train_class_dir)
        
        for fpath in ls_train:
            fname = fpath.split('/')[1]
            src_path = os.path.join(src_img_dir, fpath + ".jpg")
            dest_path = os.path.join(dest_train_class_dir, fname + ".jpg")
            shutil.copyfile(src_path, dest_path)  

        # val ----
        dest_val_class_dir = os.path.join(dest_home_dir, "val", name + f"@{k}")
        if not os.path.exists(dest_val_class_dir):
            os.makedirs(dest_val_class_dir)
        
        for fpath in ls_val:
            fname = fpath.split('/')[1]
            src_path = os.path.join(src_img_dir, fpath + ".jpg")
            dest_path = os.path.join(dest_val_class_dir, fname + ".jpg")
            shutil.copyfile(src_path, dest_path)      

    return


if __name__ == '__main__':
    config_path = '../data_paths.yml'
    config = load_config(config_path)
    food101_download_dir = config['dataset-home-dir']
    if not os.path.exists(food101_download_dir): os.makedirs(food101_download_dir)
    download_dataset(food101_download_dir)

    test_split_path = os.path.join(
        food101_download_dir, 'food-101-original', 'food-101', 'food-101',
        'meta','test.json')
    train_split_path = os.path.join(
        food101_download_dir, 'food-101-original', 'food-101', 'food-101',
        'meta','train.json')
    
    src_image_dir = os.path.join(
        food101_download_dir, 'food-101-original', 'food-101', 'food-101',
        'images')
    
    dest_home_dir = config['food-101']

    # test split
    dest_test_dir = os.path.join(dest_home_dir, 'test')
    create_food_101_test_set(src_image_dir, dest_test_dir, test_split_path)

    # val split
    val_img_per_class = 250
    create_food_101_train_val_split_from_src_train(
        src_image_dir, 
        dest_home_dir, 
        train_split_path, 
        val_img_per_class
    )


