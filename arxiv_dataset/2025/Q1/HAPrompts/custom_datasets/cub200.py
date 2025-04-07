"""
download cub-200 ~7 GB
curl from Kaggle:
https://www.kaggle.com/api/v1/datasets/download/wenewone/cub2002011 
"""

import os
import sys
import shutil
import requests
from tqdm import tqdm

sys.path.append('../')
from utils.directory import load_config



def download_dataset(dest_dir):
    # download dataset
    url = "https://www.kaggle.com/api/v1/datasets/download/wenewone/cub2002011"
    download_path = os.path.join(dest_dir, "cub200.zip")

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
            raise RuntimeError("ERROR, something went wrong while downloading the cub-200 dataset")

    # unzip .gz file
    unzip_path = os.path.join(dest_dir, "cub-200-original")
    if not os.path.exists(unzip_path):
        shutil.unpack_archive(download_path, unzip_path)
    print(f'done, unzipped at {unzip_path}')

    return


def create_cub_200_test_set(src_img_dir, dest_home_dir, img_id_to_name_path, src_train_test_split_path):
    with open(img_id_to_name_path, "r") as f:
        img_id_to_name = list(f)

    with open(src_train_test_split_path, "r") as f:
        train_test_split = list(f)
    
    is_train = {int(line.split(' ')[0]): int(line.split(' ')[1]) for line in train_test_split}


    for line in tqdm(img_id_to_name, total=len(img_id_to_name)):
        # 1, '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg\n'
        img_id, img_src_relative_path = line.split(' ')
        img_id = int(img_id)
        img_src_relative_path = img_src_relative_path[:-1]

        # skip training set imgs
        if is_train[img_id]: continue
        
        # 001.Black_footed_Albatross, Black_Footed_Albatross_0046_18.jpg
        src_class_name, src_file_name = img_src_relative_path.split('/')
        
        # 001.Black_footed_Albatross -> Black_footed_Albatross@001
        class_number, class_name = src_class_name.split('.')
        # NOTE: kaggle dataset has a WRONG label: Artic_Tern should be Arctic_Tern
        if class_name == 'Artic_Tern':
            class_name = 'Arctic_Tern'
        dest_class_name = f'{class_name}@{class_number}'

        dest_test_dir = os.path.join(dest_home_dir, "test", dest_class_name)
        if not os.path.exists(dest_test_dir):
            os.makedirs(dest_test_dir)
        
        # copy
        img_src_path = os.path.join(src_img_dir, src_class_name, src_file_name)
        img_dest_path = os.path.join(dest_test_dir, src_file_name)
        shutil.copyfile(img_src_path, img_dest_path)

    return


def create_cub_200_val_set(src_img_dir, dest_home_dir, img_id_to_name_path, src_train_test_split_path, num_val):
    """
        fetch the first num_val examples in the training set for each class to be our validation examples
    """
    with open(img_id_to_name_path, "r") as f:
        img_id_to_name = list(f)

    with open(src_train_test_split_path, "r") as f:
        train_test_split = list(f)
    
    is_train = {int(line.split(' ')[0]): int(line.split(' ')[1]) for line in train_test_split}

    val_count = dict()

    for line in tqdm(img_id_to_name, total=len(img_id_to_name)):
        # 1, '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg\n'
        img_id, img_src_relative_path = line.split(' ')
        img_id = int(img_id)
        img_src_relative_path = img_src_relative_path[:-1]

        # skip testing set imgs
        if not is_train[img_id]: continue
        
        # 001.Black_footed_Albatross, Black_Footed_Albatross_0046_18.jpg
        src_class_name, src_file_name = img_src_relative_path.split('/')
        
        if src_class_name not in val_count.keys(): # 1st val example for this class
            val_count[src_class_name] = 1
        elif val_count[src_class_name] >= num_val: # enough val example for this class
            continue
        else:                                      # 1 more val example for this class
            val_count[src_class_name] += 1
        
        # 001.Black_footed_Albatross -> Black_footed_Albatross@001
        class_number, class_name = src_class_name.split('.')
        # NOTE: kaggle dataset has a WRONG label: Artic_Tern should be Arctic_Tern
        if class_name == 'Artic_Tern':
            class_name = 'Arctic_Tern'
        dest_class_name = f'{class_name}@{class_number}'

        dest_val_dir = os.path.join(dest_home_dir, "val", dest_class_name)
        if not os.path.exists(dest_val_dir):
            os.makedirs(dest_val_dir)
        
        # copy
        img_src_path = os.path.join(src_img_dir, src_class_name, src_file_name)
        img_dest_path = os.path.join(dest_val_dir, src_file_name)
        shutil.copyfile(img_src_path, img_dest_path)

    return


if __name__ == '__main__':
    config_path = '../data_paths.yml'
    config = load_config(config_path)
    cub200_download_dir = config['dataset-home-dir']
    if not os.path.exists(cub200_download_dir): os.makedirs(cub200_download_dir)
    download_dataset(cub200_download_dir)

    src_data_dir = os.path.join(cub200_download_dir, 'cub-200-original', 'CUB_200_2011', 'images')
    dest_data_dir = config['cub-200']

    img_id_to_name_path = os.path.join(cub200_download_dir, 'cub-200-original', 'CUB_200_2011', 'images.txt')
    src_train_test_split_path = os.path.join(cub200_download_dir, 'cub-200-original', 'CUB_200_2011', 'train_test_split.txt')

    create_cub_200_test_set(
        src_data_dir, 
        dest_data_dir, 
        img_id_to_name_path, 
        src_train_test_split_path,
    )
    
    create_cub_200_val_set(
        src_data_dir, 
        dest_data_dir, 
        img_id_to_name_path, 
        src_train_test_split_path,
        num_val=20,                  # fetch first 20 example from original train as our val
    )

    print('done')