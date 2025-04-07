"""
download SUN-397 dataset ~ 39.1 GB
SUN-397 url:
http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz

filter 73 classes with multiple parents based on label hierarchy provided by SUN-397
construct SUN-324 
"""
import os
import sys
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
from utils.directory import load_config, get_filename_list

# fix RNG seed
np.random.seed(10086)


def download_dataset(dest_dir):
    # download dataset
    url = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    download_path = os.path.join(dest_dir, "SUN397.tar.gz")

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
            raise RuntimeError("ERROR, something went wrong while downloading the SUN-397 dataset")

    # unzip .gz file
    unzip_path = os.path.join(dest_dir, "sun-397-original")
    if not os.path.exists(unzip_path):
        shutil.unpack_archive(download_path, unzip_path)
    print(f'done, unzipped at {unzip_path}')
    return


def download_sun397_hierarchy(sun397_download_dir):
    url = 'https://vision.princeton.edu/projects/2010/SUN/hierarchy_three_levels.zip'
    download_path = os.path.join(sun397_download_dir, "hierarchy_three_levels.zip")

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
            raise RuntimeError("ERROR, something went wrong while downloading the SUN-397 dataset")

    # unzip .gz file
    unzip_path = os.path.join(sun397_download_dir, "hierarchy_three_levels")
    if not os.path.exists(unzip_path):
        shutil.unpack_archive(download_path, unzip_path)
    print(f'done, unzipped at {unzip_path}')
    return

def category_path_to_classname(cate_paths):
    ret = []
    for path in cate_paths:
        path_tags = path.split('/')
        if len(path_tags)>3:
            # remove 1st tag: 
            # e.g., /c/car_interior/backseat -> ['car_interior', 'backseat']
            path_tags = path_tags[2:]

            # replace _ with space
            path_tags = [tp.replace('_', ' ') for tp in path_tags]

            # connect tags into one name string
            name = path_tags[0]
            for tag in path_tags[1:]:
                name = name + ' ' + tag
        else:
            name = path_tags[2].replace('_', ' ')
        ret.append(name)
    return ret


# filter out ambiguous classes
def filter_classes(df):
    '''
        df: dataframe from reading SUN397's excel file with label hierarchy info
        cls_names: class names with ambiguous classes filtered out
    '''
    cate_paths = list(df[df.columns[0]])[1:]
    cls_names = category_path_to_classname(cate_paths)

    # from class name to row idx in dataframe
    name_to_df_row_idx = {name: k+1 for k, name in enumerate(cls_names)}

    # keep a counter for ambiguous classes
    ambiguous_cnt = 0
    delete_classes = []
    for classname in cls_names:
        row_idx = name_to_df_row_idx[classname]
        hie_labels = df.iloc[row_idx].to_list()[1:] # exclude category in df
        if sum(hie_labels) != 2:
            ambiguous_cnt += 1
            delete_classes.append(classname)
    print(f'{ambiguous_cnt} classes in SUN397 have ambiguous hierarchical label')

    # delete/filter ambiguous classes
    for classname in delete_classes:
        cls_names.remove(classname)
    N = len(cls_names)
    print(f'number of remaining classes after filtering: {N}')
    
    return cls_names


def construct_sun324_val_test_splits(src_data_dir, dest_home_dir, excel_path, N_val=25, N_test=50):

    assert 0< N_val <=100 and 0< N_test <=100 and N_val+N_test <= 100, f"error N_val={N_val}, N_test={N_test}"
    
    df = pd.read_excel(excel_path, sheet_name="SUN397")
    cate_dirs = list(df[df.columns[0]])[1:]
    cls_names = category_path_to_classname(cate_dirs)

    # from cls_name to cate_paths
    cls_to_path = {cls_name : cate_dir for cls_name, cate_dir in zip(cls_names, cate_dirs)}

    # filter ambiguous classes
    cls_names = filter_classes(df)

    # sample examples from source dir
    for classname in tqdm(cls_names, total=len(cls_names)):
        cate_dir = cls_to_path[classname]
        src_cls_dir = src_data_dir + cate_dir
        fnames = get_filename_list(src_cls_dir, "*.jpg")
        
        # shuffle examples
        np.random.shuffle(fnames)
        val_names = fnames[:N_val] # 0:N_val-1 -> val
        test_names = fnames[N_val:N_val+N_test] # N_val: N_val+N_test-1 -> test

        # copy val images
        dest_val_dir = os.path.join(dest_home_dir, 'val', classname)
        if not os.path.exists(dest_val_dir):
            os.makedirs(dest_val_dir)
        for val_name in val_names:
            src_img_path = os.path.join(src_cls_dir, val_name)
            dest_img_path = os.path.join(dest_val_dir, val_name)
            shutil.copyfile(src_img_path, dest_img_path)
        
        # copy test images
        dest_test_dir = os.path.join(dest_home_dir, 'test', classname)

        if not os.path.exists(dest_test_dir):
            os.makedirs(dest_test_dir)
        for test_name in test_names:
            src_img_path = os.path.join(src_cls_dir, test_name)
            dest_img_path = os.path.join(dest_test_dir, test_name)
            shutil.copyfile(src_img_path, dest_img_path)
    
    return


if __name__ == '__main__':
    config_path = '../data_paths.yml'
    config = load_config(config_path)
    sun397_download_dir = config['dataset-home-dir']

    # download SUN-397 dataset
    if not os.path.exists(sun397_download_dir): os.makedirs(sun397_download_dir)
    download_dataset(sun397_download_dir)

    # download SUN-397 original hierarchy
    hierarchy_dir = os.path.join(sun397_download_dir, 'sun-397-original')
    download_sun397_hierarchy(hierarchy_dir)
    excel_path = os.path.join(
        sun397_download_dir, 'sun-397-original', 
        'hierarchy_three_levels', 'hierarchy_three_levels',
        'three_levels.xlsx')
    
    # val/test split
    src_data_dir = os.path.join(sun397_download_dir, 'sun-397-original', 'SUN397')
    dest_home_dir = config['sun-324']
    construct_sun324_val_test_splits(
        src_data_dir, 
        dest_home_dir, 
        excel_path,
        N_val=25,
        N_test=50,
    )
    