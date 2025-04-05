import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from shutil import copyfile

COD_PATH = './sub_COD10K-v3'
CAMO_PATH = './sub_CAMO'
MAS_PATH = './MAS3K'
ROOT_SAVE = './MERGE_OCEAN'


def group_class_cod10k(subfolder: list):
    for folder in subfolder:
        list_images = glob(folder + '/*')
        data_type = folder.split('/')[2]
        for image_path in list_images:
            if 'NonCAM' in image_path:
                continue
            category = image_path.split('/')[-1].split('-')[5].lower()

            save_folder = os.path.join(
                ROOT_SAVE, data_type, 'COD10K', category)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            copyfile(image_path,
                     os.path.join(save_folder, image_path.split('/')[-1]))


def group_class_mas3k(subfolder: list):
    for folder in subfolder:
        list_images = glob(folder + '/*')
        data_type = folder.split('/')[2]
        for image_path in list_images:
            img_id = image_path.split('/')[-1]
            category = img_id.split('_')[2].lower()

            save_folder = os.path.join(
                ROOT_SAVE, data_type.capitalize(), 'MAS3K', category)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            copyfile(image_path,
                     os.path.join(save_folder, image_path.split('/')[-1]))

            # break


def class_statistic(df_save_path):
    datasets = ['CAMO', 'COD10K', 'MAS3K']
    dataset_types = ['Train', 'Test']
    classes = []
    [classes.extend(os.listdir(os.path.join(ROOT_SAVE, 'Train', dataset)))
     for dataset in datasets]
    classes = list(sorted(set(classes)))
    classes = [a.lower() for a in classes]

    # for dataset_type in dataset_types:
    #     for dataset in datasets:
    #         list_folder = glob(os.path.join(ROOT_SAVE, dataset_type, dataset) + '/*')
    #         for folder in list_folder:
    #             folder_id = folder.split('/')[-1].lower()
    #             os.rename(folder, os.path.join(ROOT_SAVE, dataset_type, dataset, folder_id))

    print(classes)
    print(len(classes))
    df = pd.DataFrame()
    df['Category'] = classes

    for data_type in dataset_types:
        for dataset in datasets:
            num_images = []
            for category in classes:
                path = os.path.join(ROOT_SAVE, data_type, dataset, category)
                if os.path.exists(path):
                    num_images.append(len(os.listdir(path)))
                else:
                    num_images.append(0)

            columns = '{} - {}'.format(dataset, data_type)
            df[columns] = num_images

    df.to_csv(df_save_path, index=False)


def main():

    group_class_cod10k(['./sub_COD10K-v3/Train/Image',
                        './sub_COD10K-v3/Test/Image'])
    # group_class_cod10k(['./sub_COD10K-v3/Test/Image'])
    # group_class_mas3k(['./sub_MAS3K/train/Image',
    #                    './sub_MAS3K/test/Image'])
    # class_statistic('class_stat.csv')


if __name__ == '__main__':
    main()
