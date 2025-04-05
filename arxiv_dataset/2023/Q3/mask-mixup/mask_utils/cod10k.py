import os
import json
from glob import glob
from tqdm import tqdm
from shutil import copyfile


ROOT_PATH = './COD10K-v3'
INFO_PATH = './COD10K-v3/Info'
SUB_PATH = './sub_COD10K-v3'
SUB_INFO = './sub_COD10K-v3/Info'


def extract_info():
    if not os.path.exists(SUB_INFO):
        os.makedirs(SUB_INFO)

    info_texts = glob(INFO_PATH + '/*.txt')
    for txt_file in info_texts:
        if 'NonCAM' in txt_file:
            continue

        file_id = txt_file.split('/')[-1]
        filter_lines = []
        with open(txt_file, 'r') as f:
            for line in f:
                if 'Aquatic' in line:
                    filter_lines.append(line)
        f.close()

        with open(os.path.join(SUB_INFO, file_id), 'w') as f:
            f.writelines(filter_lines)
        f.close()


def extract_image():
    dataset_types = ['Train', 'Test']
    cam_types = ['CAM', 'NonCAM']

    image_types = ['GT_Edge', 'GT_Instance', 'GT_Object', 'Image']
    for data_type in dataset_types:
        for img_type in image_types:
            folder_path = os.path.join(ROOT_PATH, data_type, img_type)
            dest_path = os.path.join(SUB_PATH, data_type, img_type)

            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            image_list = os.listdir(folder_path)
            for image_file in image_list:

                if 'Aquatic' in image_file and 'NonCAM' not in image_file:
                    copyfile(os.path.join(folder_path, image_file),
                             os.path.join(dest_path, image_file))


def extract_annotation():
    dataset_types = ['Train', 'Test']

    for data_type in dataset_types:
        json_path = os.path.join(ROOT_PATH, data_type,
                                 'CAM_Instance_{}.json'.format(data_type))

        with open(json_path, 'r') as json_file:
            annotations = json.load(json_file)

            filter_annotation = dict()
            filter_annotation['info'] = annotations['info']
            filter_annotation['licenses'] = annotations['licenses']
            filter_annotation['categories'] = annotations['categories']

            filter_images = []
            filter_anno = []

            for image_info in annotations['images']:
                image_filename = image_info['file_name']
                image_id = image_info['id']

                if 'Aquatic' in image_filename and 'NonCAM' not in image_filename:
                    filter_images.append(image_info)
                    for anno_info in annotations['annotations']:
                        img_id = anno_info['image_id']

                        if image_id == img_id:
                            filter_anno.append(anno_info)

            filter_annotation['images'] = filter_images
            filter_annotation['annotations'] = filter_anno

            json_outpath = os.path.join(SUB_PATH, data_type,
                                        'CAM_Instance_{}.json'.format(data_type))

            with open(json_outpath, 'w') as json_out:
                json.dump(filter_annotation, json_out)
            json_out.close()
        json_file.close()


def extract_text():
    dataset_types = ['Train', 'Test']

    for data_type in dataset_types:
        txt_path = os.path.join(ROOT_PATH, data_type,
                                'CAM-NonCAM_Instance_{}.txt'.format(data_type))
        out_path = os.path.join(SUB_PATH, data_type,
                                'CAM-NonCAM_Instance_{}.txt'.format(data_type))
        filter_lines = []
        with open(txt_path, 'r') as f:
            for line in f:
                if 'Aquatic' in line:
                    filter_lines.append(line)

        f.close()

        with open(out_path, 'w') as f:
            f.writelines(filter_lines)
        f.close()


def main():
    extract_info()
    extract_image()
    extract_annotation()


if __name__ == '__main__':
    main()
