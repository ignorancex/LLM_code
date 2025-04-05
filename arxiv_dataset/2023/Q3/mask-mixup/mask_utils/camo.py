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
ROOT_PATH = './CAMO'
ROOT_IMAGE = './CAMO/Images/camo_1.0'
ROOT_ANNO = './CAMO/Annotations'
SUB_PATH = './sub_CAMO/Annotations'
aquatic_root = './CAMO/AquaticAnimal_CAMO++/AquaticAnimal_CAMO++'
aquatic_save = './CAMO/AquaticAnimal_CAMO++'
dataset_types = ['train', 'test']
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (0, 0, 0), (255, 255, 255)]

dict_ocean = dict()
dict_ocean['fish'] = ['flounder', 'stingaree', 'grouper',
                      'electricray', 'carpetshark', 'salmon', 'squid']
dict_ocean['seahorse'] = ['leafyseadragon', 'ghostpipefish', 'pipefish']
dict_ocean['crab'] = ['pagurian']
dict_ocean['octopus'] = ['scorpionfish',
                         'cuttlefish', 'leafyseadragon', 'squid', 'crocodilefish', 'flounder']


def extract_annotation():
    for data_type in dataset_types:
        anno_path = os.path.join(
            ROOT_ANNO, 'camo_{}_1.0.json'.format(data_type))
        with open(anno_path, 'r') as json_file:
            annotations = json.load(json_file)
            filter_annotation = dict()
            filter_annotation['categories'] = annotations['categories']

            filter_images = []
            filter_anno = []

            for image_info in annotations['images']:
                image_filename = image_info['file_name']
                image_id = image_info['id']

                if 'Aquatic' in image_filename:
                    filter_images.append(image_info)
                    for anno_info in annotations['annotations']:
                        img_id = anno_info['image_id']

                        if image_id == img_id:
                            filter_anno.append(anno_info)

            filter_annotation['images'] = filter_images
            filter_annotation['annotations'] = filter_anno

            json_outpath = os.path.join(
                SUB_PATH, 'camo_{}_1.0.json'.format(data_type))

            with open(json_outpath, 'w') as json_out:
                json.dump(filter_annotation, json_out)
            json_out.close()

        json_file.close()


def split_train_test():
    train_anno_path = os.path.join(ROOT_ANNO, 'camo_train_1.0.json')
    test_anno_path = os.path.join(ROOT_ANNO, 'camo_test_1.0.json')

    with open(train_anno_path, 'r') as json_file:
        train_anno = json.load(json_file)
    json_file.close()

    with open(test_anno_path, 'r') as json_file:
        test_anno = json.load(json_file)
    json_file.close()

    train_list = [item['file_name'].split(
        '/')[-1] for item in train_anno['images']]
    test_list = [item['file_name'].split('/')[-1]
                 for item in test_anno['images']]

    list_imgs = glob(aquatic_root + '/**/*.jpg')
    print(len(list_imgs))

    for img in list_imgs:
        category, img_id = img.split('/')[-2:]

        if img_id in train_list:
            data_type = 'Train'
        elif img_id in test_list:
            data_type = 'Test'
        else:
            data_type = 'Test'
            print('bug : ', img)

        save_folder = os.path.join(aquatic_save, data_type, category)
        save_path = os.path.join(save_folder, img_id)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        copyfile(img, save_path)


def mask2polygon():
    save_path = os.path.join(aquatic_save, 'camo_test_fix_mask.json')
    train_anno_path = os.path.join(ROOT_ANNO, 'camo_train_1.0.json')
    test_anno_path = os.path.join(ROOT_ANNO, 'camo_test_1.0.json')
    save_folder = os.path.join(aquatic_save, 'fix_mask')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(train_anno_path, 'r') as json_file:
        train_anno = json.load(json_file)
    json_file.close()

    with open(test_anno_path, 'r') as json_file:
        test_anno = json.load(json_file)
    json_file.close()

    train_list = [item['file_name'] for item in train_anno['images']]
    test_list = [item['file_name'] for item in test_anno['images']]

    list_imgs = glob(aquatic_root + '/**/*.jpg')
    print(len(list_imgs))

    annotations = dict()
    annotations['categories'] = [
        {
            "id": 1,
            "name": "camourflage"
        }
    ]
    annotations['images'] = []
    annotations['annotations'] = []

    count_img = 0
    count_anno = 0

    for img_file in tqdm(list_imgs):
        category, img_id = img_file.split('/')[-2:]
        key_img = cv2.imread(img_file)

        flag = True
        for img_path in train_list + test_list:
            value_id = img_path.split('/')[-1]

            if value_id == img_id:
                value_img = cv2.imread('CAMO' + img_path.split('..')[1])

                if key_img.shape == value_img.shape and key_img[0, 0, 0] == value_img[0, 0, 0]:
                    flag = False
                    break
        if flag == False:
            continue

        # if img_id not in train_list:
        #     continue
        # need extract polygon
        img_id = img_id.split('.')[0]

        img_path = os.path.join(img_file)
        mask_path = img_file.split('.jpg')[0] + '.png'
        img = cv2.imread(img_path)
        np_mask = np.array(Image.open(mask_path))
        h, w = np_mask.shape

        image_dict = dict()
        image_dict['id'] = count_img
        image_dict['file_name'] = img_path
        image_dict['width'] = w
        image_dict['height'] = h
        annotations['images'].append(image_dict)

        # polygons = Mask(np_mask).polygons()
        # poly_list = polygons.segmentation

        uni_val = np.unique(np_mask)
        # poly_list = list(
        #     sorted(poly_list, key=lambda a: len(a), reverse=True))

        for i in range(len(uni_val)-1):
            tmp_mask = np_mask == uni_val[i+1]
            tmp_mask = tmp_mask.astype(int)
            polygons = Mask(tmp_mask).polygons()
            poly_list = polygons.segmentation

            poly_list = list(
                sorted(poly_list, key=lambda a: len(a), reverse=True))

            if (len(poly_list[0]) < 50):
                continue
            poly = poly_list[0]

            points = [[poly[2*i], poly[2*i+1]]
                      for i in range(len(poly)//2)]

            np_poly = np.array([[[int(poly[2*i]), int(poly[2*i+1])]
                                 for i in range(len(poly)//2)]]).astype(np.int32)
            np_points = np.array(points)
            poly_np = Polygon(points)
            area = float(poly_np.area)

            x1, y1 = float(np.min(np_points[:, 0])), float(
                np.min(np_points[:, 1]))
            x2, y2 = float(np.max(np_points[:, 0])), float(
                np.max(np_points[:, 1]))

            w_box, h_box = x2-x1, y2-y1
            bbox = [x1, y1, w_box, h_box]

            anno_dict = dict()
            anno_dict['id'] = count_anno
            anno_dict['image_id'] = count_img
            anno_dict['category_id'] = 1
            anno_dict['segmentation'] = [poly]
            anno_dict['area'] = area
            anno_dict['bbox'] = bbox
            anno_dict['iscrowd'] = 0

            annotations['annotations'].append(anno_dict)
            img = cv2.polylines(img, np_poly, True,
                                colors[i % len(colors)], 4)

            count_anno += 1
        count_img += 1
        cv2.imwrite(os.path.join(
            save_folder, img_id + '.jpg'), img)

    with open(save_path, 'w') as json_file:
        json.dump(annotations, json_file)

    json_file.close()


def extract_ocean_annotations():
    train_anno_path = os.path.join(ROOT_ANNO, 'camo_train_1.0.json')
    test_anno_path = os.path.join(ROOT_ANNO, 'camo_test_1.0.json')
    test_fix_path = os.path.join(aquatic_save, 'camo_test_fix_mask.json')

    with open(train_anno_path, 'r') as json_file:
        train_anno = json.load(json_file)
    json_file.close()

    with open(test_anno_path, 'r') as json_file:
        test_anno = json.load(json_file)
    json_file.close()

    with open(test_fix_path, 'r') as json_file:
        test_fix = json.load(json_file)
    json_file.close()

    train_list = [item['file_name'] for item in train_anno['images']]
    test_list = [item['file_name'] for item in test_anno['images']]

    test_fix_list = [item['file_name'] for item in test_fix['images']]
    already = []
    list_imgs = glob(aquatic_root + '/**/*.jpg')
    print(len(list_imgs))

    annotations = deepcopy(train_anno)

    filter_annotation = dict()
    filter_annotation['categories'] = annotations['categories']

    filter_images = []
    filter_anno = []
    img_count = 0
    ins_count = 0
    for file_id in tqdm(list_imgs):
        flag = True
        for image_info in annotations['images']:
            image_filename = image_info['file_name']
            image_id = image_info['id']
            image_file_id = image_filename.split('/')[-1]

            if image_file_id == file_id.split('/')[-1]:

                img_key = cv2.imread(file_id)
                val_path = 'CAMO/' + image_filename.split('../')[1]

                img_val = cv2.imread(val_path)

                if img_key.shape == img_val.shape and img_key[0, 0, 0] == img_val[0, 0, 0]:
                    flag = False
                    for anno_info in annotations['annotations']:
                        img_id = anno_info['image_id']

                        if image_id == img_id:
                            anno_info['image_id'] = img_count
                            anno_info['id'] = ins_count
                            filter_anno.append(deepcopy(anno_info))
                            ins_count += 1

                    image_info['id'] = img_count
                    filter_images.append(deepcopy(image_info))
                    img_count += 1

                    save_folder = os.path.join(
                        aquatic_save, 'Train', file_id.split('/')[-2].lower())
                    save_path = os.path.join(
                        save_folder, file_id.split('/')[-1])

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    copyfile(file_id, save_path)
    print(img_count, len(filter_images))

    assert img_count == len(filter_images)

    filter_annotation['images'] = filter_images
    filter_annotation['annotations'] = filter_anno

    json_outpath = os.path.join(
        SUB_PATH, 'ocean_camo_train.json')

    with open(json_outpath, 'w') as json_out:
        json.dump(filter_annotation, json_out)
    json_out.close()

    # ----------------------------------------------------------------------------------
    # Test
    img_count = 0
    ins_count = 0
    annotations = deepcopy(test_anno)

    filter_annotation = dict()
    filter_annotation['categories'] = annotations['categories']

    filter_images = []
    filter_anno = []

    for file_id in tqdm(list_imgs):
        flag = True
        for image_info in annotations['images']:
            image_filename = image_info['file_name']
            image_id = image_info['id']
            image_file_id = image_filename.split('/')[-1]

            if image_file_id == file_id.split('/')[-1]:

                img_key = cv2.imread(file_id)
                val_path = 'CAMO/' + image_filename.split('../')[1]

                img_val = cv2.imread(val_path)

                if img_key.shape == img_val.shape and img_key[0, 0, 0] == img_val[0, 0, 0]:
                    flag = False
                    for anno_info in annotations['annotations']:
                        img_id = anno_info['image_id']

                        if image_id == img_id:
                            anno_info['image_id'] = img_count
                            anno_info['id'] = ins_count
                            filter_anno.append(deepcopy(anno_info))
                            ins_count += 1

                    image_info['id'] = img_count
                    filter_images.append(deepcopy(image_info))
                    img_count += 1

                    save_folder = os.path.join(
                        aquatic_save, 'Test', file_id.split('/')[-2].lower())
                    save_path = os.path.join(
                        save_folder, file_id.split('/')[-1])

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    copyfile(file_id, save_path)
    annotations = deepcopy(test_fix)

    for file_id in tqdm(list_imgs):
        for image_info in annotations['images']:
            image_filename = image_info['file_name']
            image_id = image_info['id']

            if image_filename.split('/')[-1] == file_id.split('/')[-1]:
                for anno_info in annotations['annotations']:
                    img_id = anno_info['image_id']

                    if image_id == img_id:
                        anno_info['image_id'] = img_count
                        anno_info['id'] = ins_count
                        filter_anno.append(deepcopy(anno_info))
                        ins_count += 1

                image_info['id'] = img_count
                filter_images.append(deepcopy(image_info))
                img_count += 1

                save_folder = os.path.join(
                    aquatic_save, 'Test', file_id.split('/')[-2].lower())
                save_path = os.path.join(
                    save_folder, file_id.split('/')[-1])

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                copyfile(file_id, save_path)

    filter_annotation['images'] = filter_images
    filter_annotation['annotations'] = filter_anno
    print(img_count, len(filter_images))
    assert len(filter_images) == img_count

    json_outpath = os.path.join(
        SUB_PATH, 'ocean_camo_test.json')

    with open(json_outpath, 'w') as json_out:
        json.dump(filter_annotation, json_out)
    json_out.close()


def main():
    # extract_annotation()
    # split_train_test()
    # mask2polygon()
    extract_ocean_annotations()


if __name__ == '__main__':
    main()
