import os
import cv2
import numpy as np
import shutil
import json
from imantics import Mask, Polygons
from PIL import Image
from glob import glob
from tqdm import tqdm
from iteration_utilities import duplicates
from shapely.geometry import Polygon

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


def mask_image():
    for data_type in dataset_types:
        image_folder = os.path.join(ROOT_PATH, data_type, 'Image')
        mask_folder = os.path.join(ROOT_PATH, data_type, 'Mask')
        image_mask_path = os.path.join(ROOT_PATH, data_type, mask_image_folder)
        wrong_shape_path = os.path.join(ROOT_PATH, data_type, wrong_shape)

        if not os.path.exists(image_mask_path):
            os.makedirs(image_mask_path)

        if not os.path.exists(wrong_shape_path):
            os.makedirs(wrong_shape_path)

        image_list = os.listdir(image_folder)

        for image_file in image_list:
            img_id = image_file.split('.')[0]
            mask_path = os.path.join(mask_folder, img_id + '.png')
            img_path = os.path.join(image_folder, image_file)
            img_mask_path = os.path.join(image_mask_path, image_file)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            if img.shape == mask.shape:
                mask_img = img & mask
                cv2.imwrite(img_mask_path, mask_img)

            else:
                shutil.move(img_path,
                            os.path.join(wrong_shape_path, image_file))
                shutil.move(mask_path,
                            os.path.join(wrong_shape_path, img_id + 'png'))


def filter_instances():
    for data_type in dataset_types:
        mask_folder = os.path.join(ROOT_PATH, data_type, 'Mask')
        image_mask_path = os.path.join(ROOT_PATH, data_type, mask_image_folder)
        many_instances_folder = os.path.join(ROOT_PATH,
                                             data_type,
                                             many_instances)
        mask_many_instances_folder = os.path.join(ROOT_PATH,
                                                  data_type,
                                                  mask_many_instances)

        if not os.path.exists(many_instances_folder):
            os.makedirs(many_instances_folder)

        mask_list = os.listdir(mask_folder)

        for mask_file in tqdm(mask_list):
            mask_path = os.path.join(mask_folder, mask_file)
            np_mask = np.array(Image.open(mask_path))
            polygons = Mask(np_mask).polygons()
            img_id = mask_file.split('.')[0]
            poly_list = polygons.segmentation
            poly_list = [poly for poly in poly_list if len(poly) > 100]
            if len(poly_list) > 1:
                print(' '.join([str(len(poly))
                                for poly in poly_list]), ' ', img_id)
                mask_img_path = os.path.join(
                    image_mask_path, img_id + '.jpg')
                num_instance_folder = os.path.join(
                    many_instances_folder, str(len(poly_list)))

                if os.path.exists(mask_img_path):
                    if not os.path.exists(num_instance_folder):
                        os.makedirs(num_instance_folder)

                    shutil.move(mask_img_path,
                                os.path.join(num_instance_folder, img_id + '.jpg'))


def connectComponent():
    for data_type in dataset_types:
        mask_folder = os.path.join(ROOT_PATH, data_type, 'Mask')
        image_mask_path = os.path.join(ROOT_PATH, data_type, mask_image_folder)
        many_instances_folder = os.path.join(ROOT_PATH,
                                             data_type,
                                             many_instances)
        mask_many_instances_folder = os.path.join(ROOT_PATH,
                                                  data_type,
                                                  mask_many_instances)
        connect_component_path = os.path.join(ROOT_PATH,
                                              data_type,
                                              connect_component_folder)
        if not os.path.exists(many_instances_folder):
            os.makedirs(many_instances_folder)
        if not os.path.exists(connect_component_path):
            os.makedirs(connect_component_path)

        mask_list = os.listdir(mask_folder)

        for mask_file in mask_list:
            mask_path = os.path.join(mask_folder, mask_file)
            np_mask = np.array(Image.open(mask_path))
            img_id = mask_file.split('.')[0]
            img = cv2.imread(os.path.join(
                image_mask_path, img_id + '.jpg'))
            ret, thresh = cv2.threshold(
                np_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                thresh, 4)
            if n_labels > 2:
                size_thresh = 1
                for i in range(1, n_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
                        # print(stats[i, cv2.CC_STAT_AREA])
                        x = stats[i, cv2.CC_STAT_LEFT]
                        y = stats[i, cv2.CC_STAT_TOP]
                        w = stats[i, cv2.CC_STAT_WIDTH]
                        h = stats[i, cv2.CC_STAT_HEIGHT]
                        img = cv2.rectangle(img, (x, y), (x+w, y+h),
                                            colors[i % len(colors)], thickness=1)

                cv2.imwrite(os.path.join(
                    connect_component_path, img_id + '.jpg'), img)

                # break

        break


def get_largest_instance():

    for data_type in dataset_types:
        mask_folder = os.path.join(ROOT_PATH, data_type, 'Mask')
        image_mask_path = os.path.join(ROOT_PATH, data_type, mask_image_folder)
        many_instances_folder = os.path.join(ROOT_PATH,
                                             data_type,
                                             many_instances)
        mask_many_instances_folder = os.path.join(ROOT_PATH,
                                                  data_type,
                                                  mask_many_instances)

        save_folder = os.path.join(ROOT_PATH,
                                   data_type,
                                   largest_instance)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        mask_img_list = glob(many_instances_folder + '/**/*')

        for mask_img_file in tqdm(mask_img_list):
            img = cv2.imread(mask_img_file)
            img_id = mask_img_file.split('/')[-1].split('.')[0]
            mask_path = os.path.join(mask_folder, img_id + '.png')
            np_mask = np.array(Image.open(mask_path))
            polygons = Mask(np_mask).polygons()
            poly_list = polygons.segmentation
            # poly_list = [poly for poly in poly_list if len(poly) > 100]
            poly_list = list(sorted(poly_list, key=lambda a: len(a)))
            if len(poly_list) > 0:
                poly = poly_list[-1]
                poly_np = np.array([[[int(poly[2*i]), int(poly[2*i+1])]
                                     for i in range(len(poly)//2)]]).astype(np.int32)
                img = cv2.polylines(img, poly_np, True, colors[2], 4)
                cv2.imwrite(os.path.join(save_folder, img_id + '.jpg'), img)


def check_duplicate():
    data_type = 'train'
    many_instances_folder = os.path.join(ROOT_PATH,
                                         data_type,
                                         'Many_Instances_True')
    overlap_folder = os.path.join(ROOT_PATH,
                                  data_type,
                                  'overlap')

    largest_folder = os.path.join(ROOT_PATH,
                                  data_type,
                                  'largest_instance')

    img_folder = os.path.join(ROOT_PATH,
                              data_type,
                              'Image')
    wrongshape_folder = os.path.join(ROOT_PATH,
                                     data_type,
                                     'wrong_shape')

    list_overlap = os.listdir(overlap_folder)
    list_largest = os.listdir(largest_folder)
    list_true = [img.split('/')[-1]
                 for img in glob(many_instances_folder + '/**/*')]
    list_img = os.listdir(img_folder) + os.listdir(wrongshape_folder)

    super_cate = []
    class_cate = []
    for img in list_img:
        super_c, sub_c = img.split('_')[1: 3]
        super_cate.append(super_c)
        class_cate.append(sub_c)

    super_cate = set(super_cate)
    class_cate = set(class_cate)

    print(super_cate, len(super_cate))
    print(class_cate, len(class_cate))

    '''
    for i in list_overlap:
        if i in list_true:
            print('overlap in true : ', i)

    for i in list_largest:
        if i in list_true:
            print('largest in true : ', i)

    for i in list_overlap:
        if i in list_largest:
            print('overlap in largest : ', i)

    for i in list_true:
        if i in list_largest:
            print('true in largest : ', i)

    for i in list_largest:
        if i in list_overlap:
            print('largest in overlap: ', i)

    for i in list_true:
        if i in list_overlap:
            print('true in overlap: ', i)

    all_extract = list_largest + list_overlap + list_true
    if len(set(all_extract)) == len(all_extract):
        print('true')
    if len(set(list_true)) == len(list_true):
        print('true')
    for i in all_extract:
        if i not in list_img:
            print(i)
    for i in list_img:
        if i not in all_extract:
            print(i)
    '''
    for data_type in dataset_types:
        mask_folder = os.path.join(ROOT_PATH, data_type, 'Mask')
        image_mask_path = os.path.join(ROOT_PATH, data_type, mask_image_folder)
        many_instances_folder = os.path.join(ROOT_PATH,
                                             data_type,
                                             many_instances)
        mask_many_instances_folder = os.path.join(ROOT_PATH,
                                                  data_type,
                                                  mask_many_instances)

        save_folder = os.path.join(ROOT_PATH,
                                   data_type,
                                   largest_instance)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        mask_img_list = glob(many_instances_folder + '/**/*')

        for mask_img_file in tqdm(mask_img_list):
            img = cv2.imread(mask_img_file)
            img_id = mask_img_file.split('/')[-1].split('.')[0]
            mask_path = os.path.join(mask_folder, img_id + '.png')
            np_mask = np.array(Image.open(mask_path))
            polygons = Mask(np_mask).polygons()
            poly_list = polygons.segmentation
            # poly_list = [poly for poly in poly_list if len(poly) > 100]
            poly_list = list(sorted(poly_list, key=lambda a: len(a)))
            if len(poly_list) > 0:
                poly = poly_list[-1]
                poly_np = np.array([[[int(poly[2*i]), int(poly[2*i+1])]
                                     for i in range(len(poly)//2)]]).astype(np.int32)
                img = cv2.polylines(img, poly_np, True, colors[2], 4)
                cv2.imwrite(os.path.join(save_folder, img_id + '.jpg'), img)


def need_relabel(relabel_folder: list, dataset_type: str):

    images_relabel = []
    [images_relabel.extend(glob(os.path.join(
        ROOT_PATH, dataset_type, folder + '/*.jpg'))) for folder in relabel_folder]

    images_id = [filename.split('/')[-1] for filename in images_relabel]

    for img_type in ['Image', 'Mask']:
        save_folder = os.path.join(
            ROOT_PATH, 'need_relabel', dataset_type, img_type)
        if img_type == 'Image':
            ext = '.jpg'
        else:
            ext = '.png'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for img_id in images_id:
            img_path = os.path.join(
                ROOT_PATH, dataset_type, img_type, img_id.split('.')[0] + ext)
            if os.path.exists(img_path):
                shutil.copyfile(img_path,
                                os.path.join(save_folder, img_id.split('.')[0] + ext))


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


def extract_annotations():
    for data_type in dataset_types:
        mask_folder = os.path.join(ROOT_PATH, data_type, 'Mask')
        image_folder = os.path.join(ROOT_PATH, data_type, 'Image')
        image_mask_path = os.path.join(ROOT_PATH, data_type, mask_image_folder)
        true_instances_folder = os.path.join(ROOT_PATH,
                                             data_type,
                                             true_instances)
        save_path = os.path.join(ROOT_PATH, data_type,
                                 'mas3k_{}.json'.format(data_type))
        mask_img_list = glob(true_instances_folder + '/**/*')

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

        for mask_img_file in tqdm(mask_img_list):
            img_id = mask_img_file.split('/')[-1].split('.')[0]
            poly_type = mask_img_file.split('/')[-2]

            img_path = os.path.join(image_folder, img_id + '.jpg')
            mask_path = os.path.join(mask_folder, img_id + '.png')

            img = cv2.imread(img_path)
            np_mask = np.array(Image.open(mask_path))
            h, w = np_mask.shape

            image_dict = dict()
            image_dict['id'] = count_img
            image_dict['file_name'] = img_path
            image_dict['width'] = w
            image_dict['height'] = h
            annotations['images'].append(image_dict)

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
                points = [[poly[2*i], poly[2*i+1]]
                          for i in range(len(poly)//2)]
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

                count_anno += 1
            count_img += 1

        with open(save_path, 'w') as json_file:
            json.dump(annotations, json_file)

        json_file.close()


def extract_images():
    for data_type in dataset_types:
        image_folder = os.path.join(ROOT_PATH, data_type, 'Image')
        image_mask_path = os.path.join(ROOT_PATH, data_type, mask_image_folder)
        true_instances_folder = os.path.join(ROOT_PATH,
                                             data_type,
                                             true_instances)
        save_folder = os.path.join(SUB_PATH, data_type, 'Image')

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        mask_img_list = glob(true_instances_folder + '/**/*')

        for mask_img_file in tqdm(mask_img_list):
            img_id = mask_img_file.split('/')[-1].split('.')[0]

            img_path = os.path.join(image_folder, img_id + '.jpg')
            save_path = os.path.join(save_folder, img_id + '.jpg')

            shutil.copyfile(img_path, save_path)


def main():
    # mask_image()
    # filter_instances()
    # get_largest_instance()
    # check_duplicate()
    # connectComponent()
    # relabel_folder_train = ['largest_instance', 'overlap', 'wrong_shape']
    # relabel_folder_test = ['largest_instance', 'wrong_shape']
    # need_relabel(relabel_folder_train, 'train')
    # need_relabel(relabel_folder_test, 'test')
    # extract_polygons()
    # extract_annotations()
    extract_images()


if __name__ == '__main__':
    main()
