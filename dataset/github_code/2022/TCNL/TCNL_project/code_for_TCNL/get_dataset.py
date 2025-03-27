from fileinput import filename
import os
from random import Random, random, shuffle
import codecs
from turtle import shape
from tqdm import tqdm
import cv2
import shutil
from PIL import Image
import numpy as np
import json


def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)


def mask_process(mask_array: np.ndarray, label_name: str):
    if label_name == 'aeroplane':
        index_0 = np.where(mask_array != 1)
        index_1 = np.where(mask_array == 1)
    if label_name == 'bicycle':
        index_0 = np.where(mask_array != 2)
        index_1 = np.where(mask_array == 2)
    if label_name == 'bird':
        index_0 = np.where(mask_array != 3)
        index_1 = np.where(mask_array == 3)
    if label_name == 'boat':
        index_0 = np.where(mask_array != 4)
        index_1 = np.where(mask_array == 4)
    if label_name == 'bottle':
        index_0 = np.where(mask_array != 5)
        index_1 = np.where(mask_array == 5)
    if label_name == 'bus':
        index_0 = np.where(mask_array != 6)
        index_1 = np.where(mask_array == 6)
    if label_name == 'car':
        index_0 = np.where(mask_array != 7)
        index_1 = np.where(mask_array == 7)
    if label_name == 'cat':
        index_0 = np.where(mask_array != 8)
        index_1 = np.where(mask_array == 8)
    if label_name == 'chair':
        index_0 = np.where(mask_array != 9)
        index_1 = np.where(mask_array == 9)
    if label_name == 'cow':
        index_0 = np.where(mask_array != 10)
        index_1 = np.where(mask_array == 10)
    if label_name == 'diningtable':
        index_0 = np.where(mask_array != 11)
        index_1 = np.where(mask_array == 11)
    if label_name == 'dog':
        index_0 = np.where(mask_array != 12)
        index_1 = np.where(mask_array == 12)
    if label_name == 'horse':
        index_0 = np.where(mask_array != 13)
        index_1 = np.where(mask_array == 13)
    if label_name == 'motorbike':
        index_0 = np.where(mask_array != 14)
        index_1 = np.where(mask_array == 14)
    if label_name == 'person':
        index_0 = np.where(mask_array != 15)
        index_1 = np.where(mask_array == 15)
    if label_name == 'pottedplant':
        index_0 = np.where(mask_array != 16)
        index_1 = np.where(mask_array == 16)
    if label_name == 'sheep':
        index_0 = np.where(mask_array != 17)
        index_1 = np.where(mask_array == 17)
    if label_name == 'sofa':
        index_0 = np.where(mask_array != 18)
        index_1 = np.where(mask_array == 18)
    if label_name == 'train':
        index_0 = np.where(mask_array != 19)
        index_1 = np.where(mask_array == 19)
    if label_name == 'tvmonitor':
        index_0 = np.where(mask_array != 20)
        index_1 = np.where(mask_array == 20)
    mask_array[index_0] = 0
    mask_array[index_1] = 1

    # index_1 = np.where(mask_array == 2)

    # mask_array[index_1] = 0
    return mask_array


def segmentation(mask_array: np.ndarray, image_array: np.ndarray, label_name: str):
    new_mask_array = mask_process(mask_array, label_name)
    image_array[:, :, 0] = image_array[:, :, 0] * new_mask_array
    image_array[:, :, 1] = image_array[:, :, 1] * new_mask_array
    image_array[:, :, 2] = image_array[:, :, 2] * new_mask_array
    new_image_array = image_array
    return new_image_array


def get_image_array(image_path: str) -> np.ndarray:
    array = np.array(Image.open(image_path))
    return array


def crop_poly_test_scene(img_path:str, json_path:str):
    img = cv2.imread(img_path)
    json_info = json.load(codecs.open(json_path, 'r', 'utf-8-sig'))
    bed_coordinates = list()
    bedsidetable_coordinates = list()
    lamp_coordinates = list()
    sofa_coordinates = list()
    chair_coordinates = list()
    table_coordinates = list()
    shelf_coordinates = list()
    seat_coordinates = list()
    screen_coordinates = list()
    stage_coordinates = list()
    for shapes in json_info['shapes']:
        if shapes['label'] == 'bed':
            bed_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'bedside_table':
            bedsidetable_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'lamp':
            lamp_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'sofa':
            sofa_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'chair':
            chair_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'table':
            table_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'shelf':
            shelf_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'seat':
            seat_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'screen':
            screen_coordinates.append(np.array(shapes['points']).astype(np.int))
        if shapes['label'] == 'stage':
            stage_coordinates.append(np.array(shapes['points']).astype(np.int))
            

    bed_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(bed_mask, bed_coordinates, 1, 255)
    cv2.fillPoly(bed_mask, bed_coordinates, 255)
    bed_dst = cv2.bitwise_and(img, img, mask=bed_mask)

    bedsidetable_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(bedsidetable_mask, bedsidetable_coordinates, 1, 255)
    cv2.fillPoly(bedsidetable_mask, bedsidetable_coordinates, 255)
    bedsidetable_dst = cv2.bitwise_and(img, img, mask=bedsidetable_mask)

    lamp_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(lamp_mask, lamp_coordinates, 1, 255)
    cv2.fillPoly(lamp_mask, lamp_coordinates, 255)
    lamp_dst = cv2.bitwise_and(img, img, mask=lamp_mask)

    sofa_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(sofa_mask, sofa_coordinates, 1, 255)
    cv2.fillPoly(sofa_mask, sofa_coordinates, 255)
    sofa_dst = cv2.bitwise_and(img, img, mask=sofa_mask)

    chair_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(chair_mask, chair_coordinates, 1, 255)
    cv2.fillPoly(chair_mask, chair_coordinates, 255)
    chair_dst = cv2.bitwise_and(img, img, mask=chair_mask)

    table_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(table_mask, table_coordinates, 1, 255)
    cv2.fillPoly(table_mask, table_coordinates, 255)
    table_dst = cv2.bitwise_and(img, img, mask=table_mask)

    shelf_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(shelf_mask, shelf_coordinates, 1, 255)
    cv2.fillPoly(shelf_mask, shelf_coordinates, 255)
    shelf_dst = cv2.bitwise_and(img, img, mask=shelf_mask)

    seat_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(seat_mask, seat_coordinates, 1, 255)
    cv2.fillPoly(seat_mask, seat_coordinates, 255)
    seat_dst = cv2.bitwise_and(img, img, mask=seat_mask)

    seat_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(seat_mask, seat_coordinates, 1, 255)
    cv2.fillPoly(seat_mask, seat_coordinates, 255)
    seat_dst = cv2.bitwise_and(img, img, mask=seat_mask)

    screen_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(screen_mask, screen_coordinates, 1, 255)
    cv2.fillPoly(screen_mask, screen_coordinates, 255)
    screen_dst = cv2.bitwise_and(img, img, mask=screen_mask)

    stage_mask = np.zeros(shape=img.shape[: 2], dtype=np.uint8)
    cv2.polylines(stage_mask, stage_coordinates, 1, 255)
    cv2.fillPoly(stage_mask, stage_coordinates, 255)
    stage_dst = cv2.bitwise_and(img, img, mask=stage_mask)



    return bed_dst, bedsidetable_dst, lamp_dst, sofa_dst, chair_dst, table_dst, shelf_dst, seat_dst, screen_dst, stage_dst




# class_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class_name = ['cat', 'cow', 'dog', 'horse', 'panda']
root_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/val_aug'
file_list = list()
dir_list = list()

get_file_path(root_path, file_list, dir_list)
print(len(file_list))
shuffle(file_list)

train_txt = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/val_aug.txt'
# val_txt = '/root/XCSGCNN/VOCdevkit/reformed_data/val.txt'


def reform():
    for name in class_name:
        root_path = os.path.join('/root/XCSGCNN/VOCdevkit/my_exp_data/object', name)
        file_list = list()
        dir_list = list()
        get_file_path(root_path, file_list, dir_list)
        for i in range(int(0.9*len(file_list))):
            train_path = '/root/XCSGCNN/VOCdevkit/my_exp_data/train'
            object_path = file_list[i]
            image_path = object_path.replace('object', 'Image')
            head_path = object_path.replace('object', 'head')
            target_image_path = train_path + '/Image' + '/' + name + '/' + image_path.split('/')[-1]
            target_object_path = train_path + '/object' + '/' + name + '/' + object_path.split('/')[-1]
            target_head_path = train_path + '/head' + '/' + name + '/' + head_path.split('/')[-1]
            if not os.path.exists(os.path.split(target_image_path)[0]):
                os.makedirs(os.path.split(target_image_path)[0])
            if not os.path.exists(os.path.split(target_object_path)[0]):
                os.makedirs(os.path.split(target_object_path)[0])
            if not os.path.exists(os.path.split(target_head_path)[0]):
                os.makedirs(os.path.split(target_head_path)[0])
            shutil.copy(image_path, target_image_path)
            shutil.copy(object_path, target_object_path)
            shutil.copy(head_path, target_head_path)
        for i in range(int(0.9*len(file_list)), len(file_list)):
            train_path = '/root/XCSGCNN/VOCdevkit/my_exp_data/val'
            object_path = file_list[i]
            image_path = object_path.replace('object', 'Image')
            head_path = object_path.replace('object', 'head')
            target_image_path = train_path + '/Image' + '/' + name + '/' + image_path.split('/')[-1]
            target_object_path = train_path + '/object' + '/' + name + '/' + object_path.split('/')[-1]
            target_head_path = train_path + '/head' + '/' + name + '/' + head_path.split('/')[-1]
            if not os.path.exists(os.path.split(target_image_path)[0]):
                os.makedirs(os.path.split(target_image_path)[0])
            if not os.path.exists(os.path.split(target_object_path)[0]):
                os.makedirs(os.path.split(target_object_path)[0])
            if not os.path.exists(os.path.split(target_head_path)[0]):
                os.makedirs(os.path.split(target_head_path)[0])
            shutil.copy(image_path, target_image_path)
            shutil.copy(object_path, target_object_path)
            shutil.copy(head_path, target_head_path)


def reform_scene(class_name):
    root_path = '/root/XCSGCNN/VOCdevkit/scene/origin_data/' + class_name
    file_list = list()
    dir_list = list()
    get_file_path(root_path, file_list, dir_list)
    image_list = list()

    for file in file_list:
        if file.endswith('.jpg'):
            image_list.append(file)
    shuffle(image_list)
    print(len(image_list))
    

    for i in range(int(0.9*len(image_list))):
        json_file = image_list[i].replace('.jpg', '.json')
        target_image_dir = '/root/XCSGCNN/VOCdevkit/scene/train/' + '/image/' + class_name
        target_annotation_dir = '/root/XCSGCNN/VOCdevkit/scene/train/' + '/annotation/' + class_name
        if not os.path.exists(target_image_dir):
            os.makedirs(target_image_dir)
        if not os.path.exists(target_annotation_dir):
            os.makedirs(target_annotation_dir)
        shutil.copy(image_list[i], os.path.join(target_image_dir, image_list[i].split('/')[-1]))
        shutil.copy(json_file, os.path.join(target_annotation_dir, json_file.split('/')[-1]))
    for i in range(int(0.9*len(image_list)), len(image_list)):
        json_file = image_list[i].replace('.jpg', '.json')
        target_image_dir = '/root/XCSGCNN/VOCdevkit/scene/val/' + '/image/' + class_name
        target_annotation_dir = '/root/XCSGCNN/VOCdevkit/scene/val/' + '/annotation/' + class_name
        if not os.path.exists(target_image_dir):
            os.makedirs(target_image_dir)
        if not os.path.exists(target_annotation_dir):
            os.makedirs(target_annotation_dir)
        shutil.copy(image_list[i], os.path.join(target_image_dir, image_list[i].split('/')[-1]))
        shutil.copy(json_file, os.path.join(target_annotation_dir, json_file.split('/')[-1]))




with open(train_txt, 'w') as f:
    for object_file in file_list:
        label_name = object_file.split('/')[-2]
        label = class_name.index(label_name)
        file_name = object_file.split('/')[-2] + '/' + object_file.split('/')[-1]
        object = 'object/' + file_name
        bed = 'bed/' + file_name
        bedsidetable = 'bedsidetable/' + file_name
        lamp = 'lamp/' + file_name
        sofa = 'sofa/' + file_name
        chair = 'chair/' + file_name
        table = 'table/' + file_name
        shelf = 'shelf/' + file_name
        seat = 'seat/' + file_name
        screen = 'screen/' + file_name
        stage = 'stage/' + file_name
        content = object + ' ' + str(label) + ' ' + '\n'
        f.write(content)




# for file in tqdm(file_list):
#     if file.endswith('.jpg'):
#         target_bed_path = file.replace('image', 'bed')
#         target_bedsidetable_path = file.replace('image', 'bedsidetable')
#         target_lamp_path = file.replace('image', 'lamp')
#         target_sofa_path = file.replace('image', 'sofa')
#         target_chair_path = file.replace('image', 'chair')
#         target_table_path = file.replace('image', 'table')
#         target_shelf_path = file.replace('image', 'shelf')
#         target_seat_path = file.replace('image', 'seat')
#         target_screen_path = file.replace('image', 'sreen')
#         target_stage_path = file.replace('image', 'stage')

#         if not os.path.exists(os.path.split(target_bed_path)[0]):
#             os.makedirs(os.path.split(target_bed_path)[0])
#         if not os.path.exists(os.path.split(target_bedsidetable_path)[0]):
#             os.makedirs(os.path.split(target_bedsidetable_path)[0])
#         if not os.path.exists(os.path.split(target_lamp_path)[0]):
#             os.makedirs(os.path.split(target_lamp_path)[0])
#         if not os.path.exists(os.path.split(target_sofa_path)[0]):
#             os.makedirs(os.path.split(target_sofa_path)[0])
#         if not os.path.exists(os.path.split(target_chair_path)[0]):
#             os.makedirs(os.path.split(target_chair_path)[0])
#         if not os.path.exists(os.path.split(target_table_path)[0]):
#             os.makedirs(os.path.split(target_table_path)[0])
#         if not os.path.exists(os.path.split(target_shelf_path)[0]):
#             os.makedirs(os.path.split(target_shelf_path)[0])
#         if not os.path.exists(os.path.split(target_seat_path)[0]):
#             os.makedirs(os.path.split(target_seat_path)[0])
#         if not os.path.exists(os.path.split(target_screen_path)[0]):
#             os.makedirs(os.path.split(target_screen_path)[0])
#         if not os.path.exists(os.path.split(target_stage_path)[0]):
#             os.makedirs(os.path.split(target_stage_path)[0])
        
#         json_file = file.replace('image', 'annotation')
#         json_file = json_file.replace('.jpg', '.json')
#         bed_dst, bedsidetable_dst, lamp_dst, sofa_dst, chair_dst, table_dst, shelf_dst, seat_dst, screen_dst, stage_dst = crop_poly_test(file, json_file)
#         cv2.imwrite(target_bed_path, bed_dst)
#         cv2.imwrite(target_bedsidetable_path, bedsidetable_dst)
#         cv2.imwrite(target_lamp_path, lamp_dst)
#         cv2.imwrite(target_sofa_path, sofa_dst)
#         cv2.imwrite(target_chair_path, chair_dst)
#         cv2.imwrite(target_table_path, table_dst)
#         cv2.imwrite(target_shelf_path, shelf_dst)
#         cv2.imwrite(target_seat_path, seat_dst)
#         cv2.imwrite(target_screen_path, screen_dst)
#         cv2.imwrite(target_stage_path, stage_dst)


# reform_scene('bedroom')
# reform_scene('livingroom')
# reform_scene('store')
# reform_scene('theater')
# reform_scene('diningroom')
        

    


# with open(train_txt, 'w') as f:
#     for object_file in file_list:
#         label_name = object_file.split('/')[-2]
#         label = class_name.index(label_name)
#         file_name = object_file.split('/')[-2] + '/' + object_file.split('/')[-1]
#         object = 'image_new/' + file_name
#         bed = 'bed/' + file_name
#         bedsidetable = 'bedsidetable/' + file_name
#         lamp = 'lamp/' + file_name
#         sofa = 'sofa/' + file_name
#         chair = 'chair/' + file_name
#         table = 'table/' + file_name
#         shelf = 'shelf/' + file_name
#         seat = 'seat/' + file_name
#         screen = 'screen/' + file_name
#         stage = 'stage/' + file_name
#         content = object + ' ' + bed + ' ' + bedsidetable + ' ' + lamp + ' ' + sofa + ' ' + chair + ' ' + table + ' ' + shelf + ' ' + seat + ' ' + screen + ' ' + stage + ' ' + str(label) + ' ' + '\n'
#         f.write(content)

# for name in class_name:
#     root_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/reformed_data/val/image/' + name
#     tmp_file_list = list()
#     tmp_dir_list = list()
#
#     get_file_path(root_path, tmp_file_list, tmp_dir_list)
#
#     file_length = len(tmp_file_list)
#     cnt = 0
#     for file in tmp_file_list:
#         target_image_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/reformed_data/train/image/' + name + '/' + file.split('/')[-1]
#         object_path = file.replace('image', 'object')
#         target_object_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/reformed_data/train/object/' + name + '/' + file.split('/')[-1]
#         shutil.copy(file, target_image_path)
#         shutil.copy(object_path, target_object_path)
#         os.remove(file)
#         os.remove(object_path)
#         cnt += 1
#         if cnt >= (0.667*file_length):
#             break






# for name in tqdm(class_name):
#     train_txt_path = '/root/XCSGCNN/VOCdevkit/VOC2012/ImageSets/Main/' + name + '_train.txt'
#     val_txt_path = '/root/XCSGCNN/VOCdevkit/VOC2012/ImageSets/Main/' + name + '_val.txt'
#     image_dir = '/root/XCSGCNN/VOCdevkit/VOC2012/JPEGImages'
#     mask_dir = '/root/XCSGCNN/VOCdevkit/VOC2012/SegmentationClass'
#     mask_list = [mask.split('.')[0] for mask in os.listdir(mask_dir)]
#     target_train_image_dir = '/root/XCSGCNN/VOCdevkit/reformed_data/train/image/' + name
#     target_train_object_dir = '/root/XCSGCNN/VOCdevkit/reformed_data/train/object/' + name
#     target_val_image_dir = '/root/XCSGCNN/VOCdevkit/reformed_data/val/image/' + name
#     target_val_object_dir = '/root/XCSGCNN/VOCdevkit/reformed_data/val/object/' + name

#     if not os.path.exists(target_train_image_dir):
#         os.makedirs(target_train_image_dir)
#     if not os.path.exists(target_train_object_dir):
#         os.makedirs(target_train_object_dir)
#     if not os.path.exists(target_val_image_dir):
#         os.makedirs(target_val_image_dir)
#     if not os.path.exists(target_val_object_dir):
#         os.makedirs(target_val_object_dir)

#     train_record_list = list()
#     val_record_list = list()

#     with open(train_txt_path, 'r') as f:
#         for line in f.readlines():
#             line = line.strip('\n')
#             if line.split(' ')[-1] == '1':
#                 line = line.split(' ')[0]
#                 if line in mask_list:
#                     train_record_list.append(line)

#     with open(val_txt_path, 'r') as f:
#         for line in f.readlines():
#             line = line.strip('\n')
#             if line.split(' ')[-1] == '1':
#                 line = line.split(' ')[0]
#                 if line in mask_list:
#                     val_record_list.append(line)

#     for train_record in train_record_list:
#         image_path = os.path.join(image_dir, train_record+'.jpg')
#         mask_path = os.path.join(mask_dir, train_record + '.png')
#         target_image_path = os.path.join(target_train_image_dir, train_record+'.jpg')
#         target_object_path = os.path.join(target_train_object_dir, train_record+'.jpg')

#         image_array = get_image_array(image_path)
#         mask_array = get_image_array(mask_path)
#         main_object = segmentation(mask_array, image_array, name)
#         main_object = Image.fromarray(main_object)

#         shutil.copy(image_path, target_image_path)
#         main_object.save(target_object_path)

#     for val_record in val_record_list:
#         image_path = os.path.join(image_dir, val_record+'.jpg')
#         mask_path = os.path.join(mask_dir, val_record+'.png')
#         target_image_path = os.path.join(target_val_image_dir, val_record+'.jpg')
#         target_object_path = os.path.join(target_val_object_dir, val_record+'.jpg')

#         image_array = get_image_array(image_path)
#         mask_array = get_image_array(mask_path)
#         main_object = segmentation(mask_array, image_array, name)
#         main_object = Image.fromarray(main_object)

#         shutil.copy(image_path, target_image_path)
#         main_object.save(target_object_path)




