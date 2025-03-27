"""final version """
import math
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import json
from random import random
from util.prep import image_norm, make_onehot_label

class DataGen_CLS_CMG_P_SEG(object):
    def __init__(self, file_path, gt_path, id_list):
        self.lesion_threshold =1
        self.foreground_threshold = 500
        self.batch_size = 32
        self.cube_size = 32
        self.lesion_label = 1
        self.file_list = []
        for ids in id_list:
            datas = {}
            subject_name = 'subject-{}-'.format(ids)
            print("load image file: {}".format(subject_name))
            T1 = os.path.join(file_path, subject_name + 'T1.nii')
            label = os.path.join(gt_path, subject_name + 'label.nii')

            t1_data = nib.load(T1).get_fdata()
            label_data = nib.load(label).get_fdata()
            label_data[label_data > 0.5] = 1
            label_data = label_data.astype(np.int8)

            t1_data = image_norm(t1_data)
            mask = np.array(t1_data > 0, dtype=np.int8)

            datas['images'] = t1_data
            datas['label'] = label_data
            datas['mask'] = mask

            shape = datas['label'].shape
            if len(shape) == 3:
                datas['images'] = tf.expand_dims(datas['images'], -1)
                datas['mask'] = tf.expand_dims(datas['mask'], -1)
                datas['label'] = tf.expand_dims(datas['label'], -1)

            self.file_list.append(datas)

    def make_gen_CLS(self):
        while True:
            images_cubes = []
            label_cubes = []
            label_class = []
            """first helf batch cubes: positive(contain lesion)"""
            curr_batch_idx = 0
            while curr_batch_idx < (self.batch_size)/2:
                file_idx = np.random.randint(0, len(self.file_list))
                random_file = self.file_list[file_idx]
                h, w, d, _ = random_file['images'].shape

                while True:
                    # randomly select cube position (h,w,d)
                    random_hidx = np.random.randint(0, h-self.cube_size)
                    random_widx = np.random.randint(0, w-self.cube_size)
                    random_didx = np.random.randint(0, d-self.cube_size)
                    mask_cube = random_file['mask'][random_hidx:random_hidx+self.cube_size,
                                                    random_widx:random_widx+self.cube_size,
                                                    random_didx:random_didx+self.cube_size, :]

                    label_cube = random_file['label'][random_hidx:random_hidx + self.cube_size,
                                                      random_widx:random_widx + self.cube_size,
                                                      random_didx:random_didx + self.cube_size, :]

                    if (np.sum(mask_cube) >= self.foreground_threshold) & (np.sum(label_cube == self.lesion_label) >= self.lesion_threshold):

                        break

                random_images_cube = np.expand_dims(random_file['images'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)
                random_label_cube = np.expand_dims(random_file['label'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)

                images_cubes.append(random_images_cube)
                label_cubes.append(random_label_cube)

                lesion_num = np.sum(random_label_cube == self.lesion_label)
                if lesion_num >= self.lesion_threshold:
                    class_label = 1
                else:
                    class_label = 0
                label_class.append(class_label)

                curr_batch_idx += 1
            curr_batch_idx = 0
            """another helf batch cubes: negative(not contain lesion)"""
            while curr_batch_idx < (self.batch_size) / 2:
                file_idx = np.random.randint(0, len(self.file_list))
                random_file = self.file_list[file_idx]
                h, w, d, _ = random_file['images'].shape

                while True:
                    random_hidx = np.random.randint(0, h - self.cube_size)
                    random_widx = np.random.randint(0, w - self.cube_size)
                    random_didx = np.random.randint(0, d - self.cube_size)

                    mask_cube = random_file['mask'][random_hidx:random_hidx + self.cube_size,
                                random_widx:random_widx + self.cube_size,
                                random_didx:random_didx + self.cube_size, :]
                    label_cube = random_file['label'][random_hidx:random_hidx + self.cube_size,
                                                      random_widx:random_widx + self.cube_size,
                                                      random_didx:random_didx + self.cube_size, :]
                    if (np.sum(mask_cube) >= self.foreground_threshold) & (np.sum(label_cube == self.lesion_label) == 0):
                        break

                random_images_cube = np.expand_dims(random_file['images'][random_hidx:random_hidx + self.cube_size,
                                                    random_widx:random_widx + self.cube_size,
                                                    random_didx:random_didx + self.cube_size, :], axis=0)
                random_label_cube = np.expand_dims(random_file['label'][random_hidx:random_hidx + self.cube_size,
                                                   random_widx:random_widx + self.cube_size,
                                                   random_didx:random_didx + self.cube_size, :], axis=0)

                images_cubes.append(random_images_cube)
                label_cubes.append(random_label_cube)
                lesion_num = np.sum(random_label_cube == self.lesion_label)

                if lesion_num >= self.lesion_threshold:
                    class_label = 1
                else:
                    class_label = 0

                label_class.append(class_label)

                curr_batch_idx += 1


            images_cubes = np.concatenate(images_cubes, axis=0)
            label_cubes = np.concatenate(label_cubes, axis=0)
            label_class = np.array(label_class)
            yield (
                {'input': images_cubes},
                {'output': label_class}
            )

    def make_gen_CMG(self):
        while True:
            images_cubes = []
            masks_cubes = []
            label_cubes = []
            label_class = []
            """first helf batch cubes: positive(contain lesion)"""
            curr_batch_idx = 0
            while curr_batch_idx < (self.batch_size)/2:
                file_idx = np.random.randint(0, len(self.file_list))
                random_file = self.file_list[file_idx]
                h, w, d, _ = random_file['images'].shape

                while True:
                    # randomly select cube position (h,w,d)
                    random_hidx = np.random.randint(0, h-self.cube_size)
                    random_widx = np.random.randint(0, w-self.cube_size)
                    random_didx = np.random.randint(0, d-self.cube_size)
                    mask_cube = random_file['mask'][random_hidx:random_hidx+self.cube_size,
                                                    random_widx:random_widx+self.cube_size,
                                                    random_didx:random_didx+self.cube_size, :]

                    label_cube = random_file['label'][random_hidx:random_hidx + self.cube_size,
                                                      random_widx:random_widx + self.cube_size,
                                                      random_didx:random_didx + self.cube_size, :]

                    if (np.sum(mask_cube) >= self.foreground_threshold) & (np.sum(label_cube == self.lesion_label) >= self.lesion_threshold):
                        break

                random_images_cube = np.expand_dims(random_file['images'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)
                random_label_cube = np.expand_dims(random_file['label'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)
                random_mask_cube = np.expand_dims(random_file['mask'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)

                images_cubes.append(random_images_cube)
                label_cubes.append(random_label_cube)
                masks_cubes.append(random_mask_cube)

                lesion_num = np.sum(random_label_cube == self.lesion_label)
                if lesion_num >= self.lesion_threshold:
                    class_label = 1
                else:
                    class_label = 0
                label_class.append(class_label)

                curr_batch_idx += 1

            curr_batch_idx = 0
            """another helf batch cubes: negative(not contain lesion)"""
            while curr_batch_idx < (self.batch_size) / 2:
                file_idx = np.random.randint(0, len(self.file_list))
                random_file = self.file_list[file_idx]
                h, w, d, _ = random_file['images'].shape

                while True:
                    random_hidx = np.random.randint(0, h - self.cube_size)
                    random_widx = np.random.randint(0, w - self.cube_size)
                    random_didx = np.random.randint(0, d - self.cube_size)

                    mask_cube = random_file['mask'][random_hidx:random_hidx + self.cube_size,
                                random_widx:random_widx + self.cube_size,
                                random_didx:random_didx + self.cube_size, :]
                    label_cube = random_file['label'][random_hidx:random_hidx + self.cube_size,
                                                      random_widx:random_widx + self.cube_size,
                                                      random_didx:random_didx + self.cube_size, :]

                    if (np.sum(mask_cube) >= self.foreground_threshold) & (np.sum(label_cube == self.lesion_label) < self.lesion_threshold):
                        break

                random_images_cube = np.expand_dims(random_file['images'][random_hidx:random_hidx + self.cube_size,
                                                    random_widx:random_widx + self.cube_size,
                                                    random_didx:random_didx + self.cube_size, :], axis=0)
                random_label_cube = np.expand_dims(random_file['label'][random_hidx:random_hidx + self.cube_size,
                                                   random_widx:random_widx + self.cube_size,
                                                   random_didx:random_didx + self.cube_size, :], axis=0)
                random_mask_cube = np.expand_dims(random_file['mask'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)

                images_cubes.append(random_images_cube)
                label_cubes.append(random_label_cube)
                masks_cubes.append(random_mask_cube)

                lesion_num = np.sum(random_label_cube == self.lesion_label)

                if lesion_num >= self.lesion_threshold:
                    class_label = 1
                else:
                    class_label = 0

                label_class.append(class_label)

                curr_batch_idx += 1
            images_cubes = np.concatenate(images_cubes, axis=0)
            masks_cubes = np.concatenate(masks_cubes, axis=0)
            label_cubes = np.concatenate(label_cubes, axis=0)
            label_class = np.array(label_class)
            yield (
                {'input': images_cubes},
                {'class_label': label_class},
                {'mask': masks_cubes}
            )

    def make_gen_P_SEG(self):
        while True:
            images_cubes = []
            masks_cubes = []
            label_cubes = []
            label_class = []
            """all: positive(contain lesion)"""
            curr_batch_idx = 0
            while curr_batch_idx < (self.batch_size):
                file_idx = np.random.randint(0, len(self.file_list))
                random_file = self.file_list[file_idx]
                h, w, d, _ = random_file['images'].shape

                while True:
                    # randomly select cube position (h,w,d)
                    random_hidx = np.random.randint(0, h-self.cube_size)
                    random_widx = np.random.randint(0, w-self.cube_size)
                    random_didx = np.random.randint(0, d-self.cube_size)
                    mask_cube = random_file['mask'][random_hidx:random_hidx+self.cube_size,
                                                    random_widx:random_widx+self.cube_size,
                                                    random_didx:random_didx+self.cube_size, :]

                    label_cube = random_file['label'][random_hidx:random_hidx + self.cube_size,
                                                      random_widx:random_widx + self.cube_size,
                                                      random_didx:random_didx + self.cube_size, :]

                    if (np.sum(mask_cube) >= self.foreground_threshold) & (np.sum(label_cube == self.lesion_label) >= self.lesion_threshold):
                        break

                random_images_cube = np.expand_dims(random_file['images'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)
                random_label_cube = np.expand_dims(random_file['label'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)
                random_mask_cube = np.expand_dims(random_file['mask'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)

                images_cubes.append(random_images_cube)
                label_cubes.append(random_label_cube)
                masks_cubes.append(random_mask_cube)

                lesion_num = np.sum(random_label_cube == self.lesion_label)
                if lesion_num >= self.lesion_threshold:
                    class_label = 1
                else:
                    class_label = 0
                label_class.append(class_label)

                curr_batch_idx += 1

            images_cubes = np.concatenate(images_cubes, axis=0)
            masks_cubes = np.concatenate(masks_cubes, axis=0)
            label_cubes = np.concatenate(label_cubes, axis=0)
            label_class = np.array(label_class)
            yield (
                {'input': images_cubes},
                {'P_label': label_cubes},
                {'class_label': label_class},
                {'mask': masks_cubes}
            )

class DataGen_T_SEG(object):
    def __init__(self, file_path, gt_path, id_list):
        self.batch_size = 16
        self.cube_size = 32
        self.file_list = []
        for ids in id_list:
            datas = {}
            subject_name = 'subject-{}-'.format(ids)
            print("load image file: {}".format(subject_name))
            T1 = os.path.join(file_path, subject_name + 'T1.nii')
            label = os.path.join(gt_path, subject_name + 'label.nii')
            t1_data = nib.load(T1).get_fdata()
            mask = np.array(t1_data > 0, dtype=np.int8)

            label_data = make_onehot_label(nib.load(label).get_fdata(), 4)
            t1_data = image_norm(t1_data)
            datas['images'] = t1_data

            datas['label'] = label_data
            datas['mask'] = mask

            self.file_list.append(datas)

    def make_gen(self):
        while True:
            curr_batch_idx = 0
            images_cubes = []
            label_cubes = []
            while curr_batch_idx < self.batch_size:
                file_idx = np.random.randint(0, len(self.file_list))
                random_file = self.file_list[file_idx]
                h, w, d, _ = random_file['images'].shape

                while True:
                    random_hidx = np.random.randint(0, h - self.cube_size)
                    random_widx = np.random.randint(0, w - self.cube_size)
                    random_didx = np.random.randint(0, d - self.cube_size)
                    mask_cube = random_file['mask'][random_hidx:random_hidx + self.cube_size,
                                random_widx:random_widx + self.cube_size,
                                random_didx:random_didx + self.cube_size, :]
                    if np.sum(mask_cube) >= 50:
                        break

                random_images_cube = np.expand_dims(random_file['images'][random_hidx:random_hidx + self.cube_size,
                                                    random_widx:random_widx + self.cube_size,
                                                    random_didx:random_didx + self.cube_size, :], axis=0)
                random_label_cube = np.expand_dims(random_file['label'][random_hidx:random_hidx + self.cube_size,
                                                   random_widx:random_widx + self.cube_size,
                                                   random_didx:random_didx + self.cube_size, :], axis=0)

                images_cubes.append(random_images_cube)
                label_cubes.append(random_label_cube)
                curr_batch_idx += 1

            images_cubes = np.concatenate(images_cubes, axis=0)
            label_cubes = np.concatenate(label_cubes, axis=0)

            yield (
                {'input': images_cubes},
                {'output': label_cubes}
            )


