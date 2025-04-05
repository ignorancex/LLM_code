#
"""
@author:  Ehsan Yaghoubi
@contact: Ehsan.yaghoubi@gmail.com
"""

import glob
import re
import os
from .bases import BaseImageDataset
import mat4py
from RAP_script.rap_data_loading import load_reid_data
from RAP_script.rap_data_loading import load_rap_dataset
from config import cfg

class RAP (BaseImageDataset):

    dataset_dir = 'RAP_resized_imgs256x256'
    rap_mat_file = 'Anchor_level_rap/rap_annotations/RAP_annotation.mat'
    rap_json_data = "/media/ehsan/48BE4782BE476810/AA_GITHUP/forked_reid_baseline/RAP_script/rap_data.json"

    def __init__(self, root='/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper', verbose=True, **kwargs):
        super(RAP, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.rap_mat_file = os.path.join(root, self.rap_mat_file)
        self._check_before_run()
        self.rap_data = load_rap_dataset(rap_attributes_filepath=self.rap_mat_file, rap_keypoints_json=self.rap_json_data,
                                    load_from_file=True)

        train, query, gallery, rap_data = self._process_dir(self.rap_mat_file, self.dataset_dir, relabel= True)

        self.train = train
        self.query = query
        self.gallery = gallery



        if verbose:
            print("=> RAP is loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        #assert self.num_train_pids == 1295


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.rap_mat_file):
            raise RuntimeError("'{}' is not available".format(self.rap_mat_file))


    def _process_dir(self, rap_mat_file, data_dir, relabel):
        image_names, pids, train_indices, gallery_indices, query_indices=load_reid_data(rap_mat_file)
        #rap_data = load_rap_dataset(rap_attributes_filepath=self.rap_mat_file, rap_keypoints_json=self.rap_json_data, load_from_file=True)
        pid_container = set()
        for index, IMG_index1 in enumerate(train_indices):
            person_id = pids[IMG_index1-1]
            if person_id == -1: continue  # junk images are just ignored
            pid_container.add(person_id)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        train = []
        ids = []
        cam_ids = []
        flag = 0
        for index, IMG_index1 in enumerate(train_indices):
            try:
                person_id = pids[IMG_index1-1]
                if person_id == -1: continue  # junk images are just ignored
                img_name = image_names[IMG_index1-1]
                img_labels = self.rap_data[img_name]['attrs']
                img_full_path = os.path.join(data_dir, img_name)
                camera = img_name.split("-")[0] # e.g. ['CAM01', '2014', '02', '15', '20140215161032', '20140215162620', 'tarid0', 'frame218', 'line1.png']
                camera_id = int(re.findall('\d+',camera)[0])
                assert 1 <= person_id <= 2589  # There are 2589 person identities in RAP dataset
                assert 1 <= camera_id <= 31 # There are 23 cameras with labels between 1 to 31
                cam_ids.append(camera_id)
                ids.append(person_id)
                if relabel: person_id = pid2label[person_id] # train ids must be relabelled from zero
                train.append((img_full_path, person_id, camera_id, img_labels))
            except KeyError:
                flag += 1
                #print(" Warning at Train set {}: information of below image is not found in rap_data.\n{}".format(flag, img_name))
        print(" Warning at Train set: information of {} images are not found in rap_data".format(flag))
        # print(">> Number of train Images: ", len(set(train)))
        # print(">> Number of train IDs: ",len(set(ids)))
        # print(">> Number of train Camera IDs: ",len(set(cam_ids)))

        query= []
        ids = []
        cam_ids = []
        flag = 0
        for i2, IMG_index2 in enumerate(query_indices):
            try:
                person_id = pids[IMG_index2-1]
                if person_id == -1: continue  # junk images are just ignored
                img_name = image_names[IMG_index2-1]
                img_labels = self.rap_data[img_name]['attrs']
                img_full_path = os.path.join(data_dir, img_name)
                camera = img_name.split("-")[0] # e.g. ['CAM01', '2014', '02', '15', '20140215161032', '20140215162620', 'tarid0', 'frame218', 'line1.png']
                camera_id = int(re.findall('\d+',camera)[0])
                assert 1 <= person_id <= 2589  # There are 2589 person identities in RAP dataset
                assert 1 <= camera_id <= 31 # There are 23 cameras with labels between 1 to 31
                cam_ids.append(camera_id)
                ids.append(person_id)
                query.append((img_full_path, person_id , camera_id, img_labels))
            except KeyError:
                flag += 1
                #print(" Warning at Query set {}: information of below image is not found in rap_data.\n{}".format(flag, img_name))
        print(" Warning at Train set: information of {} images are not found in rap_data".format(flag))
        # print(">>>> Number of query Images: ",len(set(query)))
        # print(">>>> Number of query IDs: ",len(set(ids)))
        # print(">>>> Number of query Camera IDs: ",len(set(cam_ids)))

        gallery= []
        ids = []
        cam_ids = []
        flag = 0
        for i3, IMG_index3 in enumerate(gallery_indices):
            try:
                person_id = pids[IMG_index3-1]
                if person_id == -1: continue  # junk images are just ignored
                img_name = image_names[IMG_index3-1]
                img_labels = self.rap_data[img_name]['attrs']
                img_full_path = os.path.join(data_dir, img_name)
                camera = img_name.split("-")[0] # e.g. ['CAM01', '2014', '02', '15', '20140215161032', '20140215162620', 'tarid0', 'frame218', 'line1.png']
                camera_id = int(re.findall('\d+',camera)[0])
                assert 1 <= person_id <= 2589  # There are 2589 person identities in RAP dataset
                assert 1 <= camera_id <= 31 # There are 23 cameras with labels between 1 to 31
                cam_ids.append(camera_id)
                ids.append(person_id)
                gallery.append((img_full_path, person_id , camera_id, img_labels))
            except KeyError:
                flag += 1
                #print(" Warning at Gallery set {}: information of below image is not found in rap_data.\n{}".format(flag, img_name))
        print(" Warning at Train set: information of {} images are not found in rap_data".format(flag))
        # print(">>>>>> Number of gallery Images: ",len(set(gallery)))
        # print(">>>>>> Number of gallery IDs: ",len(set(ids)))
        # print(">>>>>> Number of gallery Camera IDs: ",len(set(cam_ids)))

        return  train, query, gallery, self.rap_data

