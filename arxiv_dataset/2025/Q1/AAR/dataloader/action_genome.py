import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
from scipy.misc import imread
import numpy as np
import pickle
import os
import copy
import csv
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from argparse import ArgumentParser


class AG(Dataset):

    def __init__(self, mode, data_path=None, filter_nonperson_box_frame=True, num_frames=2, infer_last=False, task="set"):

        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()

        # putting / between words that mean the same thing
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()

        # putting _ to act as space
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]


        print('-------loading annotations---------slowly-----------')

        with open(root_path + 'annotations/person_bbox.pkl', 'rb') as f:
            person_bbox = pickle.load(f)
        f.close()
        with open(root_path+'annotations/object_bbox_and_relationship.pkl', 'rb') as f:
            object_bbox = pickle.load(f)
        f.close()
        with open('dataloader/frame_action_list.pkl', 'rb') as f:
            frame_level_action = pickle.load(f)
        f.close()
        with open('dataloader/frame_action_seq.pkl', 'rb') as f:
            frame_level_seq_action = pickle.load(f)

        print('--------------------finish!-------------------------')

        # get train / test videos and check if frame is valid
        video_dict = {}
        for i in person_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: # check train or testing set
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True

                if frame_valid and i in frame_level_action:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]

        self.video_list = []
        self.video_size = [] # (w,h)
        self.gt_annotations = []
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0
        self.mem_gap = 1
        self.num_frames = num_frames # 5 including current frame
        self.task = task
        self.infer_last = infer_last
        self.valid_num_frames = 0

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''

        for i in video_dict.keys():
            video = []
            gt_annotation_video = []

            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    # no person bounding box
                    if person_bbox[j]['bbox'].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1
                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox']}]
                # each frames's objects and human
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                        k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
                        k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
                        k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            # get frame-level action classes of current and previous frames
            if self.num_frames != -1:
                for i in range(len(gt_annotation_video)):
                    frame_idx = list(filter(lambda x: x >=0, range(i, i-self.mem_gap*(self.num_frames), -self.mem_gap)))[::-1]
                    frame_idx = sorted(list(set(frame_idx)))

                    frame = list(map(video.__getitem__, frame_idx))

                    action_class = []

                    for j in frame:
                        action_class.append(frame_level_action[j])

                    actions = sorted(list(set([action for inner_list in action_class for action in inner_list])))

                    gt_annotation_video[i].append({'action_class': actions})
            else:
                # get all previous actions up to current time including current frame actions
                dataset_list = []
                for i in range(len(gt_annotation_video)):

                    frame_idx = list(filter(lambda x: x >=0, range(i, -self.mem_gap*(i+1), -self.mem_gap)))[::-1]
                    frame_idx = sorted(list(set(frame_idx)))

                    frame = list(map(video.__getitem__, frame_idx))
                    
                    action_class = []

                    for j in frame:
                        action_class.append(frame_level_action[j])

                    actions = sorted(list(set([action for inner_list in action_class for action in inner_list])))
                    
                    if self.task == 'sequence':
                        actions = frame_level_seq_action[frame[i]]
                        actions.append(157) # 157 denotes end of sequence charades 0 to 156 class. 

                    gt_annotation_video[i].append({'action_class': actions})
                    
            # check length of video, only keep video with more than or equal to 2 frames - minimum need 2 frames to do AAI
            if len(video) >= 2:
                if self.infer_last == False:
                    # remove first frame cause it does not have past frame and therefore, actions
                    video = video[1:]
                    gt_annotation_video = gt_annotation_video[1:]
                    
                    self.valid_num_frames = self.valid_num_frames + len(video)
                    
                    self.video_list.append(video)
                    self.video_size.append(person_bbox[j]['bbox_size'])
                    self.gt_annotations.append(gt_annotation_video)

                else:
                    # only take last frame and all its past actions for inference
                    last_frame_of_video = []
                    last_frame_gt_of_video = []
                    last_frame_of_video.append(video[-1])
                    last_frame_gt_of_video.append(gt_annotation_video[-1])

                    self.valid_num_frames = self.valid_num_frames + len(last_frame_of_video)
                    
                    self.video_list.append(last_frame_of_video)
                    self.video_size.append(person_bbox[j]['bbox_size'])
                    self.gt_annotations.append(last_frame_gt_of_video)
 
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x'*60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_num_frames))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_num_frames))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(non_heatmap_nums))
        print('x' * 60)

    def __getitem__(self, index):

        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):
            im = imread(os.path.join(self.frames_path, name)) # channel h,w,3
            im = im[:, :, ::-1] # rgb -> bgr
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im_scales.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index

    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]

def parse_charades_csv(filename):
    labels = {}
    vids = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')] # split individual actions then their class and start and end time
                actions = [{'class': x, 'start': float(
                    y), 'end': float(z)} for x, y, z in actions]
            labels[vid] = actions
            vids.append(vid)
    return vids, labels

def cls2int(x):
    return int(x[1:])

def sort_indices(list_of_dicts, key):
    return [i for i, j in sorted(enumerate(list_of_dicts), key=lambda x:x[1][key])]

def sort_list(list_of_dicts, key):
    sorted_indices = sort_indices(list_of_dicts, key)
    updated_dict = [list_of_dicts[i] for i in sorted_indices]

    # convert str to int too
    for dictionary in updated_dict:
        dictionary['class'] = cls2int(dictionary['class'])

    return updated_dict

def get_sequence(list_of_dicts, key, action_list, current_time):

    sequence = []

    for dictionary in list_of_dicts:
        if dictionary[key] in action_list:
            if current_time >= dictionary['start']: # keep those actions that already happened
                sequence.append(dictionary[key])

    return sequence