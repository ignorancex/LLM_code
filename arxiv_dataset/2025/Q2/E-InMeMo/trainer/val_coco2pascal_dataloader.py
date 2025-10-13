"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
# from evaluate.mae_utils import PURPLE, YELLOW
import json
import sys
import matplotlib.pyplot as plt



class DatasetCOCO2PASCAL(Dataset):
    def __init__(self, pascal_datapath, coco_datapath, fold, split, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False,
                 purple: bool = False, cluster: bool = False, feature_name: str = 'features_vit-laion2b_no_cls_trn',
                 percentage: str = '', seed: int = 0, mode: str = '', arr: str = 'a1'):
        self.fold = fold
        self.split = split
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20  # 20
        self.ncluster = 200
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.cluster = cluster
        self.use_original_imgsize = use_original_imgsize

        self.pascal_img_path = os.path.join(pascal_datapath, 'VOC2012/JPEGImages/')
        self.pascal_ann_path = os.path.join(pascal_datapath, 'VOC2012/SegmentationClassAug/')
        self.coco_img_path = os.path.join(coco_datapath, 'train2014/')
        self.coco_ann_path = os.path.join(coco_datapath, 'annotations/train2014/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata_val = self.build_img_metadata('val')
        self.img_metadata_trn = self.build_coco_img_metadata('trn')
        self.feature_name = feature_name
        self.seed = seed
        self.percentage = percentage
        self.images_top50_val = self.get_top50_images_for_validation()
        self.images_top50_trn = self.get_top50_images_trn()
        self.mode = mode
        self.arr = arr

    def __len__(self):
        return 1000
        # return len(self.img_metadata_val)

    def get_top50_images_for_validation(self):
        print('feature name for val: ', self.feature_name[:-4] + '_val')
        with open(f"./pascal-5i/VOC2012/{self.feature_name[:-4]}_val/folder{self.fold}_coco_top50-similarity.json") as f:
            images_top50 = json.load(f)

        images_top50_new = {}
        for img_name, img_class in self.img_metadata_val:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['top50'] = images_top50[img_name]
            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    # def get_top50_images_trn(self):
    #     images_top50_new = {}
    #     for img_name, img_class in self.img_metadata_trn:
    #         if img_name not in images_top50_new:
    #             images_top50_new[img_name] = {}
    #
    #         images_top50_new[img_name]['class'] = img_class
    #
    #     return images_top50_new

    # def get_top50_images_trn(self):
    #     images_top50_new = {}
    #     for img_name, img_class in self.img_metadata_trn:
    #         if img_name not in images_top50_new:
    #             images_top50_new[img_name] = {'class': []}
    #
    #         # Check if img_class is not already in the list to avoid duplicates.
    #         if img_class not in images_top50_new[img_name]['class']:
    #             images_top50_new[img_name]['class'].append(img_class)
    #
    #     return images_top50_new

    def get_top50_images_trn(self):
        images_top50_new = {}
        for img_name, img_class in self.img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def create_gradiant_grid_images(self, support_img, support_mask, query_img, query_mask, arr):
        # create grid image for suppot images and query image.
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))

        content_list = [support_img, support_mask, query_img, query_mask]

        if arr == 'a1':
            support_img = content_list[0]
            support_mask = content_list[1]
            query_img = content_list[2]
            query_mask = content_list[3]

        elif arr == 'a2':
            support_img = content_list[1]
            support_mask = content_list[0]
            query_img = content_list[3]
            query_mask = content_list[2]

        elif arr == 'a3':
            support_img = content_list[3]
            support_mask = content_list[2]
            query_img = content_list[1]
            query_mask = content_list[0]

        elif arr == 'a4':
            support_img = content_list[2]
            support_mask = content_list[3]
            query_img = content_list[0]
            query_mask = content_list[1]

        elif arr == 'a5':
            support_img = content_list[1]
            support_mask = content_list[3]
            query_img = content_list[0]
            query_mask = content_list[2]

        elif arr == 'a6':
            support_img = content_list[3]
            support_mask = content_list[1]
            query_img = content_list[2]
            query_mask = content_list[0]

        elif arr == 'a7':
            support_img = content_list[2]
            support_mask = content_list[0]
            query_img = content_list[3]
            query_mask = content_list[1]

        elif arr == 'a8':
            support_img = content_list[0]
            support_mask = content_list[2]
            query_img = content_list[1]
            query_mask = content_list[3]

        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
        canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
        canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def create_arr_grid_from_images(self, support_img, support_mask, query_img, query_mask, positions):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        # subimg_size = 111
        # images = [support_img, support_mask, query_img, query_mask]
        # coordinates = [(0, 0), (0, 1), (1, 0), (1, 1)]
        #
        # for i, pos in enumerate(positions):
        #     x = coordinates[i][0] * support_img.shape[1]
        #     y = coordinates[i][1] * support_img.shape[2]
        #     canvas[:, x:x + support_img.shape[1], y:y + support_img.shape[2]] = images[pos]
        # if positions == 'a1':
        #     canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        #     canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
        #     canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
        #     canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
        if positions == 'a2':
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_mask
        elif positions == 'a3':
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_mask
        elif positions == 'a4':
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_mask
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_mask
        elif positions == 'a5':
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_mask
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_mask
        elif positions == 'a6':
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_mask
        elif positions == 'a7':
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_mask
        elif positions == 'a8':
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def create_all_grids(self, support_img, support_mask, query_img, query_mask):
        canvas_list = []
        # List of all possible arrangements
        # arrangements = [[0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0], [2, 3, 0, 1],
        #                 [2, 0, 3, 1], [3, 1, 2, 0], [1, 3, 0, 2], [0, 2, 1, 3]]
        arrangements = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']

        for arr in arrangements:
            canvas = self.create_ensemble_grid_from_images(support_img, support_mask, query_img, query_mask, arr)
            canvas_list.append(canvas)

        # print("canvas_list: ", canvas_list[0])
        # print("length of canvas_list: ", canvas_list[0].shape)

        return canvas_list

    def __getitem__(self, idx):
        idx %= len(self.img_metadata_val)  # for testing, as n_images < 1000
        # print('idx: ', idx)
        grid_stack = torch.tensor([]).cuda()

        for sim_idx in range(1):
            query_name, support_name, class_sample_query, class_sample_support = self.sample_episode(idx, sim_idx)
            query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name,
                                                                                                 support_name)

            if self.image_transform:
                query_img = self.image_transform(query_img)
                query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query,
                                                                       purple=self.purple)
                # query_mask.save(f'./{idx}_query_mask.png')
            if self.mask_transform:
                query_mask = self.mask_transform(query_mask)

            if self.image_transform:
                support_img = self.image_transform(support_img)
            support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support,
                                                                       purple=self.purple)
            # support_mask.save(f'./{idx}support_mask.png')

            if self.mask_transform:
                support_mask = self.mask_transform(support_mask)

            if self.arr != 'ensemble':
                grid = self.create_gradiant_grid_images(support_img, support_mask, query_img, query_mask, self.arr)

                # for i in range(len(grid)):
                #     grid[i] = grid[i].unsqueeze(0)
            else:
                grid = self.create_all_grids(support_img, support_mask, query_img, query_mask)
                # grid = grid.unsqueeze(0)

            # print("canvas_list: ", grid)
            # print("length of canvas_list: ", grid.shape)

            if len(grid_stack) == 0:
                grid_stack = grid
            else:
                grid_stack = torch.cat((grid_stack, grid))

        # print('grid stack: ', grid_stack.shape)
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'support_img': support_img,
                 'support_mask': support_mask,
                 'grid_stack': grid_stack,
                 'support_name': support_name,
                 'query_name': query_name
                 }

        return batch

    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary

    def extract_ignore_idx_coco(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id] = 0
            mask[mask == class_id] = 255
            return Image.fromarray(mask), boundary

    def load_frame(self, query_name, support_name):
        # import pdb;pdb.set_trace()
        query_img = self.read_pascal_img(query_name)
        query_mask = self.read_pascal_mask(query_name)
        support_img = self.read_coco_img(support_name)
        support_mask = self.read_coco_mask(support_name)
        org_qry_imsize = query_img.size
        support_img = support_img.convert('RGB')

        return query_img, query_mask, support_img, support_mask, org_qry_imsize

    def read_pascal_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.pascal_ann_path, img_name) + '.png')
        return mask

    def read_pascal_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.pascal_img_path, img_name) + '.jpg')

    def read_coco_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.coco_ann_path, img_name) + '.png')
        return mask

    def read_coco_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.coco_img_path, img_name) + '.jpg')

    def sample_episode(self, idx, sim_idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata_val[idx]

        support_name = self.images_top50_val[query_name]['top50'][sim_idx]

        # support_classes = self.images_top50_trn[support_name]['class']
        # if class_sample in support_classes:
        #     support_class = class_sample
        # else:
        #     support_class = support_classes[0]
        support_class = self.images_top50_trn[support_name]['class']

        if support_name == query_name:
            print('support_name = query_name ' + support_name)
            return self.sample_episode(idx, sim_idx + 1)

        return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = './evaluate'

            if self.cluster:
                fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold_cluster%d.txt' % (split, fold_id))
            else:
                fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()

            # print("fold_n_metadata: ", fold_n_metadata)
            if self.cluster:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1, int(data.split('__')[2]) - 1] for
                                   data in fold_n_metadata]
            else:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]

            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_coco_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = './tools'

            fold_n_metadata_path = os.path.join(cwd, f'coco/{split}/fold{fold_id}.txt')

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()

            # print("fold_n_metadata: ", fold_n_metadata)
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]

            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        if len(self.img_metadata[0]) != 3:
            for img_name, img_class in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]
        else:
            for img_name, img_class, _ in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]

        return img_metadata_classwise


class DatasetPASCALLargeCanvas(Dataset):
    def __init__(self, pascal_datapath, coco_datapath, fold, split, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False,
                 purple: bool = False, cluster: bool = False, feature_name: str = 'features_vit-laion2b_no_cls_trn',
                 percentage: str = '', seed: int = 0, mode: str = '', arr: str = 'a1', cls_base: bool = False,
                 selected_label: int = -1, num_prompt: int = 7):
        self.fold = fold
        self.split = split
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20  # 20
        self.ncluster = 200
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.cluster = cluster
        self.use_original_imgsize = use_original_imgsize
        self.cls_base = cls_base
        self.selected_label = selected_label

        self.pascal_img_path = os.path.join(pascal_datapath, 'VOC2012/JPEGImages/')
        self.pascal_ann_path = os.path.join(pascal_datapath, 'VOC2012/SegmentationClassAug/')
        self.coco_img_path = os.path.join(coco_datapath, 'train2014/')
        self.coco_ann_path = os.path.join(coco_datapath, 'annotations/train2014/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata_val = self.build_img_metadata('val')
        self.all_img_metadata_trn = self.build_coco_img_metadata('trn')
        self.feature_name = feature_name
        self.seed = seed
        self.percentage = percentage
        self.images_top50_val = self.get_top50_images_for_validation()
        self.images_top50_trn = self.get_top50_images_trn()
        self.mode = mode
        self.arr = arr
        self.simidx = num_prompt

    def __len__(self):
        return 1000

    def get_top50_images_for_validation(self):
        print('feature name for val: ', self.feature_name[:-4] + '_val')
        with open(
                f"./pascal-5i/VOC2012/{self.feature_name[:-4]}_val/folder{self.fold}_new_coco_top50-similarity.json") as f:
            images_top50 = json.load(f)

        images_top50_new = {}
        for img_name, img_class in self.img_metadata_val:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['top50'] = images_top50[img_name]
            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def get_top50_images_trn_cls(self):
        images_top50_new = {}
        for img_name, img_class in self.all_img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {'class': []}

            # Check if img_class is not already in the list to avoid duplicates.
            if img_class not in images_top50_new[img_name]['class']:
                images_top50_new[img_name]['class'].append(img_class)

        return images_top50_new

    def get_top50_images_trn(self):
        images_top50_new = {}
        for img_name, img_class in self.all_img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def create_gradiant_cross_grid_images(self, support_imgs, support_masks, query_img, query_mask, arr):
        # Create grid image for support images and query image
        canvas = torch.ones((3, 224, 224))

        # Place the query image and mask at the bottom right
        canvas[:, -48:, -48:] = query_mask
        canvas[:, -48:, -98:-50] = query_img

        # Calculate the positions for placing support images and masks
        support_positions = [
            (126, 126), (176, 0), (126, 0), (50, 126), (0, 126), (50, 0), (0, 0)
        ]

        # Place support images and masks on the canvas
        for i, (support_img, support_mask) in enumerate(zip(support_imgs, support_masks)):
            pos = support_positions[i]
            canvas[:, pos[0]:pos[0] + 48, pos[1]:pos[1] + 48] = support_img
            canvas[:, pos[0]:pos[0] + 48, pos[1] + 50:pos[1] + 98] = support_mask

        # Set remaining positions to 0 if any
        for j in range(i + 1, len(support_positions)):
            pos = support_positions[j]
            canvas[:, pos[0]:pos[0] + 48, pos[1]:pos[1] + 48] = 0
            canvas[:, pos[0]:pos[0] + 48, pos[1] + 50:pos[1] + 98] = 0

        return canvas

    def create_gradiant_compact_grid_images(self, support_imgs, support_masks, query_img, query_mask, arr):
        # Create grid image for support images and query image
        canvas = torch.ones((3, 224, 224))

        # Define the positions for placing images within the 192x192 area
        support_positions = [
            (126, 126), (176, 0), (126, 0), (50, 126), (0, 126), (50, 0), (16, 16)
        ]

        # Place support images and masks on the canvas
        for i, (support_img, support_mask) in enumerate(zip(support_imgs, support_masks)):
            pos = support_positions[i]
            canvas[:, pos[0]:pos[0] + 48, pos[1]:pos[1] + 48] = support_img
            canvas[:, pos[0]:pos[0] + 48, pos[1] + 48:pos[1] + 96] = support_mask

        # Place the query image and mask at the bottom right of the 192x192 area
        canvas[:, 192 - 48:192, 192 - 48:192] = query_mask
        canvas[:, 192 - 48:192, 192 - 96:192 - 48] = query_img

        # Set remaining positions to 0 if any
        for j in range(i + 1, len(support_positions)):
            pos = support_positions[j]
            canvas[:, pos[0]:pos[0] + 48, pos[1]:pos[1] + 48] = 0
            canvas[:, pos[0]:pos[0] + 48, pos[1] + 48:pos[1] + 96] = 0

        return canvas

    def __getitem__(self, idx):
        idx %= len(self.img_metadata_val)  # for testing, as n_images < 1000
        grid_stack = torch.tensor([]).cuda()

        support_imgs = []
        support_masks = []

        for sim_idx in range(self.simidx):
            query_name, support_name, class_sample_query, class_sample_support = self.sample_episode(idx, sim_idx)
            query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name, support_name)

            if self.image_transform:
                query_img = self.image_transform(query_img)
                query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query, purple=self.purple)
            if self.mask_transform:
                query_mask = self.mask_transform(query_mask)

            if self.image_transform:
                support_img = self.image_transform(support_img)
            support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support, purple=self.purple)
            if self.mask_transform:
                support_mask = self.mask_transform(support_mask)

            support_imgs.append(support_img)
            support_masks.append(support_mask)

        grid = self.create_gradiant_cross_grid_images(support_imgs, support_masks, query_img, query_mask, self.arr)

        if len(grid_stack) == 0:
            grid_stack = grid
        else:
            grid_stack = torch.cat((grid_stack, grid))

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'query_name': query_name,
                 'support_name': support_name,
                 'grid_stack': grid_stack
                 }

        return batch

    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary

    def extract_ignore_idx_coco(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id] = 0
            mask[mask == class_id] = 255
            return Image.fromarray(mask), boundary

    def load_frame(self, query_name, support_name):
        # import pdb;pdb.set_trace()
        query_img = self.read_pascal_img(query_name)
        query_mask = self.read_pascal_mask(query_name)
        support_img = self.read_coco_img(support_name)
        support_mask = self.read_coco_mask(support_name)
        org_qry_imsize = query_img.size
        support_img = support_img.convert('RGB')

        return query_img, query_mask, support_img, support_mask, org_qry_imsize

    def read_pascal_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.pascal_ann_path, img_name) + '.png')
        return mask

    def read_pascal_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.pascal_img_path, img_name) + '.jpg')

    def read_coco_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.coco_ann_path, img_name) + '.png')
        return mask

    def read_coco_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.coco_img_path, img_name) + '.jpg')

    def sample_episode(self, idx, sim_idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata_val[idx]

        if self.cls_base:
            support_name = self.images_top50_val[query_name]['top50'][sim_idx]
            support_class = self.images_top50_trn[support_name]['class']
            while support_class != class_sample:
                sim_idx += 1
                if sim_idx >= len(self.images_top50_val[query_name]['top50']):
                    break
                support_name = self.images_top50_val[query_name]['top50'][sim_idx]
                support_class = self.images_top50_trn[support_name]['class']
        else:
            support_name_list = []
            support_class_list = []
            support_name = self.images_top50_val[query_name]['top50'][sim_idx]
            support_name_list.append(support_name)
            # support_classes = self.images_top50_trn[support_name]['class']
            # if class_sample in support_classes:
            #     support_class = class_sample
            # else:
            #     support_class = support_classes[0]
            support_class = self.images_top50_trn[support_name]['class']
            support_class_list.append(support_class)

        if support_name == query_name:
            print('support_name = query_name ' + support_name)
            return self.sample_episode(idx, sim_idx + 1)

        if sim_idx >= len(self.images_top50_val[query_name]['top50']):
            print('query name: ', query_name)
            sim_idx = 0
            return self.sample_episode(idx + 1, sim_idx)

        return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = './evaluate'

            if self.cluster:
                fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold_cluster%d.txt' % (split, fold_id))
            else:
                fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()

            # print("fold_n_metadata: ", fold_n_metadata)
            if self.cluster:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1, int(data.split('__')[2]) - 1] for
                                   data in fold_n_metadata]
            else:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]

            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_coco_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = './tools'

            fold_n_metadata_path = os.path.join(cwd, f'coco/{split}/fold{fold_id}.txt')

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()

            # print("fold_n_metadata: ", fold_n_metadata)
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]

            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        if len(self.img_metadata[0]) != 3:
            for img_name, img_class in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]
        else:
            for img_name, img_class, _ in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]

        return img_metadata_classwise
