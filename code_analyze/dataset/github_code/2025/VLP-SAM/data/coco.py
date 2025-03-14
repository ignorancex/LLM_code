r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle
import random   
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import model.CLIP.clip.clip as clip


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, vlmtransform, model, args):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.vlmtransform = vlmtransform

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

        # encode text (only VLM model)
        if model.module.backbone_type in ['ViT-L/14', 'RN50', 'ViT-B/16', 'CS-ViT-B/16', 'CS-ViT-L/14', 'CS-RN50']:
            text_feat_path = os.path.join(args.weightpath, 'CLIP/')
            if not os.path.exists(text_feat_path): os.makedirs(text_feat_path)
            text_feat_path = os.path.join(text_feat_path, 'text/')
            if not os.path.exists(text_feat_path): os.makedirs(text_feat_path)
            files_name = os.listdir(text_feat_path)
            save_name = f'coco_clip_text_{model.module.backbone_type}.npy'.replace('/', '-')
            if save_name not in files_name:
                label = [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
                    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
                    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
                    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
                    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
                    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
                    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
                    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
                    "toothbrush"]
                input_text = [f"a photo of a {i}" for i in label]
                text = clip.tokenize(input_text).to(model.module.device)
                with torch.no_grad():
                    text_features = model.module.clipmodel.encode_text(text)
                text_features_np = text_features.detach().cpu().numpy()
                np.save(text_feat_path + save_name, text_features_np)
            self.text_features = torch.from_numpy(np.load(text_feat_path + save_name)) # [80, 768] ('ViT-L/14')
        else:
            self.text_features = torch.zeros(self.nclass) # tentative

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        sam_query_img = self.transform(query_img)
        vlm_query_img = self.vlmtransform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), sam_query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.vlmtransform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        text_feature = self.text_features[class_sample]

        batch = {'query_img': sam_query_img,
                 'vlm_query_img': vlm_query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample),
                 'text_feature': text_feature}

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        if self.split == 'trn':
            split = 'train'
        else:
            split = 'val'
        fold_n_subclsdata = os.path.join('data/splits/lists/coco/fss_list/%s/sub_class_file_list_%d.txt' % (split, self.fold))
            
        with open(fold_n_subclsdata, 'r') as f:
            fold_n_subclsdata = f.read()
            
        sub_class_file_list = eval(fold_n_subclsdata)
        img_metadata_classwise = {}
        for sub_cls in sub_class_file_list.keys():
            img_metadata_classwise[sub_cls-1] = [data[0].split('/')[-1].split('.')[0] for data in sub_class_file_list[sub_cls]]
        return img_metadata_classwise

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('data/splits/lists/coco/fss_list/%s/data_list_%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
                
            fold_n_metadata = [data.split(' ')[0].split('/')[-1].split('.')[0] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            img_metadata += read_metadata('train', self.fold)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations')
        if 'val2014' in name:
            mask_path = os.path.join(mask_path, 'val2014')
        else:
            mask_path = os.path.join(mask_path, 'train2014')
        mask_path = os.path.join(mask_path, name)
        mask = torch.tensor(np.array(Image.open(mask_path + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        # import pdb; pdb.set_trace()
        if 'val2014' in query_name:
            base_path = os.path.join(self.base_path, 'val2014')
        else:
            base_path = os.path.join(self.base_path, 'train2014')
        query_img = Image.open(os.path.join(base_path, query_name + '.jpg')).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            if 'val2014' in support_name:
                base_path_s = os.path.join(self.base_path, 'val2014')
            else:
                base_path_s = os.path.join(self.base_path, 'train2014')
            support_imgs.append(Image.open(os.path.join(base_path_s, support_name + '.jpg')).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize