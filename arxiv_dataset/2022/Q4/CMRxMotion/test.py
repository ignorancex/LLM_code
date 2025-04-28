import os
import csv
import glob
import numpy as np
import nibabel as nib
from PIL import Image

import torch
import torch.nn.functional as F

from framework_pytorch.utils import utils as U
from framework_pytorch.utils.data_augmentation import RandomForegroundCrop
from framework_pytorch.modelnet.model_vnet_nointer import VNet
from customize import _tools as T
from customize.timm_models import EfficientNet, ResNet, ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_num_samples = 50
slice_ratio_test = [2,3,4,5,6,7,8]
crop_size = [224,224]
foreground_ratio = 0.8
aug = RandomForegroundCrop(tar_size=crop_size, foreground_ratio=foreground_ratio)
slice_votem = T.BVote([0.4, 0.3])
# slice_votem = T.WBVote([1,2,8])
merge_votem = T.MVote()

def seg_process(x):
    x = np.squeeze(x)
    x = U.CenterCrop([256,256,20])(x)
    x = U.min_max(x)
    x = U.ExpandDim(0)(x)
    x = U.ExpandDim(0)(x)
    return x

def cls_process(x, mag):
    if mag:
        x = U.Magnitude()(x)
    else:
        x = U.ExpandDim(-1)(x)
    x = U.Transpose([2, 0, 1])(x)
    return x

class PatchGenerater:
    def __init__(self, filelist):
        data = []
        for f in filelist:
            img = np.array(Image.open(f), np.float32)
            lab = np.array(Image.open(f.replace('_org.tif', '_lab.png')))
            data.append([img, lab])
        self.data = data
    
    def gen(self, mag):
        imgs = []
        for d in self.data:
            img, _ = aug(d[0], d[1])
            img = cls_process(img, mag)
            imgs.append(img)
        imgs = np.array(imgs)
        return imgs

def fill_imgs(datapath, ratio=None, suffix='*org.tif'):
    data_list = glob.glob(datapath + '/' + suffix)
    data_list = sorted(data_list)
    tmp_list = []
    for sub_file in data_list:
        if int(os.path.basename(sub_file).split('_')[0]) in ratio:
            tmp_list.append(sub_file)
    data_list = tmp_list
    return data_list
            

def segto2d(input_path, slice_path):
    if not os.path.exists(slice_path):
        os.mkdir(slice_path)

    # init models:
    model_seg = VNet(input_channel=1, class_number=2)
    model_seg.load_state_dict(torch.load('./model/seg.pt'))
    model_seg.to(device=device)
    model_seg.eval()

    org_list = glob.glob(input_path + '/*.nii.gz')
    for org_path in org_list:
        org = np.array(nib.load(org_path).dataobj).astype(np.float32)
        org_forseg = seg_process(org)
        org_forseg = torch.from_numpy(org_forseg)

        org = np.squeeze(org)
        org = U.min_max(org)

        with torch.no_grad():
            seg_logits = model_seg(org_forseg.to(device=device))

        seg_prob = F.softmax(seg_logits, dim=1)
        seg_pred = np.argmax(seg_prob.cpu().numpy(), 1)[0]
        seg_pred = U.getLargestCC(seg_pred)
        seg_pred = U.CenterPadding(org.shape)(seg_pred)

        fname = os.path.basename(org_path).split('.nii.gz')[0]
        tpath = f'{slice_path}/{fname}'
        if not os.path.exists(tpath):
            os.makedirs(tpath)
        for slice_idx in range(org.shape[-1]):
            lab2d = np.uint8(seg_pred[..., slice_idx])
            if np.sum(lab2d) == 0:
                continue
            org2d = org[..., slice_idx]
            Image.fromarray(lab2d*255).save(f'{tpath}/{slice_idx}_lab.png')
            Image.fromarray(org2d).save(f'{tpath}/{slice_idx}_org.tif')


def test(input_path, output_path):
    # segmentation
    slice_path = './slices'
    segto2d(input_path, slice_path)
    test_set = glob.glob(f'{slice_path}/**')
    test_set = sorted(test_set)

    # inference
    # model = ['effnet_mag', 'effnet_int', 'resnet_mag', 'resnet_int', 'vit_mag', 'vit_int']
    models = ['effnet_mag']

    case_id_pred = {}
    for mpath in models:
        pred_cross_fold = {}
        for ifold in range(1, 6):
            if 'effnet' in mpath:
                tmp_model = EfficientNet(num_classes=3, input_channels=1)
            elif 'resnet' in mpath:
                tmp_model = ResNet(num_classes=3, input_channels=1)
            elif 'vit' in mpath:
                tmp_model = ViT(num_classes=3, input_channels=1)
            else:
                pass
            tmp_model.load_state_dict(torch.load(f'./model/{mpath}/{ifold}.pt'))
            tmp_model.to(device=device)
            tmp_model.eval()

            for t in test_set:
                sub_test_list = fill_imgs(t, slice_ratio_test, '*_org.tif')
                sub_test_list = sorted(sub_test_list)
                pg = PatchGenerater(sub_test_list)           

                slice_probs = []
                for _ in range(test_num_samples):
                    tmp_batch = pg.gen(mag='mag' in mpath)
                    tmp_batch = torch.from_numpy(tmp_batch)
                    tmp_batch = tmp_batch.to(device=device)
                    
                    with torch.no_grad():
                        logits = tmp_model(tmp_batch)
                        prob = F.softmax(logits, dim=1)
                        slice_probs.append(prob.cpu().numpy())
                
                slice_probs = np.array(slice_probs)
                slice_probs = np.sum(slice_probs, 0)
                slice_preds = np.argmax(slice_probs, 1)
                if os.path.basename(t) not in pred_cross_fold:
                    pred_cross_fold[os.path.basename(t)] = []
                pred_cross_fold[os.path.basename(t)].append(slice_preds)
                print(f'{mpath} - {ifold} - {t}')
        for key in pred_cross_fold:
            tmp_pred = np.array(pred_cross_fold[key])
            pred_vote_slice = merge_votem([slice_votem(tmp_pred[i]) for i in range(tmp_pred.shape[0])])
            tmp_pred = np.reshape(tmp_pred, -1)
            # pred_vote_case = slice_votem(tmp_pred)
            if key not in case_id_pred:
                case_id_pred[key] = []
            case_id_pred[key].append(pred_vote_slice)
    
    for key in case_id_pred:
        case_id_pred[key] = merge_votem(case_id_pred[key])

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(output_path + f'/output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Label'])
        for key in case_id_pred:
            writer.writerow([key, str(case_id_pred[key]+1)])
    pass

