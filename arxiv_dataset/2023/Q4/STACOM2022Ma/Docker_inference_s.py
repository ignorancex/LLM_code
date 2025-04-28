import imp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import shutil
from PIL import Image
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from network.network import my_net
from utils.utils import get_device, check_accuracy, dice_loss, im_convert, label_to_onehot
from dataset import get_meta_split_data_loaders
from config import default_config
from utils.data_utils import save_image
from utils.dice_loss import dice_coeff
import imageio
import nibabel as nib

os.environ['WANDB_MODE'] = 'dryrun'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
gpus = default_config['gpus']
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

save_dir = '/home/wmaag/UROP3100/Dataset/CMR_pred/'
Val_data_dir = '/home/wmaag/UROP3100/Dataset/CMR_Validation_2D/data/'
save_dir2 = '/home/wmaag/UROP3100/Dataset/CMR_Validation2/'
submission_dir = '/home/wmaag/UROP3100/Dataset/CMR_Submission/'
model_CLS_path = '/home/wmaag/UROP3100/EPL_SemiDG-master/CLS.pt'

save_names2 = sorted(os.listdir(save_dir2))
save_names = sorted(os.listdir(save_dir)) 
wandb.init(project='CMR', entity='wmaag',
           config=default_config, name=default_config['train_name'])
config = wandb.config

def pre_data(batch_size, num_workers):

    domain_1_labeled_dataset, domain_2_labeled_dataset,domain_3_unlabeled_dataset,test_dataset = get_meta_split_data_loaders()

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset])

    label_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset])

    print("before length of label_dataset", len(label_dataset))

    label_loader = DataLoader(dataset=label_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=False)


    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=True, drop_last=True, pin_memory=False)

    print("after length of label_dataset", len(label_dataset))
    print("length of val_dataset", len(val_dataset))
    print("length of test_dataset", len(test_dataset))

    return label_loader, val_loader, test_loader, len(label_dataset), len(test_dataset)

def save_np(img, path, name):
    np.savez_compressed(os.path.join(path, name), img)
        

def inference(model_path_l, model_path_r,model_path_l2, model_path_r2, test_loader):
    Validation_names = sorted(os.listdir(Val_data_dir))
    model_l = torch.load(model_path_l, map_location=device)
    model_l = model_l.to(device)
    model_l.eval()

    model_r = torch.load(model_path_r, map_location=device)
    model_r = model_r.to(device)
    model_r.eval()
    
    # model_CLS = torch.load(model_CLS_path,map_location=device)
    # model_CLS = model_CLS.to(device)
    # model_CLS.eval()
    
    model_l2 = torch.load(model_path_l2, map_location=device)
    model_l2 = model_l2.to(device)
    model_l2.eval()
    
    model_r2 = torch.load(model_path_r2, map_location=device)
    model_r2 = model_r2.to(device)
    model_r2.eval()
    
    

    for minibatch in tqdm(test_loader):
        imgs = minibatch['image']
        path = minibatch['path']
        inn = minibatch['inn']
        k = minibatch['deep_inn']
        
        h = imgs.shape[2]
        w = imgs.shape[3]
        imgs = imgs.to(device)
        # import pdb; pdb.set_trace()
        if h%32 == 0 and w%32 ==0:
            with torch.no_grad():
                 logits_l, _ = model_l(imgs)
                 logits_r, _ = model_r(imgs)
                 logits_l2 = model_l2(imgs)
                 logits_r2 = model_r2(imgs)
        
        #get prediction
            sof_l = F.softmax(logits_l, dim=1)
            sof_r = F.softmax(logits_r, dim=1)
            sof_l2 = F.softmax(logits_l2, dim=1)
            sof_r2 = F.softmax(logits_r2, dim=1)
            pred1 = (sof_l + sof_r) / 2
            pred2 = (sof_l2 + sof_r2) / 2
            pred = (pred1 + pred2)/2
        else:
            with torch.no_grad():
                 logits_l, _ = model_l(imgs)
                 logits_r, _ = model_r(imgs)
            sof_l = F.softmax(logits_l, dim=1)
            sof_r = F.softmax(logits_r, dim=1)
            pred = (sof_l + sof_r) / 2
            print('from origin')    
        
        pred = pred.cpu().numpy()    
        pred = np.transpose(pred,(2,3,0,1))
        # import pdb; pdb.set_trace()
        # print('pred shape after: ',pred.shape)
        # assert (1==0) #debug only
        name_part = path[0].split('/')[-1]
        k_2 = int(k[0])
        name = name_part.split('.')[0] +'-%03d.npz'%k_2
        # print(name)
        save_np(pred,save_dir,name)
        print(name,' : ',pred.shape)

def move():
    for j in range (0,22):
       save_new_path = save_dir2+save_names2[0]+'/'
       origin_path = save_dir+save_names[j]
       # print(origin_path)
       # assert(1==0)
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (22,44):
        save_new_path = save_dir2+save_names2[1]+'/'
        origin_path = save_dir+save_names[j]
        # print(origin_path)
        # assert(1==0)
        shutil.copy(origin_path,save_new_path)
        print('finish: ',save_names[j])

    for j in range (44,66):
        save_new_path = save_dir2+save_names2[2]+'/'
        origin_path = save_dir+save_names[j]
        # print(origin_path)
        # assert(1==0)
        shutil.copy(origin_path,save_new_path)
        print('finish: ',save_names[j])

    for j in range (66,88):
        save_new_path = save_dir2+save_names2[3]+'/'
        origin_path = save_dir+save_names[j]
        # print(origin_path)
        # assert(1==0)
        shutil.copy(origin_path,save_new_path)
        print('finish: ',save_names[j])

    for j in range (88,110):
        save_new_path = save_dir2+save_names2[4]+'/'
        origin_path = save_dir+save_names[j]
        # print(origin_path)
        # assert(1==0)
        shutil.copy(origin_path,save_new_path)
        print('finish: ',save_names[j])

    for j in range (110,132):
        save_new_path = save_dir2+save_names2[5]+'/'
        origin_path = save_dir+save_names[j]
        # print(origin_path)
        # assert(1==0)
        shutil.copy(origin_path,save_new_path)
        print('finish: ',save_names[j])

    for j in range (132,154):
        save_new_path = save_dir2+save_names2[6]+'/'
        origin_path = save_dir+save_names[j]
        # print(origin_path)
        # assert(1==0)
        shutil.copy(origin_path,save_new_path)
        print('finish: ',save_names[j])

    for j in range (154,176):
       save_new_path = save_dir2+save_names2[7]+'/'
       origin_path = save_dir+save_names[j]
       # print(origin_path)
       # assert(1==0)
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (176,200):
       save_new_path = save_dir2+save_names2[8]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (200,224):
       save_new_path = save_dir2+save_names2[9]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (224,248):
       save_new_path = save_dir2+save_names2[10]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (248,272):
       save_new_path = save_dir2+save_names2[11]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (272,296):
       save_new_path = save_dir2+save_names2[12]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (296,320):
       save_new_path = save_dir2+save_names2[13]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (320,344):
       save_new_path = save_dir2+save_names2[14]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (344,368):
       save_new_path = save_dir2+save_names2[15]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (368,392):
       save_new_path = save_dir2+save_names2[16]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (392,416):
       save_new_path = save_dir2+save_names2[17]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (416,440):
       save_new_path = save_dir2+save_names2[18]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

    for j in range (440,464):
       save_new_path = save_dir2+save_names2[19]+'/'
       origin_path = save_dir+save_names[j]
       shutil.copy(origin_path,save_new_path)
       print('finish: ',save_names[j])

def save_array_as_nii_volume(data, filename):
    if 'ED' in filename:
        data[data == 3] = -1
        data += 1
        data = data.transpose(2, 1, 0)
        data = data[..., np.newaxis]
    else:
        # data[data == 3] = -1
        # data += 1
        data = data.transpose(2, 1, 0)
        data = data[..., np.newaxis]
    # print(img.shape)
    # assert(1==0)
    img = sitk.GetImageFromArray(data)
    # import pdb; pdb.set_trace()
    sitk.WriteImage(img, filename)
    print('shape: ',filename,' :',data.shape)

def TO3D():
    for i in range(0,8):
        son_path = save_dir2+save_names2[i]+'/'
        son_name = sorted(os.listdir(son_path))
    #get shape of ED
        img_ex_name = son_path+son_name[0]
    # print(img_ex_name)
        img_ex_img = np.load(img_ex_name)
        img_ex_np = img_ex_img['arr_0']
    # print(img_ex_np.shape)
    # assert(1==0)
        h = img_ex_np.shape[0]
        w = img_ex_np.shape[1]
        EDallImg = []
    # EDallImg = np.zeros([h,w,11,1], dtype='uint8')
        ESallImg = []
    # ESallImg = np.zeros([h,w,11,1], dtype='uint8')
        for j in range(11):
            single_image_name = son_path+son_name[j]
            img_ex_img = np.load(single_image_name)
            img_ex_np = img_ex_img['arr_0']
        # img = img_ex_np.convert('L')
            img_ex_np = np.argmax(img_ex_np[:, :, 0, :], axis=-1)
            EDallImg.append(img_ex_np)
    EDallImg = np.stack(EDallImg, axis=-1)
    EDallImg = EDallImg.astype(np.int32)
    # EDallImg = EDallImg[...,  np.newaxis].shape
    # import pdb; pdb.set_trace()
    filenameED = submission_dir+save_names2[i]+'-ED'+'.nii.gz'
    save_array_as_nii_volume(EDallImg, filenameED)
    print('finish: ',filenameED)
    for k in range(11,len(son_name)):
        single_image_name = son_path+son_name[k]
        img_ex_img = np.load(single_image_name)
        img_ex_np = img_ex_img['arr_0']
        # img = img_ex_np.convert('L')
        img_ex_np = np.argmax(img_ex_np[:, :, 0, :], axis=-1)
        ESallImg.append(img_ex_np)
    filenameES = submission_dir+save_names2[i]+'-ES'+'.nii.gz'
    ESallImg = np.stack(ESallImg, axis=-1)
    ESallImg = EDallImg.astype(np.int32)
    # ESallImg = EDallImg[...,  np.newaxis].shape
    save_array_as_nii_volume(ESallImg, filenameES)
    print('finish: ',filenameES)

#re-save nii.gz as required shape

    for i in range(8,len(save_names2)):
        son_path = save_dir2+save_names2[i]+'/'
        son_name = sorted(os.listdir(son_path))
    #get shape of ED
        img_ex_name = son_path+son_name[0]
    # print(img_ex_name)
        img_ex_img = np.load(img_ex_name)
        img_ex_np = img_ex_img['arr_0']
        h = img_ex_np.shape[0]
        w = img_ex_np.shape[1]
        EDallImg = []
    # EDallImg = np.zeros([h,w,11,1], dtype='uint8')
        ESallImg = []
    # ESallImg = np.zeros([h,w,11,1], dtype='uint8')
        for j in range(12):
            single_image_name = son_path+son_name[j]
            img_ex_img = np.load(single_image_name)
            img_ex_np = img_ex_img['arr_0']
        # img = img_ex_np.convert('L')
            img_ex_np = np.argmax(img_ex_np[:, :, 0, :], axis=-1)
            EDallImg.append(img_ex_np)
        EDallImg = np.stack(EDallImg, axis=-1)
        EDallImg = EDallImg.astype(np.int32)
    # EDallImg = EDallImg[...,  np.newaxis].shape
    # import pdb; pdb.set_trace()
        filenameED = submission_dir+save_names2[i]+'-ED'+'.nii.gz'
        save_array_as_nii_volume(EDallImg, filenameED)
        print('finish: ',filenameED)
        for k in range(12,len(son_name)):
           single_image_name = son_path+son_name[k]
           img_ex_img = np.load(single_image_name)
           img_ex_np = img_ex_img['arr_0']
        # img = img_ex_np.convert('L')
           img_ex_np = np.argmax(img_ex_np[:, :, 0, :], axis=-1)
           ESallImg.append(img_ex_np)
        filenameES = submission_dir+save_names2[i]+'-ES'+'.nii.gz'
        ESallImg = np.stack(ESallImg, axis=-1)
        ESallImg = EDallImg.astype(np.int32)
    # ESallImg = EDallImg[...,  np.newaxis].shape
        save_array_as_nii_volume(ESallImg, filenameES)
        print('finish: ',filenameES)
    

        
def main():
    batch_size = 1
    num_workers = 8
    # model_path_l = '/home/wmaag/UROP3100/EPL_SemiDG-master/l_deeplab_5%_C.pt'
    # model_path_r = '/home/wmaag/UROP3100/EPL_SemiDG-master/r_deeplab_5%_C.pt'
    model_path_l = '/home/wmaag/UROP3100/EPL_SemiDG-master/l_CMR.pt'
    model_path_r = '/home/wmaag/UROP3100/EPL_SemiDG-master/r_CMR.pt'
    model_path_l2 = '/home/wmaag/UROP3100/EPL_SemiDG-master/l_sCMR.pt'
    model_path_r2 = '/home/wmaag/UROP3100/EPL_SemiDG-master/r_sCMR.pt'
    #prepare data as train
    label_loader, val_loader, test_loader, num_label_imgs, num_test_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers)
    
    inference(model_path_l, model_path_r, model_path_l2, model_path_r2, test_loader)
    move()
    # TO3D()

if __name__ == '__main__':
    main()
