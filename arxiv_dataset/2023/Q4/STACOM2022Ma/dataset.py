import os
import csv
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from PIL import Image
from utils.data_utils import fourier_transform, colorful_spectrum_mix

Validation_dir = '/home/wmaag/UROP3100/Dataset/CMR_Validation/'
Validation_names = sorted(os.listdir(Validation_dir))

def check_label(img):
    label_exit = True
    ave_lv = img[:,0,:].mean()
    ave_myo = img[:,1,:].mean()
    ave_rv = img[:,2,:].mean()
    ave_lv = int(ave_lv)
    ave_myo = int(ave_myo)
    ave_rv = int(ave_rv)
    if ave_lv == 0 & ave_myo == 0 & ave_rv == 0:
        label_exit = False
    return label_exit

def get_meta_split_data_loaders():
    domain_1_labeled_dataset = CMRxMotion_seg(domain_list=[0])
    domain_2_labeled_dataset = CMRxMotion_seg(domain_list=[1])
    domain_3_unlabeled_dataset = CMRxMotion_seg(domain_list=[2])
    test_dataset = CMRxMotion_Test()
    return domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_unlabeled_dataset,test_dataset
    

def fourier_augmentation(img, tar_img, mode='AM', alpha=0.3):
    # transfer image from PIL to numpy
    img = np.array(img)
    tar_img = np.array(tar_img)
    img = img[..., np.newaxis]
    tar_img = tar_img[..., np.newaxis]

    # the mode comes from the paper "A Fourier-based Framework for Domain Generalization"
    if mode == 'AS':
        aug_img, aug_tar_img = fourier_transform(img, tar_img, L=0.01, i=1)
    elif mode == 'AM':
        aug_img, aug_tar_img = colorful_spectrum_mix(img, tar_img, alpha=alpha)
    else:
        print("mode name error")

    aug_img = np.squeeze(aug_img)
    aug_tar_img = np.squeeze(aug_tar_img)
    return aug_img


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr

class CMRxMotion_cls(Dataset):
    def __init__(self, split='CMR_Train', test=False):
        root_dir = '/home/wmaag/UROP3100/Dataset'
        if split != 'CMR_Train':
            raise NotImplementedError
        
        self.data_list = []
        label_cnt = [0, 0, 0]
        with open(f'{root_dir}/{split}/IQA.csv') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                oid, cnt, tag = row[0].split('-')
                label = int(row[1]) - 1
                label_cnt[label] += 1
                path = f'{root_dir}/{split}/data/{oid}-{cnt}/{oid}-{cnt}-{tag}.nii.gz'
                image = read_nifti(path)
                for sub_image in image:
                    sub_image = cv2.resize(sub_image, (512, 512))
                    self.data_list.append({
                        'image': sub_image,
                        'label': label,
                    })
        
        label_cnt = np.array(label_cnt).astype(np.float32)
        weight = label_cnt / label_cnt.sum()
        self.weight = np.power(np.amax(weight) / weight, 1/3)
        
        self.transform = None
        if not test:
            self.transform = A.Compose([
                A.RandomCrop(width=384, height=384),
                A.HorizontalFlip(p=0.5)
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image, label = data['image'], data['label']

        if self.transform:
            image = self.transform(image=image)['image']
        
        image = image[np.newaxis, ...]
        return {
            'image': image,
            'label': label
        }


class CMRxMotion_seg(Dataset):
    def __init__(self, split='CMR_Train', domain_list=[0, 1, 2], test=False):
        root_dir = '/home/wmaag/UROP3100/Dataset'
        if split != 'CMR_Train':
            raise NotImplementedError
        
        self.data_list = []
        self.fourier_list = []

        label_cnt = [0, 0, 0, 0]
        with open(f'{root_dir}/{split}/IQA.csv') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                oid, cnt, tag = row[0].split('-')
                domain = int(row[1]) - 1
                image_path = f'{root_dir}/{split}/data/{oid}-{cnt}/{oid}-{cnt}-{tag}.nii.gz'
                label_path = f'{root_dir}/{split}/data/{oid}-{cnt}/{oid}-{cnt}-{tag}-label.nii.gz'
                image = read_nifti(image_path)
                # import pdb;pdb.set_trace()
                if os.path.exists(label_path):
                    label = read_nifti(label_path)
                    for i in range(4):
                        label_cnt[i] += np.sum(label == i)
                else:
                    label = np.zeros_like(image)
                
                for i in range(image.shape[0]):
                    sub_image = cv2.resize(image[i, ...], (512, 512))
                    self.fourier_list.append(sub_image)
                    if domain in domain_list:
                        sub_label = cv2.resize(label[i, ...], (512, 512))
                        self.data_list.append({
                            'image': sub_image,
                            'label': sub_label,
                            'domain': domain,
                        })
                    
        label_cnt = np.array(label_cnt).astype(np.float32)
        weight = label_cnt / label_cnt.sum()
        self.weight = np.power(np.amax(weight) / weight, 1/3)
        
        self.transform = None
        if not test:
            self.transform = A.Compose([
                A.RandomRotate90(),
                A.RandomScale(p=0.8),
                A.RandomCrop(width=384, height=384),
                A.HorizontalFlip(p=0.5),
                # A.ColorJitter(),
                # A.GaussianBlur(blur_limit = 5, sigma_limit = 0)
                #add here 
            ])
            self.fourier_transform = A.Compose([
                A.RandomCrop(width=384, height=384),
                A.HorizontalFlip(p=0.5)
            ])
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image, label = data['image'], data['label']

        if self.transform:
            f_idx = np.random.randint(len(self.fourier_list))
            fourier = self.fourier_list[f_idx]
            fourier = self.fourier_transform(image=fourier)['image']

            trans = self.transform(image=image, mask=label)
            image, label = trans['image'], trans['mask']

            fourier = fourier_augmentation(image, fourier)
        
        fourier -= fourier.min()
        fourier /= fourier.std()
        image = image[np.newaxis, ...].astype(np.float32)
        fourier = fourier[np.newaxis, ...].astype(np.float32)
        label = label.astype(np.int32)
        w, h = label.shape
        label = label.reshape(-1)
        label = (label - 1) % 4
        oh_label = np.zeros((label.size, 4))
        oh_label[np.arange(label.size), label] = 1
        oh_label = oh_label.reshape([w, h, -1])
        oh_label = oh_label.transpose(2, 0, 1)
        label_exist = check_label(oh_label)
        if label_exist == True:
            label_exist_tar = 1
        else:
            label_exist_tar = 0
        return {
            'img': image,
            'aug_img': fourier,
            'mask': oh_label,
            'label_exist_tar': label_exist_tar
        }

class CMRxMotion_Test(Dataset):
    def __init__(self):
        self.data_list = []
        for i in range(0,len(Validation_names)):
            son_path = Validation_dir + Validation_names[i] + '/'
            son_names = sorted(os.listdir(son_path))
            for j in range(0,len(son_names)):
                path = son_path + son_names[j]
                image = read_nifti(path)
                # import pdb;pdb.set_trace()
                k = 0
                for sub_image in image:
                    self.data_list.append({
                        'img': sub_image,
                        'path': path,
                        'deep_inn': k,
                        'image':image
                    })
                    k = k+1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        print('enter Test')
        data = self.data_list[index]
        # print(data)
        image = data['img']
        o_img = data['image']
        inn = index
        path = data['path']
        k = data['deep_inn']
        image = image[np.newaxis, ...]
        return {
            'image': image,
            'inn': inn,
            'path':path,
            'deep_inn':k,
            'o_img':o_img
        }

if __name__ == '__main__':
    dst = CMRxMotion_Test()
    # dst = CMRxMotion_seg(domain_list=[0, 1])
    print(1)
    import pdb; pdb.set_trace()
    print(2)
    data_loader = DataLoader(dataset=dst, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)
    dataiter = iter(data_loader)
    output = dataiter.next()
    # mask = output['mask']
    img = output['o_img']
    print('o_img',img)
    


