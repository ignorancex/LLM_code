import json
import os.path
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np


def read(img):
    img = sitk.ReadImage(img)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose(1, 2, 0)
    return img

if __name__ == '__main__':
    path = './dataset_old.json'
    new_path = './move'
    data = open(path, "r", encoding="utf-8")
    x = json.load(data)    # 字典
    data.close()

    print(x['training'])
    y = x.copy()
    y['training'] = []
    y['numTraining'] = 0

    gt_path = './nnUNetPlans_3d_fullres'
    ls = os.listdir(gt_path)
    for i in tqdm(x['training']):
        # print(i['label'].split('/')[-1])
        org = i['label'].split('/')[-1][:-12]
        name = i['label'].split('/')[-1][:-12]+'_seg.npy'

        if name in ls:
            y['training'].append(i)
            y['numTraining'] += 1
            print('add:', y['numTraining'], i)
        else:
            import shutil
            shutil.move(gt_path+'/'+org+'.pkl', new_path+'/'+org+'.pkl')
            shutil.move(gt_path+'/'+org+'.npz', new_path+'/'+org+'.npz')
            print(name)


save_json(y, './ch_dataset.json', sort_keys=True)
