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
    path = './dataset.json'
    data = open(path, "r", encoding="utf-8")
    x = json.load(data)    # 字典
    data.close()

    print(x['training'])
    y = x.copy()
    y['training'] = []
    y['numTraining'] = 0

    gt_path = './nnUNet_preprocessed/Task022_FLARE22/gt_segmentations'
    for i in tqdm(x['training']):
        # print(i['label'].split('/')[-1])
        name = i['label'].split('/')[-1]

        gt = read(os.path.join(gt_path, name))
        gt = gt.reshape(-1)
        gt = gt.astype(np.uint8)
        set = list(np.unique(gt))
        print(set)

        # if 14 not in set:
        #     y['training'].remove(i)
        #     y['numTraining'] -= 1
        #     print('delete:', i)

        if len(set) != 1:
            y['training'].append(i)
            y['numTraining'] += 1
            print('add:', y['numTraining'], i)


save_json(y, './ch_dataset.json', sort_keys=True)
