import json
import os.path
import shutil

from batchgenerators.utilities.file_and_folder_operations import *
# from check import *
from tqdm import tqdm

# path = './dataset.json'
# data = open(path, "r", encoding="utf-8")
# x = json.load(data)    # 字典
# data.close()
#
# print(x['training'])
# y = x.copy()
#
#
# gt_path = './nnUNet_preprocessed/Task022_FLARE22/gt_segmentations'


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def move():
    import shutil
    source_path = './nnUNet_raw_data_base/nnUNet_cropped_data/Task022_FLARE22'
    # source_path = './nnUNet_preprocessed/Task022_FLARE22/nnUNetData_plans_v2.1_stage0'

    target_path = './move/Task022_FLARE22'
    # target_path = './move/nnUNetData_plans_v2.1_stage0'

    ls = os.listdir(source_path)
    print(ls)
    stay = []
    for i in tqdm(x['training']):
        # print(i['label'].split('/')[-1])
        name = i['label'].split('/')[-1]
        print(name)
        stay.append(name.split('.')[0])

    for i in ls:
        if i.split('.')[0] not in stay:
            print('move:', source_path+'/'+i)
            shutil.move(source_path+'/'+i, target_path)


def move_npy():
    import shutil
    source_path = './nnUNet_preprocessed/Task022_FLARE22/nnUNetData_plans_v2.1_stage1'

    target_path = './nnUNet_preprocessed/Task022_FLARE22/nnUNetData_plans_v2.1_stage1_npy'
    # target_path = './move/nnUNetData_plans_v2.1_stage0'

    ls = os.listdir(source_path)
    print(ls)

    for i in ls:
        if i.endswith('.npy'):
            print('move:', source_path+'/'+i)
            shutil.move(source_path+'/'+i, target_path)


def move_train():
    pre_tr = './pred_tr'
    train_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/imagesTr'
    new_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/imagesTr_'
    check_dir(new_path)

    ls = os.listdir(train_path)
    exist_ls = os.listdir(pre_tr)
    # print(exist_ls)

    for i in ls:
        name = i[:-12]+'.nii.gz'
        if name not in exist_ls:
            print(name)
            shutil.copy(train_path+'/'+i, new_path+'/'+i)


def move_jia():
    source_path = '/jhcnas1/jiaxin/data/Flare23/nnUNet_preprocessed/Dataset101_Flare23/nnUNetPlans_3d_fullres/'
    target_path = '/jhcnas1/wulinshan/nnUNet/nnUNet_preprocessed/Dataset101_Flare23/nnUNetPlans_3d_fullres/'
    s_ls = os.listdir(source_path)
    t_ls = os.listdir(target_path)
    for i in s_ls:
        if i not in t_ls:
            print(i)
            shutil.copy(source_path+'/'+i, target_path+'/'+i)


if __name__ == '__main__':
    # move()
    # move_npy()
    # move_train()
    move_jia()


