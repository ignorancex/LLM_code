import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse



def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)
# epoch = 260
# root_dir = '/root/XCSGCNN/train/visulization/exp_v3/resnet_multi_concept_mammal/train'
# target_epoch = '260'
root_dir = '/home/wangzhihao/XAI/XCSGCNN/train/visualization/exp_v4/vgg_multi_concept_mammal_gan_ablation/train'
target_epoch = '300'

concept_list = ['head', 'torso', 'leg', 'shape']
concept_mse_list = list()
concept_ssim_list = list()

for concept in concept_list:
    root_path = root_dir + '/' + concept  + '/' + target_epoch
    file_list = list()
    dir_list = list()
    get_file_path(root_path, file_list, dir_list)
    curr_mse_list = list()
    curr_ssim_list = list()
    for file in tqdm(file_list):
        r_concept = Image.open(file)
        r_concept = r_concept.resize((224, 224))
        r_concept.save(file)
        r_concept = cv2.imread(file)
        if concept == 'shape':
            gt_concept_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train/' + 'outline' + '/' + file.split('/')[-2] + '/' + file.split('/')[-1]
        else:
            gt_concept_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train/' + concept + '/' + file.split('/')[-2] + '/' + file.split('/')[-1]
        gt_concept = cv2.imread(gt_concept_path)
        curr_mse_list.append(mse(gt_concept, r_concept))
        curr_ssim_list.append(ssim(gt_concept, r_concept, multichannel=True))
    concept_mse_list.append(np.average(curr_mse_list))
    concept_ssim_list.append(np.average(curr_ssim_list))
print(concept_mse_list)
print(concept_ssim_list)

        

        
        

