import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
# from skimage import io
# from skimage import color
# import cv2
import config


# 最新版本，使用边缘图像maxoper
class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, origin_img_dir, pathlistfile, edited_img_dir='', task=''):
        self.origin_img_dir = origin_img_dir  # 相对应的gtc图片的位置
        self.edited_img_dir = edited_img_dir  # 经过预处理的需要来训练的图片的位置
        self.task = task
        self.pathlist = self.loadpath(pathlistfile)
        self.count = len(self.pathlist)

    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist  # 返回 pathlistfile 文件如 train_mine.txt 中的字符串路径构成的列表list

    def __getitem__(self, index):
        frames = []
        path_code = self.pathlist[index]  # index指示的文件夹 如 00006/10
        print('pathcode: ', path_code)
        N = config.N    # 3张雨图
        nums = [168, 116, 125, 298, 256, 250, 219, 250]
        # for i in range(1, nums[int(path_code) - 1] + 1):    # edited_img_dir/path_code 某个雨图文件夹 | 生成每一帧的单帧去雨的结果 (1, 10), (1, 32)
        #     frames.append(plt.imread(os.path.join(self.edited_img_dir, path_code, '%d.jpg' % (i) ) )[0:config.h, 0:config.w, 0:3] / 255.0)
        for i in range(1, 32):  # edited_img_dir/path_code 某个雨图文件夹 | 生成每一帧的单帧去雨的结果 (1, 10), (1, 32)
            frames.append(plt.imread(os.path.join(self.edited_img_dir, path_code, 'rfc-%d.jpg' % (i) ) )[0:config.h, 0:config.w, 0:3] / 255.0 )   #
        # # frames.append(plt.imread(os.path.join(self.origin_img_dir, path_code, 'gtc-%d.jpg' % (N // 2 + 1) ) )[0:config.h, 0:config.w] / 255.0)  #

        frames = np.asarray(frames, dtype=np.float32)
        framex = np.transpose(frames[:, :, :, :], (0, 3, 1, 2))
        # framey = np.transpose(frames[-1, :, :, :], (2, 0, 1))
        # frames = np.asarray(frames, dtype=np.float32)
        # framex = np.transpose(frames[0:N, :, :, :], (0, 3, 1, 2))  # 转成 C, H, W
        # framey = np.transpose(frames[N:2*N, :, :, :], (0, 3, 1, 2))  # ground truth 对照 转成 C H W
        print('framex: ', framex.shape, framex.dtype)
        print(torch.from_numpy(framex).shape)
        # print('framey: ', framey.shape, framey.dtype)
        # tmp = torch.from_numpy(framex)
        # print('tmp_x: ', tmp.shape, tmp.dtype)
        # tmp_f = torch.from_numpy(framey)
        # print('tmp_y: ', tmp_f.shape, tmp_f.dtype)
        return torch.from_numpy(framex), path_code

    def __len__(self):
        return self.count
