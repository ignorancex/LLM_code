import torch
import torch.serialization
# from torchvision import transforms
import numpy as np
# import sys
# import getopt
import os
# import shutil
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg  # 读取图片
# from skimage import io
# from skimage import color
# from skimage.measure import compare_ssim  # 输入ndarray
# from skimage.measure import compare_psnr  # 输入ndarray
import argparse

# import datetime
# from PIL import Image
from Network import TOFlow
# import warnings
import real_config as config
# import toolbox.utils as utils
from real_multi_read_data_eval import MemoryFriendlyLoader


# warnings.filterwarnings("ignore", module="matplotlib.pyplot")
# # ------------------------------
# # I don't know whether you have a GPU.
# plt.switch_backend('agg')


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path, exist_ok=True)


def vimeo_evaluate(modelspath='', outimgs_dir='', mutil_eval=False):
    task = config.task  # 测试任务的类别
    dataset_dir = config.dataset_dir  # 测试数据集
    dataset_gtc_dir = config.dataset_gtc_dir  # 测试用到的gtc文件的路径
    # out_img_dir = config.out_img_dir  # 测试结果存放位置
    pathlistfile = config.pathlistfile  # 具体测试用例名称表
    # model_path = config.model_path  # 要测试的模型位置
    if mutil_eval:
        model_path = modelspath
        out_img_dir = outimgs_dir
    else:
        model_path = config.model_path
        out_img_dir = config.out_img_dir

    gpuID = config.gpuID
    BATCHSIZE = config.BATCH_SIZE
    h = config.h
    w = config.w
    N = config.N
    map_location = config.map_location

    # if gpuID is None:
    #     cuda_flag = False
    # else:
    #     cuda_flag = True
    #     torch.cuda.set_device(gpuID)

    mkdir_if_not_exist(out_img_dir)
    # prepare DataLoader
    Dataset = MemoryFriendlyLoader(origin_img_dir=dataset_gtc_dir, edited_img_dir=dataset_dir,
                                   pathlistfile=pathlistfile,
                                   task=task)
    train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=0)
    sample_size = Dataset.count

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="./raft-small_pytorch_old.pth",
                        help="restore checkpoint, 使用旧的，不使用pytorch1.6")
    parser.add_argument('--path', default="./demo-frames", help="dataset for evaluation")
    parser.add_argument('--small', default=True, action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # print('raft is ', args.model)

    # net = TOFlow(h, w, args, cuda_flag=cuda_flag, task=task)
    # net.load_state_dict(torch.load(model_path, map_location=map_location))
    # # net.load_state_dict(torch.load(model_path))   # 报错, 主要原因是训练和测试使用的不是一个GPU
    # # RuntimeError: Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1
    # # Please use torch.load with map_location to map your storages to an existing device.
    #
    # if cuda_flag:
    #     net.cuda().eval()
    # else:
    #     net.eval()
    net = TOFlow()
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    psnr_my = 0.0
    ssm_my = 0.0
    step = 0
    pre = 0
    processing_time_for_all = 0
    nums = [168, 116, 125, 298, 256, 250, 219, 250]
    with torch.no_grad():
        for step, (x, path_code) in enumerate(train_loader):
            # x = x.cuda()    # X: 1x31xCxHxW [0:30]]共31帧
            print('x.shape: ', x.shape, x.size(1))
            for center in range(1, 8):  # 使用连续3帧生成中间帧去雨的结果 [1, 2, 3, 4, 5, ..., 27, 28, 29]
                # for center in range(1, nums[int(path_code[0]) - 1] - 1):
                tmp = torch.zeros(size=(1, 3, 3, x.size(3), x.size(4)))
                tmp[:, :, :, :, :] = x[:, center - 1:center + 2, :, :, :]
                # tmp[:, 0, :, :, :] = x[:, center, :, :, :]   # inputs is the same!
                # tmp[:, 1, :, :, :] = x[:, center, :, :, :]
                # tmp[:, 2, :, :, :] = x[:, center, :, :, :]

                tmp = tmp.cuda()  # 直接将x送进网络, 来进行迭代是有问题的

                # if center == 1:
                #     init = True
                #     x_h = None
                #     x_o = None
                # else:
                #     init = False

                predicted_img = net(tmp)  #
                img_ndarray = predicted_img.cpu().detach().numpy()
                img_ndarray = np.transpose(img_ndarray, (0, 2, 3, 1))
                img_ndarray = img_ndarray[0]

                img_tobesaved = np.asarray(img_ndarray)
                mkdir_if_not_exist(os.path.join(out_img_dir, path_code[0]))
                # video = path_code[0].split('/')[0]  # print(path_code)    # ('00001/6',)
                # sep = path_code[0].split('/')[1]
                # plt.imsave(os.path.join(out_img_dir, path_code[0], '%d.jpg' % (center+1)), np.clip(img_ndarray, 0.0, 1.0))
                plt.imsave(os.path.join(out_img_dir, path_code[0], '%05d.jpg' % (center + 1)),
                           np.clip(img_tobesaved, 0.0, 1.0))  # 路径有问题
                # plt.imsave(os.path.join(out_img_dir, video, sep, 'middle-%d.jpg' % (center+1)),
                #            np.clip(img_tobesaved, 0.0, 1.0))  # 路径有问题

        print('*' * 40)
        print('END')
        # print('psnr_mean: ', psnr_my)
        # print('ssm_mean: ', ssm_my)
        # print('processing_time_mean: ', processing_time_for_all, 'us')

if __name__ == '__main__':
    vimeo_evaluate()
    # for modelspath in ()