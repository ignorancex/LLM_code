import argparse
import os, pdb
import torch, cv2
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import time, math, glob
import scipy.io as sio
from PIL import Image
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./checkpoint/model_denoise_stage2.pth", type=str, help="model path")
parser.add_argument("--dataset", default="./depth", type=str, help="noisy dataset name")
parser.add_argument("--save", default="./results", type=str, help="savepath, Default: results")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

model = torch.load(opt.model)["model"]

image_list = glob.glob(opt.dataset+"/*.*") 
p=0
p2=0
with torch.no_grad():
    for image_name in image_list:
        name = image_name.split('\\')
        print("Processing ", image_name)
        im_n = Image.open(image_name)
        im_n=np.array(im_n)
        im_n=im_n[:,:,0]
        im_n = np.expand_dims(im_n, 2)

        im_n = np.transpose(im_n, (2,0,1))
        im_n = np.expand_dims(im_n, 0)
        im_n = torch.from_numpy(im_n).float()/255

        im_input = Variable(im_n)

        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        start_time = time.time()

        height = int(im_input.size()[2])
        width = int(im_input.size()[3])
        M = int(height / 16)  # 行能分成几组
        N = int(width / 16)

        im_input = im_input[:,:, :M * 16, :N * 16]
        im_output = torch.zeros(3, M * 16, N * 16)
        # print(im_input.shape)
        # for i in range(M):
        #     for j in range(N):
        #         HR_4x[:, i * 16:i * 16 + 16, j * 16:j * 16 + 16] = model(
        #             im_input[:, :, i * 16:i * 16 + 16, j * 16:j * 16 + 16]).squeeze()\
        # print(im_input.shape)
        im_output = model(im_input)
        im_output = im_output.cpu()
        save_image(im_output.data,opt.save+'/'+name[-1])
