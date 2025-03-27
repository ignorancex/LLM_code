import os
import torch
import numpy as np
from torch.nn import init
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.data import DataLoader
from models.LaplacianPyramid import Lap_Pyramid
from torchvision import transforms
from utils import loaddata
from models import FCNet
from utils import Loss
import random

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        # img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.detach().numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.detach().numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def weight_init(m):
    # 可以判断是否为conv2d，使用相应的初始化方式
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
     # 是否为批归一化层
    elif classname.find('InstanceNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data,   0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


"""
#初始化参数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
"""

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    seed = config.seed
    seed_torch(seed)
    lamda=4000
    #初始化损失函数
    pyr_loss = Loss.Pyramid_Loss().cuda()
    rec_loss = Loss.Reconstruction_Loss().cuda()
    L_spa = Loss.L_spa().cuda()

    #是否载入模型
    wulalala  = FCNet.FCNet(num_high=3).cuda()
    wulalala.apply(weight_init)
    if config.load_pretrain == True:
        wulalala.load_state_dict(torch.load(config.pretrain_dir))

    #数据集处理
    train_transform = transforms.Compose([
        loaddata.BatchToTensor(),
    ])

    # 数据集路径
    #datapath = "./data/"
    # 构建数据集

    train_data = loaddata.ImageSeqDataset(csv_file=os.path.join(config.datapath, 'train.txt'),
                                 Train_img_seq_dir=config.datapath,
                                 Label_img_dir=config.labelpath,
                                 Train_transform=train_transform,
                                 Label_transform=transforms.ToTensor(),
                                 randomlist=True)

    train_loader = DataLoader(train_data,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=6)



    a_optimizer = torch.optim.Adam(wulalala.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(a_optimizer, milestones=[50,100,150,200,250], gamma=0.8, verbose=True)

    wulalala.train()

    for epoch in range(config.num_epochs):
        for step, sample_batched in enumerate(train_loader):


            train_image, label_image= sample_batched['Train'], sample_batched['Lable']

            train_image = train_image.squeeze(0).cuda()

            label_image = label_image.cuda()

            out1 , out2, out3, out4 = wulalala(train_image)

            lp = Lap_Pyramid()
            pry1 = lp.pyramid_decom(label_image)
            pry2 = lp.pyramid_recons(pry1)


            loss = rec_loss(out4, label_image) + pyr_loss(out1 , out2, out3,pry2[0],pry2[1],pry2[2])\
                   +lamda*(torch.mean(L_spa(out1,pry2[0]))+4*torch.mean(L_spa(out2, pry2[1]))\
                   +16*torch.mean(L_spa(out3, pry2[2]))+64*torch.mean(L_spa(out4, label_image)))

            a_optimizer.zero_grad()
            loss.backward()
            a_optimizer.step()

            if ((step + 1) % config.display_iter) == 0:
                print("Loss at iteration", step + 1, ":", loss.item())
        scheduler.step()
        if ((epoch + 1) % config.snapshot_iter) == 0 and (step + 1) >= 100:
            torch.save(wulalala.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--datapath', type=str, default="../trainimage/")
    parser.add_argument('--labelpath', type=str, default="../trainimage/label/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.9)
    parser.add_argument('--grad_clip_norm', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=160)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--d_snapshots_folder', type=str, default="dis_snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--d_load_pretrain', type=bool, default=False)
    parser.add_argument('--d_pretrain_dir', type=str, default="dis_snapshots/Epoch97.pth")
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch29.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train(config)





