from datetime import datetime
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mpl_toolkits.mplot3d as p3d
import math
import os
import torch.optim as optim
from scipy.special import binom
import scipy.io as io
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import conf.config as conf
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
device_ids = [0, 1]

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

PEDCC_PATH = ''
def read_pkl():
    f = open(PEDCC_PATH, 'rb')
    a = pickle.load(f)
    f.close()
    return a

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    print(norm.shape)
    output = torch.div(input, norm)
    return output


def Decorrelation(input):
    input_norm = l2_norm(input)
    res = torch.mm(input_norm, input_norm.T)
    res_loss = torch.sum(abs(res))-res.size(0)
    res_loss = res_loss/((res.size(0)-1)*res.size(0))
    return res_loss
def train(net, train_data, valid_data, cfg, criterion,criterion1, save_folder):

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
        print(net)

    prev_time = datetime.now()
    map_dict = read_pkl()
    LR = cfg['LR']
    for epoch in range(cfg['max_epoch']):
        if epoch in cfg['lr_steps']:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_acc = 0
        length, num = 0, 0
        length_test = 0
        net = net.train()
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():

                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label:
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()), 0)

                label_mse_tensor = tensor_empty.view(label.shape[0], -1)  # (batchSize, dimension)
                label_mse_tensor = label_mse_tensor.cuda()


            output, output1, output2, output3 = net(im)
            w = net.module.fc.weight

            loss1 = criterion(output, label)
            loss2 = criterion1(output2, label_mse_tensor)
            loss3 = Decorrelation(w)
            print(w.shape)
            loss = loss1 + loss2 + loss3

            length += output.pow(2).sum().item()
            num += output.shape[0]
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_acc += get_acc(output, label)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
                length_test += output.pow(2).sum().item()/im.shape[0]
            epoch_str = (
                "Epoch %d. Train Loss: %f,Train Loss1: %f,Train Loss2: %f,Train Loss3: %f,Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, length: %f, length_test: %f"
                % (epoch, train_loss / len(train_data),
                   train_loss1 / len(train_data),train_loss2 / len(train_data),train_loss3 / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, length/num, length_test/len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(save_folder+'acc.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), save_folder + 'densenet' + str(epoch+1) + '_epoch.pth')


# consieLinear层 实现了norm的fea与norm weight的点积计算，服务于margin based softmax loss
# 将w替换成pedcc，固定
# 计算余弦距离
class CosineLinear_PEDCC(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear_PEDCC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        map_dict = read_pkl()
        tensor_empty = torch.Tensor([]).cuda()
        for label_index in range(self.out_features):
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
        label_40D_tensor = label_40D_tensor.cuda()
        self.weight.data = label_40D_tensor
        #print(self.weight.data)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features
        cos_theta = x.mm(w)  # size=(B,Classnum)  x.dot(ww)


        return cos_theta  # size=(B,Classnum,1)

# AMSoftmax 层的pytorch实现，两个重要参数 scale，margin（不同难度和量级的数据对应不同的最优参数）
# 原始实现caffe->https://github.com/happynear/AMSoftmax
class AMSoftmax(nn.Module):
    def __init__(self, scale, margin, is_amp=False):
        super(AMSoftmax, self).__init__()
        self.scale = scale
        self.margin = margin
        self.is_amp = is_amp
    def forward(self, input, target):
        # self.it += 1
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index1 = ~index


        output = cos_theta * 1.0  # size=(B,Classnum)
        index = index.bool()
        output[index] = output[index] - self.margin
        if self.is_amp:
            output[index1] += self.margin
        output = output * self.scale


        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss
