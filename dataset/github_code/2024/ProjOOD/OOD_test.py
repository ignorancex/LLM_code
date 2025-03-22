import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from tqdm import tqdm
import numpy
import os


from densenet import DenseNet3
import conf.config as conf
from data.dataLoader import *
import utils


torch.set_printoptions(precision=10, threshold=100000)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device_ids = [0,1]


model_path = ''
cfg = conf.CIFAR10
test_data = CIFAR10_test_data
OOD_data = TinyImageNet_Resize_Test
utils.PEDCC_PATH = cfg['PEDCC_Type']  # 修改使用的PEDCC文OOD_test.py:24件
fmap_out = torch.Tensor([]).cuda()

f1 = open('a.txt', 'w')
f2 = open('b.txt', 'w')
f3 = open('len.txt', 'w')
f4 = open('gamma.txt', 'w')
def test():

    net = DenseNet3(depth=100, num_classes=cfg['num_classes'], feature_size=cfg['feature_size'])
    test_loader = Data.DataLoader(dataset=test_data, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        net = net.cuda()
    net = net.eval()
    w = net.module.fc.weight.data
    w = w.t()
    wt = w.t()
    P = torch.mm(wt,w)
    P = torch.linalg.inv(P)
    P = torch.mm(w,P)
    P = torch.mm(P,wt)
    for im, label in tqdm(test_loader):
        if torch.cuda.is_available():
            im, label = im.cuda(), label.cuda()

        out, outf,outnorm,f = net(im)
        len = outf.pow(2).sum(1)
        len = len.sqrt()

        score,_ = out.max(1)

        f = f.t()


        p = torch.mm(P,f)


        p = p.t()
        f = f.t()
        plen = p.pow(2).sum(1)
        plen = plen.sqrt()
        flen = f.pow(2).sum(1)
        flen = flen.sqrt()
        gamma = plen/flen

        a2 = out.pow(2).sum(1)

        a2 = a2*0.9
        a = a2.sqrt()

        #
        for elem in a:
            f1.write("{}\n".format(elem))
        for i in range(score.shape[0]):
            f2.write("{}\n".format((score[i]/a[i])))
        for elem in len:
            f3.write("{}\n".format(elem))
        for elem in gamma:
            f4.write("{}\n".format(elem))
if __name__ == '__main__':
    test()
