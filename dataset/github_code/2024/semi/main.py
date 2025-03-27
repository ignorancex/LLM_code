from one_stage_model import *
from two_stage_model import *
from data_transform import *
from util import train_nolabel
import torch
import torchvision.models

from utils.common_config import get_model, get_train_dataset, \
    get_val_dataset, \
    get_val_dataloader, \
    get_val_transformations, get_train_transformations\



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class pedccResNet18(torch.nn.Module):
    def __init__(self):
        super(pedccResNet18, self).__init__()

        #self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = torch.load('./resnet.pth')
        #self.resnet = torch.load('facebookresearch/balowtwins:main','resnet50')
        self.resnet.fc = nn.Linear(2048,512)
        self.relu = nn.ReLU()
        self.Linear_down = nn.Linear(512,outputdim)
        self.out = Softmax_PEDCC(outputdim,classnum,PKL)
    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
    def forward(self,x):
        x = self.resnet(x)
        x = self.relu(x)
        x = self.Linear_down(x)
        x1 = self.l2_norm(x)
        out = self.out(x1)
        return x1, out


base_lr = 0.001
epoches = 401
# lr_step = 70
outputdim = 100
dim = 1


import argparse
CONFIG_ENV_PATH = 'configs/env.yml'
CONFIG_EXP_PATH = 'configs/pretext/simclr_stl10.yml'
from utils.config import create_config
p = create_config(CONFIG_ENV_PATH, CONFIG_EXP_PATH)

transform1 = get_val_transformations(p)
transform2 = get_train_transformations(p)


# data1 = get_val_dataset(p,transform1,True)
data2 = get_train_dataset(p,transform1,False,True)
loader = get_val_dataloader(p,data2)

#net1 = encoder_plus_add(dim,outputdim)
net1 = pedccResNet18()
print(net1.resnet)
for param in net1.resnet.parameters():
     param.requires_grad =False

#for param in net1.resnet.layer2.parameters():
#    param.requires_grad = True
for param in net1.resnet.layer3.parameters():
    param.requires_grad = True
for param in net1.resnet.layer4.parameters():
    param.requires_grad = True
for param in net1.resnet.avgpool.parameters():
     param.requires_grad = True
for param in net1.resnet.fc.parameters():
     param.requires_grad = True





net2 = decoder_plus_add(outputdim)
#optimizer1 = torch.optim.Adam(net1.parameters(), lr=base_lr)
optimizer1 = torch.optim.Adam(
    [{'params':net1.resnet.layer2.parameters(),'lr':base_lr},
    {'params' :net1.resnet.layer3.parameters(),'lr':base_lr},
     {'params' :net1.resnet.layer4.parameters(),'lr':base_lr},
{'params' :net1.resnet.avgpool.parameters(),'lr':base_lr},
{'params' :net1.resnet.fc.parameters(),'lr':base_lr},
    # {'params':net1.Linear_down.parameters(), 'lr': base_lr},
     {'params':net1.out.parameters(),'lr':base_lr},])

optimizer2 = torch.optim.Adam(net2.parameters(), lr=base_lr)

criterion = nn.MSELoss()
criterion1= nn.L1Loss()


print(" ####Start training  ####")

for epoch in range(epoches):
    train_nolabel(net1,loader,L_test_data_1,epoch,optimizer1,optimizer2,criterion,criterion1)

print("Done!")
