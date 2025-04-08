from torchvision.models import resnet50, resnet34, resnet18
import torch.nn as nn
import torch
import numpy as np
from utils.loss import FocalLoss

def partition_assign(a, n):
    idx = np.argpartition(a,-n,axis=1)[:,-n:]
    out = np.zeros(a.shape, dtype=int)
    np.put_along_axis(out,idx,1,axis=1)
    return out
 
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.resnet = resnet18(pretrained=True)

        self.feature = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            self.resnet.avgpool)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, args.num_classes)


        norm_layer = nn.BatchNorm1d
        self.bn1 = norm_layer(256)

        self.name = 'resnet18'
        self.dropout = args.dropout
        self.dp = torch.nn.Dropout(self.dropout)
        self.tau = 1

    def model_name(self):
        return self.name

    def forward(self, x, tgt):

        


        x = self.feature(x)

        x1 = x = torch.flatten(x, 1)

        x2 = x = self.fc1(x)


        x = self.bn1(x)


        x = self.resnet.relu(x)
        
        x = self.fc2(x)

       
        # if self.dropout < 1.0 :
        #     x = self.dp(x)
        # x = self.fc1(x)

        # m = x1.shape[0]//2
        # new_labels = torch.tensor(range(m)).cuda()


        # q1 = x1[:m]
        # k1 = x1[m:]
        # q2 = x2[:m]
        # k2 = x2[m:]
        # q = x[:m]
        # k = x[m:]


        # t1 = torch.mm(q1, k1.t())
        # t2 = torch.mm(q2, k2.t())
        # tt = torch.mm(q, k.t())

        # t2t = t2[:, 1:].detach().cpu().numpy()
        # mask2 = torch.tensor(partition_assign(t2t, n=16)).cuda()
        # t2[:, 1:] = t2[:, 1:] * mask2


        # t3t = tt[:, 1:].detach().cpu().numpy()
        # mask3 = torch.tensor(partition_assign(t3t, n=4)).cuda()
        # tt[:, 1:] = tt[:, 1:] * mask3
        

        # loss1 = nn.CrossEntropyLoss()(t1/self.tau, new_labels) * 2 * self.tau
        # loss2 = nn.CrossEntropyLoss()(t2/self.tau, new_labels) * 2 * self.tau
        # loss3 = nn.CrossEntropyLoss()(tt/self.tau, new_labels) * 2 * self.tau


        # return x, loss1, loss2, loss3
        return x, nn.CrossEntropyLoss()(x, tgt.long())




#
#
#
# if __name__ == '__main__':
#     net = ResLSTM().cuda()
#     img = torch.rand(4, 3, 224, 224).cuda()
#     tgt = torch.tensor([1, 2, 3, 4]).cuda()
#     o, l = net(img, tgt)
#     print('================== train =====================')
#     print(o)
#     print(l)
#     net.eval()
#     o, l = net(img, tgt)
#     print('================== val =====================')
#     print(o)
#     print(l)