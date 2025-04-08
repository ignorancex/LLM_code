from torchvision.models import resnet50, resnet34, resnet18
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.cnn1 = resnet50(pretrained=True)

        self.fc1 = nn.Linear(1000, args.num_classes)

        self.name = 'resnet50'
        self.dropout = args.dropout
        self.dp = torch.nn.Dropout(self.dropout)


    def model_name(self):
        return self.name

    def forward(self, x, tgt):

        x = self.cnn1(x)
        # x = self.cnn1.relu(x)
        
        # if self.dropout < 1.0 :
        #     x = self.dp(x)
        # x = self.fc1(x)

        return x
        # return x, nn.CrossEntropyLoss()(x, tgt.long())




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