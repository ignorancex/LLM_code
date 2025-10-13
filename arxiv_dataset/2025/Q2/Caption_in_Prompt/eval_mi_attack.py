import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch.utils.data import ConcatDataset
from timm import utils
from imagenetv2_pytorch import ImageNetV2Dataset
from timm.models import create_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--net_arch", type=str, default="resnet50", help="network architecture")
parser.add_argument("--model_path", type=str, default="/models/imagenet1k_models/r50_real_data/last.pth.tar", help="path to load pretrained model")
parser.add_argument("--data_path", type=str, default="/data", help="path to load test set for evalution")
args = parser.parse_args()


# Print Args
print("--------args----------")
for k in list(vars(args).keys()):
    print("%s: %s" % (k, vars(args)[k]))
print("--------args----------\n")

net_arch = args.net_arch
model_path = args.model_path
data_path = args.data_path


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def membership_inference_attack(model, train_loader, test_loader, cpu=False):
    model.eval()

    result = []
    softmax = torch.nn.Softmax(dim=1)

    train_cnt = 0
    for x, y in train_loader:
        if not cpu:
            x, y = x[:2].cuda(), y[:2].cuda()

        with torch.no_grad():
            _y = softmax( model(x) )
        train_cnt += len(y)
        for i in range(len(_y)):
            result.append( [_y[i][y[i]].item(), 1] )

    test_cnt = 0
    for x, y in test_loader:
        if not cpu:
            x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _y = softmax( model(x) )
        test_cnt += len(y)
        for i in range(len(_y)):
            result.append( [_y[i][y[i]].item(), 0] )

    result = np.array(result)
    result = result[result[:,0].argsort()]
    one = train_cnt
    zero = test_cnt
    best_atk_acc = 0.0
    for i in range(len(result)):
        atk_acc = 0.5 * (one/train_cnt + (test_cnt-zero)/test_cnt)
        best_atk_acc = max(best_atk_acc, atk_acc)
        if result[i][1] == 1:
            one = one-1
        else: zero = zero-1

    return best_atk_acc


# load pretrained model
if 'resnet' in net_arch:
    net = create_model('resnet50').to(device) #torchvision.models.resnet50().to(device)
elif 'vit' in net_arch:
    net = create_model('vit_base_patch16_224', in_chans=3, num_classes=1000, drop_rate=0).to(device)
    #net = torchvision.models.vit_b_16().to(device)

net.load_state_dict(torch.load(model_path)['state_dict'], strict=True)
net.eval()

#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]
#test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

data_cfg = timm.data.resolve_data_config(net.pretrained_cfg)
test_transform = timm.data.create_transform(**data_cfg)

trainset = torchvision.datasets.ImageNet(root=os.path.join(data_path, 'ImageNet'), split="train", transform=test_transform)
testset = torchvision.datasets.ImageNet(root=os.path.join(data_path, 'ImageNet'), split="val", transform=test_transform)

train_loader =  torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=False, num_workers=2)
test_loader =  torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=2)

mi_attack_acc = membership_inference_attack(net, train_loader, test_loader)
print("Membership inference attack Accuracy: %.2f"%(mi_attack_acc))