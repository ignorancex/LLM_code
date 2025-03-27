import os
import json
import argparse

parser = argparse.ArgumentParser(
        description='Attack Evaluation')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=["cifar10", "cifar100", "tinyimagenet"],
                    help='The dataset')
parser.add_argument('--model_path',
                    help='Model for attack evaluation')
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='Name of the model')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='Input batch size for testing (default: 200)')
parser.add_argument('--epsilon', default=8/255, type=float,
                    help='Attack perturbation magnitude')
parser.add_argument('--gpu', metavar='N', type=int, nargs='+',
                    help='gpu numbers')



args = parser.parse_args()
args.gpu = ",".join([str(a) for a in args.gpu])
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torchvision
import torchvision.transforms as transforms
from torchattacks import *
from wide_resnet import *
from collections import OrderedDict
from autoattack import AutoAttack, Normalize

from tin import TImgNetDataset
from preact import PreActResNet18

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 10
n = 1024
x = 32


print("evaluating: ", args.model_path)

transform = transforms.Compose(
    [transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])


if args.dataset == "cifar10":
    num_classes = 10
    n = 1024
    x = 32
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
if args.dataset == "cifar100":
    num_classes = 100
    n = 1024
    x = 32
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
                                
if args.dataset == "tinyimagenet":
    num_classes = 200
    n = 4096
    x = 64
    with open("tiny-imagenet-classindex.json", 'r') as openfile:
        class_to_idx = json.load(openfile)
    
    testset = TImgNetDataset(img_path="./tiny-imagenet-200/val/images", gt_path="./tiny-imagenet-200/val/val_annotations.txt",
                            class_to_idx=class_to_idx, transform=transform)

batch_size = args.batch_size


testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


if args.model[0:3] =="wrn":
    parts = args.model.split('-')
    depth = int(parts[1])
    widen = int(parts[2])
    target_model = WideResNet(
                depth=depth, num_classes=num_classes, widen_factor=widen)
elif args.model == "preactresnet18":
    target_model = PreActResNet18()
else:
    print("Model not implemented!!!")
    exit()

target_model = target_model.to(device)
checkpoint = torch.load(args.model_path)['net']

try:
    target_model.load_state_dict(checkpoint)
except:
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    target_model.load_state_dict(new_state_dict, False)

target_model = torch.nn.parallel.DataParallel(target_model)
# target_model = nn.Sequential(
#     Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
#     target_model
# ).cuda()



target_model.eval()
atk = AutoAttack(target_model, eps=args.epsilon, n=n, x=x)

l = [X for (X, y) in testloader]
x_test = torch.cat(l, 0)
l = [y for (X, y) in testloader]
y_test = torch.cat(l, 0)

adv_complete = atk.run_standard_evaluation(x_test, y_test, bs=args.batch_size)


# adv_complete = atk.clean_accuracy(x_test, y_test, bs=args.batch_size)





