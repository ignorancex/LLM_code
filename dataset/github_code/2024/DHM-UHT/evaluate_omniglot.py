import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import learn2learn as l2l
import clustering
import models
from torch.utils.data import Subset
from util import AverageMeter, UnifLabelSampler
from tqdm import tqdm


def mian():
    checkpoint = torch.load('./output/checkpoint/omniglot_60000.pth')
    args=checkpoint['args']

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = models.__dict__[args.arch](num_classes=args.way)
    model.load_state_dict(checkpoint['state_dict'])
    feature_length = model.classifier[-1].in_features
    model.cuda()
    maml = l2l.algorithms.MAML(model, lr=args.meta_lr, first_order=False)
    optimizer = torch.optim.Adam(maml.parameters(), args.base_lr)
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.CrossEntropyLoss().cuda()






if __name__ == '__main__':
    main()
