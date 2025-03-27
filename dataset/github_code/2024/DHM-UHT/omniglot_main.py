# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--data', default='./Omniglot', help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'omniglotcnn', 'miniimagenetcnn'], default='omniglotcnn')
    parser.add_argument('--algo', default='MAML', choices=['MAML', 'ANIL'], help='MAML or ANIL')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC', 'DBSCAN'],
                        default='DBSCAN', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=5,
                        help='number of cluster for k-means (default: 5000)')
    parser.add_argument('--adapt_num', type=int, default=10,
                        help='number of adaptation (default: 3)')
    parser.add_argument('--meta_lr', default=0.001, type=float,
                        help='outer learning rate (default: 0.001)')
    parser.add_argument('--base_lr', default=0.05, type=float,
                        help='inner learning rate (default: 0.05)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=60000,
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--way', type=int, default=5,
                        help='way of classification')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--sample_per_task', type=int, default=256)
    parser.add_argument('--task_num', type=int, default=16)
    return parser.parse_args()


def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = models.__dict__[args.arch](num_classes=args.way)
    feature_length = model.classifier[-1].in_features
    model.cuda()
    maml = l2l.algorithms.MAML(model, lr=args.meta_lr, first_order=False)
    optimizer = torch.optim.Adam(maml.parameters(), args.base_lr)
    criterion = nn.CrossEntropyLoss().cuda()

    # load the data
    tra = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                              transforms.Resize(28),
                              transforms.ToTensor()
                              ])
    dataset = datasets.ImageFolder(args.data, transform=tra)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](k=args.nmb_cluster)

    for epoch in tqdm(range(args.epochs)):
        meta_loss = 0
        for _ in range(args.task_num):
            model = maml.clone()
            task_index = np.random.choice(len(dataset), args.sample_per_task, replace=False)
            task = Subset(dataset, task_index)
            task_imgs = [dataset.imgs[i] for i in task_index]
            dataloader = torch.utils.data.DataLoader(task,
                                                     batch_size=args.batch,
                                                     num_workers=args.workers,
                                                     pin_memory=True)

            # get the features for the whole dataset
            features = compute_features(dataloader, model.features, len(task))

            # cluster the features
            deepcluster.cluster(features)

            # assign pseudo-labels
            train_task = clustering.cluster_assign(deepcluster.images_lists,
                                                   task_imgs, tra)
            # uniformly and randomly sample per target
            sampler = UnifLabelSampler(int(args.reassign * len(train_task)),
                                       deepcluster.images_lists)

            train_dataloader = torch.utils.data.DataLoader(
                train_task,
                batch_size=int(len(train_task) / 4),
                num_workers=args.workers,
                sampler=sampler,
                pin_memory=True,
            )
            # re-initialize dynamic head
            if args.clustering == 'DBSCAN':
                model.classifier[-1] = nn.Linear(feature_length, len(deepcluster.images_lists))
                classifier_layer = model.classifier[-1]
                nn.init.kaiming_normal_(classifier_layer.weight)
                nn.init.zeros_(classifier_layer.bias)
                model.cuda()

            # train network with clusters as pseudo-labels
            meta_loss += inner_loop(train_dataloader, model, criterion, args.adapt_num)

        meta_loss /= args.task_num
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 and epoch != 0:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args
            }, './output/checkpoint/omniglot_' + str(args.epoch) + '.pth')


def inner_loop(loader, model, crit, adapt_num):
    loss = 0
    if args.algo == 'ANIL':
        for param in model.features.parameters():
            param.require_grad = False

    data_iter = iter(loader)
    for i in range(adapt_num + 1):  # Take the last loop as evaluating on the query set
        try:
            input_tensor, target = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_tensor, target = next(data_iter)

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        loss = crit(output, target_var)
        model.adapt(loss, first_order=False, allow_nograd=True)

    return loss


def compute_features(dataloader, model, N):
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux
            else:
                # special treatment for final batch
                features[i * dataloader.batch_size:] = aux

    return features


if __name__ == '__main__':
    args = parse_args()
    main(args)
