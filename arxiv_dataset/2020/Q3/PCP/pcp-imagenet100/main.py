# Baseline0.2 - multitask

import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import datasets
import models
import argparse
import os
import sys
import math
from tqdm import tqdm
import numpy as np
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch.utils.data import Dataset

from lib import clustering
from lib.utils import AverageMeter, TripletLoss
from lib.LinearAverage import LinearAverage
from lib.BatchAverage import BatchCriterion
from test import kNN
from lib.pairscore import PreScore
from lib.criterion_icr import ICRcriterion
from lib.assigncluster import cluster_assign
from lib.icr import ICRDiscovery

from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/ucsf')

def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--low-dim', default=128, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16'], default='resnet',
                        help='CNN architecture (default: resnet)')

    parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--cluster-ratio', default=0.1, type=float,
                        help='init ratio (default: 0.05)')
    parser.add_argument('--high-ratio', default=0.5, type=float,
                        help='no more than (default: 0.5)')
    parser.add_argument('--alpha', default=0.8, type=float,
                        help='decay (default: 0.8)')
    parser.add_argument('--beta', default=-1, type=float,
                        help='threshold (default: -1)')
    parser.add_argument('--warm-epoch', default=100, type=int,
                        help='warm epoch')
    parser.add_argument('--stage-update', default=180, type=int,
                        help='which stage begin to update')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--batch-t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--batch-m', default=1, type=float,
                        metavar='N', help='m for negative sum')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=1000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--margin', default=5, type=float, help='triplet loss margin')
    parser.add_argument('--model_dir', default='checkpoint/', type=str,
                        help='model save path')
    parser.add_argument('--nce-t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--nce-m', default=0.5, type=float,
                        metavar='M', help='momentum for non-parametric updates')

    return parser.parse_args()

def compute_feature(trainloader, model, N, args):
    # from networks get features


    batch_time = AverageMeter()
    end = time.time()

    model.eval()

    trainFeatures = np.zeros((N, args.low_dim), dtype='float32')
    feature_index = np.zeros(N, dtype='int')

    with torch.no_grad():
        for batch_index, (inputs, _, targets, indexes) in enumerate(trainloader):
            batchSize = inputs.size(0)
            features = model(inputs)
            # print(len(trainloader)) # 391
            # print(features.shape) #torch.Size([128, 128]) --> [batch,dim]
            if batch_index < len(trainloader) - 1:
                trainFeatures[batch_index * batchSize:batch_index * batchSize + batchSize, :] = features.data.cpu(). \
                    numpy().astype('float32')
                feature_index[batch_index * batchSize:batch_index * batchSize + batchSize] = np.array([x.item() for x in indexes])
                # print(batch_index * batchSize + batchSize)
            else:
                # print('*****************')
                # print(batch_index) # 390
                # print(batchSize) # 80
                # print(batchSize * batch_index)
                trainFeatures[batch_index * args.batch_size:, :] = features.data.cpu().numpy().astype('float32')
                feature_index[batch_index * args.batch_size: ] = np.array([x.item() for x in indexes])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print('{0} / {1}\t'
            #       'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
            #       .format(batch_index, len(trainloader), batch_time=batch_time))

    return trainFeatures, feature_index

def adjust_learning_rate(optimizer, epoch, lr, round):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr
    # if epoch >= 80:
    #     lr = lr * (0.1 ** ((epoch - 80) // 40))
    # print(lr)
    #
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    #
    # writer.add_scalar('lr', lr, epoch)


    if epoch >= 120 and epoch < 160:
        lr = lr * 0.1
    elif epoch >= 160 and epoch < 200:
        lr = lr * 0.05
    elif epoch >= 200:
        lr = lr * 0.01
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    writer.add_scalar('lr', lr, epoch + round*200)

    # writer.add_scalar('lr', lr, args.max_epoch * round + epoch)

def train(epoch, net, optimizer,lemniscate, criterion, uel_criterion, trainloader, icr, icr2, stage_update, lr, device, round):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch, lr, round)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    change_branch_epoch = 180

    if epoch < change_branch_epoch:
        t = 0.2 + epoch * (0.25 / 180)
    else:
        t = 1


    end = time.time()
    for batch_idx, (inputs1, inputs2, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs1, inputs2, targets, indexes = inputs1.to(device), inputs2.to(device), targets.to(device), indexes.to(device)
        # uel branch
        inputs = torch.cat((inputs1, inputs2), 0)

        optimizer.zero_grad()

        # cluster branch
        features = net(inputs1)
        outputs = lemniscate(features, indexes)
        loss_cluster = criterion(outputs, indexes, icr)

        if stage_update > 0:
            if epoch < stage_update:
                features = net(inputs)
                loss_uel = uel_criterion(features, indexes)
                loss = (1 - t) * loss_uel + loss_cluster * t
            else:
                loss = loss_cluster
        else:
            features = net(inputs)
            loss_uel = uel_criterion(features, indexes)
            loss = (1 - t) * loss_uel + loss_cluster * t

        loss.backward()
        optimizer.step()

        # if epoch < change_branch_epoch:
        #
        #     train_loss.update(loss.item(), inputs1.size(0))
        # else:
        #     train_loss.update(loss.item(), inputs1.size(0))
        train_loss.update(loss.item(), inputs1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            print('Round: [{}] Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                 round, epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time,
                train_loss=train_loss))

    writer.add_scalar('loss', train_loss.avg, epoch + round * 200)

    return icr2


def main(args):

    # Data
    print('==> Preparing data..')
    _resize = 256
    _size = 224
    transform_train = transforms.Compose([
        transforms.Resize(size=_resize),
        transforms.RandomResizedCrop(size=_size, scale=(0.2, 1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size=_resize),
        transforms.CenterCrop(_size),  ###
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = datasets.ImageFolderInstance('/data2/zyf/ImageNet/ILSVRC2012-100/train',
                                            transform=transform_train, two_crop=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    testset = datasets.ImageFolderInstance('/data2/zyf/ImageNet/ILSVRC2012-100/val',
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ndata = trainset.__len__()
    print('trainset length: ' + str(ndata))

    print('==> Building model..')
    # net = models.__dict__['ResNet18'](low_dim=args.low_dim)

    # net = models.__dict__['resnet18'](low_dim=args.low_dim)
    net = models.__dict__['alexnet'](out=args.low_dim)
    # net2 = models.__dict__['appendnet'](low_dim=args.low_dim)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        # net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # define loss function: inner product loss within each mini-batch
    # criterion = BatchCriterion(args.batch_m, args.batch_t, args.batch_size, ndata)
    criterion = ICRcriterion()
    # define loss function: inner product loss within each mini-batch
    uel_criterion = BatchCriterion(args.batch_m, args.batch_t, args.batch_size, ndata)

    net.to(device)
    # net2.to(device)
    criterion.to(device)
    uel_criterion.to(device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    cluster_ratio = args.cluster_ratio
    if args.test_only or len(args.resume) > 0:
        cluster_ratio = args.cluster_ratio
        # Load checkpoint.
        model_path = 'checkpoint/' + args.resume
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    # define leminiscate
    if args.test_only and len(args.resume) > 0:

        trainFeatures, feature_index = compute_feature(trainloader, net, len(trainset), args)
        lemniscate = LinearAverage(torch.tensor(trainFeatures), args.low_dim, ndata, args.nce_t, args.nce_m)

    else:

        lemniscate = LinearAverage(torch.tensor([]), args.low_dim, ndata, args.nce_t, args.nce_m)
    lemniscate.to(device)

    # define optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer2 = torch.optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


    # test acc
    if args.test_only:
        acc = kNN(0, net, trainloader, testloader, 200, args.batch_t, ndata, low_dim=args.low_dim)
        exit(0)


    if len(args.resume) > 0:
        best_acc = best_acc
        start_epoch = start_epoch + 1
    else:
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    icr2 = ICRDiscovery(ndata)

    # init_cluster_num = 20000
    for round in range(5):
        for epoch in range(start_epoch, 200):
            #### get Features

            # trainFeatures are trainloader features and shuffle=True, so feature_index is match data
            trainFeatures, feature_index = compute_feature(trainloader, net, len(trainset), args)

            if round == 0:
                y = -1 * math.log10(ndata-5) / 200 * epoch + math.log10(ndata-5)
                cluster_num = int(math.pow(10, y))
                if cluster_num <= args.nmb_cluster:
                    cluster_num = args.nmb_cluster

                print('cluster number: ' + str(cluster_num))

                ###clustering algorithm to use
                # faiss cluster
                deepcluster = clustering.__dict__[args.clustering](int(cluster_num))

                #### Features to clustering
                clustering_loss = deepcluster.cluster(trainFeatures, feature_index, verbose=args.verbose)

                L = np.array(deepcluster.images_lists)
                image_dict = deepcluster.images_dict

                print('create ICR ...')
                # icr = ICRDiscovery(ndata)

                # if args.test_only and len(args.resume) > 0:
                # icr = cluster_assign(icr, L, trainFeatures, feature_index, trainset,
                # cluster_ratio + epoch*((1-cluster_ratio)/250))
                icrtime = time.time()

                # icr = cluster_assign(epoch, L, trainFeatures, feature_index, 1, 1)
                if epoch < args.warm_epoch:
                    icr = cluster_assign(epoch, L, trainFeatures, feature_index, args.cluster_ratio, 1)
                else:
                    icr = PreScore(epoch, L, image_dict, trainFeatures, feature_index, trainset,
                                   args.high_ratio, args.cluster_ratio, args.alpha, args.beta)

                print('calculate ICR time is: {}'.format(time.time() - icrtime))
                writer.add_scalar('icr_time', (time.time() - icrtime), epoch + round * 200)

            else:
                cluster_num = args.nmb_cluster
                print('cluster number: ' + str(cluster_num))

                ###clustering algorithm to use
                # faiss cluster
                deepcluster = clustering.__dict__[args.clustering](int(cluster_num))

                #### Features to clustering
                clustering_loss = deepcluster.cluster(trainFeatures, feature_index, verbose=args.verbose)

                L = np.array(deepcluster.images_lists)
                image_dict = deepcluster.images_dict

                print('create ICR ...')
                # icr = ICRDiscovery(ndata)

                # if args.test_only and len(args.resume) > 0:
                # icr = cluster_assign(icr, L, trainFeatures, feature_index, trainset,
                # cluster_ratio + epoch*((1-cluster_ratio)/250))
                icrtime = time.time()

                # icr = cluster_assign(epoch, L, trainFeatures, feature_index, 1, 1)
                icr = PreScore(epoch, L, image_dict, trainFeatures, feature_index, trainset,
                                   args.high_ratio, args.cluster_ratio, args.alpha, args.beta)

                print('calculate ICR time is: {}'.format(time.time() - icrtime))
                writer.add_scalar('icr_time', (time.time() - icrtime), epoch + round * 200)

            # else:
            #     icr = cluster_assign(icr, L, trainFeatures, feature_index, trainset, 0.2 + epoch*0.004)

            # print(icr.neighbours)

            icr2 = train(epoch, net, optimizer, lemniscate, criterion, uel_criterion, trainloader,
                         icr, icr2, args.stage_update, args.lr, device, round)

            print('----------Evaluation---------')
            start = time.time()
            acc = kNN(0, net, trainloader, testloader, 200, args.batch_t, ndata, low_dim=args.low_dim)
            print("Evaluation Time: '{}'s".format(time.time() - start))

            writer.add_scalar('nn_acc', acc, epoch + round * 200)

            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir(args.model_dir):
                    os.mkdir(args.model_dir)
                torch.save(state, './checkpoint/ckpt_best_round_{}.t7'.format(round))
                if epoch < 200:
                    torch.save(state, './checkpoint/ckpt_200_best_round_{}.t7'.format(round))
                if epoch < 300:
                    torch.save(state, './checkpoint/ckpt_300_best_round_{}.t7'.format(round))
                best_acc = acc

            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if epoch < 200:
                torch.save(state, './checkpoint/ckpt_200_last_round_{}.t7'.format(round))
            else:
                torch.save(state, './checkpoint/ckpt_last_round_{}.t7'.format(round))

            if epoch < 300:
                torch.save(state, './checkpoint/ckpt_300_last_round_{}.t7'.format(round))

            print('[Round]: {} [Epoch]: {} \t accuracy: {}% \t (best acc: {}%)'.format(round, epoch, acc, best_acc))




if __name__ == '__main__':
    args = parse_args()
    main(args)
    writer.close()
