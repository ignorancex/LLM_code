import torch
import time
import datasets
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn
from sklearn.cluster import KMeans
from lib.utils import AverageMeter

def kNN(epoch, net, trainloader, testloader, K, sigma, ndata, low_dim=128):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    correct_t = 0
    testsize = testloader.dataset.__len__()

    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
        # print('1')
    else:
        try:
            trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
            # print('2')
        except:
            trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
            # print('3')

    trainFeatures = np.zeros((low_dim, ndata))
    C = trainLabels.max() + 1
    C = np.int(C)
    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        # print(trainloader.dataset)
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4)
        for batch_idx, (inputs, _, targets, indexes) in enumerate(temploader):
            # print(inputs.shape)
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            #
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t().cpu()

    trainloader.dataset.transform = transform_bak
    #

    trainFeatures = torch.Tensor(trainFeatures).cuda()
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            total += targets.size(0)

            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            # yd_transform = yd.clone().div_(yd.clone().max()).div_(sigma).exp_()
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)),
                              1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            # print('***********')
            # print(correct.shape)
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

    print(top1 * 100. / total)

    return top1 * 100. / total