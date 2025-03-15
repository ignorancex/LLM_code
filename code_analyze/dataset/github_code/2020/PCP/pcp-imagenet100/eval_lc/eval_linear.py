import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datasets as cifar_datasets
from lib.utils import AverageMeter
from lib.utils import learning_rate_decay, load_model
from RegLog import *
import models

from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/eval')

def parse_args():
    parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                     of frozen convolutional layers of an AlexNet.""")
    parser.add_argument('--low-dim', default=128, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--data', type=str, help='path to dataset')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--conv', type=int, choices=[1, 2, 3, 4, 5],
                        help='on top of which convolutional layer train logistic regression')
    parser.add_argument('--tencrops', action='store_true',
                        help='validation accuracy averaged over 10 crops')
    parser.add_argument('--exp', type=str, default='', help='exp folder')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=90, help='number of total epochs to run (default: 90)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=-4, type=float,
                        help='weight decay pow (default: -4)')
    parser.add_argument('--seed', type=int, default=31, help='random seed')
    parser.add_argument('--verbose', action='store_true', help='chatty')

    return parser.parse_args()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, reglog, criterion, optimizer, epoch, forward):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # freeze also batch norm layers
    model.eval()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        #adjust learning rate
        learning_rate_decay(optimizer, len(train_loader) * epoch + i, args.lr)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
        # compute output

        output = forward(input_var, model, reglog.conv)
        output = reglog(output)
        loss = criterion(output, target_var)
        # print(loss.data.item())
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data[0], input.size(0))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)


def validate(val_loader, model, reglog, criterion, forward):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    softmax = nn.Softmax(dim=1).cuda()
    end = time.time()
    for i, (input_tensor, target, _) in enumerate(val_loader):
        if args.tencrops:
            bs, ncrops, c, h, w = input_tensor.size()
            input_tensor = input_tensor.view(-1, c, h, w)
        target = target.cuda(async=True)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            # input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
            target_var = torch.autograd.Variable(target)
            # target_var = torch.autograd.Variable(target, volatile=True)
            # torch.autograd.Variable(target, requires_grad=False)

            output = reglog(forward(input_var, model, reglog.conv))

            if args.tencrops:
                # print #100
                # print(ncrops) #10
                output_central = output.view(bs, ncrops, -1)[:, int(ncrops / 2 - 1), :]
                output = softmax(output)
                output = torch.squeeze(output.view(bs, ncrops, -1).mean(1))
            else:
                output_central = output

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], input_tensor.size(0))
            top5.update(prec5[0], input_tensor.size(0))
            loss = criterion(output_central, target_var)
            losses.update(loss.data.item(), input_tensor.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose and i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                      .format(i, len(val_loader), batch_time=batch_time,
                              loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def main(args):
    # _size = 32
    _size = 224

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    best_prec1 = 0

    # load model
    print('==> Building model..')
    # net = models.__dict__['ResNet18'](low_dim=args.low_dim)
    net = models.__dict__['alexnet'](out=args.low_dim)
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    checkpoint = torch.load(args.model)
    net.load_state_dict(checkpoint['net'])

    # model = load_model(args.model)
    model = net
    model.cuda()
    cudnn.benchmark = True

    # freeze the features layers
    for param in model.parameters(): #features.
        param.requires_grad = False

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize(size=_size),
        transforms.RandomResizedCrop(size=_size, scale=(0.2, 1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if args.tencrops:
        transform_test = transforms.Compose([
            transforms.Resize(size=_size),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(size=_size),
            transforms.CenterCrop(_size),
            transforms.ToTensor(),
            normalize,
        ])

    trainset = cifar_datasets.ImageFolderInstance('/data2/zyf/ImageNet/ILSVRC2012-100/train',
                                            transform=transform_train, two_crop=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = cifar_datasets.ImageFolderInstance('/data2/zyf/ImageNet/ILSVRC2012-100/val',
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)


    # logistic regression
    reglog = AlexnetRegLog(args.conv, len(trainset.classes)).cuda()
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, reglog.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.weight_decay
    )

    for epoch in range(args.epochs):
        end = time.time()

        # train for one epoch
        train(trainloader, model, reglog, criterion, optimizer, epoch, forward_alexnet)

        # evaluate on validation set
        prec1, prec5, loss = validate(testloader, model, reglog, criterion, forward_alexnet)
        writer.add_scalar('acc', prec1, epoch)

        # loss_log.log(loss)
        # prec1_log.log(prec1)
        # prec5_log.log(prec5)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            filename = 'pcf200_model_best.pth.tar'
            print('best acc: ' + str(prec1))
        else:
            filename = 'pcf200_checkpoint.pth.tar'
        torch.save({
            'epoch': epoch + 1,
            'arch': 'net',
            'state_dict': model.state_dict(),
            'prec5': prec5,
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.exp, filename))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    writer.close()
