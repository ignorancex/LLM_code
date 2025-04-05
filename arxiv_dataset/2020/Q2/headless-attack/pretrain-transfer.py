import torch

from models import *
import argparse
import time
import os

import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--chk-dir', type=str, default="chks")
    parser.add_argument('--net', type=str, default="resnet50")

    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--extract-features", default=False, action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Preparing data...")
    # load the pre-trained models
    net = eval(args.net)(pretrained=True, num_classes=10)
    net = nn.DataParallel(net).to('cuda')
    # may use data augmentation to boost the results later
    tfms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ] if not args.extract_features else []
    tfms += [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
    transform_train = transforms.Compose(tfms)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    if args.extract_features:
        fname = "CIFAR10-{}-faetures.pth".format(args.net)
        if not os.path.exists(fname):
            feat_list = []
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=False, num_workers=2)
            net.eval()
            with torch.no_grad():
                for img, label in trainloader:
                    img, label = img.to('cuda'), label.to('cuda')
                    feats = net.module.penultimate(img)
                    feat_list += [(feat.cpu(), lab.item()) for feat, lab in zip(feats, label)]
            torch.save(feat_list, "CIFAR10-{}-faetures.pth".format(args.net))
        else:
            feat_list = torch.load(fname)
        # feat_tensor = torch.stack(feat_list)
        trainloader = torch.utils.data.DataLoader(feat_list, batch_size=args.batchsize, shuffle=True, num_workers=2)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=1000, shuffle=False, num_workers=2)


    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    param_groups = [{'params':[param for name, param in net.named_parameters() if "fc" not in name], 'lr': 0, 'weight_decay': 0},
                    {'params':[param for name, param in net.named_parameters() if "fc" in name], 'lr': args.lr, 'weight_decay': 5e-4}]
    optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # attack with the centroids, and test the results on clean

    ce_loss = nn.CrossEntropyLoss(reduction="mean")
    for epoch in range(args.epochs):
        total_corr = 0
        total_adv_corr = 0
        total_clean_loss = 0
        n_total = 0
        test_total_clean_loss, test_total_corr, test_total = 0, 0, 0
        if epoch in [50, 75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        for nb, (imgs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            imgs, labels = imgs.to('cuda'), labels.to('cuda')
            # pdb.set_trace()
            if args.extract_features:
                logits = net.module.fc(imgs)
                net.eval()
            else:
                net.train()
                logits = net(imgs)

            # pdb.set_trace()
            pred = torch.argmax(logits, 1)

            clean_loss = ce_loss(logits, labels)
            clean_loss.backward()
            optimizer.step()

            total_clean_loss += clean_loss.item() * imgs.size(0)
            n_corr = torch.sum(pred.view(-1) == labels.view(-1))
            total_corr += n_corr

            n_total += imgs.size(0)
            if nb % 50 == 0 or nb == len(trainloader) - 1:
                print("{}, Epoch {}, iter {} Train natural loss/accuracy: {}  {}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), epoch, nb, total_clean_loss/n_total,
                    float(total_corr)/n_total))
        print("\n")

        with torch.no_grad():
            net.eval()
            for nb, (imgs, labels) in enumerate(cifar_testloader):
                imgs, labels = imgs.to('cuda'), labels.to('cuda')
                # pdb.set_trace()
                logits = net(imgs)
                pred = torch.argmax(logits, 1)

                clean_loss = ce_loss(logits, labels)
                # clean_loss.backward()
                # optimizer.step()

                test_total_clean_loss += clean_loss.item() * imgs.size(0)
                n_corr = torch.sum(pred.view(-1) == labels.view(-1))
                test_total_corr += n_corr

                test_total += imgs.size(0)
                if nb % 50 == 0 or nb == len(cifar_testloader) - 1:
                    print("* {}, Epoch {}, iter {} Test natural loss/accuracy: {}  {}".format(
                        time.strftime("%Y-%m-%d %H:%M:%S"), epoch, nb, test_total_clean_loss/test_total,
                        float(test_total_corr)/test_total))
            print("\n")
        torch.save({'net': net.state_dict(), "test_acc": float(test_total_corr)/test_total}, "chks/{}-lr{}-last.pth".format(args.net, args.lr))