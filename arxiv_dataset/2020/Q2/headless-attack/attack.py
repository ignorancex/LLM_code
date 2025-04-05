import torch

from models import *
import argparse
import time
import os

import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from dataloader import SubsetOfList
from utils import *

def compute_or_load_centroids(dataloader, net, n_variants=1, gauss_std=0, out_name="centroids/init.pth", overwrite=False):
    if os.path.exists(out_name) and (not overwrite):
        return torch.load(out_name)

    # get the centroid of each class
    centroid_dict = [0 for _ in range(10)]
    count_dict = [0 for _ in range(10)]
    with torch.no_grad():
        for nv in range(n_variants):
            for nb, (input, label) in enumerate(dataloader):
                input, label = input.to('cuda'), label.to('cuda')
                if gauss_std > 0:
                    perturb = torch.zeros_like(input).normal_(0, gauss_std)
                else:
                    perturb = 0
                pen_feats = net.module.penultimate(input + perturb)
                for feat, lab in zip(pen_feats, label):
                    centroid_dict[lab.item()] += feat
                    count_dict[lab.item()] += 1
    for n in range(len(centroid_dict)):
        centroid_dict[n] /= count_dict[n]

    torch.save(centroid_dict, out_name)
    return centroid_dict

def pgd_centroid_attack(net, img_batch, label_batch, centroid_tensor, mean_tensor, std_tensor,
                        pgd_lr, pgd_steps, eps):
    perturb = torch.zeros_like(img_batch).uniform_(-eps / 0.203, eps / 0.203).to('cuda')
    perturb.requires_grad_()
    img_01 = img_batch * std_tensor + mean_tensor
    for step in range(pgd_steps):
        # batch_size x 512
        pen_feats = net.module.penultimate(img_batch + perturb)
        # compute the distance
        class_dists = torch.sqrt(torch.sum((pen_feats.unsqueeze(1) - centroid_tensor.unsqueeze(0))**2, 2))
        # class_dists.scatter_(1, label_batch.unsqueeze(1), 1e10)
        # min_targets, _ = torch.min(class_dists, 1)
        # sum_loss = torch.sum(min_targets)
        # sum_loss.backward()
        loss = nn.CrossEntropyLoss()(class_dists, label_batch)
        loss.backward()
        perturb.data -= pgd_lr * torch.sign(perturb.grad)
        # clip the range of perturbation
        perturb_01 = torch.clamp(perturb.data * std_tensor, -eps, eps)
        perturbed_01 = torch.clamp(img_01 + perturb_01, 0, 1)
        perturb.data = (perturbed_01 - img_01) / std_tensor

        net.zero_grad()
        perturb.grad[:] = 0

    return img_batch + perturb

def pgd_attack(net, img_batch, label_batch, mean_tensor, std_tensor,
                        pgd_lr, pgd_steps, eps):
    perturb = torch.zeros_like(img_batch).uniform_(-eps / 0.203, eps / 0.203).to('cuda')
    perturb.requires_grad_()
    img_01 = img_batch * std_tensor + mean_tensor
    ce_loss = nn.CrossEntropyLoss(reduction="sum")
    for step in range(pgd_steps):
        # batch_size x 512
        logits = net(img_batch + perturb)
        loss = ce_loss(logits, label_batch)
        loss.backward()
        perturb.data += pgd_lr * torch.sign(perturb.grad)
        # clip the range of perturbation
        perturb_01 = torch.clamp(perturb.data * std_tensor, -eps, eps)
        perturbed_01 = torch.clamp(img_01 + perturb_01, 0, 1)
        perturb.data = (perturbed_01 - img_01) / std_tensor

        net.zero_grad()
        perturb.grad[:] = 0

    return img_batch + perturb


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sidx', default=0, type=int, help='starting id of each class')
    parser.add_argument('--eidx', default=4800, type=int, help='ending id of each class')
    parser.add_argument('--train-dp', type=float, default=0)
    parser.add_argument('--droplayer', type=float, default=0)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--model-resume-path', default='model-chks', type=str,
                        help="Path to the pre-trained models")
    parser.add_argument('--subs-net', default="ResNet18", type=str, help='starting id of each class')
    parser.add_argument("--subs-chk-name", default='ckpt-%s-4800.t7', type=str)
    parser.add_argument("--target-net", default='ResNet18', type=str)
    parser.add_argument("--target-chk-name", default='ckpt-%s-4800.t7', type=str)
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--centroid-out-name", default="centroids/init.pth", type=str)

    # parameters for the attack
    parser.add_argument("--eps", default=8, type=float)
    parser.add_argument("--pgd-lr", default=0.05, type=float)
    parser.add_argument("--pgd-steps", default=20, type=int)

    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--pgd", default=False, action="store_true")
    parser.add_argument("--seed", default=1234, type=int)

    parser.add_argument("--n-variants", default=1, type=int)
    parser.add_argument("--gauss-std", default=0, type=float)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.eps = args.eps / 255.

    print("Preparing data...")
    # may use data augmentation to boost the results later
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    mean_tensor = torch.Tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to('cuda')
    std_tensor = torch.Tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to('cuda')
    # mean_tensor = torch.Tensor([0,0,0]).view(1, 3, 1, 1).to('cuda')
    # std_tensor = torch.Tensor([1,1,1]).view(1, 3, 1, 1).to('cuda')

    train_list = torch.load('./datasets/CIFAR10_TRAIN_Split.pth')['clean_train']
    trainset = SubsetOfList(train_list, transform=transform_train, start_idx=args.sidx, end_idx=args.eidx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=1000, shuffle=False, num_workers=2)

    # load the pre-trained models
    subs_net = load_pretrained_net(args.subs_net, args.subs_chk_name, model_chk_path=args.model_resume_path)
    subs_net = subs_net.to('cuda')

    target_net = load_pretrained_net(args.target_net, args.target_chk_name, model_chk_path=args.model_resume_path)
    target_net = target_net.to('cuda')
    subs_net.eval()
    target_net.eval()

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    centroids = compute_or_load_centroids(trainloader, subs_net, n_variants=args.n_variants, gauss_std=args.gauss_std, out_name=args.centroid_out_name, overwrite=args.overwrite)

    # attack with the centroids, and test the results on clean
    total_corr = 0
    total_adv_corr = 0
    total_clean_loss, total_adv_loss = 0, 0
    n_total = 0
    centroid_tensor = torch.stack(centroids)
    ce_loss = nn.CrossEntropyLoss(reduction="sum")
    for nb, (imgs, labels) in enumerate(cifar_testloader):
        imgs, labels = imgs.to('cuda'), labels.to('cuda')
        if args.pgd:
            adv_img = pgd_attack(subs_net, imgs, labels, mean_tensor, std_tensor,
                                          args.pgd_lr, args.pgd_steps, args.eps)
            # pdb.set_trace()
        else:
            adv_img = pgd_centroid_attack(subs_net, imgs, labels, centroid_tensor, mean_tensor, std_tensor,
                            args.pgd_lr, args.pgd_steps, args.eps)
        # pdb.set_trace()
        with torch.no_grad():
            logits = target_net(imgs)
            pred = torch.argmax(logits, 1)
            clean_loss = ce_loss(logits, labels)
            total_clean_loss += clean_loss.item()
            n_corr = torch.sum(pred.view(-1) == labels.view(-1))
            total_corr += n_corr

            logits = target_net(adv_img)
            pred = torch.argmax(logits, 1)
            adv_loss = ce_loss(logits, labels)
            total_adv_loss += adv_loss.item()
            n_corr = torch.sum(pred.view(-1) == labels.view(-1))
            total_adv_corr += n_corr
        n_total += imgs.size(0)
        if nb % 10 == 0 or nb == len(cifar_testloader) - 1:
            print("{}, natural loss/accuracy: {}  {}, adv loss/accuracy: {} {}".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), total_clean_loss/n_total,
                float(total_corr)/n_total, total_adv_loss/n_total, float(total_adv_corr)/n_total))
