"""Train CIFAR10 with PyTorch."""
import argparse
import copy
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cleverhans.torch.attacks.projected_gradient_descent import \
    projected_gradient_descent
# import wandb
from models import *
from models.aaron import Aaron
from utils import progress_bar

if not os.path.exists("./logs"):
    os.makedirs("./logs")

mylog = datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y.log')
mylogfile = os.path.join('./logs', mylog)

# get a random seed
seed = random.randint(1, 10000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--decay_rate", default=0.1, type=float, help="decay_rate")
parser.add_argument("--epochs", default=200, type=int, help="# epochs")
parser.add_argument("--wig", help="weight on ig")
parser.add_argument("--ensemble", default=10, type=int, help="# of particles")
parser.add_argument("--root",
                    default="~/data",
                    type=str,
                    help="root of dataset")
parser.add_argument("--checkpoint",
                    default="~/data",
                    type=str,
                    help="root of dataset")
parser.add_argument("--decay_epoch1",
                    default=60,
                    type=int,
                    help="weight on ig")
parser.add_argument("--decay_epoch2",
                    default=90,
                    type=int,
                    help="weight on ig")
parser.add_argument("--wd",
                    default=0.0,
                    type=float,
                    help="weight decay for optimizer")
parser.add_argument("--batch_size",
                    default=512,
                    type=int,
                    help="batch size to train")
parser.add_argument("--dataset",
                    default="cifar",
                    choices=['cifar', 'stl'],
                    help="dataset to run")
parser.add_argument("--optim",
                    default="adam",
                    type=str,
                    help="optimizer to run")

parser.add_argument("--resume",
                    "-r",
                    action="store_true",
                    help="resume from checkpoint")
args = parser.parse_args()

logger = logging.getLogger(__name__)

# create the checkpoint dir

checkpoint_dir = args.checkpoint
epochs = args.epochs
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)



device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
best_adv_acc = 0  # best test adv accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


logging.basicConfig(
    format="[%(asctime)s] - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
    filemode="w",
    filename=mylogfile,
)

logger.info(args)

logger.info("Random seed for this training is {}".format(seed))

if args.dataset == "cifar":
    NET = VGG("VGG16")
    NET = torch.nn.DataParallel(NET)
elif args.dataset == "stl":
    NET = Aaron(10)
    NET = torch.nn.DataParallel(NET)


from bayes import BayesWrap

class Object(object):
    pass


opt = Object()
opt.num_particles = args.ensemble
net = BayesWrap(opt, NET)

net = net.to(device)

# Data
print("==> Preparing data..")
logging.info("==> Preparing data..")

if args.dataset == "cifar":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
elif args.dataset == "stl":
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])


if args.dataset == "cifar":

    trainset = torchvision.datasets.CIFAR10(root=args.root,
                                            train=True,
                                            download=True,
                                            transform=transform_train)

    # split the trainset into train and val
    trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=8)

    valloader = torch.utils.data.DataLoader(valset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=8)

    # evaluate on PGD is heavier, so we use a smaller batch size
    valloader_PGD = torch.utils.data.DataLoader(valset,
                                                 batch_size=int(
                                                     args.batch_size / 2),
                                                 shuffle=False,
                                                 num_workers=8)

elif args.dataset == "stl":
    trainset = torchvision.datasets.STL10(root=args.root,
                                          split="train",
                                          download=True,
                                          transform=transform_train)

    # split the trainset into train and val
    trainset, valset = torch.utils.data.random_split(trainset, [4000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=8)

    valloader = torch.utils.data.DataLoader(valset,
                                             batch_size=50,
                                             shuffle=False,
                                             num_workers=8)
    valloader_PGD = torch.utils.data.DataLoader(valset,
                                                 batch_size=int(
                                                     args.batch_size / 2),
                                                 shuffle=False,
                                                 num_workers=8)

# Model
print("==> Building model..")
logging.info("==> Building model..")

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    logging.info("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint_dir = args.checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, 'ckpt_adv_best.pth')
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    net.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()

if args.optim == "sgd":
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                               'min', patience=6)
elif args.optim == "adam":
    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)


def PGD(test_loader, epoch, epsilon=0.1, min_val=0, max_val=1):
    """Function to evaluate the model on adverserial examples generated using PGD"""
    global best_adv_acc
    correct = 0
    adv_correct = 0
    misclassified = 0
    total = 0
    net.eval()
    test_rob_losses = []
    test_ben_losses = []
    test_gaps = []
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)


        images_adv = projected_gradient_descent(net, images, 8/255, 2/255, 10, np.inf)

        images_adv = images_adv.to(device)
        outputs = net(images)
        ben_loss = criterion(outputs, labels)
        test_ben_losses.append(ben_loss.item())

        adv_output = net(
            images_adv)  # output by the model after adding adverserial noise

        # Prediction on the clean image
        _, predicted = torch.max(outputs.data, 1)
        _, adv_predicted = torch.max(
            adv_output.data,
            1)  # Prediction on the image after adding adverserial noise

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        adv_correct += (adv_predicted == labels).sum().item()
        misclassified += (predicted != adv_predicted).sum().item()

        progress_bar(
            i,
            len(valloader),
            "Acc: %.3f%% (%d/%d) | Acc_adv: %.3f%% (%d/%d)" % (
                100.0 * correct / total,
                correct,
                total,
                100.0 * adv_correct / total,
                adv_correct,
                total,
            ),
        )

    adv_acc = 100 * adv_correct / total
    ben_acc = 100. * correct / total


    if adv_acc > best_adv_acc:
        print("Saving adv model..")
        logging.info("Saving adv model..")

        state = {
            "net": net.state_dict(),
            "acc": adv_acc,
            "optim": optimizer.state_dict(),
            "epoch": epoch,
        }
        best_adv_file = os.path.join(checkpoint_dir, "ckpt_adv_best.pth")
        torch.save(
            state,
            best_adv_file)

        best_adv_acc = adv_acc


    logging.info(
        "Accuracy of the model w/0 adverserial attack on test images is : {} %"
        .format(100 * correct / total))
    print(
        "Accuracy of the model w/0 adverserial attack on test images is : {} %"
        .format(100 * correct / total))
    logging.info(
        "Accuracy of the model with adverserial attack on test images is : {} %"
        .format(100 * adv_correct / total))
    print(
        "Accuracy of the model with adverserial attack on test images is : {} %"
        .format(100 * adv_correct / total))
    logging.info(
        "Number of misclassified examples(as compared to clean predictions): {}/{}"
        .format(misclassified, total))
    print(
        "Number of misclassified examples(as compared to clean predictions): {}/{}"
        .format(misclassified, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    kwargs = {"return_entropy": False}  # no entropy required for this
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, **kwargs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(valloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    logging.info("The accuracy on val set is {}".format(acc))
    if acc > best_acc:
        print("Saving..")
        logging.info("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "optim": optimizer.state_dict(),
            "epoch": epoch,
        }
        best_file = os.path.join(checkpoint_dir, "ckpt_best.pth")
        torch.save(
            state,
            best_file)
        best_acc = acc


# Training


def train_IG(epochs=200):
    losses, infgain = [], []
    train_ben_losses, train_rob_losses = [], []
    for epoch in range(start_epoch + 1,
                       epochs):  # loop over the dataset multiple times
        # update learning rate
        if args.optim == "sgd":
            if epoch < args.decay_epoch1:
                lr = args.lr
            elif epoch < args.decay_epoch2:
                lr = args.lr * args.decay_rate
            else:
                lr = args.lr * args.decay_rate * args.decay_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        #
        running_loss = 0.0
        logging.info("Epoch: {}".format(epoch))
        particle_ce = 0
        correct1 = 0
        correct2 = 0
        total = 0
        ig_benign = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            inputs = images.to(device)
            labels = labels.to(device)

            inputs_adv = projected_gradient_descent(
                net, inputs, 8 / 255, 2 / 255, 10, np.inf)


            #----------------------------------------------------------------------------------------------------
            # Calculate InfoGain

            kwargs = {"return_entropy": True}
            outputs, entropies = net(inputs, **kwargs)
            p_benign = torch.softmax(outputs, 1)  # expected output
            h_benign = (-p_benign * torch.log(p_benign + 1e-8)).sum(
                1)  # entropy of expected output
            _, preds1 = outputs.max(1)
            correct1 += preds1.eq(labels).sum().item()
            loss1 = net.get_losses(inputs, labels, criterion, **kwargs)  # benign loss
            train_ben_losses.append(loss1.item())

            outputs_adv, entropies_adv = net(inputs_adv, **kwargs)
            p_adv = torch.softmax(outputs_adv, 1)  # expected output
            h_adv = (-p_adv * torch.log(p_adv + 1e-8)).sum(
                1)  # entropy of expected output

            _, preds2 = outputs_adv.max(1)
            loss_adv = net.get_losses(inputs_adv, labels, criterion, **kwargs)  # adversarial loss
            train_rob_losses.append(loss_adv.item())

            total += labels.size(0)


            #----------------------------------------------------------------------------------------------------
            # calc IG loss
            loss_IG = torch.abs((h_adv - entropies_adv) -
                              (h_benign - entropies)).mean(0)  # IG loss
            

            # get the hyper-parameter to control IG
            wig = float(args.wig)

            loss = (loss_adv + wig * loss_IG)  # baseline ens + ig

            losses.append(loss1.item())
            infgain.append(loss_IG.item())

            optimizer.zero_grad()
            loss.backward()
            # SVGD update
            net.update_grads()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            progress_bar(
                i,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)" %
                (
                    running_loss / (i + 1),
                    100.0 * correct1 / total,
                    correct1,
                    total,

                ),
            )

        if args.optim == "sgd":
            scheduler.step(loss_adv)
        # write the training log
        ig_loss = sum(infgain) / float(len(infgain))
        ce_train_loss = sum(losses) / float(len(losses))
        ce_train_rob_loss = sum(train_rob_losses) / float(
            len(train_rob_losses))
        test(epoch)
        if epoch % 5 == 0:
            PGD(valloader_PGD, epoch)


if __name__ == "__main__":
    train_IG(epochs)