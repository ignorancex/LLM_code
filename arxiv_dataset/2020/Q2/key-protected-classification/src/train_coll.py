import argparse
import random
import numpy
import torch
import pickle
import copy
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from focal_loss import FocalLoss
from participant import Participant
from ops import test_clf, set_learning_rate, weights_init, approximate_key

N_MAX_CLASSES = 50
N_EPOCHS_SAVE = 2


def init_for_mnist(args):
    from networks.clf_mnist import Net
    # the number of actual classes in the dataset
    args.n_classes = 10

    # We can denote a bigger number than the actual number of classes
    # So that we can load a pre-trained checkpoint for when doing vanilla
    # GAN attack
    net = Net(N_MAX_CLASSES, args.d_key, args.fixed_layer).to(args.device)
    net.apply(weights_init)

    optimizer = optim.SGD(
        net.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    optim_scheduler_params = {
        "milestones" : [30],
        "gamma" : 1.0
    }

    train_set = datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]))

    test_loader = DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])),
        batch_size=1000,
        shuffle=False,
        pin_memory=True,
        num_workers=8)

    return net, optimizer, optim_scheduler_params, train_set, test_loader


def init_for_olivetti(args):
    from networks.clf_olivetti import Net
    # the number of actual classes in the dataset
    args.n_classes = 40

    # We can denote a bigger number than the actual number of classes
    # So that we can load a pre-trained checkpoint for when doing vanilla
    # GAN attack
    net = Net(N_MAX_CLASSES, args.d_key, args.fixed_layer).to(args.device)
    net.apply(weights_init)

    optimizer = torch.optim.RMSprop(net.parameters(), 0.0003, eps=1e-5)
    optim_scheduler_params = {
        "milestones" : [50, 100, 200, 300],
        "gamma" : 1.0,
    }

    from olivetti_faces import OlivettiFaces
    train_set = OlivettiFaces(
        '../data',
        "trainval",
        transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(64, padding=(8, 8, 8, 8)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]))

    test_loader = DataLoader(
        OlivettiFaces(
            '../data',
            "test",
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])),
        batch_size=50,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=8)

    return net, optimizer, optim_scheduler_params, train_set, test_loader


def main(args):

    if args.dataset == "mnist":
        net, optimizer, optim_scheduler_params, train_set, test_loader = init_for_mnist(args)
    elif args.dataset == "olivetti":
        net, optimizer, optim_scheduler_params, train_set, test_loader = init_for_olivetti(args)

    print("network of the classification model:", flush=True)
    print(net, flush=True)

    # if this is a key-protected collaborative learning framework:
    if args.key_prot:
        # cosine distance loss
        from ops import cosine_distance_loss
        crit = cosine_distance_loss
    else:
        # cross-entropy loss
        crit = FocalLoss(gamma=2.0).to(args.device)

    ########################################################################################################################
    # create CLF
    ########################################################################################################################

    n_classes_per_part = args.n_classes // args.n_participants
    participants = []
    for i in range(args.n_participants):
        print("participant:{}".format(i), flush=True)

        # partition the training set among participants
        classes_for_part = [c for c in range(i * n_classes_per_part, (i + 1) * n_classes_per_part)]
        print("classes in training set: {}".format(classes_for_part), flush=True)

        data = []
        targets = []
        for c in classes_for_part:
            data.append(train_set.data[train_set.targets == c])
            targets.append(train_set.targets[train_set.targets == c])
        data = torch.cat(data)
        targets = torch.cat(targets)

        train_set_for_part = copy.deepcopy(train_set)
        train_set_for_part.data = data
        train_set_for_part.targets = targets
        print("training set size: {}".format(len(train_set_for_part)), flush=True)

        train_loader_for_part = DataLoader(
            train_set_for_part,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8)

        # in vanilla GAN attack,
        # the attacker can choose which class to attack
        c_attack = -1
        c_fake = -1
        if args.c_attack >= 0 and args.c_attack not in classes_for_part:
            print("this is an attacker, targeting at {}".format(args.c_attack), flush=True)
            c_attack = args.c_attack
            c_fake = args.n_classes

            # but in key-protected classification
            # since there is no "permutation" or "order" of classes
            # i.e. classes are defined by their class keys
            # the attacker cannot choose a class to attack
            # instead, the success of its attack depends only on
            # l2 distances between the random key it generates c_random and all other
            # class keys
            if args.key_prot:

                # c_random = c_attack
                if args.epsilon == "zero":
                    print("the attacker has the key of class {}".format(c_attack))

                # c_random = random class key
                elif args.epsilon == "random":
                    print("the attacker uses a random class key to attack")
                    c_attack = args.n_classes
                    c_fake = args.n_classes + 1

                # epsilon experiment
                else:
                    # find a key for the attacker such that |c_random - c_attack| = args.epsilon
                    epsilon = float(args.epsilon)
                    apprx_key = approximate_key(net.keys[:, c_attack], epsilon)
                    # attack to this generated key
                    c_attack = args.n_classes
                    c_fake = args.n_classes + 1
                    net.keys[:, c_attack] = apprx_key

        p = Participant(i,
                        args,
                        train_loader_for_part,
                        net,
                        crit,
                        optimizer,
                        c_attack,
                        c_fake)
        participants.append(p)

        print("-" * 100, flush=True)

    ########################################################################################################################
    # perform CLF training
    ########################################################################################################################

    exp_loss = {}
    exp_acc = {}

    start_epoch = 1
    if args.checkpoint:
        print("Loading a pre-trained checkpoint from {}".format(args.checkpoint), flush=True)
        ckpt = torch.load(args.checkpoint)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        exp_loss = ckpt["exp_loss"]
        exp_acc = ckpt["exp_acc"]

    print("training starts", flush=True)
    for epoch in range(start_epoch, args.n_epochs + 1):
        print("", flush=True)
        print("*" * 150, flush=True)
        print("epoch: {}/{}".format(epoch, args.n_epochs), flush=True)

        for part in participants:
            print("-" * 100, flush=True)
            print("participant-{} is taking turn".format(part.id), flush=True)
            part.train_epoch(epoch)

        print("-" * 100, flush=True)
        loss, acc = test_clf(args, net, test_loader, crit)
        exp_loss[str(epoch)] = loss
        exp_acc[str(epoch)] = acc

        # update the learning rate
        set_learning_rate(optimizer,
                          optim_scheduler_params["milestones"],
                          optim_scheduler_params["gamma"],
                          epoch)

    # save the network and its optimizer
    torch.save(
        {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "exp_loss": exp_loss,
            "exp_acc": exp_acc,
        },
        "{}/model.pth".format(args.output_dir)
    )

    # save the logs
    with open("{}/logs.pkl".format(args.output_dir), "wb") as fid:
        pickle.dump({"exp_loss": exp_loss, "exp_acc": exp_acc}, fid)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Collaborative learning settings
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--n_participants', type=int, default=2,
                        help='number of participants in the collaborative learning framework')
    parser.add_argument('--key_prot', action="store_true",
                        help='key protection, ie. whether to allow participants to create their own keys')
    parser.add_argument('--d_key', type=int, default=-1,
                        help='dimension of class keys, valid only when key_prot is True')
    parser.add_argument('--fixed_layer', type=int, default=0, choices=[0, 1],
                        help='whether to use fixed layer in classifiers to map features to class keys')
    parser.add_argument('--c_attack', type=int, default=-1)
    parser.add_argument('--epsilon', type=str, default="-1.",
                        help="distance between the keys of c_attack the actualy class")
    parser.add_argument('--n_fake_to_trainset', type=int, default=2000,
                        help='number of fake images added to the training set by attacker')
    parser.add_argument('--n_previous_to_trainset', type=int, default=3,
                        help='')
    parser.add_argument('--n_steps_train_gen', type=int, default=250,
                        help='number of steps to train the local generator for')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--checkpoint', type=str,
                        help='path of a checkpoint for network and optimizer')

    # misc
    parser.add_argument('--output_dir', type=str, default="../output/tmp",
                        help='directory to store the program outputs')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.fixe_layer = args.fixed_layer == 1

    print("program arguments:", flush=True)
    print(args, flush=True)

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
