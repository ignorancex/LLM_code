import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torchvision import transforms

from models import REGISTRY_MODEL, get_family
from datasets import REGISTRY_INFO


def clip(images, dataset='cifar10'):
    if dataset in ['cifar10', 'cifar100', 'svhn', 'gtsrb']:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset in ['mnist', 'kmnist', 'fashionmnist']:
        mean = [0.1307]
        std = [0.3081]
        
    dtype = images.dtype
    mean_ = torch.as_tensor(mean, dtype=dtype, device=images.device).view(-1, 1, 1)
    std_ = torch.as_tensor(std, dtype=dtype, device=images.device).view(-1, 1, 1)

    images = images.mul_(std_).add_(mean_)
    images = torch.clamp(images, min=0, max=1)
    images = transforms.Normalize(mean=mean, std=std)(images)
    return images


def get_labels(X_sub: torch.Tensor, model: torch.nn.Module, pred_type="soft", device=torch.device("cuda")):
    ds = TensorDataset(X_sub)
    dl = DataLoader(ds, batch_size=128)
    ys = torch.tensor((), device=device)
    model.eval()
    coherence = torch.tensor([], device=device)

    with torch.no_grad():

        for (x,) in dl:
            x = x.to(device)
            if pred_type == "soft":
                y = model(x)
                y = F.softmax(y, dim=-1)
            else:
                y = model(x).argmax(dim=-1)
            try:
                c = model.coherence(x)
                coherence = torch.cat((coherence, c))
            except:
                pass
            ys = torch.cat((ys, y))
    if pred_type == "hard":
        ys = ys.long()
    if coherence.shape[0] > 0:
        print("coherence:{:.3f}".format(coherence.mean().item()))
    return ys


def batch_indices(batch_nb, data_length, batch_size):
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = Variable(x, requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        x_var_exp = x_var.unsqueeze(0)
        score = model(x_var_exp)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(
    dataset, model, X_sub_prev, Y_sub, lmbda=0.1, nb_classes=10, device=torch.device("cuda")
):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    model.eval()
    X_sub = torch.cat((X_sub_prev, X_sub_prev), dim=0)
    if Y_sub.ndim == 2:
        # Labels could be a posterior probability distribution. Use argmax as a proxy.
        Y_sub = torch.argmax(Y_sub, axis=1)

    # For each input in the previous' substitute training iteration
    offset = X_sub_prev.shape[0]
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x, nb_classes)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = torch.sign(torch.tensor(grad, device=device))

        # Create new synthetic point in adversary substitute training set
        X_sub[offset + ind] = x + lmbda * grad_val

    X_sub = clip(X_sub, dataset)

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub


def jacobian_tr_augmentation(
    dataset, model, X_sub_prev, Y_sub, lmbda=0.1, nb_classes=10, fgsm_iter=5, device=torch.device("cuda")
):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    model.eval()
    X_sub = torch.cat((X_sub_prev, X_sub_prev), dim=0)
    if Y_sub.ndim == 2:
        # Labels could be a posterior probability distribution. Use argmax as a proxy.
        Y_sub = torch.argmax(Y_sub, axis=1)

    # For each input in the previous' substitute training iteration
    offset = 0
    for _ in range(1):
        offset += len(X_sub_prev)
        for ind, x in enumerate(X_sub_prev):
            ind_tar = (
                ind + np.random.randint(nb_classes)
            ) % nb_classes  # pick a random target class
            for _ in range(fgsm_iter):
                grads = jacobian(model, x, nb_classes)
                # Select gradient corresponding to the label picked as the target
                grad = grads[ind_tar]

                # Compute sign matrix
                grad_val = torch.sign(torch.tensor(grad, device=device))

                # Create new synthetic point in adversary substitute training set
                x += lmbda * grad_val / fgsm_iter
                x = clip(x, dataset)
            X_sub[offset + ind] = x

    X_sub = clip(X_sub, dataset)

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub


class JBDA:
    def __init__(self, teacher, dataset, model, batch_size=256, num_seed=100, aug_rounds=6, epochs=10, mode="jbda", lmbda=0.1, pred_type="soft", lr_S=0.1, scheduler="multistep", steps=[0.1, 0.3, 0.5], scale=0.3, weight_decay=5e-4, momentum=0.9, device=torch.device("cuda"), **kwargs):
        
        self.num_classes = REGISTRY_INFO[dataset][0]
        self.batch_size = batch_size
        self.device = device

        family = get_family(dataset)
        self.student = REGISTRY_MODEL[family][model](self.num_classes).to(device)
        self.teacher = teacher
        self.dataset = dataset

        self.num_seed = num_seed
        self.aug_rounds = aug_rounds
        self.epochs = epochs
        print(f"\nnum_seed: {self.num_seed}")
        print(f"\naug_rounds: {self.aug_rounds}")
        print("Total number of epochs: ", self.epochs)

        self.opt = optim.SGD(self.student.parameters(), lr=lr_S, weight_decay=weight_decay, momentum=momentum)

        steps = sorted([int(step * self.epochs) for step in steps])
        print("Learning rate scheduling at steps: ", steps)
        print()

        if scheduler == "multistep":
            self.scheduler_S = optim.lr_scheduler.MultiStepLR(self.opt, steps, scale)
        elif scheduler == "cosine":
            self.scheduler_S = optim.lr_scheduler.CosineAnnealingLR(self.opt, self.epochs)

        self.mode = mode
        self.lmbda = lmbda
        self.pred_type = pred_type

    def run(self, test_loader, train_loader, victim_acc):
        acc_list = []

        # Label seed data
        device = self.device

        num_classes = self.num_classes
        data_iter = iter(
            DataLoader(train_loader.dataset, batch_size=self.num_seed, shuffle=False)
        )
        X_sub, _ = next(data_iter)
        X_sub = X_sub.to(device)

        Y_sub = get_labels(X_sub, self.teacher, pred_type=self.pred_type, device=self.device)
        if self.pred_type == "soft":
            criterion = torch.nn.KLDivLoss(reduction="batchmean")
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # Train the substitute and augment dataset alternatively
        self.teacher.eval()
        for aug_round in range(1, self.aug_rounds + 1):
            # model training
            # Indices to shuffle training set
            ds = TensorDataset(X_sub, Y_sub)
            dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            self.student.train()

            for _ in range(self.epochs):
                for x, y in dataloader:
                    self.opt.zero_grad()
                    x, y = x.to(device), y.to(device)
                    Sout = self.student(x)
                    Sout = F.log_softmax(Sout, dim=-1)

                    lossS = criterion(Sout, y)
                    lossS.backward()
                    self.opt.step()

            test_acc = test(self.student, test_loader, self.device)
            acc_list.append(test_acc)

            # If we are not in the last substitute training iteration, augment dataset
            if aug_round < self.aug_rounds:
                print("[{}] Augmenting substitute training data.".format(aug_round))
                # Perform the Jacobian augmentation
                if self.mode == "jbda":
                    X_sub = jacobian_augmentation(
                        self.dataset, self.student, X_sub, Y_sub, nb_classes=num_classes, lmbda=self.lmbda, device=self.device
                    )
                elif self.mode == "jbda-tr":
                    X_sub = jacobian_tr_augmentation(
                        self.dataset, self.student, X_sub, Y_sub, nb_classes=num_classes, lmbda=self.lmbda, device=self.device
                    )

                print("Labeling substitute training data.")
                Y_sub = get_labels(X_sub, self.teacher, pred_type=self.pred_type, device=self.device)
            print(
                "Aug Round {} Clone Accuracy: {:.2f}({:.2f}x)".format(
                    aug_round, test_acc * 100, test_acc / victim_acc
                )
            )

        return acc_list


def test(model, data_loader, device):
    """
    test accuracy
    """
    model = model.to(device)
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == y).sum().item()
        acc = correct / len(data_loader.dataset)
    return acc
