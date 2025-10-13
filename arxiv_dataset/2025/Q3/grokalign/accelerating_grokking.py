import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import wandb as wb
from tqdm import tqdm

from utils import GrokAlign, gradfilter_ema


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_mnist_loaders(train_points, test_points, batch_size, data_dir='./data'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: x.flatten())
    ])
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_subset = torch.utils.data.Subset(train_set, range(train_points))
    test_subset = torch.utils.data.Subset(test_set, range(test_points))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return {'train': train_loader, 'test': test_loader}


def build_model(config, device):
    layers = [nn.Linear(784, config.width, config.bias), nn.ReLU()]
    for _ in range(config.depth - 2):
        layers += [nn.Linear(config.width, config.width, config.bias), nn.ReLU()]
    layers += [nn.Linear(config.width, 10, config.bias)]

    model = nn.Sequential(*layers).to(device)
    with torch.no_grad():
        for p in model.parameters():
            p.data *= config.init_scale
    return model


def compute_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += x.size(0)
    return correct / total


def train(config, device='cuda', data_dir='./data', dtype=torch.float64):
    run_name = f"{config.loss_fn}-{'GF' if config.grokfast else ''}{'GA' if config.lambda_jac > 0 else ''}{'Wd' if config.weight_decay > 0 else ''}{'At' if config.adv_training else ''}-{config.seed}"
    run = wb.init(project='accelerating_grokking', config=vars(config), name=run_name)

    torch.set_default_dtype(dtype)
    set_seed(config.seed)

    loaders = get_mnist_loaders(config.train_points, config.test_points, config.batch_size, data_dir)
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    grokalign = GrokAlign(model) if config.lambda_jac > 0.0 else None
    one_hots = torch.eye(10, device=device)
    grads = None

    total_time = 0
    pbar = tqdm(range(config.steps + 1))
    for step in pbar:
        model.train()
        start_time = time.time()

        if config.adv_training:
            train_acc = compute_accuracy(model, loaders['train'], device)
            sigma = max(0.06 * (1 - train_acc), 0.03)

        for x, labels in loaders['train']:
            x, labels = x.to(device), labels.to(device)
            if config.adv_training:
                x += torch.randn_like(x) * sigma

            optimizer.zero_grad()
            outputs = model(x)

            if config.loss_fn == 'CrossEntropy':
                loss = nn.CrossEntropyLoss()(outputs, labels)
            elif config.loss_fn == 'MSE':
                loss = nn.MSELoss()(outputs, one_hots[labels])
            else:
                raise ValueError(f"Unsupported loss function: {config.loss_fn}")

            if grokalign:
                loss += config.lambda_jac * grokalign(x)

            loss.backward()
            if config.grokfast:
                grads = gradfilter_ema(model, grads=grads, alpha=0.8, lamb=0.1)
            optimizer.step()

        total_time += time.time() - start_time
        pbar.set_description(f"{loss.item():.4f}")

        if step % 10 == 0:
            stats = {
                'step': step,
                'loss': loss.item(),
                'optimization_time': total_time
            }
            for split in ['train', 'test']:
                stats[f'{split}_accuracy'] = compute_accuracy(model, loaders[split], device)

            wb.log(stats)

            if stats['test_accuracy'] > 0.85:
                break

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_points', type=int, default=1024)
    parser.add_argument('--test_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=100_000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lambda_jac', type=float, default=0.0)
    parser.add_argument('--grokfast', action='store_true')
    parser.add_argument('--adv_training', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--init_scale', type=float, default=8.0)
    parser.add_argument('--width', type=int, default=196)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--loss_fn', type=str, choices=['MSE', 'CrossEntropy'], default='MSE')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    train(args, device=args.device)
