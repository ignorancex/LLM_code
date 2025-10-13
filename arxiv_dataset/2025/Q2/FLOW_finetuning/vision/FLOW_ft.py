import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
import datetime
import argparse
import random
import numpy as np
import timm
from pathlib import Path
from utils import progress_bar  # Ensure this utility exists or replace with tqdm


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# Dataset configurations
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'dataset': torchvision.datasets.CIFAR10,
        'transform_train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'input_size': 224,
        'uses_train_arg': True,
        'download': True
    },
    'cifar100': {
        'num_classes': 100,
        'dataset': torchvision.datasets.CIFAR100,
        'transform_train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        'input_size': 224,
        'uses_train_arg': True,
        'download': True
    },
    'stanford_dogs': {
        'num_classes': 120,
        'dataset': lambda root, train, transform, download=False: torchvision.datasets.ImageFolder(
            root=os.path.join(root, 'stanford_dogs/Images'),
            transform=transform
        ),
        'transform_train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'input_size': 224,
        'uses_train_arg': False,  # Set to False since we will perform a random split
        'download': False  # ImageFolder does not support download
    },
    'flowers102': {
        'num_classes': 102,
        'dataset': lambda root, train, transform, download=False: (
            torch.utils.data.ConcatDataset([
                torchvision.datasets.Flowers102(
                    root=root,
                    split='train',
                    transform=transform,
                    download=download
                ),
                torchvision.datasets.Flowers102(
                    root=root,
                    split='val',
                    transform=transform,
                    download=download
                )
            ]) if train else
            torchvision.datasets.Flowers102(
                root=root,
                split='test',
                transform=transform,
                download=download
            )
        ),
        'transform_train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'input_size': 224,
        'uses_train_arg': True,  # Set to True to utilize 'train' argument
        'download': True
    },
    'stanford_cars': {
        'num_classes': 196,
        'dataset': lambda root, train, transform, download=False: torchvision.datasets.ImageFolder(
            root=os.path.join(root, 'stanford_cars/train' if train else 'stanford_cars/test'),
            transform=transform
        ),
        'transform_train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'input_size': 224,
        'uses_train_arg': True,  # Set to True since we have separate train/test directories
        'download': False  # ImageFolder does not support download
    },
    'dtd': {
        'num_classes': 47,
        'dataset': torchvision.datasets.DTD,
        'transform_train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'input_size': 224,
        'split_option': 'split_arg',
        'train_value': 'train',
        'test_value': 'test',
        'uses_train_arg': False,  # Set to False if custom splitting is required
        'download': True
    },
    'caltech101': {  # Newly added configuration for Caltech101
        'num_classes': 102,  # Caltech101 has 101 classes + background
        'dataset': lambda root, train, transform, download=False: torchvision.datasets.ImageFolder(
            root=os.path.join(root, 'caltech101_processed', 'train' if train else 'test'),
            transform=transform
        ),
        'transform_train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'input_size': 224,
        'uses_train_arg': True,  # Assuming separate train/test directories
        'download': False  # ImageFolder does not support download
    }
}

# Model configurations
MODEL_CONFIGS = {
    'vit': ['vit_base_patch32_224', 'vit_base_patch16_224', 'vit_large_patch14_224'],
    'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
    'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base'],
    'swin': ['swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224']
}

# Flatten model list for argparse choices
AVAILABLE_MODELS = [model for family in MODEL_CONFIGS.values() for model in family]

# Dataset-specific epochs
EPOCH_CONFIGS = {
    'cifar10': 20,  # 20
    'cifar100': 25,
    'stanford_dogs': 30,
    'flowers102': 25,
    'stanford_cars': 30,
    'dtd': 20,
    'caltech101': 30  # Added epoch count for Caltech101
}


def get_dataloaders(dataset_name, data_root='./data', batch_size=128, num_workers=4):
    """Set up datasets and dataloaders."""
    config = DATASET_CONFIGS[dataset_name]

    # Handle dataset initialization based on whether it uses train argument
    if config['uses_train_arg']:
        # Datasets with separate train/test directories (e.g., stanford_cars, cifar10, cifar100)
        trainset = config['dataset'](
            root=data_root,
            train=True,
            transform=config['transform_train'],
            download=config.get('download', False)
        )
        testset = config['dataset'](
            root=data_root,
            train=False,
            transform=config['transform_test'],
            download=config.get('download', False)
        )
    else:
        # Datasets without separate train/test directories (e.g., stanford_dogs, flowers102)
        # Load the entire dataset and perform a random split
        dataset = config['dataset'](
            root=data_root,
            train=False,  # 'train' argument is irrelevant here
            transform=config['transform_train'],  # Apply training transforms
            download=config.get('download', False)
        )
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return trainloader, testloader, config['num_classes']


def evaluate_imagenet(net, device):
    """Evaluate model on ImageNet validation set"""
    net.eval()
    val_dataset = torchvision.datasets.ImageFolder(
        root='/home/ss95332/src/data/vision/Imagenet/imagenet/val',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(targets.view(1, -1).expand_as(pred_top5)).any(dim=0).sum().item()

            total += targets.size(0)

            progress_bar(batch_idx, len(val_loader),
                         'ImageNet Val: Top-1: %.3f%% | Top-5: %.3f%%'
                         % (100. * correct / total, 100. * correct_top5 / total))

    return 100. * correct / total, 100. * correct_top5 / total


def setup_logger(log_dir, args):
    """Setup logging to both file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'logs_ours_{args.model}_{args.dataset}_lr{args.lr}.txt'

    def log_print(*args_print, **kwargs_print):
        """Custom print function that writes to both console and log file"""
        print(*args_print, **kwargs_print)  # Print to console
        with open(log_file, 'a') as f:
            print(*args_print, file=f, **kwargs_print)  # Print to file

    return log_print


def train(net, lp_model, trainloader, optimizer, criterion, device, epoch, log_print, temp):
    """Train for one epoch."""
    net.train()
    lp_model.eval()  # Ensure lp_model is in evaluation mode
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Step 3: Compute lp_losses using lp_model
        with torch.no_grad():
            lp_outputs = lp_model(inputs)
            lp_losses = criterion(lp_outputs, targets)  # Per-sample losses

        # Compute weights W_i
        weights = torch.exp(-lp_losses / temp)  # temp is a scalar
        outputs = net(inputs)
        losses = criterion(outputs, targets)
        Weighted_loss = torch.mean(weights * losses)

        # loss.backward()
        # optimizer.step()
        # Step 6: Backprop on Weighted_loss
        Weighted_loss.backward()
        optimizer.step()

        train_loss += Weighted_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1),
                        100. * correct / total, correct, total))

    train_acc = 100. * correct / total
    log_print(f'Epoch: {epoch} | Train Loss: {train_loss / (batch_idx + 1):.3f} | Train Acc: {train_acc:.3f}%')
    return train_acc


def test(net, testloader, criterion, device, epoch, best_acc, checkpoint_dir, log_print, args):
    """Evaluate the model and save the best checkpoint."""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Define a separate criterion for testing
    test_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = test_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1),
                            100. * correct / total, correct, total))

    acc = 100. * correct / total
    log_print(f'Epoch: {epoch} | Test Loss: {test_loss / (batch_idx + 1):.3f} | Test Acc: {acc:.3f}%')

    if acc > best_acc:
        log_print('Saving..')

        # Unwrap the model if it's wrapped in DataParallel
        if isinstance(net, torch.nn.DataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
        state = {
            'net': state_dict,
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_dir / f'{args.dataset}/ckpt_best.pth')
        log_print("Saved new best model")
        return acc
    return best_acc


def save_original_parameters(model):
    """Save original parameters of the model"""
    original_params = {
        'fc': {
            'weight': model.fc.weight.data.clone(),
            'bias': model.fc.bias.data.clone()
        },
        'bn': {}
    }

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            original_params['bn'][name] = {
                'weight': module.weight.data.clone(),
                'bias': module.bias.data.clone(),
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone()
            }

    return original_params


def main():
    parser = argparse.ArgumentParser(description='Linear Probing with Multiple Datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='dataset to use for training')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=AVAILABLE_MODELS,
                        help='model architecture (default: resnet18)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--data-root', default='/home/ss95332/src/data/vision', type=str,
                        help='path to dataset root')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--temp', default=1.0, type=float, help='temperature for weighting losses')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--checkpoint-dir', default='./checkpoints/ours',
                        type=str, help='checkpoint directory')
    parser.add_argument('--lp-dir', default='./checkpoints/linear_probing',
                        type=str, help='checkpoint directory')
    parser.add_argument('--log-dir', default='./logs/ours',
                        type=str, help='log directory')
    args = parser.parse_args()
    args = parser.parse_args()

    epochs = EPOCH_CONFIGS[args.dataset]

    # Setup logging
    log_dir = Path(args.log_dir) / f'{args.model}/{args.dataset}'
    log_print = setup_logger(log_dir, args)

    print("\n=== Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Temp: {args.temp}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Seed: {args.seed}")
    print("===================\n")

    # Log configuration
    log_print("\n=== Configuration ===")
    log_print(f"Dataset: {args.dataset}")
    log_print(f"Model: {args.model}")
    log_print(f"Epochs: {epochs}")
    log_print(f"Learning Rate: {args.lr}")
    log_print(f"Temp: {args.temp}")
    log_print(f"Batch Size: {args.batch_size}")
    log_print(f"Workers: {args.workers}")
    log_print(f"Seed: {args.seed}")
    log_print("===================\n")

    # Setup
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = Path(args.checkpoint_dir)
    lp_dir = Path(args.lp_dir)

    # Get pretrained model and evaluate on ImageNet
    pretrained_model = timm.create_model(args.model, pretrained=True)
    pretrained_model = pretrained_model.to(device)
    if device == 'cuda':
        pretrained_model = torch.nn.DataParallel(pretrained_model)

    print(f"\n==> Evaluating pretrained {args.model} on ImageNet...")
    top1_acc, top5_acc = evaluate_imagenet(pretrained_model, device)
    print(f"Pretrained Model ImageNet - Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")

    # Save original parameters
    if isinstance(pretrained_model, torch.nn.DataParallel):
        original_params = save_original_parameters(pretrained_model.module)
    else:
        original_params = save_original_parameters(pretrained_model)

    # Create model for target dataset
    num_classes = DATASET_CONFIGS[args.dataset]['num_classes']
    net = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes
    )
    # net = nn.DataParallel(net)

    # 2) Load checkpoint from step 1
    lp_path = lp_dir / f'{args.model}/{args.dataset}/ckpt_linear_best.pth'
    lp_checkpoint = torch.load(lp_path)
    print(f"Loading linear-probe checkpoint from {lp_path}")

    # 3) (Option A) Load entire checkpoint, ignoring mismatch for final layer
    net.load_state_dict(lp_checkpoint['net'], strict=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    # cudnn.benchmark = True

    # Load lp_model (linear probe model)
    lp_model = timm.create_model(args.model, pretrained=False, num_classes=DATASET_CONFIGS[args.dataset]['num_classes'])
    lp_path = lp_dir / f'{args.model}/{args.dataset}/ckpt_linear_best.pth'
    lp_checkpoint = torch.load(lp_path)
    print(f"Loading linear-probe checkpoint in {lp_path}")
    # lp_model.load_state_dict(lp_checkpoint['net'])
    lp_model = lp_model.to(device)
    # if device == 'cuda':
    #     lp_model = torch.nn.DataParallel(lp_model)
    lp_model.load_state_dict(lp_checkpoint['net'], strict=True)
    lp_model.eval()

    # Get dataloaders
    print(f'\n==> Preparing {args.dataset} dataset')
    trainloader, testloader, num_classes = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers
    )
    print(f"Number of classes: {num_classes}")

    # **New Code: Print and Log Train/Test Set Sizes**
    train_size = len(trainloader.dataset)
    test_size = len(testloader.dataset)
    print(f"Train set size: {train_size}")
    print(f"Test set size: {test_size}")

    log_print(f"Train set size: {train_size}")
    log_print(f"Test set size: {test_size}")

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    log_print(f'\n==> Training {args.model} on {args.dataset}')
    best_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        train_acc = train(net, lp_model, trainloader, optimizer, criterion, device, epoch + 1, log_print, args.temp)
        best_acc = test(net, testloader, criterion, device, epoch + 1,
                        best_acc, checkpoint_dir, log_print, args)
        scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log_print(f'Training time: {total_time_str}')
    log_print(f'Best accuracy: {best_acc}%')

    # Load best checkpoint for ImageNet evaluation
    log_print("Evaluating best checkpoint on ImageNet...")
    checkpoint_path = checkpoint_dir / f'{args.dataset}/ckpt_best.pth'
    if not checkpoint_path.exists():
        log_print(f"Checkpoint {checkpoint_path} does not exist. Skipping ImageNet evaluation.")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create new model for ImageNet evaluation (1000 classes)
        imagenet_model = timm.create_model(args.model, pretrained=False, num_classes=1000).to(device)
        if device == 'cuda':
            imagenet_model = torch.nn.DataParallel(imagenet_model)

        checkpoint_dict = checkpoint['net']
        imagenet_dict = imagenet_model.state_dict()

        # Copy all layers except fc and BatchNorm from checkpoint
        for k in checkpoint_dict.keys():
            if not ('fc' in k or 'head' in k or 'bn' in k or 'downsample.1' in k):
                if k in imagenet_dict:
                    imagenet_dict[k] = checkpoint_dict[k]
                elif f'module.{k}' in imagenet_dict:
                    imagenet_dict[f'module.{k}'] = checkpoint_dict[k]

        # Restore original ImageNet parameters
        if isinstance(imagenet_model, torch.nn.DataParallel):
            # Restore fc weights
            imagenet_dict['module.fc.weight'] = original_params['fc']['weight'].to(device)
            imagenet_dict['module.fc.bias'] = original_params['fc']['bias'].to(device)

            # Restore BatchNorm parameters
            for name, params in original_params['bn'].items():
                imagenet_dict[f'module.{name}.weight'] = params['weight'].to(device)
                imagenet_dict[f'module.{name}.bias'] = params['bias'].to(device)
                imagenet_dict[f'module.{name}.running_mean'] = params['running_mean'].to(device)
                imagenet_dict[f'module.{name}.running_var'] = params['running_var'].to(device)
        else:
            # Restore fc weights
            imagenet_dict['fc.weight'] = original_params['fc']['weight']
            imagenet_dict['fc.bias'] = original_params['fc']['bias']

            # Restore BatchNorm parameters
            for name, params in original_params['bn'].items():
                imagenet_dict[f'{name}.weight'] = params['weight']
                imagenet_dict[f'{name}.bias'] = params['bias']
                imagenet_dict[f'{name}.running_mean'] = params['running_mean']
                imagenet_dict[f'{name}.running_var'] = params['running_var']

        imagenet_model.load_state_dict(imagenet_dict)

        # Evaluate on ImageNet
        top1_acc, top5_acc = evaluate_imagenet(imagenet_model, device)
        log_print("\n=== Final Results ===")
        log_print(f"Dataset: {args.dataset}")
        log_print(f"Temp: {args.temp}")
        log_print(f"Model: {args.model}")
        log_print(f"Best accuracy on {args.dataset}: {best_acc:.2f}%")
        log_print(f"ImageNet accuracy after {args.dataset} linear-probe:")
        log_print(f"Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")
        log_print("===================")
        print("\n=== Final Results ===")
        print(f"Dataset: {args.dataset}")
        print(f"Temp: {args.temp}")
        print(f"Model: {args.model}")
        print(f"Best accuracy on {args.dataset}: {best_acc:.2f}%")
        print(f"ImageNet accuracy after {args.dataset} ours finetuning:")
        print(f"Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")
        print("===================")


if __name__ == '__main__':
    main()
