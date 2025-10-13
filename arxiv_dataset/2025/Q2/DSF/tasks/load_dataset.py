



def load_dataset(config):
    print("Loading dataset...")

    if config.dataset.origin == 'huggingface':
        from tasks.huggingface_dataset import load_huggingface_dataset
        
        train_loader, valid_loader, info = load_huggingface_dataset(config)

    elif config.dataset.name == 'ptb':
        from tasks.ptb import get_ptb_dataloaders
        
        train_loader, valid_loader, test_loader, info = get_ptb_dataloaders(config)

    elif config.dataset.name == 'cifar10':
        from tasks.cifar10 import get_cifar10_dataloaders

        train_loader, valid_loader, info = get_cifar10_dataloaders(config)

    else:
        raise ValueError(f"Invalid dataset with origin {config.dataset.origin} and name {config.dataset.name}")
    
    return train_loader, valid_loader, info
