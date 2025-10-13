from .dataset import Vein, SUN_SEG
from .function import *
from torch.utils.data import DataLoader


def get_loader(args):
    if args.dataset == 'vein':
        train_dataset = Vein(args, transform=MVA_aug())
        test_dataset = Vein(args, mode='val')
    elif args.dataset == 'sun-seg':
        train_dataset = SUN_SEG(args, transform=MVA_aug())
        test_dataset = SUN_SEG(args, mode='test')
    else:
        raise ValueError(f'Dataset {args.dataset} not found')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=VideoShuffleSampler(train_dataset), drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    return train_loader, test_loader