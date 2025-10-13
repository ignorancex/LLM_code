from utils import seed_everything, get_exp_name
from models import build_model
import torch
from utils.cfg import parse_args
from dataset import get_loader
from torch.utils.tensorboard import SummaryWriter
from models.function import train_model, test_model
import os
import time


args = parse_args()
torch.cuda.set_device(args.gpu_device)
seed_everything(2024)


def main():
    model = build_model(args)
    exp_name = get_exp_name()
    log_dir = os.path.join(args.log_dir, args.dataset, time.strftime('%Y%m%d_%H%M%S') + '_' + exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, test_loader = get_loader(args)
    max_dice = 0
    
    for epoch in range(args.epoch):
        train_model(model, optimizer, train_loader, epoch, writer)
        if not epoch % args.val_freq:
            dice = test_model(model, test_loader, epoch, writer)
            if dice > max_dice:
                max_dice = dice
                torch.save({'model': model.state_dict()}, os.path.join(log_dir, f'best_vvs_in_{args.dataset}.pth'))
    writer.close()
    
    print(f'max dice: {max_dice}')


if __name__ == '__main__':
    main()