import torch
from util.torch_dist_sum import *
from data.dataloader import *
from data.transform_ovlp import CustomDataAugmentation
import torch.nn as nn
from util.meter import *
from network.wcl import WCL
import time
import math
from util.LARS import LARS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size-pergpu', type=int, default=50)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument("--image_size", default=224, type=int)
parser.add_argument("--min_scale", default=0.2, type=float)
parser.add_argument("--patch_size", default=16, type=int)
parser.add_argument("--save_path", default="./checkpoints/ckp_dino/", type=str)
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument("--save_epoch", default=20, type=int)
parser.add_argument("--warm_up", default=10, type=int)
parser.add_argument("--pretrained_path", default="dino_deitsmall16_pretrain.pth", type=str)
args = parser.parse_args()
print(args)

epochs = args.epochs
warm_up = args.warm_up

def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, device, criterion, optimizer, epoch, iteration_per_epoch, base_lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    graph_losses = AverageMeter('graph', ':.4e')

    # switch to train mode
    model.train()

    end = time.time()
    for i, pack in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        data_time.update(time.time() - end)
        try:
            crops, coords, flags = pack
        except:
            print("ERROR and WHY????")
        img1 = crops[0]
        img2 = crops[1]

        img1 = img1.cuda()
        img2 = img2.cuda()

        # compute output
        output, target, graph_loss, loss_ovlp = model(img1, img2, coords, flags)
        ce_loss = criterion(output, target)
        
        losses = loss_ovlp + graph_loss + ce_loss

        graph_losses.update(graph_loss.item(), img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        rank = torch.distributed.get_rank()
        if i % 50 == 0 and rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: {epoch} | Iter: {i} | loss: {losses} | graph: {graph_loss} | ce: {ce_loss} | ovlp: {loss_ovlp} | lr: {lr} ")


def main():
    from torch.nn.parallel import DistributedDataParallel
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    batch_size = args.batch_size_pergpu
    num_workers = 8
    base_lr = 0.0075

    model = WCL(args.pretrained_path, device, args.patch_size).cuda()
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)
    
    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, 'ignore': True },
                                {'params': rest_params, 'weight_decay': 1e-6, 'ignore': False}], 
                                lr=base_lr, momentum=0.9, weight_decay=1e-6)

    optimizer = LARS(optimizer, eps=0.0)
    
    torch.backends.cudnn.benchmark = True

    transform = CustomDataAugmentation(args.image_size, args.min_scale)
    weak_aug_train_dataset = ImagenetContrastive(aug=transform, max_class=1000)
    weak_aug_train_sampler = torch.utils.data.distributed.DistributedSampler(weak_aug_train_dataset)
    weak_aug_train_loader = torch.utils.data.DataLoader(
        weak_aug_train_dataset, batch_size=batch_size, shuffle=(weak_aug_train_sampler is None),
        num_workers=num_workers, pin_memory=False, sampler=weak_aug_train_sampler, drop_last=True)

    train_dataset = ImagenetContrastive(aug=transform, max_class=1000)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=False, sampler=train_sampler, drop_last=True)
    
    iteration_per_epoch = train_loader.__len__()
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    
    model.train()
    for epoch in range(start_epoch, epochs):
        
        if epoch < warm_up:
            weak_aug_train_sampler.set_epoch(epoch)
            train(weak_aug_train_loader, model, device, criterion, optimizer, epoch, iteration_per_epoch, base_lr)
        else:
            train_sampler.set_epoch(epoch)
            train(train_loader, model, device, criterion, optimizer, epoch, iteration_per_epoch, base_lr)
        
        os.makedirs(args.save_path, exist_ok=True)
        checkpoint_path = args.save_path + '/wcl-16-{}.pth'.format(epoch+1)

        if (epoch+1) % args.save_epoch == 0 or epoch==epochs:
            torch.save(
            {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)

if __name__ == "__main__":
    main()
