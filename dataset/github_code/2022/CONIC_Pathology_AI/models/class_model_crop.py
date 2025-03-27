import os
import sys
sys.path.append("./")
import numpy as np
import cv2
from argparse import ArgumentParser

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import models
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F

from datasets.classification_datasets import NucleiDatasetCrop
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--port", type=str, default="12353", help="port of distributed")
parser.add_argument("--device", type=int, default=0, help="device id")
parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--max_epoch", type=int, default=12, help="max epoch number")
parser.add_argument("--fold_idx", type=int, default=4, help="fold idx to class")
parser.add_argument("--input_dim", type=int, default=11, help="input dimension")
parser.add_argument("--resnet", type=str, default="resnet18", help="name of resnet used")


args = parser.parse_args()

def setup(rank, world_size):
    r"""
    Set up the environment for distributed parallel training
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    r"""
    Destroy the parallel training environment and clean up the training
    """
    dist.destroy_process_group()


def run_demo(demo_fn, world_size):
    r"""
    Generate processes for the parallel training with each process running demo_fn
    """
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def demo_basic(rank, world_size):
    r"""
    The basic function to train the feature encoder with our framework.
    The framework is composed of three branches: Structured Triplet, 
                                                 Attribute Learning, 
                                                 and Self-Supervision.
    We add up the loss of the three branches to lead the learning.
    """
    setup(rank, world_size)

    if args.resnet == "resnet18":
        model = resnet18(pretrained=True)
    elif args.resnet == "resnet50":
        model = resnet50(pretrained=True)
    elif args.resnet == "efficientnet_b2":
        resnet = models.efficientnet_b2(pretrained=True)

    model.conv1 = nn.Conv2d(args.input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    
    if args.resnet == "resnet18":
        model.fc = nn.Linear(512, 6)
    elif args.resnet == "resnet50":
        model.fc = nn.Linear(512*4, 6)

    model = model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8], gamma=0.1)

    if os.path.exists('checkpoints/{}/{}/latest.pth.tar'.format(args.resnet, args.fold_idx)):
        args.resume = 'checkpoints/{}/{}/latest.pth.tar'.format(args.resnet, args.fold_idx)

    start_epoch = 0

    imagenet = NucleiDatasetCrop(split_name="valid", folder_idx=args.fold_idx, input_dim=args.input_dim)

    sampler = DistributedSampler(imagenet)
    imagenet_dataloader = torch.utils.data.DataLoader(dataset=imagenet, batch_size=args.batch_size,
                                                      num_workers=2, sampler=sampler)
    print(len(imagenet))

    imagenet_test = NucleiDatasetCrop(split_name="valid", folder_idx=args.fold_idx, input_dim=args.input_dim)

    sampler_test = DistributedSampler(imagenet_test)
    imagenet_dataloader_test = torch.utils.data.DataLoader(dataset=imagenet_test, batch_size=args.batch_size,
                                                      num_workers=2, sampler=sampler_test)

    dist.barrier()
    if rank == 0:
        writer = SummaryWriter("logs/{}/{}".format(args.resnet, args.fold_idx))

    cross_entropy_loss = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.max_epoch):
        # train
        model.train()
        loss_sum = 0
        total = 0
        correct = 0
        sampler.set_epoch(epoch)

        for local_batch, gt_labels in imagenet_dataloader:
            
            if args.input_dim==11 and rank == 0:
                for idx, (item, type_idx_this) in enumerate(zip(local_batch, gt_labels)):
                    print("item.shape", item.shape)
                    os.makedirs(f"visualize_crop/linear/{type_idx_this}", exist_ok=True)
                    img = (item[:3, :, :] +  1) / 2
                    img = torchvision.transforms.functional.to_pil_image(img)
                    img.save(f"visualize_crop/linear/{type_idx_this}/{idx}_img.png")
                    for tp in range(7):
                        type_map = item[3+tp, :, :]
                        # type_map = torch.argmax(type_map, dim=0)
                        type_map = type_map.cpu().numpy()
                        img_vis = np.array(type_map*255, dtype=np.uint8)
                        cv2.imwrite(f"visualize_crop/linear/{type_idx_this}/{idx}_prob_{tp}.png", img_vis)

                    """
                    type_map = item[13:, :, :]
                    type_map = torch.argmax(type_map, dim=0)
                    type_map = type_map.cpu().numpy()
                    img_vis = np.zeros((224, 224, 3), dtype=np.uint8)
                    colors = [[0  ,   0,   0], [255,   0,   0], [0  , 255,   0], 
                                [0  ,   0, 255], [255, 255,   0], [255, 165,   0], [165, 0, 165]]
                    """

                    type_map = item[10, :, :]
                    # type_map = torch.argmax(type_map, dim=0)
                    type_map = type_map.cpu().numpy()
                    img_vis = np.array(type_map*255, dtype=np.uint8)
                    cv2.imwrite(f"visualize_crop/linear/{type_idx_this}/{idx}_prob_mask.png", img_vis)

                    """
                    print("type map shape: ", type_map.shape)
                    print("unique types: ", np.unique(type_map))
                    for tp in range(7):
                        tp_indexes = type_map == tp
                        img_vis[tp_indexes] = colors[tp]
                    
                    cv2.imwrite(f"visualize_crop/linear/{type_idx_this}/{idx}_prob_mask.png", img_vis)
                    """

                input("checkpoint")

            local_batch, gt_labels = local_batch.to(rank).float(), gt_labels.to(rank)
            output = model(local_batch)
            loss = cross_entropy_loss(output, gt_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += gt_labels.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == gt_labels).sum().item()

            if rank == 0:
                loss_sum += loss.detach().cpu().numpy()

        scheduler.step()
        if rank == 0:
            print(f"Epoch: {epoch}, loss: {loss_sum / total},  acc: {correct / total}")

            writer.add_scalar('Loss/loss', loss_sum / total, epoch)
            writer.add_scalar('ACC', correct / total, epoch)
            
        # check point save
        if rank == 0:
            os.makedirs("checkpoints/{}/{}".format(args.resnet, args.fold_idx), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoints/{}/{}/latest.pth.tar'.format(args.resnet, args.fold_idx))

            if epoch % 2 == 0 or epoch == (args.max_epoch - 1):
                checkpoint_path = "checkpoints/{}/{}/improved-net_{}.pt".format(args.resnet, args.fold_idx, epoch)
                torch.save(model.module.state_dict(), checkpoint_path)
        
        if epoch % 2 == 0:
            model.eval()
            total_test = 0
            correct_test = 0
            pred_num = [0, 0, 0, 0, 0, 0]
            gt_num = [0, 0, 0, 0, 0, 0]

            with torch.no_grad():
                for local_batch, gt_labels in imagenet_dataloader_test:
                    local_batch, gt_labels = local_batch.to(rank).float(), gt_labels.to(rank)
                    output = model(local_batch)

                    total_test += gt_labels.size(0)
                    _, predicted = torch.max(output.data, 1)
                    correct_test += (predicted == gt_labels).sum().item()

                    for tp_idx in range(6):
                        pred_num[tp_idx] += ((predicted == gt_labels) * (predicted == tp_idx)).sum().item()
                        gt_num[tp_idx] += (gt_labels == tp_idx).sum().item()

                print("valid_acc: ", correct_test / total_test)
                print("pred num: ", pred_num)
                print("gt num: ", gt_num)
            
            print(f"====================={epoch}====================")


        dist.barrier()

    if rank == 0:
        writer.close()

    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run_demo(demo_basic, n_gpus)