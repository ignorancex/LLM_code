import os
import warnings
warnings.filterwarnings('ignore')
DEBUG = False
os.environ["OMP_NUM_THREADS"] = "1" 
import time
import random
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from util import config

from util.build import get_dataloader, get_model, get_optimizer
from util.util import find_free_port, save_checkpoint, AverageMeter, intersectionAndUnionGPU
from util.logger import get_logger
from util.lr import CamLR

categories_scannet = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture"
]

categories_s3dis = ["ceiling", "floor", "wall", "beam", "column", "window",
               "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]

categories = {
    "scannet_cross": categories_scannet,
    "s3dis_cross": categories_s3dis,
}

def get_parser():
    parser = argparse.ArgumentParser(description='WSegPC')
    parser.add_argument('--config', type=str, default='config/scannet/WSegPC_cmg_rpc_seg.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet/WSegPC_cmg_rpc_seg.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    args.sleep_time = args.get("sleep_time", 0)
    time.sleep(args.sleep_time)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    torch.cuda.empty_cache()

    args.only_feature = args.get("only_feature", False)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    torch.backends.cudnn.enabled = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)

    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    model = get_model(args)
    optimizer_2d, optimizer_3d = get_optimizer(args, model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)
    
    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        # logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))   

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        if args.sync_bn_2d:
            if main_process():
                logger.info("using DDP synced BN for 2D")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.sync_bn_3d:
            if main_process():
                logger.info("using DDP synced BN for 3D")
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=False)
    else:
        model = model.cuda()    
    
    train_loader, train_sampler, val_loader, val_sampler = get_dataloader(args)
    scheduler = CamLR(args, optimizer_2d, optimizer_3d, max_iter_2d=len(train_loader)*args.epochs, max_iter_3d=len(train_loader)*args.epochs)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    # ####################### Train ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
        if main_process():
            logger.info(f"current best train: {best_iou:.4f}")
        mIoU_train, iou_class_train = train(train_loader, model, criterion, optimizer_3d, epoch, scheduler)
        epoch_log = epoch + 1

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            mIoU_val, iou_class_val  = validate(val_loader, model, criterion)

            if main_process():
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)
                logger.info("train --- val ---")
                for i in range(len(categories[args.data_name])):
                    cls_name = categories[args.data_name][i]
                    iou_train = iou_class_train[i]
                    iou_val = iou_class_val[i]
                    info_str = f"{cls_name:15} --- {iou_train:.4f} - {iou_val:.4f} "
                    logger.info(info_str)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            logger.info('Saving checkpoint to: ' + os.path.join(args.save_path, 'model'))
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer_3d.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )
    if main_process():
        # writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch, scheduler):
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        (coords, feat, _, _, _, _, _, _, label_3d, _, _, _, _) = batch_data
        # For some networks, making the network invariant to even, odd coords is important
        coords[:, :3] += (torch.rand(3) * 100).type_as(coords)
        sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        if args.pseudo_label_3d:
            label = label_3d[:,-1].cuda(non_blocking=True)
        else:
            label = label_3d[:,0].cuda(non_blocking=True)
        output = model(sinput)
        # pdb.set_trace()
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, label.detach(), args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Adjust lr
        current_iter = epoch * len(train_loader) + i + 1
        cur_lr = scheduler.get_last_lr()["3d"][0]
        if args.scheduler_update == 'step':
            scheduler.step(current_iter)

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.empty_cache_freq == 0:
            torch.cuda.empty_cache()

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr {lr:.6f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          lr=cur_lr,
                                                          accuracy=accuracy))


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
    return mIoU, iou_class


def validate(val_loader, model, criterion):
    torch.backends.cudnn.enabled = True  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    torch.cuda.empty_cache()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            (coords, feat, label_3d, _, _, _, inds_reverse, _) = batch_data
            # (coords, feat, label, inds_reverse) = batch_data
            sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            label = label_3d[:,0].cuda(non_blocking=True)
            output = model(sinput)
            # pdb.set_trace()
            output = output[inds_reverse, :]
            loss = criterion(output, label)

            output = output.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, label.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return mIoU, iou_class

if __name__ == '__main__':
    main()