from ast import arg
import os
import warnings
warnings.filterwarnings('ignore')
DEBUG = False
os.environ["OMP_NUM_THREADS"] = "1" 
import shutil
import time
import random
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from util import config

from util.build import get_dataloader, get_model, get_optimizer, get_loss
from util.util import find_free_port, AverageMeter, intersectionAndUnionGPU
from util.logger import get_logger
from util.lr import CamLR
from collections import OrderedDict

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
    parser.add_argument('--config', type=str, default='config/scannet/WSegPC_cmg.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet/WSegPC_cmg.yaml for all options', default=None,
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
    args.ts_model_update_freq = args.get("ts_model_update_freq", 1)
    time.sleep(args.sleep_time)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

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


def update_teacher_model(model, model_teacher, keep_rate=0.996):
    if args.distributed:
        student_model_dict = {
            key[7:]: value for key, value in model.state_dict().items()
        }
    else:
        student_model_dict = model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    model_teacher.load_state_dict(new_teacher_dict)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args = argss
    best_iou_3D, best_iou_2D = 0, 0
    not_improve = 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    model = get_model(args)
    optimizer_2d, optimizer_3d = get_optimizer(args, model)
    criterion = get_loss(args)
    if args.ts_model:
        ema_model = get_model(args)

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

    if args.ts_model:
        ema_model = ema_model.cuda()
        update_teacher_model(model, ema_model, keep_rate=0)
    else:
        ema_model = None

    train_loader, train_sampler, val_loader, val_sampler = get_dataloader(args)
    scheduler = CamLR(args, optimizer_2d, optimizer_3d, max_iter_2d=len(train_loader)*args.epochs, max_iter_3d=len(train_loader)*args.epochs)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            if args.distributed:
                weight_model_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if "module." not in key:
                        key = "module."+key
                    weight_model_dict[key] = value
            else:
                weight_model_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if "module." in key:
                        key = key[7:]
                    weight_model_dict[key] = value
                weight_model_dict = checkpoint['state_dict']
            load_state_info = model.load_state_dict(weight_model_dict, strict=False)
            if main_process():
                logger.info("=> loaded weight '{}' from {} epoch".format(args.weight, checkpoint['epoch']))
                logger.info(f"Missing keys: {load_state_info[0]}")
            if args.ts_model and 'ema_state_dict' in checkpoint:
                load_state_info = ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False) 
                if main_process():
                    logger.info(f"Missing keys: {load_state_info[0]}")

            best_iou_3D = checkpoint['best_iou_3D'] if 'best_iou_3D' in checkpoint else 0.0
            best_iou_2D = checkpoint['best_iou_2D'] if 'best_iou_2D' in checkpoint else 0.0
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
            logger.info(f"current best_iou_3D: {best_iou_3D:.4f}, best_iou_2D: {best_iou_2D:.4f}")
        last_lr = scheduler.get_last_lr()
        if main_process():
            logger.info("lr_2d: {}, lr_3d: {}".format(last_lr["2d"], last_lr["3d"]))
        
        mIoU_3d, iou_class_3d, mIoU_2d, iou_class_2d = train_cross(train_loader, model, criterion, optimizer_2d, optimizer_3d, epoch, scheduler, ema_model)
        if args.scheduler_update == 'epoch':
            scheduler.step(epoch)
        epoch_log = epoch + 1
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            mIoU_3d_val, iou_class_3d_val, mIoU_2d_val, iou_class_2d_val = validate_cross(val_loader, ema_model)
            if main_process():
                for i in range(len(categories[args.data_name])):
                    cls_name = categories[args.data_name][i]
                    iou_3d = iou_class_3d_val[i]
                    iou_2d = iou_class_2d_val[i]
                    info_str = f"{cls_name:15} --- {iou_3d:.4f} - {iou_2d:.4f} "
                    logger.info(info_str)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            logger.info('Saving checkpoint to: ' + os.path.join(args.save_path, 'model'))
            is_best_val = mIoU_3d_val > best_iou_3D
            best_iou_3D = max(best_iou_3D, mIoU_3d_val)
            best_iou_2D = max(best_iou_2D, mIoU_2d_val)
            
            save_dict = {
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict() if ema_model else "",
                'optimizer_2d': optimizer_2d.state_dict() if optimizer_2d else "",
                'optimizer_3d': optimizer_3d.state_dict() if optimizer_3d else "",
                'best_iou_3D': best_iou_3D,
                "best_iou_2D": best_iou_2D,
                'scheduler': scheduler.state_dict()
            }
            sav_path = os.path.join(args.save_path, 'model')
            filename = os.path.join(sav_path, 'model_last.pth.tar')
            torch.save(save_dict, filename)
            if is_best_val:
                shutil.copyfile(filename, os.path.join(sav_path, 'model_best.pth.tar'))
                not_improve = 0
            else:
                not_improve += 1
                if not_improve == args.early_stop_epoch:
                    logger.info('==>Training early stop!\n')
                    logger.info(f"best_iou_3D: {best_iou_3D:.4f}, best_iou_2D: {best_iou_2D:.4f}")
                    os._exit(0)
        if not_improve == args.early_stop_epoch:
            os._exit(0)

    if main_process():
        # writer.close()
        logger.info('==>Training done!\n')
        logger.info(f"best_iou_3D: {best_iou_3D:.4f}, best_iou_2D: {best_iou_2D:.4f}")


def train_cross(train_loader, model, criterion, optimizer_2d, optimizer_3d, epoch, scheduler, ema_model):

    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter, loss_meter_cam_3d, loss_meter_cam_2d = AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter_seg_3d, loss_meter_seg_2d = AverageMeter(), AverageMeter()
    loss_meter_cmg_2d, loss_meter_cmg_3d = AverageMeter(), AverageMeter()
    loss_meter_rpc_2d, loss_meter_rpc_3d = AverageMeter(), AverageMeter()

    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    if args.ts_model and epoch == args.ts_model_epoch:
        update_teacher_model(model, ema_model, keep_rate=0.00)
        ema_model.eval()

    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        current_iter = epoch * len(train_loader) + i + 1

        (coords_s, feat_s, coords_t, feat_t, ind_s, ind_recons_s, ind_t, ind_recons_t,
         label_3d, labels_cls, color, label_2d, link) = batch_data
        # For some networks, making the network invariant to even, odd coords is important
        coords_student = coords_s[:,0:4]
        coords_teacher = coords_t[:,0:4]
        coords_student[:, 1:4] += (torch.rand(3) * 100).type_as(coords_student)
        coords_teacher[:, 1:4] += (torch.rand(3) * 100).type_as(coords_teacher)

        sinput_student = SparseTensor(feat_s[:,0:6].cuda(non_blocking=True), coords_student.cuda(non_blocking=True))
        sinput_teacher = SparseTensor(feat_t[:,0:6].cuda(non_blocking=True), coords_teacher.cuda(non_blocking=True))

        color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
        label_3d, label_2d = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
        labels_cls = labels_cls.cuda(non_blocking=True)
        
        color_student = color[:,0]
        color_teacher = color[:,1]

        output_3d, output_2d = model(sinput_student, color_student, link)

        with torch.no_grad():
            if args.ts_model and epoch >= args.ts_model_epoch:
                output_3d_ema, output_2d_ema = ema_model(sinput_teacher, color_teacher)
                feat_3d = output_3d_ema[0].F
                feat_3d = feat_3d[ind_recons_t][ind_s]
                output_3d_ema[0] = SparseTensor(features=feat_3d,
                                    coordinate_map_key=output_3d[0].coordinate_map_key,
                                    coordinate_manager=output_3d[0].coordinate_manager
                                    )
            else:
                output_3d_ema, output_2d_ema = model(sinput_teacher, color_teacher)

        loss_dict, output_3d, output_2d = criterion(output_3d, output_2d, sinput_student, output_3d_ema, output_2d_ema, label_3d, label_2d, labels_cls, link)
        loss = loss_dict["loss"]

        if optimizer_2d is not None:
            optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

        if optimizer_2d is not None:
            optimizer_2d.step()
        optimizer_3d.step()

        if args.ts_model and epoch >= args.ts_model_epoch and i % args.ts_model_update_freq == 0:
            update_teacher_model(model, ema_model, keep_rate=args.ts_model_keep_rate)

        # ############ 3D ############ #
        if output_3d is not None:
            output_3d = output_3d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d[:,0].detach(), args.classes,
                                                                args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3d.update(intersection)
            union_meter_3d.update(union)
            target_meter_3d.update(target)

        # ############ 2D ############ #
        if output_2d is not None:
            output_2d = output_2d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d[:,0].contiguous().detach(), args.classes,
                                                                args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_2d.update(intersection)
            union_meter_2d.update(union)
            target_meter_2d.update(target)

        loss_meter.update(loss.item(), args.batch_size)
        loss_meter_cam_2d.update(loss_dict["cam_2d"].item() if "cam_2d" in loss_dict else 0.0, args.batch_size)
        loss_meter_cam_3d.update(loss_dict["cam_3d"].item() if "cam_3d" in loss_dict else 0.0, args.batch_size)
        loss_meter_seg_3d.update(loss_dict["seg_3d"].item() if "seg_3d" in loss_dict else 0.0, args.batch_size)
        loss_meter_seg_2d.update(loss_dict["seg_2d"].item() if "seg_2d" in loss_dict else 0.0, args.batch_size)
        loss_meter_cmg_2d.update(loss_dict["cmg_2d"].item() if "cmg_2d" in loss_dict else 0.0, args.batch_size)
        loss_meter_cmg_3d.update(loss_dict["cmg_3d"].item() if "cmg_3d" in loss_dict else 0.0, args.batch_size)
        loss_meter_rpc_2d.update(loss_dict["rpc_2d"].item() if "rpc_2d" in loss_dict else 0.0, args.batch_size)
        loss_meter_rpc_3d.update(loss_dict["rpc_3d"].item() if "rpc_3d" in loss_dict else 0.0, args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Adjust lr
        cur_lr_3d = scheduler.get_last_lr()["3d"][0]
        cur_lr_2d = scheduler.get_last_lr()["2d"][0]
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
            # torch.backends.cudnn.enabled = True
            # torch.cuda.empty_cache()
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} '
                        'Batch {batch_time.val:.3f} '
                        'Remain {remain_time} '
                        'Lr_3 {lr3:.5f} '
                        'Lr_2 {lr2:.5f} '
                        'L {loss_meter.val:.3f} '
                        'CAM_2d {loss_cam_2d.val:.3f} '
                        'CAM_3d {loss_cam_3d.val:.3f} '
                        'Seg_2d {loss_seg_2d.val:.3f} '
                        'Seg_3d {loss_seg_3d.val:.3f} '
                        'CMG_2d {loss_cmg_2d.val:.3f} '
                        'CMG_3d {loss_cmg_3d.val:.3f} '
                        'RPC_2d {loss_rpc_2d.val:.3f} '
                        'RPC_3d {loss_rpc_3d.val:.3f} '\
                        .format(epoch + 1, args.epochs, i + 1, len(train_loader),
                        batch_time=batch_time, data_time=data_time,
                        remain_time=remain_time,
                        lr3=cur_lr_3d,
                        lr2=cur_lr_2d,
                        loss_meter=loss_meter,
                        loss_cam_2d=loss_meter_cam_2d,
                        loss_cam_3d=loss_meter_cam_3d,
                        loss_seg_2d=loss_meter_seg_2d,
                        loss_seg_3d=loss_meter_seg_3d,
                        loss_cmg_2d=loss_meter_cmg_2d,
                        loss_cmg_3d=loss_meter_cmg_3d,
                        loss_rpc_2d=loss_meter_rpc_2d,
                        loss_rpc_3d=loss_meter_rpc_3d,
                        ))

    iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
    mIoU_3d = np.mean(iou_class_3d)
    mAcc_3d = np.mean(accuracy_class_3d)
    if output_3d is not None:
        allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)
    else:
        allAcc_3d = 0

    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = np.mean(iou_class_2d)
    mAcc_2d = np.mean(accuracy_class_2d)
    if output_2d is not None:
        allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
    else:
        allAcc_2d = 0

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: 3D 2D mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}, {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                           mIoU_3d, mAcc_3d, allAcc_3d,
                                                                                           mIoU_2d, mAcc_2d, allAcc_2d))
    return mIoU_3d, iou_class_3d, mIoU_2d, iou_class_2d


def validate_cross(val_loader, model):

    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    # torch.backends.cudnn.enabled = True  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            (coords, feat, label_3d, color, label_2d, link, inds_reverse, _) = batch_data
            
            sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
            label_3d, label_2d = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)#.squeeze()
            label_3d = label_3d[:,0]
            output_3d, output_2d = model(sinput, color)
            inverse_map = sinput.inverse_mapping
            if output_3d is not None:
                cam_3d = output_3d[0]
                output_3d = cam_3d.F[inverse_map][inds_reverse]
                # print(output_3d.shape, label_3d.shape)
                if args.val_with_cls:
                    cam_3d_coord = cam_3d.C[inverse_map][inds_reverse]
                    bs = cam_3d_coord[-1, 0] + 1
                    cls_3D_label = torch.zeros((bs, args.classes)).cuda()
                    cls_3D_label_all = torch.ones_like(output_3d).cuda()
                    for b_i in range(bs):
                        inds = cam_3d_coord[:, 0] == b_i
                        per_scene_label = label_3d[inds]
                        for cls in torch.unique(per_scene_label):
                            if cls != 255: # ignore 255
                                cls_3D_label[b_i, cls] = 1
                        cls_3D_label_all[inds] = cls_3D_label_all[inds] * cls_3D_label[b_i]
                    output_3d = output_3d + (1-cls_3D_label_all) * -1000000
                output_3d = output_3d.detach().max(1)[1]
                intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                                    args.ignore_label)
                if args.multiprocessing_distributed:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter_3d.update(intersection)
                union_meter_3d.update(union)
                target_meter_3d.update(target)

            if output_2d is not None: 
                cam_2d = output_2d[0]
                V_NUM = link.shape[-1]
                V_B, C, H, W = cam_2d.shape
                cam_2d = cam_2d.view(V_B//V_NUM, V_NUM, C, H, W)
                cam_2d = cam_2d.permute(0, 2, 3, 4, 1)
                if args.val_with_cls:
                    cam_2d = cam_2d + (1-cls_3D_label.unsqueeze(2).unsqueeze(3).unsqueeze(-1)) * -1000000
                output_2d = cam_2d.detach().max(1)[1]
                intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
                                                                    args.ignore_label)
                if args.multiprocessing_distributed:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter_2d.update(intersection)
                union_meter_2d.update(union)
                target_meter_2d.update(target)
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('[{}/{}] '.format(i + 1, len(val_loader)))

    iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
    mIoU_3d = np.mean(iou_class_3d)
    mAcc_3d = np.mean(accuracy_class_3d)
    if output_3d is not None:
        allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)
    else:
        allAcc_3d = 0

    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = np.mean(iou_class_2d)
    mAcc_2d = np.mean(accuracy_class_2d)
    if output_2d is not None: 
        allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
    else:
        allAcc_2d = 0
    
    if main_process():
        logger.info(
            'Val result: 3D mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}, 2D mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
                mIoU_3d, mAcc_3d, allAcc_3d, mIoU_2d, mAcc_2d, allAcc_2d))
    return mIoU_3d, iou_class_3d, mIoU_2d, iou_class_2d


if __name__ == '__main__':
    main()