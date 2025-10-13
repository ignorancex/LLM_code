import os
os.environ["OMP_NUM_THREADS"] = "1" 
import random
import numpy as np
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch_scatter
import torch.utils.data
import torch.distributed as dist

from MinkowskiEngine import SparseTensor
from util import config

from util.build import get_model
from util.util import AverageMeter, intersectionAndUnionGPU
from util.logger import get_logger
from tqdm import tqdm

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)
class2id = np.array(VALID_CLASS_IDS_20)
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
    parser.add_argument('--config', type=str, default='config/scannet/WSegPC_stage_2.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet/WSegPC_stage_2.yaml for all options', default=None,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    torch.cuda.empty_cache()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)

    # if len(args.train_gpu) == 1:
    args.sync_bn = False
    args.distributed = False
    args.multiprocessing_distributed = False
    main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    model = get_model(args)

    if main_process():
        global logger, writer
        logger = get_logger(os.path.join(args.save_path, "test_log.txt"))
        # logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))   
    
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if os.path.isfile(args.model_path):
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        
        if args.distributed:

            loaded_dict = checkpoint['state_dict']
            sd = model.state_dict()
            for k in model.state_dict():
                # k_module = "module." + k
                k_module = k[7:]
                if k_module in loaded_dict and sd[k].size() == loaded_dict[k_module].size():
                    sd[k] = loaded_dict[k_module]
                else:
                    print(k)
            load_state_info = model.load_state_dict(sd)

        else:
            loaded_dict = checkpoint['ema_state_dict'] if args.use_ema else checkpoint['state_dict']
            # save_dict = {
            #     'ema_state_dict': loaded_dict,
            # }
            # torch.save(save_dict, "/root/WSegPC/exp/s3dis/WSegPC_cmg_rpc/model/s3dis.pth.tar")
            # return
            sd = model.state_dict()
            for k in model.state_dict():
                k_module = k #if args.use_ema else "module." + k
                if k_module in loaded_dict and sd[k].size() == loaded_dict[k_module].size():
                    sd[k] = loaded_dict[k_module]
                else:
                    print(k)
            load_state_info = model.load_state_dict(sd)
        if main_process():
            # print(checkpoint['best_iou'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
            logger.info(f"Missing keys: {load_state_info[0]}")
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    if args.data_name == 'scannet_cross':
        from dataset.ScanNetCross import ScanNetCross, collation_fn, collation_fn_eval_all, collation_fn_test_all
        val_data = ScanNetCross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split=args.split, aug=False,
                                eval_all=True, view_num=args.view_num, val_benchmark=True, 
                                data_root_img=args.data_root_img, data_root_sp=args.data_root_sp)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                    shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                    drop_last=False, collate_fn=collation_fn_eval_all if args.split != "test" else collation_fn_test_all,
                                                    sampler=val_sampler)
    if args.data_name == 's3dis_cross':
        from dataset.S3DIS_Cross import S3DIS_Cross, collation_fn, collation_fn_eval_all
        val_data = S3DIS_Cross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split=args.split, aug=False,
                                eval_all=True, view_num=args.view_num, val_benchmark=True, 
                                data_root_img=args.data_root_img, data_root_sp=args.data_root_sp)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                    shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                    drop_last=False, collate_fn=collation_fn_eval_all,
                                                    sampler=val_sampler)
        
    if args.arch in ['mink_14A', 'mink_18A', 'mink_34C']:
        test_3d(model, val_loader)
    else:
        test_3d_2d(model, val_loader)


def test_3d_2d(model, val_data_loader):
    torch.backends.cudnn.enabled = True  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        if main_process():
            pbar = tqdm(total=len(val_data_loader), ncols=120)
        for i, batch_data in enumerate(val_data_loader):
            
            if main_process():
                pbar.update(1)

            (coords, feat, label_3d, color, label_2d, link, inds_reverse, _) = batch_data

            sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
            label_3d, label_2d = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)#.squeeze()
            label_3d_sp = label_3d[:,1]
            label_3d = label_3d[:,0]
            output_3d, output_2d = model(sinput, color, links=None)
            inverse_map = sinput.inverse_mapping
            
            if output_3d is not None:
                cam_3d = output_3d[0]
                output_3d = cam_3d.F[inverse_map][inds_reverse]
                # print(output_3d.shape, label_3d.shape)
                if args.use_cls:
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
                output_3d_softmax = torch.softmax(output_3d, dim=1)
                if args.use_sp:
                    label_3d_sp_mask = label_3d_sp >= 0
                    output_3d_softmax_self = output_3d_softmax[label_3d_sp_mask]
                    label_3d_sp_self = label_3d_sp[label_3d_sp_mask]
                    output_3d_softmax_self_mean = torch_scatter.scatter_mean(output_3d_softmax_self, label_3d_sp_self, dim=0)
                    output_3d_softmax_self_mean = output_3d_softmax_self_mean[label_3d_sp_self]
                    # print(output_3d_softmax_self_mean.shape, label_3d_sp_self.shape, output_3d_softmax.shape, label_3d_sp.shape)
                    output_3d_softmax[label_3d_sp_mask] = output_3d_softmax_self_mean

                output_3d_prob, output_3d = output_3d_softmax.detach().max(1)
                # label_3d[output_3d_prob<0.5] = args.ignore_label
                intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes, args.ignore_label)
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
                if args.use_cls:
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

            if main_process() and args.save_3d:

                for b_i in range(bs):
                    inds = cam_3d_coord[:, 0] == b_i

                    output_3d_softmax_i = output_3d_softmax[inds]
                    output_3d_prob_i, output_3d_i = output_3d_softmax_i.max(1)
                    output_3d_i = output_3d_i.cpu().numpy()
                    output_3d_prob_i = output_3d_prob_i.cpu().numpy()

                    scan_index = i*val_data_loader.batch_size + b_i
                    scan_id = val_data_loader.dataset.data_paths[scan_index]
                    if args.data_name == 's3dis_cross':
                        scan_id = scan_id.split("/")
                        scan_id = scan_id[-2]+"__"+scan_id[-1].split(".")[0]
                    else:
                        scan_id = os.path.basename(scan_id).split(".")[0]
                    save_path = os.path.join(args.save_folder, '{}.npz'.format(scan_id))
                    np.savez(save_path, pred_label=output_3d_i.astype(np.uint8))

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
            pbar.close()
            # print(cls_diff.avg, cls_diff_3d.avg, cls_diff_2d.avg)
            for i in range(len(categories[args.data_name])):
                a = categories[args.data_name][i]
                b = iou_class_3d[i]
                c = iou_class_2d[i]
                logger.info(f"{a:20} - {b:.3f} - {c:.3f}")
            logger.info(
                'Val result: 3D mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}, 2D mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
                    mIoU_3d, mAcc_3d, allAcc_3d, mIoU_2d, mAcc_2d, allAcc_2d))


def test_3d(model, val_data_loader):
    torch.backends.cudnn.enabled = True  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    intersection_meter_3d = AverageMeter()
    union_meter_3d = AverageMeter()
    target_meter_3d = AverageMeter()
    with torch.no_grad():
        model.eval()
        if main_process():
            pbar = tqdm(total=len(val_data_loader), ncols=120)
        for i, batch_data in enumerate(val_data_loader):
            if main_process():
                pbar.update(1)
            if args.split != "test":
                (coords, feat, label_3d, color, label_2d, link, inds_reverse, _) = batch_data
            else:
                (coords, feat, label_3d, inds_reverse, _) = batch_data

            sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            # color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
            label_3d = label_3d.cuda(non_blocking=True)
            if args.split != "test":
                label_3d_sp = label_3d[:,1]
            label_3d = label_3d[:,0]    
            output_3d = model(sinput)
            inverse_map = sinput.inverse_mapping
            
            if output_3d is not None:
                output_3d = output_3d[inverse_map][inds_reverse]
                cam_3d_coord = sinput.C[inverse_map][inds_reverse]
                bs = cam_3d_coord[-1, 0] + 1
                output_3d_softmax = torch.softmax(output_3d, dim=1)
                if args.use_sp:
                    label_3d_sp_mask = label_3d_sp >= 0
                    output_3d_softmax_self = output_3d_softmax[label_3d_sp_mask]
                    label_3d_sp_self = label_3d_sp[label_3d_sp_mask]
                    output_3d_softmax_self_mean = torch_scatter.scatter_mean(output_3d_softmax_self, label_3d_sp_self, dim=0)
                    output_3d_softmax_self_mean = output_3d_softmax_self_mean[label_3d_sp_self]
                    if args.use_sp_with_self:
                        output_3d_softmax[label_3d_sp_mask] = (output_3d_softmax[label_3d_sp_mask] + output_3d_softmax_self_mean) / 2
                    else:
                        output_3d_softmax[label_3d_sp_mask] = output_3d_softmax_self_mean

                output_3d_prob, output_3d = output_3d_softmax.detach().max(1)
                # label_3d[output_3d_prob<0.5] = args.ignore_label
                # print(output_3d.shape, label_3d.detach().shape)
                intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes, args.ignore_label)
                if args.multiprocessing_distributed:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter_3d.update(intersection)
                union_meter_3d.update(union)
                target_meter_3d.update(target)
                # accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)
            
            if main_process() and args.save_3d:

                for b_i in range(bs):
                    inds = cam_3d_coord[:, 0] == b_i

                    output_3d_softmax_i = output_3d_softmax[inds]
                    output_3d_prob_i, output_3d_i = output_3d_softmax_i.max(1)
                    output_3d_i = output_3d_i.cpu().numpy()
                    output_3d_prob_i = output_3d_prob_i.cpu().numpy()

                    scan_index = i*val_data_loader.batch_size + b_i
                    scan_id = val_data_loader.dataset.data_paths[scan_index]
                    if args.data_name == 's3dis_mix':
                        scan_id = scan_id.split("/")
                        scan_id = scan_id[-2]+"__"+scan_id[-1].split(".")[0]
                    else:
                        scan_id = os.path.basename(scan_id).split(".")[0]
                    save_path = os.path.join(args.save_folder, '{}.npz'.format(scan_id))
                    np.savez(save_path, pred_label=output_3d_i.astype(np.uint8))
            
            if args.get("submit", False):
                for b_i in range(bs):
                    inds = cam_3d_coord[:, 0] == b_i

                    output_3d_softmax_i = output_3d_softmax[inds]
                    output_3d_prob_i, output_3d_i = output_3d_softmax_i.max(1)
                    output_3d_i = output_3d_i.cpu().numpy()
                    output_3d_prob_i = output_3d_prob_i.cpu().numpy()

                    scan_index = i*val_data_loader.batch_size + b_i
                    scan_id = val_data_loader.dataset.data_paths[scan_index]
                    scan_id = os.path.basename(scan_id).split(".")[0]
                    # print(scan_id)
                    np.savetxt(
                        os.path.join(args.save_folder, "{}.txt".format(scan_id)),
                        class2id[output_3d_i.astype(np.uint8)].reshape([-1, 1]),
                        fmt="%d",
                    )
        iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
        accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
        mIoU_3d = np.mean(iou_class_3d)
        mAcc_3d = np.mean(accuracy_class_3d)
        if output_3d is not None:
            allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)
        else:
            allAcc_3d = 0
        
        if main_process():
            pbar.close()
            for i in range(len(categories[args.data_name])):
                a = categories[args.data_name][i]
                b = iou_class_3d[i]
                logger.info(f"{a:20} - {b:.3f}")
            logger.info(
                'Val result: 3D mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
                    mIoU_3d, mAcc_3d, allAcc_3d))


if __name__ == '__main__':
    main()