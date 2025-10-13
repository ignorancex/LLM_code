import argparse
import os
from multiprocessing import Process

import numpy as np
import torch
from mmcv.ops import points_in_boxes_cpu
from mmdet3d.datasets.convert_utils import NuScenesNameMapping
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d import Box3DMode
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from torch.nn import functional as F

OCC_L = 200
OCC_W = 200
OCC_H = 16

METAINFO = {
    'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
}

map_from_bbox_label_to_occ_label = {
    0: 4,   # car
    1: 10,  # truck
    2: 9,   # trailer
    3: 3,   # bus
    4: 5,   # construction_vehicle
    5: 2,   # bicycle
    6: 6,   # motorcycle
    7: 7,   # pedestrian
    8: 8,   # traffic_cone
    9: 1    # barrier
}

PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    
xs = (torch.linspace(0.5, OCC_L - 0.5, OCC_L, dtype=torch.float32)
      .view(OCC_L, 1, 1).expand(OCC_L, OCC_W, OCC_H) / OCC_L)
ys = (torch.linspace(0.5, OCC_W - 0.5, OCC_W, dtype=torch.float32)
      .view(1, OCC_W, 1).expand(OCC_L, OCC_W, OCC_H) / OCC_W)
zs = (torch.linspace(0.5, OCC_H - 0.5, OCC_H, dtype=torch.float32)
      .view(1, 1, OCC_H).expand(OCC_L, OCC_W, OCC_H) / OCC_H)
OCC_GRID_NORM = torch.stack((xs, ys, zs), 3)

OCC_GRID = OCC_GRID_NORM.clone()
OCC_GRID[..., 0:1] = (OCC_GRID_NORM[..., 0:1] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0])
OCC_GRID[..., 1:2] = (OCC_GRID_NORM[..., 1:2] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1])
OCC_GRID[..., 2:3] = (OCC_GRID_NORM[..., 2:3] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2])

# decided based on IoUgeo and mIoU on mini val set (gave more importance to truck and pedestrian since they are less
# frequent in the dataset)
EXTRA_WITH = 0.30

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# alternative worker: no low reso, store only objects voxels centers in grid coordinates
def worker(p, nusc, occupancy_path, chunk, args):

    # define low reso grid shape
    lr_occ_l = int(OCC_L * 1 / args.scale_factors[0])
    lr_occ_w = int(OCC_W * 1 / args.scale_factors[1])
    lr_occ_h = int(OCC_H * 1 / args.scale_factors[2])

    # loop over scenes inside occupancy path
    for s, scene in enumerate(chunk):

        # compute percentage progress
        print("Progress of process {}: {:.2f}%".format(p, s / len(chunk) * 100))

        scene_path = os.path.join(occupancy_path, scene)

        # loop over samples inside scene path
        for sample_token in os.listdir(scene_path):

            results = {}

            sample = nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)

            # extract category name and align bbox labels to occ labels
            names = [NuScenesNameMapping[b.name] if b.name in NuScenesNameMapping else -1 for b in boxes]
            gt_labels_3d = [METAINFO['classes'].index(n) if n != -1 else -1 for n in names]
            gt_labels_3d = np.array([map_from_bbox_label_to_occ_label[l] if l != -1 else -1 for l in gt_labels_3d])

            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)

            # objects labeled with -1 are removed from annotations in the dataset passed to the dataloader
            # with the method _remove_dontcare in Det3DDataset
            filter_mask = gt_labels_3d != -1
            gt_boxes = gt_boxes[filter_mask]
            gt_labels_3d = gt_labels_3d[filter_mask]

            gt_boxes = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                            origin=(0.5, 0.5, 0.5)).convert_to(Box3DMode.LIDAR)

            # move bboxes from lidar to ego reference system
            lidar2ego_translation = np.array(cs_record['translation'])
            lidar2ego_rotation = np.array(cs_record['rotation'])

            lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

            gt_boxes.rotate(np.linalg.inv(lidar2ego_rotation))
            gt_boxes.translate(lidar2ego_translation)

            # load occupancy map
            occ_labels = np.load(os.path.join(scene_path, sample_token, 'labels.npz'))

            gt_occ_map = torch.tensor(occ_labels['semantics'])
            gt_occ_map_mask_camera = torch.tensor(occ_labels['mask_camera']).bool()

            # interpolate occupancy map to low reso grid
            results['gt_occ_map_low_reso'] = F.interpolate(gt_occ_map[None, None, ...].float(),
                                                           size=(lr_occ_l, lr_occ_w, lr_occ_h),
                                                           mode='nearest')[0, 0, ...].int().numpy()

            results['gt_occ_map_mask_camera_low_reso'] = F.interpolate(gt_occ_map_mask_camera[None, None, ...].float(),
                                                                       size=(lr_occ_l, lr_occ_w, lr_occ_h),
                                                                       mode='nearest')[0, 0, ...].bool().numpy()

            points = OCC_GRID.view(-1, 3).unsqueeze(0)  # create dummy batch dimension
            gt_boxes = gt_boxes.enlarged_box(EXTRA_WITH)
            gt_boxes = gt_boxes.tensor[:, :7].unsqueeze(0)  # remove velocity if present and dummy batch dimension

            if gt_boxes.shape[1] > 0:  # check if there are any boxes
                box_idx = points_in_boxes_cpu(points, gt_boxes).squeeze(0)  # remove dummy batch dimension
                # find voxels centers that are not inside any box
                not_into = torch.sum(box_idx, dim=1) == 0
                # take for simplicity the first box where the point is inside (most of the time it will be inside only one box)
                panoptic_occ_map = torch.argmax(box_idx, dim=1)
                # put box_idx to -1 for voxel centers that are not inside any box
                panoptic_occ_map[not_into] = -1
                # reshape box_idx to the shape of the voxel grid and convert to numpy
                panoptic_occ_map = panoptic_occ_map.view(OCC_L, OCC_W, OCC_H)
                # find intersection with original gt_occ_map (keep only voxels inside bboxes that are actually labeled with
                # the same class as the bbox in the original gt_occ_map)
                # gt_occ_map = torch.tensor(gt_occ_map)
                # from a panoptic map with indices to a panoptic map with classes
                obj_occ_map = torch.tensor(gt_labels_3d)[panoptic_occ_map]
                # remove spurious indexation
                obj_occ_map[panoptic_occ_map == -1] = -1
                # verify which voxels inside the bboxes labeled with the same class as the bbox are actually labeled the same
                # in the original occ map (this removes eventual voxels inside the bbox which are empty or from another class
                # e.g. drivable_surface)
                mask_eq = gt_occ_map == obj_occ_map
                panoptic_occ_map[~mask_eq] = -1
            else:
                panoptic_occ_map = torch.zeros((OCC_L, OCC_W, OCC_H), dtype=torch.long) - 1

            results['gt_panoptic_occ_map'] = panoptic_occ_map

            # create objects occupancy masks
            if args.mask_camera:
                # for objects that have at least one visible voxel by the cameras
                # mask_camera = torch.tensor(gt_occ_map_mask_camera, dtype=torch.bool)
                gt_visible_obj_idx = torch.unique(panoptic_occ_map[gt_occ_map_mask_camera])
            else:
                gt_visible_obj_idx = torch.unique(panoptic_occ_map)

            # remove index -1 if present
            gt_visible_obj_idx = gt_visible_obj_idx[gt_visible_obj_idx != -1].int()

            visible_obj_occ_grid = []
            visible_obj_occ_grid_idx = []
            idx_counter = 0
            for i in gt_visible_obj_idx:
                obj_mask = panoptic_occ_map == i
                visible_obj_occ_grid_idx.append(idx_counter)
                visible_obj_occ_grid.append(OCC_GRID[obj_mask])
                idx_counter += obj_mask.sum().item()

            if len(visible_obj_occ_grid) > 0:
                visible_obj_occ_grid = torch.cat(visible_obj_occ_grid)
            else:
                visible_obj_occ_grid = torch.empty((0, 3))

            results['gt_occ_map_visible_obj_occ_grid'] = visible_obj_occ_grid.numpy()
            results['gt_occ_map_visible_obj_occ_grid_start_idx'] = np.array(visible_obj_occ_grid_idx)
            results['gt_occ_map_visible_obj_idx'] = gt_visible_obj_idx.numpy()

            # save results to a npz file
            dest = os.path.join(scene_path, sample_token, 'aux_labels.npz')
            np.savez(dest, **results)


def main(args):
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.root_path, verbose=True)

    occupancy_path = os.path.join(args.root_path, 'occupancy')

    if not os.path.exists(occupancy_path):
        raise Exception("occupancy path does not exist")

    scene_list = os.listdir(occupancy_path)
    # split between different processes
    scene_list = list(split(scene_list, args.nproc))

    processes = []
    for i, chunk in enumerate(scene_list):
        p = Process(target=worker, args=(i, nusc, occupancy_path, chunk, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nuscene panoptic arg parser')
    parser.add_argument('--root-path', type=str, default='./data/nuscenes', help='specify the root path of dataset')
    parser.add_argument("--scale_factors", type=float, nargs='+', default=[2.0, 2.0, 2.0],
                        help='scale factors for low resolution occ map')
    parser.add_argument("--mask_camera", default=False, action='store_true', help='mask camera')
    parser.add_argument("--nproc", type=int, default=4, help='number of processes to be used')
    args = parser.parse_args()

    main(args)
