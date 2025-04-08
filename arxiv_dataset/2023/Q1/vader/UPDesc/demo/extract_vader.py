from __future__ import print_function
from __future__ import division

import argparse
import os.path as osp
import sys
import random
import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), 'extraction')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.updesc import MODELS  # noqa: E402
from pointclouds import PointCloudDataset  # noqa: E402
from utils.io import may_create_folder  # noqa: E402


def main(cfg):
    with open(cfg.hparams, 'r') as fh:
        hparams = yaml.full_load(fh)

    may_create_folder(cfg.out_root)

    print('Load data', cfg.data_root)
    dataset = PointCloudDataset(data_root=cfg.data_root,
                                voxel_size=cfg.voxel_size,
                                num_points_per_sample=hparams['data.num_points_per_sample'],
                                sample_radius=hparams['data.sample_radius'],
                                file_suffix=cfg.suffix,
                                scale_factor=cfg.scale)

    model = MODELS[cfg.model](**hparams)
    print('Load model from {}, trained on {}'.format(cfg.ckpt, hparams['data.type']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(cfg.ckpt, map_location=device)['state_dict'])
    model.to(device)
    model.eval()

    seed = hparams['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = hparams['data.num_samples']
    l2_normalize = True

    for idx in range(len(dataset)):
        data = dataset[idx]
        pcd = torch.from_numpy(data['pcd']).to(device)
        kpts = torch.from_numpy(data['kpts']).to(device)
        patches = torch.from_numpy(data['patches']).to(device)
        lrfs = torch.from_numpy(data['lrfs']).to(device)
        name = data['name']

        all_descs = list()
        num_kpts = len(kpts)
        print('Processing point cloud {}/{} with {} keypoints'.format(idx, len(dataset), num_kpts))
        for i in tqdm(range(0, num_kpts, batch_size), desc="computing features"):
            j = min(i + batch_size, num_kpts)
            with torch.no_grad():
                descs = model([pcd], [kpts[i:j, :]], [patches[i:j, :, :]], [lrfs[i:j, :, :]],
                              l2_normalize)
                descs = descs[0]
            all_descs.append(descs.cpu().numpy())
        all_descs = np.concatenate(all_descs, axis=0)
        # Save to
        np.save(osp.join(cfg.out_root, '{}.desc.npy'.format(name)),
                all_descs,
                allow_pickle=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--hparams', required=True)

    parser.add_argument('--data_root', required=True)
    parser.add_argument('--suffix', default='.off')
    parser.add_argument('--scale', type=float, default=4.0)
    parser.add_argument('--voxel_size', type=float, default=0.3)

    parser.add_argument('--out_root', required=True)

    return parser.parse_args()


if __name__ == '__main__':
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # fix seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    print("seed fixed!")

    cfg = parse_args()
    main(cfg)
