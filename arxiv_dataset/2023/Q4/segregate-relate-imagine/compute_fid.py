# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import os
from os import path as osp
from tqdm import tqdm
import pathlib
import shutil

import torch
import numpy as np
from PIL import Image

from third_party.genesis_v2.geco import GECO
from sri.model import SRI
from third_party.pytorch_fid import fid_score as FID
from third_party.genesis_v2.shapestacks_config import ShapeStacksDataset
import argparse

local_rank = os.environ['LOCAL_RANK']
torch.set_printoptions(threshold=10000, linewidth=300)


def restore_from_checkpoint(config, checkpoint):
    global local_rank
    map_location = {'cuda:0': local_rank}
    state = torch.load(checkpoint, map_location=map_location)
    
    model = SRI(config['K'], config['pixel_bound'], config['z_dim'], 
                config['semiconv'], config['kernel'], config['pixel_std'],
                config['dynamic_K'], config['image_likelihood'], False,
                config['img_size'], config['s_dim'], config['L'])

    model.load_state_dict(state['model_state_dict'])
    print(f'loaded {checkpoint}')
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank)
    return model


def tensor_to_png(tensor, save_dir, count, stop):
    np_images = tensor.cpu().numpy()
    np_images = np.moveaxis(np_images, 1, 3)
    for i in range(len(np_images)):
        im = Image.fromarray(np.uint8(255*np_images[i]))
        fn = osp.join(save_dir, str(count).zfill(6) + '.png')
        im.save(fn)
        count += 1
        if count >= stop:
            return count
    return count


def fid_from_model(model, test_loader, ckpt, results_file=None, batch_size=10,
                    num_images=10000, feat_dim=2048, img_dir='/tmp', temp=1.0,
                    verbose=True, model_name='SRI'):
    # Save images from test set as pngs
    
    test_dir = osp.join(img_dir, 'test_images')
    if not os.path.isdir(test_dir):
        if verbose:
            print("Saving images from test set as pngs.")
        os.makedirs(test_dir)
        count = 0
        for bidx, batch in enumerate(test_loader):
            batch['imgs'] = (batch['imgs'] + 1) / 2.
            count = tensor_to_png(batch['imgs'], test_dir, count, num_images)
            if count >= num_images:
                break

    # Generate images and save as pngs
    print("Generate images and save as pngs.")
    gen_dir = osp.join(img_dir, 'generated_images')
    # Delete the folder if it already exists
    if os.path.isdir(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir)
    count = 0
    for _ in tqdm(range(num_images // batch_size + 1), disable=not verbose):
        with torch.no_grad():
            recons, _ = model.module.sample(batch_size=batch_size, temp=temp, K_steps=model.module.K_steps)

        count = tensor_to_png(recons, gen_dir, count, num_images)
        if count >= num_images:
            break
    # Compute FID
    if verbose:
        print("Computing FID.")
    gpu = next(model.parameters()).is_cuda
    fid_value = FID.calculate_fid_given_paths(
        [test_dir, gen_dir], batch_size, gpu, feat_dim, verbose)
    
    print(f"FID: {fid_value}")

    model.train()
    if results_file is not None:
        with open(results_file, 'a+') as f:
            f.write('{},{}\n'.format(ckpt,fid_value))
    return fid_value


def fid_from_folder(model_folder, test_loader, results_file, img_dir='/tmp', batch_size=10, num_images=10000,
                    feat_dim=2048, verbose=False):
     # Save images from test set as pngs
    
    test_dir = osp.join(img_dir, 'test_images')
    if not os.path.isdir(test_dir):
        if verbose:
            print("Saving images from test set as pngs.")
        os.makedirs(test_dir)
        count = 0
        for bidx, batch in enumerate(test_loader):
            batch['imgs'] = (batch['imgs'] + 1) / 2.
            count = tensor_to_png(batch['imgs'], test_dir, count, num_images)
            if count >= num_images:
                break
    # Compute FID
    if verbose:
        print("Computing FID.")

    fid_value = FID.calculate_fid_given_paths(
        [test_dir, model_folder], batch_size, True, feat_dim, verbose)
    
    print(f"FID: {fid_value}")

    if results_file is not None:
        with open(results_file, 'a+') as f:
            f.write('{},{}\n'.format(model_folder,fid_value))
    return fid_value


def run(config, seed):
    global local_rank

    # Fix random seed
    print(f'setting random seed to {seed}')
    
    # Auto-set by sacred
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    local_rank = 'cuda:{}'.format(local_rank)
    assert local_rank == 'cuda:0', 'Eval should be run with a single process'

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(config['DDP_port'])
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    te_dataset = ShapeStacksDataset(data_dir=config['data_dir'], mode=config['mode'], load_instances=False)
    te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=config['batch_size'],
                                                shuffle=True, num_workers=config['num_workers'],
                                                drop_last=True)
    checkpoint = pathlib.Path(config['checkpoint_dir'], config['checkpoint'])
    
    if config['from_folder'] == "":
        model = restore_from_checkpoint(config, checkpoint)
        model.eval()
        # Compute FID
        fid_from_model(model, te_dataloader, config['checkpoint'], 
                    config['results_file'], config['batch_size'],
                    config['num_fid_images'], config['feat_dim'], config['img_dir'],
                    1.0, model_name='SRI')
    else:
        fid_from_folder(config['from_folder'], te_dataloader, config['results_file'],config['img_dir'],
                        config['batch_size'], config['num_fid_images'], config['feat_dim'], verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('segregate-reglate-imagine FID')

    parser.add_argument('--DDP_port', type=int, default=29500,
                        help='torch.distributed config')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Workers for dataloading')
    
    parser.add_argument('--from_folder', type=str, default='',
                        help='Provide a folder containing generated images. Used ' \
                             'for computing FID for a model defined external to this repo.')
    parser.add_argument('--feat_dim', type=int, default=2048,
                        help='FID net feature dim')
    parser.add_argument('--num_fid_images', type=int, default=10000,
                        help='# of images to use for FID calculation')
    parser.add_argument('--img_dir', type=str, default='/tmp',
                        help='.png directory')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Generate images in batches')
    parser.add_argument('--mode', type=str, default='test',
                        help='Name of dataset split')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Parent folder of data')
                        
    parser.add_argument('--checkpoint_dir', type=str, default='./model',
                        help='The dir containing the provided checkpoint file')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Trained model weights')
    parser.add_argument('--results_file', type=str, default='results_FID.txt',
                        help='File to store FID score in')
    
    ### Model
    parser.add_argument('--z_dim', type=int, default=64,
                        help='slot dim')
    parser.add_argument('--s_dim', type=int, default=128,
                        help='s dim')
    parser.add_argument('--L', type=int, default=3,
                        help='number of relation layers in scene encoder')
    parser.add_argument('--K', type=int, default=9,
                        help='number of slots')       
    ###########################################                                                             
    parser.add_argument('--kernel', type=str, default='gaussian',
                        help='{laplacian,gaussian,epanechnikov}')
    parser.add_argument('--semiconv', action='store_true', default=True,
                        help='semiconv for GENESISv2')
    parser.add_argument('--dynamic_K', action='store_true', default=False,
                        help='dynamic_K for GENESISv2')
    parser.add_argument('--pixel_bound',  action='store_true', default=True,
                        help='constrain decoder outputs')
    parser.add_argument('--img_size', type=int, default=64,
                        help='side length of square image inputs')
    parser.add_argument('--pixel_std', type=float, default=0.7,
                        help='Global std dev for image likelihood')
    parser.add_argument('--image_likelihood', type=str, default='MoG-Normed',
                        help='MoG or MoG-Normed (use normalization constant)')
                        
    args = vars(parser.parse_args())

    run(args, args['seed'])
