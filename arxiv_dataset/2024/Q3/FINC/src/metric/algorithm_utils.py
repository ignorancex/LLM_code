import torch
import numpy as np
from torch.linalg import eigh, eigvalsh, eigvals
from torch.distributions import Categorical
from torchvision.utils import save_image
import os
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm

def visualize_mode_by_eigenvectors_rff(test_feats, test_dataset, test_idxs, args):
    args.logger.info('Start compute covariance matrix')
    x_cov, _, x_feature = cov_rff(test_feats, args.rff_dim, args.sigma, args.batchsize)

    test_idxs = test_idxs.to(dtype=torch.long)

    args.logger.info('Start compute eigen-decomposition')
    eigenvalues, eigenvectors = torch.linalg.eigh(x_cov)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)

    m, max_id = eigenvalues.topk(args.num_visual_mode)

    now_time = args.current_time

    for i in range(args.num_visual_mode):

        top_eigenvector = eigenvectors[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (x_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        topk_id = s_value.topk(args.num_img_per_mode)[1]
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}/{}_{}/'.format(args.backbone, args.visual_name, now_time), 'top{}'.format(i+1))
        os.makedirs(save_folder_name)
        summary = []

        for j, idx in enumerate(test_idxs[topk_id.cpu()]):
            top_imgs = transform(test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        save_image(summary, os.path.join(save_folder_name, 'summary.png'.format(j)), nrow=5)

def visualize_novel_mode_by_eigenvectors_rff(test_feats, ref_feats, test_dataset, test_idxs, args):
    args.logger.info('Start compute covariance matrix')
    x_cov, y_cov, _, x_feature, _ = cov_diff_rff(test_feats, ref_feats, args.rff_dim, args.sigma, args.batchsize)
    diff_cov = x_cov - args.eta * y_cov

    test_idxs = test_idxs.to(dtype=torch.long)

    args.logger.info('Start compute eigen-decomposition')
    eigenvalues, eigenvectors = torch.linalg.eigh(diff_cov)
    args.logger.info('Finish compute eigen-decomposition')
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)

    m, max_id = eigenvalues.topk(args.num_visual_mode)

    now_time = args.current_time

    args.logger.info('Start visualizing novel modes')

    for i in range(args.num_visual_mode):

        top_eigenvector = eigenvectors[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (x_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        topk_id = s_value.topk(args.num_img_per_mode)[1]
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}/{}_{}/'.format(args.backbone, args.visual_name, now_time), 'top{}'.format(i+1))
        os.makedirs(save_folder_name)
        summary = []

        for j, idx in enumerate(test_idxs[topk_id.cpu()]):
            top_imgs = transform(test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        save_image(summary, os.path.join(save_folder_name, 'summary.png'.format(j)), nrow=5)

def cov_rff2(x, feature_dim, std, batchsize=16, presign_omeaga=None):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product) # [B, feature_dim]
    batched_rff_sin = torch.sin(product) # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim) # [B, 2 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]

    cov = torch.zeros((2 * feature_dim, 2 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 2

    return cov, batched_rff.squeeze()

def cov_diff_rff(x, y, feature_dim, std, batchsize=16):
    assert len(x.shape) == len(y.shape) == 2 # [B, dim]

    B, D = x.shape
    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to('cuda' if torch.cuda.is_available() else 'cpu')

    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)
    y_cov, y_feature = cov_rff2(y, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)

    return x_cov, y_cov, omegas, x_feature, y_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim], [B, 2 * feature_dim]

def cov_rff(x, feature_dim, std, batchsize=16):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D = x.shape
    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)

    return x_cov, omegas, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]
