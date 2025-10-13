import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from utils import Flowers102ImagesDataset, CocoImagesDataset
from wmppc import all_correlations
from argparse import ArgumentParser
from rich import print

def get_last_layer(run):
    """
    Get n° of the last layer, to compute Comparative Sharedeness
    Usually 23,-1 or 12
    """
    subdirs = glob(f'sae_features/{run}/*')
    if len(subdirs) == 1: return subdirs[0]
    elif f'sae_features/{run}/layer-1.pt' in subdirs: return f'sae_features/{run}/layer-1.pt'
    return sorted(subdirs)[-1]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--source', default=None, type=str)
    parser.add_argument('--min_group', default=None,nargs='+', type=str)
    parser.add_argument('--max_group', default=None,nargs='+', type=str)
    args = parser.parse_args()


    M = torch.load(get_last_layer(args.source), map_location=device).to(torch.float32)
    S = M.sum(axis=0, dtype=torch.float32)


    rho_MG = torch.zeros_like(S)
    for enc in args.min_group:
        G_i = torch.load(get_last_layer(enc), map_location=device).to(torch.float32)
        corrs = all_correlations(M, G_i)
        rho_current = corrs.max(axis=1)[0]
        rho_MG = torch.minimum(rho_MG, rho_current)
        del G_i
        del corrs


    rho_MH = torch.zeros_like(S)
    for enc in args.max_group:
        H_i = torch.load(get_last_layer(enc), map_location=device).to(torch.float32)
        corrs = all_correlations(M, H_i)
        rho_current = corrs.max(axis=1)[0]
        rho_MH = torch.maximum(rho_MH, rho_current)
        del H_i
        del corrs


    Delta = S * (rho_MG**2 - rho_MH**2)

    _, top_feats = torch.topk(Delta, k=int(len(Delta) / 100))

    print("Top features : ", top_feats)