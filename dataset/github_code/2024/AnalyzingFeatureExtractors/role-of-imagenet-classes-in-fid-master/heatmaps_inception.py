"""Visualize regions of a generated image to which is the most sensitive to."""

import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, \
                   Optional

from dataset import get_dataloader
from fid_heatmaps.heatmap_utils import ActivationsAndGradients, \
                                       compute_sensitivity_heatmap
from utils import bar_chart, \
                  create_grid, \
                  generate_images, \
                  load_feature_network, \
                  load_imagenet_labels, \
                  load_stylegan2, \
                  create_results_dir
import ast
from glob import glob

from torchvision.utils import save_image

import sys
sys.path.append("..\\cleanfid")
from utils_resize_dataset import ResizeDataset
from inception_torchscript import MyInceptionV3W,InceptionV3W
import platform
path = "./" if platform.system() == "Windows" else "/tmp"
device="cuda:0"
inception_v3_temp = load_feature_network(network_name='inception_v3_tf').to(device)
inception_v3 = MyInceptionV3W(path, inception_v3_temp, download=True).to(device)
inception_v3.eval()


def visualize_heatmaps(real_features,gen_features,gen_images_dataset,results_dir) -> None:

    #----------------------------------------------------------------------------
    # Load Inception-V3.
    #inception_v3 = load_feature_network(network_name='inception_v3_tf').to(device)
        
    #----------------------------------------------------------------------------
    # Compute feature statistics.
    print('Computing feature statistics...')
    mean_reals = torch.from_numpy(np.mean(real_features, axis=0)).to(device)
    cov_reals = torch.from_numpy(np.cov(real_features, rowvar=False)).to(device)
    mean_gen = torch.from_numpy(np.mean(gen_features, axis=0)).to(device)
    cov_gen = torch.from_numpy(np.cov(gen_features, rowvar=False)).to(device)

    #----------------------------------------------------------------------------
    # Register forward and backward hooks to get activations and gradients, respectively.
    acts_and_gradients = ActivationsAndGradients(network=inception_v3,network_kwargs=None)

    #----------------------------------------------------------------------------

    print('Visualizing heatmaps...')
    for gen_image,img_path in tqdm(gen_images_dataset):

        #----------------------------------------------------------------------------
        
        # Compute and visualize a sensitivity map.
        gen_image = gen_image.to(device).unsqueeze(0)

        overlay_heatmap,data = compute_sensitivity_heatmap(gen_image=gen_image,
                                                      acts_and_gradients=acts_and_gradients,
                                                      mean_reals=mean_reals,
                                                      cov_reals=cov_reals,
                                                      mean_gen=mean_gen,
                                                      cov_gen=cov_gen,
                                                      num_images=gen_features.shape[0])

        #----------------------------------------------------------------------------
        #D:\ACADEMIC\gen_imgs_50k\stylegan2-ffhq-256x256\seed0000.png
        img_name = img_path.split("\\")[-1][:-4]
        #save_image(gen_image[0]/255.0,os.path.join(results_dir, f'{img_name}_gen_image.png'))
        overlay_heatmap.save(os.path.join(results_dir, f'{img_name}_heatmap.png'))
        np.save(os.path.join(results_dir, f'{img_name}_data.npy'),data)



def main() -> None:

    results_dir = create_results_dir(results_root='my_result_heatmaps_inception',
                                     description="stylegan2-ffhq-256x256")
    real_features = np.load("<your path>.npy")
    gen_features = np.load("<your path>.npy")

    dataset = ResizeDataset(glob("<your path>\\*.png"), fdir=None, mode="clean")
    # Visualize FID sensitivity heatmaps.
    visualize_heatmaps(real_features,gen_features,dataset,results_dir)


if __name__ == "__main__":
    main()
