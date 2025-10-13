
import torch
import torchvision
import os

from pathlib import Path

from core.digipath_utils import load_decoder
from core.visualization import generate_gan_visualizations

from core.config import OUTPUT_PATH, DEVICE


decoder = load_decoder("version_299")
feature_extractor = decoder.feature_extractor

#folder_path = "/home/digipath2/projects/xai/digipath_xai_fast_api/highest_pixels/version_299_nmf_full_ds_decomp/"
#folder_path = os.path.join(OUTPUT_PATH, "version_299_layer3_nmf_decomp")

folder_path = os.path.join(OUTPUT_PATH, "contrasting", "version_299")

for class_combination_folder in os.listdir(folder_path):
    basis_path = os.path.join(folder_path, class_combination_folder, "nmf", "nmf_basis.pt")
    activation_vectors = torch.load(basis_path).to(DEVICE)
    print("generate visualizations...")
    visualizations_layer_0, visualizations_layer_3 = generate_gan_visualizations(activation_vectors, decoder, feature_extractor)

    torchvision.utils.save_image(visualizations_layer_0, os.path.join(folder_path, class_combination_folder, "nmf", "basis_0_"+str(1)+".jpg"))
    torchvision.utils.save_image(visualizations_layer_3, os.path.join(folder_path, class_combination_folder, "nmf", "basis_3_"+str(1)+".jpg"))








