import os

# Set environment variables for OpenBLAS threading
# sklearn only supports up to 64 threads, it crashes if more are used
# (i.e. some server has 256 threads and it would crash there without limiting the number of threads)
# To still use all 256 threads, the training of the gmm models is done in parallel per disease class,
# since they are trained independently anyway. Training of the gmm models is relatively slow
# for even a small number of samples (60 patches at 1536x1536),
# therefore use all available processes to speed it up.
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"

import numpy as np
import pickle
from core.train_gmm import train_gmm, gmm_density_score, train_gmm_from_data
from tqdm import tqdm
#from multiprocessing import Pool
from pathlib import Path

from core.config import OUTPUT_PATH

#n_components_global = 10

def train_gmms_from_saved_activations(target_channel, n_components, load_folder, save_folder):
    load_path = os.path.join(load_folder, str(target_channel) + ".npy")

    if os.path.exists(load_path):
        above_cutoff_activations = np.load(load_path)
        print("activations shape: {}".format(above_cutoff_activations.shape))
        #above_cutoff_activations = above_cutoff_activations[:100]
        gmm = train_gmm_from_data(above_cutoff_activations, n_components=n_components)
        with open(os.path.join(save_folder, str(n_components) + "__" +str(target_channel) + ".pkl"), "wb") as file:
            pickle.dump(gmm, file)


# Define a helper function to wrap arguments for `train_gmms_from_saved_activations`
def process_gmm(class_to_use, n_components, load_folder, save_folder):
    train_gmms_from_saved_activations(class_to_use, n_components, load_folder, save_folder)

def train_gmm_from_activations(classes_to_use, n_components, load_folder, save_folder):
    #global n_components_global
    #n_components_global = n_components

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    for class_to_use in classes_to_use:
        process_gmm(class_to_use, n_components, load_folder, save_folder)

    # List of tuples to use in parallel processing




    #tuples_to_use = [[1], [32]]

    # Use a multiprocessing pool to run the function in parallel
    #with Pool(processes=2) as pool:
    #    list(tqdm(pool.imap(process_gmm, tuples_to_use, n_components, load_folder, save_folder), total=len(tuples_to_use), ascii=True))

model_name = "resnet50_robust"

load_folder = os.path.join(OUTPUT_PATH, "activations_for_gmm_training", model_name)
save_folder = os.path.join(OUTPUT_PATH, "gmm_models", model_name)

train_gmm_from_activations(classes_to_use=[248, 249, 250, 282, 430, 949, 950, 951], n_components=10, load_folder=load_folder, save_folder=save_folder)
