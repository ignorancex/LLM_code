import random
import warnings
import copy
import torch
import gc
import pickle
import os
import sys
import argparse
import json
sys.path.append("../models")
sys.path.append("../helpers")

from vgg import VGG, make_layers
from resnet import ResNet, BasicBlock, Bottleneck

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from extract_activations_dist import ExtractAct

from noise import add_gaussian_noise, add_salt_and_pepper, add_gaussian_blur, add_shot_noise, add_impulse_noise, add_frost
from save_images import save_img

import numpy as np
import pandas as pd

from vgg import VGG, make_layers
from resnet import ResNet, BasicBlock, Bottleneck

from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import seaborn as sns

from data_classes import class_count


warnings.filterwarnings('ignore')


print("Imports done")

MODEL_PATH_ROOT = "PATH_TO_MODELS_DIR"
RESULTS_PATH_ROOT = "PATH_TO_SAVE_RESULTS"


###################################################################################################################
###################################################################################################################


transform_toTensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
transform_toPIL = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
transform_toTensor_Normalize = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)), #Resize to 64x64 if using TinyImageNet
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #Change mean and standard deviation to (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) if using TinyImageNet



def output_to_std_and_file(directory, file_name, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    print(data)
    with open(file_path, "a") as file:
        file.write(data)


class DataTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def create_dataframe(groundtruth, features):
    data = {'GroundTruth' : groundtruth}

    for i in range(features.shape[1]):
        col_name = f"Value_{i}"
        data[col_name] = features[:, i]

    df = pd.DataFrame(data)
    cluster_centroids = df.groupby('GroundTruth').mean().reset_index()

    return df, cluster_centroids


def cluster_w_centroids(dataframe, centroids_np, classes):
    dataframe_features = dataframe.iloc[:, 1:]
    kmeans = KMeans(n_clusters=classes, init=centroids_np, n_init=1, random_state=0)
    cluster_labels = kmeans.fit_predict(dataframe_features)
    dataframe.insert(loc=dataframe.columns.get_loc("GroundTruth") + 1, column="ClusterLabels", value=cluster_labels)

    ground_truth = dataframe["GroundTruth"]
    predicted_labels = dataframe["ClusterLabels"]
    ari_score = adjusted_rand_score(ground_truth, predicted_labels)

    return ari_score


def cluster_kmeans_plusplus(dataframe, classes):
    dataframe.drop('ClusterLabels', axis=1, inplace=True)
    dataframe_features = dataframe.iloc[:, 1:]
    kmeans = KMeans(n_clusters=classes, init='k-means++', n_init=1, random_state=0)
    cluster_labels = kmeans.fit_predict(dataframe_features)
    dataframe.insert(loc=dataframe.columns.get_loc("GroundTruth") + 1, column="ClusterLabels", value=cluster_labels)

    ground_truth = dataframe["GroundTruth"]
    predicted_labels = dataframe["ClusterLabels"]
    ari_score = adjusted_rand_score(ground_truth, predicted_labels)

    return ari_score


def tsne_plot(dataframe, save_loc, save_name):
    dataframe_features = dataframe.iloc[:, 1:]
    labels = dataframe['GroundTruth']

    tsne = TSNE(n_components=2, random_state=1)
    tsne_result = tsne.fit_transform(dataframe_features)

    tsne_df = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['GroundTruth'] = labels

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_df['Dimension 1'], tsne_df['Dimension 2'], c=tsne_df['GroundTruth'])
    plt.title('t-SNE Plot')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(handles=scatter.legend_elements()[0], title='GroundTruth')

    plt.savefig(f'{save_loc}/{save_name}.png')


###################################################################################################################
###################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, help='path to model')
parser.add_argument('-s', type=str, help='Save model and data location')
parser.add_argument('-n_t', type=str, help='Noise type', default=None)
parser.add_argument('-n_l', type=float, help='Noise level', default=None)
parser.add_argument('-d', type=str, help='Dataset')
parser.add_argument('-rs', type=int, help='Random seed')
parser.add_argument('-cl', type=str, help='Final conv or embedding layer')
parser.add_argument('-mt', type=str, help='Model type')
parser.add_argument('-tsne', type=str, help='Plot t-SNE ?', default="False")

args = parser.parse_args()

model_path = os.path.join(MODEL_PATH_ROOT, args.m)
save_to = os.path.join(RESULTS_PATH_ROOT, args.s)
noise_type = args.n_t
noise_level = args.n_l
data = args.d
random_seed = args.rs
final_conv_layer = args.cl
model_type = args.mt
tsne_flag = "t" in args.tsne.lower()

std_string = f"Model path: {model_path}\n"
std_string += f"Save results to: {save_to}\n"
std_string += f"Noise type: {noise_type}\n"
std_string += f"Noise level: {noise_level}\n"
std_string += f"Dataset: {data}\n"
std_string += f"Random seed: {random_seed}\n"
std_string += f"Final embedding layer name: {final_conv_layer}\n"
std_string += f"Model type: {model_type}\n"
std_string += f"plot t-SNE: {tsne_flag}\n"

output_to_std_and_file(save_to, "standard_output.txt", std_string)

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False


###################################################################################################################
###################################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_to_std_and_file(save_to, "standard_output.txt", f"\n{device}")

model = pickle.load(open(model_path, "rb"))
model = model["model"]
model.to(device) 


###################################################################################################################
###################################################################################################################


if data.lower() == "cifar10":
    dataset = "CIFAR10"
    trainset = torchvision.datasets.CIFAR10(root='PATH_TO_CIFAR10', train=True, download=True)
elif data.lower() == "cifar100":
    dataset = "CIFAR100"
    trainset = torchvision.datasets.CIFAR100(root='PATH_TO_CIFAR100', train=True, download=True)
elif data.lower() == "gtsrb":
    dataset = "GTSRB"
    trainset = torchvision.datasets.GTSRB(root='PATH_TO_GTRSB', split='train', download=True)
elif data.lower() == "stl10":
    dataset = "STL10"
    trainset = torchvision.datasets.STL10(root='PATH_TO_STL10', split='train', download=True)
elif data.lower() == "mnist":
    dataset = "MNIST"
    trainset = torchvision.datasets.MNIST(root='PATH_TO_MNIST', train=True, download=True)
    transform_toTensor_Normalize = transform_gray
elif data.lower() == "emnist":
    dataset = "EMNIST"
    trainset = torchvision.datasets.EMNIST(root='PATH_TO_EMNIST', split='balanced', train=True, download=True)
    transform_toTensor_Normalize = transform_gray
elif data.lower() == "fashionmnist" or data.lower() == "fashmnist":
    dataset = "FashionMNIST"
    trainset = torchvision.datasets.FashionMNIST(root='PATH_TO_FashionMNIST', train=True, download=True)
    transform_toTensor_Normalize = transform_gray
elif data.lower() == "svhn":
    dataset = "SVHN"
    trainset = torchvision.datasets.SVHN(root='PATH_TO_SVHN', split='train', download=True)
elif data.lower() == "food101":
    dataset = "Food101"
    trainset = torchvision.datasets.Food101(root='PATH_TO_Food101', split='train', download=True)

#If tiny ImageNet, use follwoing code
#trainset = datasets.ImageFolder(tinyimagenet_train_data_path)

labels, classes = class_count(trainset)
index_positions = [int(len(trainset) * (i + 1) / 6) for i in range(5)]
batch_size_options = [32, 20, 16, 15, 10, 8, 5, 4, 2]

for each_size in batch_size_options:
    if len(trainset)%each_size == 0:
        batch_size = each_size
        break

dataset += f"\nTrain size: {len(trainset)}"
dataset += f"\nClasses: {classes}"
std_string += f"\nBatch size: {batch_size}\n"

output_to_std_and_file(save_to, "standard_output.txt", dataset)


###################################################################################################################
###################################################################################################################


save_img(trainset, os.path.join(save_to, "clean_imgs"), index_positions)
trainset_norm_tensor = DataTransform(trainset, transform=transform_toTensor_Normalize)
trainloader_clean = DataLoader(trainset_norm_tensor, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


###################################################################################################################
###################################################################################################################


noise_type = noise_type.lower()
if "blur" in noise_type:
    std_string += "\nGaussian blur"
    noise_function = add_gaussian_blur
elif "sp" in noise_type or "salt_pepper" in noise_type:
    std_string += "\nSalt-and-pepper noise"
    noise_function = add_salt_and_pepper
elif "shot" in noise_type or "shot_noise" in noise_type or "poisson" in noise_type or "poisson_noise" in noise_type:
    std_string += "\nShot noise"
    noise_function = add_shot_noise
elif "gauss" in noise_type:
    std_string += "\nGaussian noise"
    noise_function = add_gaussian_noise
elif "impulse" in noise_type:
    std_string += "\nImpulse noise"
    noise_function = add_impulse_noise
elif "frost_2" in noise_type:
    std_string += "\nFrost noise 2"
    noise_function = add_frost_2
elif "frost" in noise_type:
    std_string += "\nFrost noise"
    noise_function = add_frost

trainset_tensor =  DataTransform(trainset, transform=transform_toTensor)
trainset_noise = [(noise_function(x, noise_level), y) for x, y in trainset_tensor]

trainset = DataTransform(trainset_noise, transform=transform_toPIL)
save_img(trainset, os.path.join(save_to, "noise_imgs"), index_positions)

trainset_norm_tensor = DataTransform(trainset, transform=transform_toTensor_Normalize)
trainloader_noise = DataLoader(trainset_norm_tensor, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


###################################################################################################################
###################################################################################################################


extractObject = ExtractAct(model, device, final_conv_layer, classes, trainloader_clean, trainloader_noise, [16, 32, 64], model_type)
clean_final_embeddings, noise_final_embeddings, groundtruths, final_distances, distances = extractObject.forward_pass()


###################################################################################################################
###################################################################################################################


df1_clean, centroids1_clean = create_dataframe(groundtruths, clean_final_embeddings)
centroids1_clean_np = centroids1_clean.iloc[:, 1:].values

df1_noise, centroids1_noise = create_dataframe(groundtruths, noise_final_embeddings)
centroids1_noise_np = centroids1_noise.iloc[:, 1:].values


###################################################################################################################
###################################################################################################################


df1_noise_2 = copy.deepcopy(df1_noise)

if tsne_flag:
    tsne_plot(df1_clean, save_to, "clean")
    tsne_plot(df1_noise, save_to, "noise")

ari_score_df1_clean_C = cluster_w_centroids(df1_clean, centroids1_clean_np, classes)
ari_score_df1_clean_Kpp = cluster_kmeans_plusplus(df1_clean, classes)
output_to_std_and_file(save_to, "standard_output.txt", "\nDone with clustering 1")
df1_clean = None
del df1_clean
gc.collect()

ari_score_df1_noise_C = cluster_w_centroids(df1_noise, centroids1_noise_np, classes)
ari_score_df1_noise_Kpp = cluster_kmeans_plusplus(df1_noise, classes)
output_to_std_and_file(save_to, "standard_output.txt", "\nDone with clustering 2")
df1_noise = None
del df1_noise
gc.collect()

ari_score_df1_noise_2_C = cluster_w_centroids(df1_noise_2, centroids1_clean_np, classes)
output_to_std_and_file(save_to, "standard_output.txt", "\nDone with clustering 3")
df1_noise_2 = None
del df1_noise_2
gc.collect()

std_string += f"\n\nARI Score, explicits centroids:"
std_string += f"\nARI Score DF1 Clean: {ari_score_df1_clean_C}"
std_string += f"\nARI Score DF1 Noise_1: {ari_score_df1_noise_C}"
std_string += f"\nARI Score DF1 Noise_2: {ari_score_df1_noise_2_C}"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

std_string = f"\n\nARI Score, Kmeans++:"
std_string += f"\nARI Score DF1 Clean: {ari_score_df1_clean_Kpp}"
std_string += f"\nARI Score DF1 Noise: {ari_score_df1_noise_Kpp}"
output_to_std_and_file(save_to, "standard_output.txt", std_string)


###################################################################################################################
###################################################################################################################


final_results = {}

final_results["distances"] = final_distances
final_results["inv_ari"] = {}

final_results["inv_ari"]["explicit_centroids"] = {
    "clean_df1" : 1 - ari_score_df1_clean_C,
    "noise_1_df1" : 1 - ari_score_df1_noise_C, 
    "noise_2_df1" : 1 - ari_score_df1_noise_2_C, 
}

final_results["inv_ari"]["Kmeans_pluplus"] = {
    "clean_df1" : 1 - ari_score_df1_clean_Kpp,
    "noise_1_df1" : 1 - ari_score_df1_noise_Kpp, 
}

with open(os.path.join(save_to, "all_measures.json"), "w") as f:
    json.dump(final_results, f)


###################################################################################################################
###################################################################################################################