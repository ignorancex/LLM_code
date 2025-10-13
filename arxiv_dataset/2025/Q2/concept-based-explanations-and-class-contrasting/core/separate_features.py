import torch
import numpy as np

from tqdm import tqdm
import os
from pathlib import Path

from core.linear_classifier import train_linear_classifier


def load_activation_vecs_from_components(feature, model_folder_dir):
    activation_vecs = []
    folder_dir = os.path.join(model_folder_dir, str(feature), "vectors")

    if os.path.exists(folder_dir):
        for component, component_folder in enumerate(sorted(os.listdir(folder_dir))):
            if os.path.isdir(os.path.join(folder_dir, component_folder)):
                for file in sorted(os.listdir(os.path.join(folder_dir, component_folder))):
                    activation_vec = torch.load(os.path.join(folder_dir, component_folder, file))
                    activation_vecs.append(activation_vec)
        if len(activation_vecs) > 0:
            return torch.stack(activation_vecs, dim=0).float()
    return None


def load_activation_vecs(feature, folder_dir):
    file_path = os.path.join(folder_dir, str(feature) + ".npy")
    if os.path.exists(file_path):
        return torch.from_numpy(np.load(file_path)).float()
    return None


def train_linear_classifiers(activation_vecs_folder, output_folder, features_list=[i for i in range(1000)],
                             load_function=load_activation_vecs):
    #activation_vec = torch.load("highest_pixels/version_99_nmf_full_ds_decomp/1/vectors/0/0.pt")

    activation_vecs = []
    targets = []

    print("load activation vecs...")
    activation_vecs_per_feature = {}
    for feature in tqdm(features_list, ascii=True):
        activation_vecs = load_function(feature, activation_vecs_folder)
        if activation_vecs is not None:
            activation_vecs_per_feature[feature] = activation_vecs

            dummy_hyperplane_normal = torch.zeros(activation_vecs.shape[1], device="cuda")
            dummy_hyperplane_bias = torch.tensor(0.0, device="cuda")
            dummy_hyperplane_accuracy = torch.tensor(0.0, device="cuda")


    for i in tqdm(features_list, ascii=True):
        hyperplane_normal_list = []
        hyperplane_bias_list = []
        hyperplane_accuracy_list = []

        for j in features_list:
            is_dummy = False
            if not i in activation_vecs_per_feature:
                hyperplane_normal_list.append(dummy_hyperplane_normal)
                hyperplane_bias_list.append(dummy_hyperplane_bias)
                hyperplane_accuracy_list.append(dummy_hyperplane_accuracy)
                is_dummy = True
                continue
            if not j in activation_vecs_per_feature:
                hyperplane_normal_list.append(dummy_hyperplane_normal)
                hyperplane_bias_list.append(dummy_hyperplane_bias)
                hyperplane_accuracy_list.append(dummy_hyperplane_accuracy)
                is_dummy = True
                continue

            if not is_dummy:
                training_activation_vecs = torch.cat([activation_vecs_per_feature[i], activation_vecs_per_feature[j]], dim=0).to("cuda")
                targets = torch.zeros(len(activation_vecs_per_feature[i]), device="cuda")
                targets_other_feature = torch.ones(len(activation_vecs_per_feature[j]), device="cuda")
                training_targets = torch.cat([targets, targets_other_feature], dim=0)

                hyperplane_normal, hyperplane_bias, accuracy = train_linear_classifier(training_activation_vecs, training_targets)
                hyperplane_normal_list.append(hyperplane_normal)
                hyperplane_bias_list.append(hyperplane_bias)
                hyperplane_accuracy_list.append(accuracy)

        hyperplane_normal_list = torch.stack(hyperplane_normal_list, dim=0)
        hyperplane_bias_list = torch.stack(hyperplane_bias_list, dim=0)
        hyperplane_accuracy_list = torch.stack(hyperplane_accuracy_list, dim=0)

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        torch.save(hyperplane_normal_list, os.path.join(output_folder, str(i) + ".pt"))
        torch.save(hyperplane_bias_list, os.path.join(output_folder, "bias_" + str(i) + ".pt"))
        torch.save(hyperplane_accuracy_list, os.path.join(output_folder, "accuracy_" + str(i) + ".pt"))


    return



#run()
