from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from core.core import extract_attribution_filtered_activation_vectors, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model, generate_prediction_list, get_dataset
from core.config import DEVICE, OUTPUT_PATH

from core.separate_features import train_linear_classifiers
import torch


#OUTPUT_PATH = "/home/digipath2/projects/xai/class_contrasting/out/"
OUTPUT_PATH = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution")



def generate_data(model_name="resnet50_robust", layer_name="layer3[5]", use_noise_tunnel=True, num_samples=60):
    for class_idx in tqdm(range(1000), ascii=True):
        #for class_idx in [249, 250]:
        dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx, use_train_ds=True)
        if dataset is not None:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

            model = load_model(model_name).eval().to(DEVICE)
            #print(model)
            layer = eval("model."+layer_name) #model.layer3[2]

            torch.manual_seed(42)
            patches = collect_patches(dataloader)
            if len(patches) > num_samples:
                patches = patches[:num_samples]
            if len(patches) > 0:
                dataset = TensorDataset(patches)

                save_folder_path = os.path.join(OUTPUT_PATH, "activations_for_classifier_training", model_name + "_" + layer_name)

                above_cutoff_activations = extract_attribution_filtered_activation_vectors(dataset, target_channel=class_idx,
                                                model=model, wrapped_model=model, layer=layer,
                                                batch_size_attribution=16, batch_size_activation=256,
                                                attribution_cutoff=0.25,
                                                use_noise_tunnel=use_noise_tunnel)

                save_name = os.path.join(save_folder_path, str(class_idx) + ".npy")
                Path(save_folder_path).mkdir(parents=True, exist_ok=True)
                np.save(save_name, above_cutoff_activations.cpu().numpy())


def run_data_generation(model_name="resnet50_robust", layer_name="layer3[5]", use_noise_tunnel=True, num_samples=60):

    generate_data(model_name=model_name, layer_name=layer_name, use_noise_tunnel=use_noise_tunnel, num_samples=num_samples)

    train_linear_classifiers(activation_vecs_folder=os.path.join(OUTPUT_PATH, "activations_for_classifier_training", model_name + "_" + layer_name),
                            output_folder=os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", model_name + "_" + layer_name),
                            features_list=[i for i in range(1000)])


run_data_generation(model_name="resnet50", layer_name="layer4[2]", use_noise_tunnel=False, num_samples=500)
run_data_generation(model_name="resnet50", layer_name="layer3[5]", use_noise_tunnel=False, num_samples=500)

