from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from core.core import extract_attribution_filtered_activation_vectors, collect_patches
from core.linear_classifier import train_linear_classifier

from core.imagenet_utils import get_dataset_for_class, load_model, generate_prediction_list, get_dataset
from core.config import DEVICE, OUTPUT_PATH

from core.separate_features import train_linear_classifiers
import torch
from core.core import nmf_decomp_hyperplane_ood_filtered_activations, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model
from core.config import DEVICE, OUTPUT_PATH


#OUTPUT_PATH_ = "/home/digipath2/projects/xai/class_contrasting/out/"


def do_nmf_decomp(model_name, layer_name, hyperplane_normal, hyperplane_bias, class_idx, other_class, use_train_ds=False, num_samples=60, n_nmf_components=6):
    print("do nmf decomp")

    dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx, use_train_ds=use_train_ds)

    print("hyperplane normal: {}".format(hyperplane_normal))

    if dataset is not None:
        print("dataset is valid")
        if len(dataset) > 0:
            print("dataset len greater 0")
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

            model = load_model(model_name).eval().to(DEVICE)
            layer = model.layer4[2]

            patches = collect_patches(dataloader)
            if len(patches) > num_samples:
                patches = patches[:num_samples]
            if len(patches) > 0:
                print("num patches greater 0")
                dataset = TensorDataset(patches)
                # batch size has to be 1, code does not produce correct results with larger batch sizes yet (assumes batch size to be 1)
                dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

                save_folder_path = os.path.join(OUTPUT_PATH, "contrasting", model_name + layer_name, str(class_idx) + "_" + str(other_class))

                """
                gmm_path = os.path.join(OUTPUT_PATH, "gmm_models", model_name, "10__" + str(class_idx) + ".pkl")
                #gmm_path = os.path.join("/home/digipath2/projects/xai/digipath_xai_fast_api/gmm_models_diff_comps", "10__" + str(class_idx) + "_.pkl")
                with open(gmm_path, "rb") as file:
                    gmm_model = pickle.load(file)
                """

                print("run actual nmf decomp...")
                nmf_decomp_hyperplane_ood_filtered_activations(dataloader, model=model, layer=layer,
                                                            hyperplane_normal=hyperplane_normal,
                                                            hyperplane_bias=hyperplane_bias,
                                                            hyperplane_additional_bias=0.0,
                                                save_folder=save_folder_path, n_nmf_components=n_nmf_components,
                                                gmms=None)#[gmm_model])



#OUTPUT_PATH = "/home/digipath2/projects/xai/class_contrasting/out/"
#OUTPUT_PATH = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution")



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

                save_folder_path = os.path.join(OUTPUT_PATH, "activations_for_gmm_training", model_name + "_" + layer_name)

                above_cutoff_activations = extract_attribution_filtered_activation_vectors(dataset, target_channel=class_idx,
                                                model=model, wrapped_model=model, layer=layer,
                                                batch_size_attribution=16, batch_size_activation=256,
                                                attribution_cutoff=0.25,
                                                use_noise_tunnel=use_noise_tunnel)

                save_name = os.path.join(save_folder_path, str(class_idx) + ".npy")
                Path(save_folder_path).mkdir(parents=True, exist_ok=True)
                np.save(save_name, above_cutoff_activations.cpu().numpy())


def run_data_generation(class_to_explain, against_class, model_name="resnet50_robust", layer_name="layer3[5]"):
    #dataset = get_dataset(return_original_sample=False, use_train_ds=False)
    #generate_prediction_list(model_name=model_name, dataset=dataset)

    activations = np.load(os.path.join(OUTPUT_PATH, "activations_for_gmm_training", model_name + "_" + layer_name, str(class_to_explain) + ".npy"))
    activations_other = np.load(os.path.join(OUTPUT_PATH, "activations_for_gmm_training", model_name + "_" + layer_name, str(against_class) + ".npy"))

    activations = torch.from_numpy(activations)
    activations_other = torch.from_numpy(activations_other)

    training_activation_vecs = torch.cat([activations, activations_other], dim=0).to("cuda")
    targets = torch.zeros(len(activations), device="cuda")
    targets_other_feature = torch.ones(len(activations_other), device="cuda")
    training_targets = torch.cat([targets, targets_other_feature], dim=0)

    hyperplane_normal, hyperplane_bias, accuracy = train_linear_classifier(training_activation_vecs, training_targets)

    do_nmf_decomp(model_name, layer_name, -hyperplane_normal, hyperplane_bias, class_to_explain, against_class, use_train_ds=True, num_samples=500, n_nmf_components=8)

    print(hyperplane_normal)
    print(hyperplane_normal.shape)


if __name__ == "__main__":
    #run_data_generation(model_name="resnet50_robust", layer_name="layer3[5]")
    #run_data_generation(model_name="resnet50_robust", layer_name="layer4[2]")
    #run_data_generation(model_name="resnet50_robust", layer_name="layer2[3]")

    run_data_generation(model_name="resnet50_robust", layer_name="layer4[2]", class_to_explain=249, against_class=250)
    run_data_generation(model_name="resnet50_robust", layer_name="layer4[2]", class_to_explain=250, against_class=249)
    #run_data_generation(model_name="resnet50_robust", layer_name="layer4[2]", use_noise_tunnel=False, num_samples=500)

    #run_data_generation(model_name="resnet34", layer_name="layer3[5]")
    #run_data_generation(model_name="resnet50_robust", layer_name="layer3[5]")


    #generate_data("resnet50")
    #generate_data("resnet34")


    #train_linear_classifiers(activation_vecs_folder=os.path.join(OUTPUT_PATH, "activations_for_gmm_training", "resnet50"),
    #                         output_folder=os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "resnet50"),
    #                         features_list=[i for i in range(1000)])


