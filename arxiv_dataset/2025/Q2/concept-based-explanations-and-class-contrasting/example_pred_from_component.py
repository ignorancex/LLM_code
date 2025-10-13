from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
import random

from core.core import extract_attribution_filtered_activation_vectors, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model, generate_prediction_list, get_dataset, get_dataset_excluding_class, CroppedImageNetDataset
from core.config import DEVICE, OUTPUT_PATH

from core.separate_features import train_linear_classifiers
from example_generate_basis_visualization_from_data_imagenet import sample_closest_image_patches, generate_data, generate_data_including_prediction

from example_nmf_decomp_contrasting_imagenet import do_nmf_decomp




def get_concept_patches(activation_vectors, activations, image_dataset):
    all_concept_patches = []
    for component_index in range(len(activation_vectors)):
        concept_patches = sample_closest_image_patches(activation_vectors[component_index], activations, image_dataset, n_to_sample=1, is_image_dataset_list=True)
        all_concept_patches.append(concept_patches)

    return torch.cat(all_concept_patches, dim=2)



def get_pred(class_idx, activations, image_dataset, activation_vectors):
    concept_patches = get_concept_patches(activation_vectors, activations, image_dataset).to(DEVICE) #
    concept_patches = torch.cat([concept_patches[i] for i in range(len(concept_patches))], dim=-1).unsqueeze(dim=0)

    with torch.no_grad():
        pred = F.softmax(model(concept_patches), dim=1)
        pred_class = torch.mean(pred[:, class_idx])
        max_class = torch.argmax(torch.mean(pred, dim=0))
        max_pred = torch.max(torch.mean(pred, dim=0))

        print("class idx: {}".format(class_idx))
        print("pred class: {}, max_class: {}, max_pred: {}".format(pred_class, max_class, max_pred))




    return pred_class, max_class, max_pred


def run_test(activations_all, prediction_classes, save_name, exclude_target_class_from_patches=False, save_output=True,
             folder_path=None, class_frequency=10, craft_num_comps=6):

    if folder_path is None:
        folder_path = os.path.join(OUTPUT_PATH, model_name + layer_name)
    folder_path_contrasting = os.path.join(OUTPUT_PATH, "contrasting", model_name + layer_name)

    class_preds = []
    matches_desired_pred = []


    for class_idx in tqdm(range(1000), ascii=True):
        if class_idx % class_frequency != 0:
            continue

        if exclude_target_class_from_patches:
            indices = np.where(prediction_classes != class_idx)[0]
            image_dataset = torch.utils.data.Subset(dataset, indices)
            activations = activations_all[indices]
        else:
            image_dataset = dataset
            activations = activations_all

        #image_dataset = dataset

        class_combination_folder = str(class_idx)
        basis_path = os.path.join(folder_path, class_combination_folder, "nmf", "nmf_basis.pt")
        if os.path.exists(basis_path):
            activation_vectors = torch.load(basis_path).to(DEVICE)
        else:
            # in case CRAFT is used, for comparison with our method
            basis_path = os.path.join(folder_path, class_combination_folder, "nmf", "nmf.npy")
            activation_vectors = np.load(basis_path)

            importances_path = os.path.join(folder_path, class_combination_folder, "nmf", "importances.npy")
            importances = np.load(importances_path)
            most_important_concepts = np.argsort(importances)[::-1]#[:5]
            activation_vectors = activation_vectors[most_important_concepts[:craft_num_comps]]
            activation_vectors = torch.from_numpy(activation_vectors).to(DEVICE)


        pred_class, max_class, max_pred = get_pred(class_idx, activations, image_dataset, activation_vectors)

        class_preds.append(pred_class)
        if max_class == class_idx:
            matches_desired_pred.append(torch.tensor(1.0))
        else:
            matches_desired_pred.append(torch.tensor(0.0))



    class_preds = torch.stack(class_preds, dim=0)
    matches_desired_pred = torch.stack(matches_desired_pred, dim=0)

    out_save_path = save_name
    Path(out_save_path).mkdir(parents=True, exist_ok=True)

    print("average pred: {}".format(torch.mean(class_preds)))
    print("average matches_desired_pred: {}".format(torch.mean(matches_desired_pred)))

    if save_output:
        if not exclude_target_class_from_patches:
            torch.save(class_preds, os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
            torch.save(matches_desired_pred, os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))
        else:
            torch.save(class_preds, os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
            torch.save(matches_desired_pred, os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))





model_name="resnet50"
layer_name="_layer4[2]"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]

dataset = get_dataset(use_train_ds=True)
dataset = CroppedImageNetDataset(dataset)
dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=8)

# Needs to run all 11.5 million image crops, takes around 2 hours on an RTX2080Ti. Will also take some RAM (likely more than 46GB)
# Builds the vector database in order of sampling the closest dataset patches (for visualizing an NMF basis vector, we compare
# it via cosine similarity with each of the 11.5 million vectors in the database and sample the n closest ones)
# This needs to be repeated if a different layer is used, since the NMF basis vectors are computed at a specific layer and the
# vector database needs to be at that same layer
# The predictions for each of the 11.5 million crops is also computed, since we optionally exclude crops that are predicted
# as the class that should be explained
activations_all, prediction_classes = generate_data_including_prediction(model, layer, dataloader)
prediction_classes = prediction_classes.cpu().numpy()

#---------------------------------------------------------------------------------


# ResNet50 model Layer4.2 all classes
# For Table 1 and 2

folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer4[2]_4_comps")
save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer4[2]_4_comps")

run_test(activations_all, prediction_classes,
            exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
        save_name=save_name, class_frequency=1)
run_test(activations_all, prediction_classes,
            exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=1)



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

# CRAFT as comparison, for the ResNet50 model Layer4.2 every 10th class using 1 to 10 components filtered by CRAFT's importance score
# For Figure 4 and 13

folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer4[2]_craft")

for num_comp in range(10):
    save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer4[2]_craft_" + str(num_comp) + "_comps")
    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10, craft_num_comps=num_comp)
    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
                save_name=save_name, class_frequency=10, craft_num_comps=num_comp)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

# Vary the number of samples, ResNet50 Layer4.2
# For Figure 11

number_of_samples = [50] + [(i+1)*100 for i in range(9)]
for num_samples in number_of_samples:
    folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer4[2]"+"_4_comps_"+str(num_samples)+"_samples")
    save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer4[2]"+str(num_samples)+"_samples")

    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10)
    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
                save_name=save_name, class_frequency=10)


#---------------------------------------------------------------------------------


# Vary the number of components between 1 and 10, ResNet50 Layer4.2
# (code calling this for layer3.5, layer2.3 and layer1.2 is further below)
# For Figure 4 and 13

for num_comp in range(10):
    folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer4[2]"+"_"+str(num_comp)+"_comps_grid_search")
    save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer4[2]"+"_"+str(num_comp)+"_comps_grid_search")

    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10)
    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
                save_name=save_name, class_frequency=10)





#---------------------------------------------------------------------------------

# Change the attribution method to Gradient times Activation
# For Table 6 and 7

folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer4[2]_4_comps_gradientXactivation")
save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer4[2]_4_comps_gradientXactivation")

run_test(activations_all, prediction_classes,
            exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
        save_name=save_name, class_frequency=10)
run_test(activations_all, prediction_classes,
            exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10)



# Change the attribution method to DeepLift with SmoothGrad
# For Table 6 and 7

folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer4[2]_4_comps_deeplift_smoothgrad")
save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer4[2]_4_comps_deeplift_smoothgrad")

run_test(activations_all, prediction_classes,
            exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
        save_name=save_name, class_frequency=10)
run_test(activations_all, prediction_classes,
            exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10)






#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------




model_name="resnet50"
layer_name="_layer3[5]"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]

dataset = get_dataset(use_train_ds=True)
dataset = CroppedImageNetDataset(dataset)
dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=8)

# Needs to run all 11.5 million image crops, takes around 2 hours on an RTX2080Ti. Will also take some RAM (likely more than 46GB)
# We need to rerun this because in the following we use layer3.5 instead of layer4.2
activations_all, prediction_classes = generate_data_including_prediction(model, layer, dataloader)
prediction_classes = prediction_classes.cpu().numpy()


#---------------------------------------------------------------------------------

# Change the attribution method to Gradient times Activation, for layer3.5
# For Table 6 and 7

folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer3[5]_4_comps_gradientXactivation")
save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer3[5]_4_comps_gradientXactivation")

run_test(activations_all, prediction_classes,
            exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
        save_name=save_name, class_frequency=10)
run_test(activations_all, prediction_classes,
            exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10)


#---------------------------------------------------------------------------------

# Vary the number of components between 1 and 10, ResNet50 Layer3.5
# For Figure 4 and 13

for num_comp in range(10):
    folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer3[5]"+"_"+str(num_comp)+"_comps_grid_search")
    save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer3[5]"+"_"+str(num_comp)+"_comps_grid_search")

    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10)
    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
                save_name=save_name, class_frequency=10)





#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


model_name="resnet50"
layer_name="_layer2[3]"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer2[3]

dataset = get_dataset(use_train_ds=True)
dataset = CroppedImageNetDataset(dataset)
dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=8)

# Needs to run all 11.5 million image crops, takes around 2 hours on an RTX2080Ti. Will also take some RAM (likely more than 46GB)
activations_all, prediction_classes = generate_data_including_prediction(model, layer, dataloader)
prediction_classes = prediction_classes.cpu().numpy()

#---------------------------------------------------------------------------------

# Vary the number of components between 1 and 10, ResNet50 Layer2.3
# For Figure 4 and 13

for num_comp in range(10):
    folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer2[3]"+"_"+str(num_comp)+"_comps_grid_search")
    save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer2[3]"+"_"+str(num_comp)+"_comps_grid_search")

    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10)
    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
                save_name=save_name, class_frequency=10)






#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


model_name="resnet50"
layer_name="_layer1[2]"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer1[2]

dataset = get_dataset(use_train_ds=True)
dataset = CroppedImageNetDataset(dataset)
dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=8)

# Needs to run all 11.5 million image crops, takes around 2 hours on an RTX2080Ti. Will also take some RAM (likely more than 46GB)
activations_all, prediction_classes = generate_data_including_prediction(model, layer, dataloader)
prediction_classes = prediction_classes.cpu().numpy()

#---------------------------------------------------------------------------------

# Vary the number of components between 1 and 10, ResNet50 Layer1.2
# For Figure 4 and 13

for num_comp in range(10):
    folder_path = os.path.join(OUTPUT_PATH, "resnet50_layer1[2]"+"_"+str(num_comp)+"_comps_grid_search")
    save_name = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "resnet50_layer1[2]"+"_"+str(num_comp)+"_comps_grid_search")

    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=True, save_output=True, folder_path=folder_path,
            save_name=save_name, class_frequency=10)
    run_test(activations_all, prediction_classes,
                exclude_target_class_from_patches=False, save_output=True, folder_path=folder_path,
                save_name=save_name, class_frequency=10)





