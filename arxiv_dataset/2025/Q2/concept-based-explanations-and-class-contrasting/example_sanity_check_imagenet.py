

from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
import random
import pickle

from core.core import collect_patches
from core.sanity_checks import test_move_along_hyperplane, test_mask_input, visualize_masking
from core.imagenet_utils import get_dataset_for_class, load_model
from core.config import DEVICE, OUTPUT_PATH
from core.utils import access_activations_forward_hook

offsets = [0.0, -0.5, -1.0, -2.0, -10.0, -15.0, -20.0, -35.0, -50.0, -200.0]

additional_bias_values = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
#offsets = [0.0, 0.5, 1.0, 2.0, 10.0, 15.0, 20.0, 35.0, 50.0, 200.0]

#offsets = [offset * 10000000 for offset in offsets]



def run_masking_test_images(model, layer, model_name):

    class_idx = 773
    other_class = 611


    dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx)
    if dataset is not None:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

        patches = collect_patches(dataloader)
        if len(patches) > 60:
            patches = patches[:60]
        if len(patches) > 0:
            dataset = TensorDataset(patches)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

            hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", model_name)
            hyperplane_path = os.path.join(hyperplane_folder, str(class_idx) + ".pt")
            hyperplane_normals = torch.load(hyperplane_path).to(DEVICE)

            hyperplane_path = os.path.join(hyperplane_folder, "bias_" + str(class_idx) + ".pt")
            hyperplane_biases = torch.load(hyperplane_path).to(DEVICE)

        #other_class = random.randint(1, 999)

        hyperplane_normal = hyperplane_normals[other_class].to(DEVICE)
        hyperplane_bias = hyperplane_biases[other_class].to(DEVICE)

        patches = patches.to(DEVICE)
        activations = access_activations_forward_hook([patches], model, layer)
        visualize_masking(patches, patches, activations, hyperplane_normal, hyperplane_bias, model, class_idx, other_class)





def run_masking_test(model, layer, model_name, save_output=False, debug_prints=False, n_runs=5):

    for test_number in range(n_runs):
        all_adjusted_preds = []
        class_tuples = []

        tested_combinations = set()
        #model = load_model(model_name).eval().to(DEVICE)
        #layer = model.layer3[2]

        for i in tqdm(range(1000)):
            #for class_idx in tqdm(range(1000), ascii=True):
            #if class_idx != 1:
            #    continue
            #if class_idx == 0:
            #    continue

            class_idx = random.randint(0, 999)

            dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx)
            if dataset is not None:
                dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

                patches = collect_patches(dataloader)
                if len(patches) > 60:
                    patches = patches[:60]
                if len(patches) > 0:
                    dataset = TensorDataset(patches)
                    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

                    hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", model_name)
                    hyperplane_path = os.path.join(hyperplane_folder, str(class_idx) + ".pt")
                    hyperplane_normals = torch.load(hyperplane_path).to(DEVICE)

                    hyperplane_path = os.path.join(hyperplane_folder, "bias_" + str(class_idx) + ".pt")
                    hyperplane_biases = torch.load(hyperplane_path).to(DEVICE)


                for j in range(10):
                    #for other_class in tqdm(range(1000), ascii=True):
                    other_class = random.randint(1, 999)
                    #continue
                    #if other_class == 0:
                    #    continue
                    if class_idx == other_class:
                        continue
                    if (class_idx, other_class) in tested_combinations:
                        continue
                    tested_combinations.add((class_idx, other_class))

                    hyperplane_normal = hyperplane_normals[other_class]
                    hyperplane_bias = hyperplane_biases[other_class]
                    adjusted_preds, _ = test_mask_input(dataloader, model, layer, hyperplane_normal, hyperplane_bias,
                                                                                    other_class=other_class,
                                                                                    additional_bias_values=additional_bias_values, do_prints=debug_prints,
                                                                                    replacement_value=0.0)


                    # reverse classes, print for original class to check whether the dataset is build correctly
                    #adjusted_preds, max_adjusted_preds = test_move_along_hyperplane(dataloader, model, layer, hyperplane_normal,
                    #                                                                from_class=other_class, other_class=class_idx,
                    #                                                                use_relu=False, offsets=offsets)

                    all_adjusted_preds.append(torch.tensor(adjusted_preds))
                    class_tuples.append((class_idx, other_class))
                    #all_adjusted_preds_dict[str(class_idx) + "_" + str(other_class)] = adjusted_preds
                    #all_adjusted_preds_dict["max_" + str(class_idx) + "_" + str(other_class)] = max_adjusted_preds

                if debug_prints:
                    print(torch.mean(torch.stack(all_adjusted_preds, dim=0), dim=0))

            #break
        if save_output:
            Path(os.path.join(OUTPUT_PATH, "sanity_check", model_name)).mkdir(parents=True, exist_ok=True)
            all_adjusted_preds_ = torch.stack(all_adjusted_preds, dim=0)
            out_file = os.path.join(OUTPUT_PATH, "sanity_check", model_name, "masking_adjusted_preds_"+str(test_number)+".pickle")
            with open(out_file, 'wb') as handle:
                pickle.dump((all_adjusted_preds_.cpu().numpy(), class_tuples), handle)
            #np.save(out_file, (all_adjusted_preds.cpu().numpy(), class_tuples))




def run_test(model, layer, model_name, layer_name="", save_output=True, debug_prints=False, n_runs=5):
    for test_number in range(n_runs):
        all_adjusted_preds = []
        all_adjusted_preds_original_class = []
        class_tuples = []

        tested_combinations = set()
        #model = load_model(model_name).eval().to(DEVICE)
        #layer = model.layer3[2]

        for i in tqdm(range(1000)):
            #for class_idx in tqdm(range(1000), ascii=True):
            #if class_idx != 1:
            #    continue
            #if class_idx == 0:
            #    continue

            class_idx = random.randint(0, 999)

            dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx, use_train_ds=True)
            if dataset is not None:
                dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

                torch.manual_seed(42)
                patches = collect_patches(dataloader, num_to_collect=60)
                if len(patches) > 60:
                    patches = patches[:60]
                if len(patches) > 0:
                    dataset = TensorDataset(patches)
                    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

                    hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "hyperplanes_from_attribution", model_name + layer_name)
                    hyperplane_path = os.path.join(hyperplane_folder, str(class_idx) + ".pt")
                    hyperplane_normals = torch.load(hyperplane_path).to(DEVICE)

                for j in range(10):
                    #for other_class in tqdm(range(1000), ascii=True):
                    other_class = random.randint(1, 999)
                    #continue
                    #if other_class == 0:
                    #    continue
                    if class_idx == other_class:
                        continue
                    if (class_idx, other_class) in tested_combinations:
                        continue
                    tested_combinations.add((class_idx, other_class))

                    hyperplane_normal = hyperplane_normals[other_class]
                    #print(hyperplane_normal)
                    adjusted_preds, max_adjusted_preds = test_move_along_hyperplane(patches.to("cuda"), model, layer, hyperplane_normal,
                                                                                    from_class=class_idx, other_class=other_class,
                                                                                    offsets=offsets, use_relu=True)

                    #adjusted_preds_original_class, max_adjusted_preds = test_move_along_hyperplane(dataloader, model, layer, hyperplane_normal,
                    #                                                                from_class=other_class, other_class=class_idx,
                    #                                                                offsets=offsets, use_relu=True)

                    # reverse classes, print for original class to check whether the dataset is build correctly
                    #adjusted_preds, max_adjusted_preds = test_move_along_hyperplane(dataloader, model, layer, hyperplane_normal,
                    #                                                                from_class=other_class, other_class=class_idx,
                    #                                                                use_relu=False, offsets=offsets)

                    all_adjusted_preds.append(torch.tensor(adjusted_preds))
                    class_tuples.append((class_idx, other_class))
                    #all_adjusted_preds_original_class.append(torch.tensor(adjusted_preds_original_class))
                    #all_adjusted_preds_dict[str(class_idx) + "_" + str(other_class)] = adjusted_preds
                    #all_adjusted_preds_dict["max_" + str(class_idx) + "_" + str(other_class)] = max_adjusted_preds

                if debug_prints:
                    print(torch.mean(torch.stack(all_adjusted_preds, dim=0), dim=0))

                #break
            #break
        if save_output:
            Path(os.path.join(OUTPUT_PATH, "sanity_check", "shifting_test", model_name + layer_name + "_more_steps")).mkdir(parents=True, exist_ok=True)
            all_adjusted_preds_ = torch.stack(all_adjusted_preds, dim=0)
            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "shifting_test", model_name + layer_name, "adjusted_preds_" + str(test_number) + "_more_steps" + ".pickle")
            with open(out_file, 'wb') as handle:
                pickle.dump((all_adjusted_preds_.cpu().numpy(), class_tuples), handle)


                """
                Path(os.path.join(OUTPUT_PATH, "sanity_check", model_name)).mkdir(parents=True, exist_ok=True)
                all_adjusted_preds = torch.stack(all_adjusted_preds, dim=0)
                out_file = os.path.join(OUTPUT_PATH, "sanity_check", model_name, "adjusted_preds.pt")
                np.save(out_file, all_adjusted_preds.cpu().numpy())
                #out_file = os.path.join(OUTPUT_PATH, "sanity_check", model_name, "adjusted_preds_original_class.npy")
                #np.save(out_file, all_adjusted_preds_original_class.cpu().numpy())
                """
"""
model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
#layer = model.encoder.layers.encoder_layer_7
layer = model.layer3[2]

run_test(model=model, layer=layer, model_name=model_name, save_output=True, debug_prints=False)
#run_test(model_name="resnet34", save_output=True, debug_prints=False)


model_name = "resnet34"
model = load_model(model_name).eval().to(DEVICE)
#layer = model.encoder.layers.encoder_layer_7
layer = model.layer3[2]

run_test(model=model, layer=layer, model_name=model_name, save_output=True, debug_prints=False)
"""


def run_masking():
    model_name = "resnet50_robust"
    model = load_model(model_name).eval().to(DEVICE)
    #layer = model.encoder.layers.encoder_layer_7
    layer = model.layer3[5]
    layer_name = "_layer3[5]"

    run_masking_test(model=model, layer=layer, model_name=model_name, layer_name=layer_name, save_output=True, debug_prints=False)
    #run_test(model_name="resnet34", save_output=True, debug_prints=False)

    model_name = "resnet50"
    model = load_model(model_name).eval().to(DEVICE)
    #layer = model.encoder.layers.encoder_layer_7
    layer = model.layer3[5]

    run_masking_test(model=model, layer=layer, model_name=model_name, layer_name=layer_name, save_output=True, debug_prints=False)
    #run_test(model_name="resnet34", save_output=True, debug_prints=False)

    model_name = "resnet34"
    model = load_model(model_name).eval().to(DEVICE)
    #layer = model.encoder.layers.encoder_layer_7
    layer = model.layer3[5]

    run_masking_test(model=model, layer=layer, model_name=model_name, layer_name=layer_name, save_output=True, debug_prints=False)
    #run_test(model_name="resnet34", save_output=True, debug_prints=False)


def run_shifting_pred():
    layer_name = "_layer4[2]"
    model_name = "resnet50"
    model = load_model(model_name).eval().to(DEVICE)
    #layer = model.encoder.layers.encoder_layer_7
    layer = model.layer4[2]

    run_test(model=model, layer=layer, model_name=model_name, layer_name=layer_name, save_output=True, debug_prints=True)



    layer_name = "_layer3[5]"
    model_name = "resnet50"
    model = load_model(model_name).eval().to(DEVICE)
    #layer = model.encoder.layers.encoder_layer_7
    layer = model.layer3[5]

    run_test(model=model, layer=layer, model_name=model_name, layer_name=layer_name, save_output=True, debug_prints=True)



    #run_test(model_name="resnet34", save_output=True, debug_prints=False)


    """
    model_name = "resnet50_robust"
    model = load_model(model_name).eval().to(DEVICE)
    #layer = model.encoder.layers.encoder_layer_7
    layer = model.layer4[2]

    run_test(model=model, layer=layer, model_name=model_name, layer_name=layer_name, save_output=True, debug_prints=False)
    #run_test(model_name="resnet34", save_output=True, debug_prints=False)
    """


    #model_name = "resnet34"
    #model = load_model(model_name).eval().to(DEVICE)
    ##layer = model.encoder.layers.encoder_layer_7
    #layer = model.layer3[5]

    #run_test(model=model, layer=layer, model_name=model_name, layer_name=layer_name, save_output=True, debug_prints=False)




run_shifting_pred()
