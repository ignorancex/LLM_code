

from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path

from core.core import extract_attribution_filtered_activation_vectors, generate_activations
from core.sanity_checks import test_move_along_hyperplane, test_mask_input, visualize_masking, test_shifting_from_attribution_cutoff_only_wrapper
from core.digipath_utils import collect_dataset, get_model, collect_patches_from_prediction, AttributionLimitToTargetClassWrapper
from core.config import DEVICE, OUTPUT_PATH
from core.utils import access_activations_forward_hook


offsets = [0.0, -0.5, -1.0, -2.0, -10.0, -15.0, -20.0, -35.0, -50.0, -200.0, -500.0]
#additional_bias_values = [-4.0, -3.0, -2.0, -1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

all_adjusted_preds_dict = {}
all_adjusted_preds = torch.zeros((46, 46, len(offsets)))
all_adjusted_preds_max = torch.zeros((46, 46, len(offsets)))

all_adjusted_preds = []#torch.zeros((46, 46, len(offsets)))
all_adjusted_preds_max = []#torch.zeros((46, 46, len(offsets)))


model = get_model("version_299").eval().to(DEVICE)
layer = getattr(model.model.encoder.down_blocks, 'down block 3').resnet_blocks[2]


def run_masking_test_images():

    class_idx = 8
    other_class = 24


    dataset = collect_dataset(target_class=class_idx)
    if dataset is not None:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

        patches = collect_patches_from_prediction(dataloader, model, target_channel=class_idx, num_to_sample=60)
        if len(patches) >= 60:
            dataset = TensorDataset(patches)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

            hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "version_299")
            #hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_nmf_components", "version_299")
            hyperplane_path = os.path.join(hyperplane_folder, str(class_idx) + ".pt")
            hyperplane_normals = torch.load(hyperplane_path).to(DEVICE)

            hyperplane_path = os.path.join(hyperplane_folder, "bias_" + str(class_idx) + ".pt")
            hyperplane_biases = torch.load(hyperplane_path).to(DEVICE)

        patches = patches.to(DEVICE)
        #activations = access_activations_forward_hook([patches], model, layer)

        activations = generate_activations(model, dataloader, layer)

        hyperplane_normal = hyperplane_normals[other_class]
        hyperplane_bias = hyperplane_biases[other_class]

        visualize_masking(patches, patches, activations, hyperplane_normal, hyperplane_bias, model, class_idx, other_class,
                            replacement_value=1.0)




def run_test(save_output=True):
    mean_masked_preds = torch.zeros((46, 46, 10, 46)) - 1.0
    mean_max_masked_preds = torch.zeros((46, 46, 10, 46)) - 1.0

    for class_idx in tqdm(range(46), ascii=True):
        if class_idx == 0:
            continue

        dataset = collect_dataset(target_class=class_idx)
        if dataset is not None:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

            patches = collect_patches_from_prediction(dataloader, model, target_channel=class_idx, num_to_sample=60)
            if len(patches) >= 60:
                dataset = TensorDataset(patches)
                dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

                hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "version_299")
                #hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_nmf_components", "version_299")
                hyperplane_path = os.path.join(hyperplane_folder, str(class_idx) + ".pt")
                hyperplane_normals = torch.load(hyperplane_path).to(DEVICE)

                hyperplane_path = os.path.join(hyperplane_folder, "bias_" + str(class_idx) + ".pt")
                hyperplane_biases = torch.load(hyperplane_path).to(DEVICE)

            else:
                continue


            for other_class in tqdm(range(46), ascii=True):
                if other_class == 0:
                    continue
                if class_idx == other_class:
                    continue


                hyperplane_normal = hyperplane_normals[other_class]

                adjusted_preds, max_adjusted_preds = test_move_along_hyperplane(dataloader, model, layer, hyperplane_normal,
                                                                                from_class=class_idx, other_class=other_class)

                mean_masked_preds[class_idx, other_class] = torch.tensor(adjusted_preds).cpu()
                mean_max_masked_preds[class_idx, other_class] = torch.tensor(max_adjusted_preds).cpu()

                #all_adjusted_preds.append(torch.tensor(adjusted_preds))
                #all_adjusted_preds_max.append(torch.tensor(max_adjusted_preds))

        if save_output:
            #all_adjusted_preds_save = torch.stack(all_adjusted_preds, dim=0)
            #all_adjusted_preds_max_save = torch.stack(all_adjusted_preds_max, dim=0)

            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "version_299", "adjusted_preds_tensor_list.pt")
            #np.save(out_file, all_adjusted_preds_save.cpu().numpy())
            torch.save(mean_masked_preds, out_file)
            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "version_299", "adjusted_preds_max_tensor_list.pt")
            torch.save(mean_max_masked_preds, out_file)
            #np.save(out_file, all_adjusted_preds_max_save.cpu().numpy())

                #hyperplane_bias = hyperplane_biases[other_class]
                #adjusted_preds, max_adjusted_preds = test_mask_input(dataloader, model, layer, hyperplane_normal, hyperplane_bias,
                #                                class_idx, additional_bias_values=additional_bias_values, do_prints=True)

                #all_adjusted_preds_dict[str(class_idx) + "_" + str(other_class)] = adjusted_preds
                #all_adjusted_preds_dict["max_" + str(class_idx) + "_" + str(other_class)] = max_adjusted_preds

                #all_adjusted_preds[class_idx, other_class] = torch.tensor(adjusted_preds)
                #all_adjusted_preds_max[class_idx, other_class] = torch.tensor(max_adjusted_preds)

        """
                all_adjusted_preds.append(torch.tensor(adjusted_preds))
                all_adjusted_preds_max.append(torch.tensor(max_adjusted_preds))

        if save_output:
            all_adjusted_preds_save = torch.stack(all_adjusted_preds, dim=0)
            all_adjusted_preds_max_save = torch.stack(all_adjusted_preds_max, dim=0)

            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "version_299", "adjusted_preds_tensor_list.npy")
            np.save(out_file, all_adjusted_preds_save.cpu().numpy())
            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "version_299", "adjusted_preds_max_tensor_list.npy")
            np.save(out_file, all_adjusted_preds_max_save.cpu().numpy())
        """


def run_test_masking_input(save_output=True, do_prints=False):
    #i = 0

    mean_masked_preds = torch.zeros((46, 46, 11, 46)) - 1.0
    mean_max_masked_preds = torch.zeros((46, 46, 11, 46)) - 1.0

    for class_idx in tqdm(range(46), ascii=True):
        if class_idx == 0:
            continue

        dataset = collect_dataset(target_class=class_idx)
        if dataset is not None:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

            patches = collect_patches_from_prediction(dataloader, model, target_channel=class_idx, num_to_sample=60)
            if len(patches) >= 60:
                dataset = TensorDataset(patches)
                dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

                hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "version_299")
                #hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_nmf_components", "version_299")
                hyperplane_path = os.path.join(hyperplane_folder, str(class_idx) + ".pt")
                hyperplane_normals = torch.load(hyperplane_path).to(DEVICE)

                hyperplane_path = os.path.join(hyperplane_folder, "bias_" + str(class_idx) + ".pt")
                hyperplane_biases = torch.load(hyperplane_path).to(DEVICE)

            else:
                continue


            for other_class in tqdm(range(46), ascii=True):
                if other_class == 0:
                    continue
                if class_idx == other_class:
                    continue

                hyperplane_normal = hyperplane_normals[other_class]
                if torch.sum(hyperplane_normal) == 0.0:
                    continue


                #i += 1

                #if i < 10:
                #    continue


                print("other class: {}".format(other_class))

                #adjusted_preds, max_adjusted_preds = test_move_along_hyperplane(dataloader, model, layer, hyperplane_normal,
                #                                                                from_class=class_idx, other_class=other_class)

                hyperplane_bias = hyperplane_biases[other_class]
                adjusted_preds, max_adjusted_preds = test_mask_input(dataloader, model, layer, hyperplane_normal, hyperplane_bias,
                                                other_class, do_prints=do_prints)

                #all_adjusted_preds_dict[str(class_idx) + "_" + str(other_class)] = adjusted_preds
                #all_adjusted_preds_dict["max_" + str(class_idx) + "_" + str(other_class)] = max_adjusted_preds

                #all_adjusted_preds[class_idx, other_class] = torch.tensor(adjusted_preds)
                #all_adjusted_preds_max[class_idx, other_class] = torch.tensor(max_adjusted_preds)

                mean_masked_preds[class_idx, other_class] = torch.tensor(adjusted_preds).cpu()
                mean_max_masked_preds[class_idx, other_class] = torch.tensor(max_adjusted_preds).cpu()

                #all_adjusted_preds.append(torch.tensor(adjusted_preds))
                #all_adjusted_preds_max.append(torch.tensor(max_adjusted_preds))

        if save_output:
            #all_adjusted_preds_save = torch.stack(all_adjusted_preds, dim=0)
            #all_adjusted_preds_max_save = torch.stack(all_adjusted_preds_max, dim=0)

            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "version_299", "masking_input_adjusted_preds_tensor_list.pt")
            #np.save(out_file, all_adjusted_preds_save.cpu().numpy())
            torch.save(mean_masked_preds, out_file)
            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "version_299", "masking_input_adjusted_preds_max_tensor_list.pt")
            torch.save(mean_max_masked_preds, out_file)
            #np.save(out_file, all_adjusted_preds_max_save.cpu().numpy())

        #break



def run_test_function(save_name, save_output=True, do_prints=False, function_to_run=test_shifting_from_attribution_cutoff_only_wrapper):
    mean_masked_preds = torch.zeros((46, 46, 11, 46)) - 1.0
    mean_max_masked_preds = torch.zeros((46, 46, 11, 46)) - 1.0

    #print("run test function...")

    for class_idx in tqdm(range(46), ascii=True):
        if class_idx != 1:
            continue

        dataset = collect_dataset(target_class=class_idx)
        if dataset is not None:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

            #print("dataset is not None")

            patches = collect_patches_from_prediction(dataloader, model, target_channel=class_idx, num_to_sample=6)
            if len(patches) >= 6:
                wrapped_model = AttributionLimitToTargetClassWrapper(model, target_class=class_idx)

                dataset = TensorDataset(patches)
                dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

                hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "version_299")
                #hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_nmf_components", "version_299")
                hyperplane_path = os.path.join(hyperplane_folder, str(class_idx) + ".pt")
                hyperplane_normals = torch.load(hyperplane_path).to(DEVICE)

                hyperplane_path = os.path.join(hyperplane_folder, "bias_" + str(class_idx) + ".pt")
                hyperplane_biases = torch.load(hyperplane_path).to(DEVICE)

            else:
                continue


            for other_class in tqdm(range(46), ascii=True):
                if other_class != 32:
                    continue
                if class_idx == other_class:
                    continue

                hyperplane_normal = hyperplane_normals[other_class]
                if torch.sum(hyperplane_normal) == 0.0:
                    continue


                print("other class: {}".format(other_class))

                hyperplane_bias = hyperplane_biases[other_class]
                adjusted_preds, max_adjusted_preds = function_to_run(dataloader=dataloader, model=model, layer=layer, hyperplane_normal=hyperplane_normal,
                                                                     hyperplane_bias=hyperplane_bias, class_idx=class_idx,
                                                                     other_class=other_class, do_prints=do_prints, offsets=offsets, replacement_value=1.0,
                                                                     wrapped_model=wrapped_model)

                mean_masked_preds[class_idx, other_class] = torch.tensor(adjusted_preds).cpu()
                mean_max_masked_preds[class_idx, other_class] = torch.tensor(max_adjusted_preds).cpu()

        if save_output:

            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "version_299", save_name + ".pt")
            torch.save(mean_masked_preds, out_file)
            out_file = os.path.join(OUTPUT_PATH, "sanity_check", "version_299", save_name + "_max.pt")
            torch.save(mean_max_masked_preds, out_file)


run_test_function(save_output=True, save_name="adjust_activations_attribution_cutoff",
         function_to_run=test_shifting_from_attribution_cutoff_only_wrapper,
         do_prints=True)

#run_test_masking_input(save_output=True, do_prints=True)

#run_masking_test_images()