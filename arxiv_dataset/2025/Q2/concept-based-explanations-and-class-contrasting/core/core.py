import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision

import os
import numpy as np
from tqdm import tqdm
import captum
from pathlib import Path
from sklearn.decomposition import NMF

from core.train_gmm import gmm_density_score

from core.config import DEVICE, SEED
from core.utils import access_activations_forward_hook






def collect_patches(dataloader, num_to_collect=500):
    all_images = []
    for data in tqdm(dataloader, ascii=True, desc="Collect Images", total=num_to_collect):
        images = data[0]
        all_images.append(images)
        if len(all_images) >= num_to_collect:
            return torch.cat(all_images, dim=0)
    return torch.cat(all_images, dim=0)


def generate_activations(model, dataloader, layer):
    with torch.no_grad():
        activations = []
        model = model.to(DEVICE).eval()

        for data in dataloader:
            images = data[0]
            images = images.to(DEVICE)

            activation = access_activations_forward_hook([images], model, layer)#.cpu()
            activations.append(activation)

    return torch.cat(activations, dim=0)



def compute_attribution(dataloader, model, target_channel, layer,
                               attribution_method="deep_lift", use_noise_tunnel=True):

    attributions = []

    if attribution_method == "deep_lift":
        AttributionMethod = captum.attr.LayerDeepLift(
            model, layer, multiply_by_inputs=True
        )
    elif attribution_method == "integrated_gradients":
        AttributionMethod = captum.attr.LayerIntegratedGradients(
            model, layer, multiply_by_inputs=True
        )
    elif attribution_method == "gradient":
        #print("using gradient")
        AttributionMethod = captum.attr.LayerGradientXActivation(
            model, layer, multiply_by_inputs=True
        )

    if use_noise_tunnel:
        AttributionMethod = captum.attr.NoiseTunnel(AttributionMethod)


    baselines = 1.0  # torch.ones_like(patch)#torch.tensor(1.0).to(DEVICE)
    # baselines.requires_grad = True

    nt_samples = 20
    # cpu is much slower than gpu, set nt_samples to 5 there to reduce time
    if DEVICE == "cpu":
        nt_samples = 5

    with torch.enable_grad():
        for data in tqdm(dataloader, ascii=True, desc="Compute Attribution"):
            images = data[0]
            images = images.to(DEVICE)
            images.requires_grad = True
            if use_noise_tunnel:
                if attribution_method == "deep_lift":
                    attribution = AttributionMethod.attribute(
                        images,
                        target=int(target_channel),
                        nt_samples=nt_samples,
                        stdevs=0.25,
                        nt_samples_batch_size=1,
                        baselines=baselines,
                    ).detach()
                elif attribution_method == "integrated_gradients":
                    attribution = AttributionMethod.attribute(
                        images,
                        target=int(target_channel),
                        nt_samples=nt_samples,
                        stdevs=0.25,
                        nt_samples_batch_size=1,
                        internal_batch_size=10,
                        baselines=baselines,
                    ).detach()
            else:
                if attribution_method == "deep_lift":
                    #print("patch requires grad: {}".format(patch.requires_grad))
                    attribution = AttributionMethod.attribute(
                        images, target=int(target_channel), baselines=baselines
                    ).detach()
                elif attribution_method == "integrated_gradients":
                    attribution = AttributionMethod.attribute(
                        images,
                        target=int(target_channel),
                        internal_batch_size=1,
                        baselines=baselines,
                    ).detach()
                elif attribution_method == "gradient":
                    #print("using gradient run")
                    attribution = AttributionMethod.attribute(
                        images, target=int(target_channel)
                    ).detach()

            attributions.append(attribution.detach())

    return torch.cat(attributions, dim=0)



def extract_attribution_cutoff_mask(dataloader, target_channel, wrapped_model,
                                            layer,
                                            attribution_cutoff=0.25):
    #dataloader_attribution = DataLoader(dataset, shuffle=False, batch_size=batch_size_attribution)
    attributions = compute_attribution(dataloader, wrapped_model, target_channel, layer)

    attribution_masks = F.relu(attributions)
    attribution_masks = torch.mean(attributions, dim=[1]).unsqueeze(dim=1)

    #print("attribution masks shape: {}".format(attribution_masks.shape))

    b = attribution_masks.shape[0]

    attribution_masks_flat_per_image = attribution_masks.reshape(b, -1)
    max_attribution_per_image = torch.max(attribution_masks_flat_per_image, dim=1)[0]

    #print("max_attribution_per_image shape: {}".format(max_attribution_per_image.shape))

    attribution_masks = torch.where(attribution_masks > max_attribution_per_image.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)*attribution_cutoff, 1.0, 0.0)

    #print("attribution masks shape: {}".format(attribution_masks.shape))

    return attribution_masks








def extract_attribution_filtered_activation_vectors(dataset, target_channel, wrapped_model, model,
                                            layer,
                                            batch_size_attribution=4,
                                            batch_size_activation=16,
                                            attribution_cutoff=0.25,
                                            use_noise_tunnel=True):

    dataloader_attribution = DataLoader(dataset, shuffle=False, batch_size=batch_size_attribution)
    attributions = compute_attribution(dataloader_attribution, wrapped_model, target_channel, layer,
                                       use_noise_tunnel=use_noise_tunnel)

    dataloader_activations = DataLoader(dataset, shuffle=False, batch_size=batch_size_activation)
    #print("generate activations")
    activations = generate_activations(model, dataloader_activations, layer)
    #print("finished genrate activations!")

    attribution_masks = F.relu(attributions)
    attribution_masks = torch.mean(attributions, dim=[1]).unsqueeze(dim=1)

    b = activations.shape[0]
    c = activations.shape[1]

    attribution_masks_flat_per_image = attribution_masks.reshape(b, -1)
    max_attribution_per_image = torch.max(attribution_masks_flat_per_image, dim=1)[0]

    # CNN
    if len(activations.shape) == 4:
        activations_flat = activations.permute(0, 2, 3, 1)
    #ViT
    else:
        activations_flat = activations.permute(0, 2, 1)

    activations_flat = activations_flat.reshape(-1, c)  # Shape: (b*h*w, c)

    masks_flat = torch.where(attribution_masks_flat_per_image > max_attribution_per_image.unsqueeze(dim=1)*attribution_cutoff, 1.0, 0.0)
    #print(torch.mean(masks_flat))
    masks_flat = masks_flat.reshape(-1)
    above_cutoff_indice = torch.nonzero(masks_flat)[:, 0]
    #print("index above cutoff actiations")
    above_cutoff_activations = activations_flat[above_cutoff_indice]
    #print("finished index abocve cutoff activations")

    return above_cutoff_activations







def nmf_decompose_batch(input_activation_bchw, n_components=6):
    input_for_nmf = input_activation_bchw
    b, c, h, w = input_for_nmf.shape

    input_for_nmf = torch.permute(input_for_nmf, (1, 0, 2, 3))

    input_for_nmf = torch.reshape(
        input_for_nmf,
        shape=(
            input_for_nmf.shape[0],
            input_for_nmf.shape[1] * input_for_nmf.shape[2] * input_for_nmf.shape[3],
        ),
    )
    input_for_nmf = torch.permute(input_for_nmf, (1, 0))

    nmf = NMF(n_components=n_components)

    X_embedded = nmf.fit_transform(input_for_nmf)
    basis = nmf.components_

    X_embedded = torch.from_numpy(X_embedded)
    X_embedded = torch.permute(X_embedded, (1, 0))
    #print("X_embedded.shape: {}".format(X_embedded.shape))
    X_embedded = torch.reshape(X_embedded, shape=(n_components, b, h, w))
    X_embedded = torch.permute(X_embedded, (1, 0, 2, 3))

    return X_embedded, torch.from_numpy(basis)






def apply_hyperplane(dataloader, model, layer, hyperplane_normal, hyperplane_bias, output_path=None):
    for data in dataloader:
        images = data[0]

        activations = access_activations_forward_hook([images.to(DEVICE)], model, layer)




def extract_overlay_channel_activations(activation, patch, channel_indize):
    channel_activations = []
    patch = patch.to(DEVICE)
    activation = activation.to(DEVICE)
    for channel_index in channel_indize:
        channel_activation = activation[:, channel_index, :, :]
        channel_activation = torch.stack([channel_activation for i in range(3)], dim=1)
        channel_activation = torch.clamp(
            channel_activation / torch.max(channel_activation), min=0.0, max=0.75
        )
        channel_activation += 0.25
        channel_activations.append(channel_activation[0])
    channel_activations = torch.stack(channel_activations, dim=0)#.to(DEVICE)
    #print("channel activations shape: {}".format(channel_activations.shape))
    channel_activations = torch.nn.functional.interpolate(
        channel_activations, size=patch.shape[-1]
    )
    #channel_activations = channel_activations.to(DEVICE) * patch.to(DEVICE) + (1.0 - channel_activations.to(DEVICE))
    #channel_activations = channel_activations.to(DEVICE) * patch.to(DEVICE)
    channel_activations = channel_activations*patch
    channel_activations = channel_activations.cpu()

    #activation = activation.cpu()
    #patch = patch.cpu()

    return channel_activations


def nmf_attribution_whole_ds_decomp(dataset, target_channel, model, layer, save_folder_path,
                                           n_components=6,
                                           save_images=False,
                                           batch_size=8,
                                           use_noise_tunnel=True,
                                           attribution_method="deep_lift",
                                           save_data=True):
    #save_model_name = model_name+name_append
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    activations = []
    original_activations = []
    patches = []
    #n_components = 6

    attributions = compute_attribution(dataloader, model, target_channel, layer, use_noise_tunnel=use_noise_tunnel,
                                       attribution_method=attribution_method)
    attributions = torch.mean(F.relu(attributions), dim=1).unsqueeze(dim=1)

    original_activations = generate_activations(model, dataloader, layer)
    activations = attributions * original_activations

    #activations = activations.cpu()
    #original_activations = original_activations.cpu()
    patches = collect_patches(dataloader)


    print("nmf decomposition...")
    X_embedded, basis = nmf_decompose_batch(
        activations.cpu(), n_components=n_components
    )
    X_embedded = X_embedded.to(DEVICE)

    if save_data:
        Path(os.path.join(save_folder_path, str(target_channel), "nmf")).mkdir(parents=True, exist_ok=True)
        for i in range(n_components):
            Path(os.path.join(save_folder_path, str(target_channel), "vectors", str(i))).mkdir(parents=True, exist_ok=True)
            if save_images:
                Path(os.path.join(save_folder_path, str(target_channel), "patches", str(i))).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(save_folder_path, str(target_channel), "mask", str(i))).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(save_folder_path, str(target_channel), "patches_original", str(i))).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(save_folder_path, str(target_channel), "vectors_visualized", str(i))).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(save_folder_path, str(target_channel), "visualizations", str(i))).mkdir(parents=True, exist_ok=True)

        if save_images:
            print("save images...")
            image_index = 0
            for i, patch in tqdm(enumerate(patches), ascii=True):
                channel_activation = extract_overlay_channel_activations(
                    X_embedded[i].unsqueeze(dim=0), patch, [j for j in range(n_components)]
                )
                for k, overlay in enumerate(channel_activation):
                    #print(original_activations[i].device)
                    #print(X_embedded[i, k].device)
                    activation_vec = torch.mean(
                        original_activations[i] * X_embedded[i, k].unsqueeze(dim=0).unsqueeze(dim=0),
                        dim=[0, 2, 3],
                    ) / torch.mean(X_embedded[i, k])
                    mask = X_embedded[i][k]
                    #print("activation vec shape: {}".format(activation_vec.shape))

                    torch.save(activation_vec, os.path.join(save_folder_path, str(target_channel), "vectors", str(k%n_components), str(image_index) + ".pt"))
                    mask = mask - torch.min(mask)
                    mask = mask / torch.max(mask)

                    torchvision.utils.save_image(mask, os.path.join(save_folder_path, str(target_channel), "mask", str(k%n_components), str(image_index) + ".png"))
                    torchvision.utils.save_image(overlay, os.path.join(save_folder_path, str(target_channel), "patches", str(k%n_components), str(image_index) + ".jpg"))
                    torchvision.utils.save_image(patch, os.path.join(save_folder_path, str(target_channel), "patches_original", str(k%n_components), str(image_index) + ".jpg"))
                    image_index += 1

        torch.save(basis, os.path.join(save_folder_path, str(target_channel), "nmf", "nmf_basis.pt"))
    return basis



def generate_mask_from_gmm(activation, gmms, gmm_cutoff_score=-300.0):
    b, c, h, w = activation.shape
    activations_flat = activation.permute(0, 2, 3, 1).reshape(-1, c)  # Shape: (b*h*w, c)
    activations_flat = activations_flat.cpu()

    in_distribution_scores_models = []
    for gmm in gmms:
        in_distribution_scores = gmm_density_score(activations_flat.cpu(), gmm).float().to(DEVICE)

        in_distribution_scores = in_distribution_scores.view(b, h, w, 1).permute(0, 3, 1, 2)
        in_distribution_scores_models.append(in_distribution_scores)

    # Initialize the reduction with the first tensor in the list
    in_distribution_scores = in_distribution_scores[0]

    if len(in_distribution_scores) > 1:
        # Iterate over the rest of the tensors and apply torch.maximum
        for other_indistrubtion_score in in_distribution_scores[1:]:
            in_distribution_scores = torch.maximum(in_distribution_scores, other_indistrubtion_score)

    in_distribution_scores = torch.where(in_distribution_scores > gmm_cutoff_score, 1.0, 0.0)
    return in_distribution_scores


def generate_masks_hyperplane_pred(model, dataloader, layer, hyperplane_normal, hyperplane_bias, hyperplane_additional_bias=0.0,
                                    gmms=None, gmm_cutoff_score=-300.0):
    masks = []
    model = model.to(DEVICE).eval()
    for data in dataloader:
        images = data[0].to(DEVICE)

        activation = access_activations_forward_hook([images], model, layer).to(DEVICE)
        activation_in_dir = activation * hyperplane_normal.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).to(DEVICE)
        mask = torch.sum(activation_in_dir, dim=[1]).unsqueeze(dim=1).to(DEVICE) + hyperplane_bias.to(DEVICE) + hyperplane_additional_bias

        # only use the positive part, i.e. the feature contributing positively towards the class
        mask = F.relu(mask)
        #mask = mask / torch.max(mask)


        if gmms is not None:
            in_distribution_mask = generate_mask_from_gmm(activation, gmms, gmm_cutoff_score=gmm_cutoff_score)

            mask = mask*in_distribution_mask
            #mask = mask / torch.max(mask)
            masks.append(mask)
        else:
            masks.append(mask)

    return torch.cat(masks, dim=0)



def nmf_decomp_hyperplane_ood_filtered_activations(dataloader, model, layer, hyperplane_normal, hyperplane_bias, hyperplane_additional_bias,
                                                   n_nmf_components=6,
                                                   save_folder=None,
                                                   scale_masks_individually_for_saving=False,
                                                   gmms=None):

    patches = collect_patches(dataloader)
    activations = generate_activations(model, dataloader, layer)

    # default cutoff score was -300 (24 comps joined training of gmm -> i.e. one gmm for both classes)
    masks = generate_masks_hyperplane_pred(model, dataloader, layer, hyperplane_normal, hyperplane_bias, hyperplane_additional_bias,
                                            gmms, gmm_cutoff_score=-350.0)

    masked_activations = activations*masks

    X_embedded, basis = nmf_decompose_batch(
        masked_activations.cpu(), n_components=n_nmf_components
    )

    print("save folder path: {}".format(save_folder))
    Path(os.path.join(save_folder, "nmf")).mkdir(parents=True, exist_ok=True)
    torch.save(basis, os.path.join(save_folder, "nmf", "nmf_basis.pt"))

    return

    if save_folder is not None:
        Path(os.path.join(save_folder, "patches")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_folder, "patches_original")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_folder, "masks")).mkdir(parents=True, exist_ok=True)

        for c in range(X_embedded.shape[1]):
            Path(os.path.join(save_folder, "nmf", str(c))).mkdir(parents=True, exist_ok=True)

        masks = masks.to(DEVICE)
        if scale_masks_individually_for_saving:
            masks = [masks[i]/torch.max(masks[i]) for i in range(len(masks))]
            masks = torch.stack(masks, dim=0)
        else:
            #print("max masks: {}".format(torch.max(masks)))
            masks = masks / torch.max(masks)
            #masks = masks / 8.102898597717285


        masks_for_saving = masks
        masks = F.interpolate(masks, size=patches.shape[-1])
        masks = torch.clamp(masks, min=0.25, max=1.0)

        overlays = patches.to(DEVICE)*masks

        for image_index in tqdm(range(len(patches)), ascii=True):
            overlay = overlays[image_index]
            patch = patches[image_index]
            mask = masks_for_saving[image_index]

            torchvision.utils.save_image(overlay, os.path.join(save_folder, "patches", str(image_index) + ".jpg"))
            torchvision.utils.save_image(patch, os.path.join(save_folder, "patches_original", str(image_index) + ".jpg"))
            torchvision.utils.save_image(mask, os.path.join(save_folder, "masks", str(image_index) + ".png"))

            for c, component in enumerate(X_embedded[image_index]):
                component = component/torch.max(component)
                component = torch.clamp(component, min=0.25, max=1.0)
                overlay = patch * F.interpolate(component.unsqueeze(dim=0).unsqueeze(dim=0), size=patch.shape[-1])
                torchvision.utils.save_image(overlay, os.path.join(save_folder, "nmf", str(c), str(image_index) + ".jpg"))



