
import torch
import torch.nn.functional as F
import os
import numpy as np

from core.core import collect_patches, generate_activations, extract_attribution_cutoff_mask
from core.utils import get_pred_replace_activations
from core.config import DEVICE, OUTPUT_PATH

from core.utils import access_activations_forward_hook

import torchvision


def create_masks(distance_from_hyperplane, N=10):
    b, _, h, w = distance_from_hyperplane.shape
    A = distance_from_hyperplane
    A_flat_global = A.view(-1)  # Shape: (b * h * w)

    # Sort globally
    sorted_vals, _ = torch.sort(A_flat_global)  # Shape: (b * h * w,)

    # Determine global thresholds for each step
    threshold_indices = (torch.arange(1, N + 1) * (A_flat_global.size(0) / N)).long() - 1
    threshold_indices = threshold_indices.clamp(max=A_flat_global.size(0) - 1)  # Ensure indices are in bounds
    thresholds = sorted_vals[threshold_indices]  # Shape: (N,)

    #print("thresholds: {}".format(thresholds))
    #print("thresholds: {}".format(threshold_indices))
    #print(distance_from_hyperplane)
    #raise RuntimeError

    # Create masks for each step using global thresholds
    masks = []
    for i in range(N):
        mask = (A >= thresholds[i])  # Apply global threshold
        masks.append(mask.float())  # Shape: (b, 1, h, w)

    # Stack masks to form a tensor of shape (N, b, 1, h, w)
    masks = [torch.ones_like(masks[0])] + masks
    masks = torch.stack(masks, dim=0)

    return masks


def get_masks(original_activations, hyperplane_normal, hyperplane_bias):
    hyperplane_normal = hyperplane_normal.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)

    distance_from_hyperplane = original_activations * hyperplane_normal
    distance_from_hyperplane = torch.sum(distance_from_hyperplane, dim=[1]).unsqueeze(dim=1) + hyperplane_bias
    masks = create_masks(distance_from_hyperplane)

    return masks


def visualize_masking(images, unnormalized_images, original_activations, hyperplane_normal, hyperplane_bias, model, original_class, other_class,
                      replacement_value=1.0):

    masks = get_masks(original_activations, hyperplane_normal, hyperplane_bias)

    with torch.no_grad():
        for i, mask in enumerate(masks):
            mask = F.interpolate(mask, size=images.shape[-1])
            imgs = images*mask + replacement_value*(1.0-mask)
            unnormalized_imgs = unnormalized_images*mask + torch.tensor([0.485, 0.456, 0.406], device=DEVICE).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)*(1.0-mask)

            pred = F.softmax(model(imgs), dim=1)[:, other_class]

            if len(pred.shape) == 3:

                pred = torch.max(torch.max(pred, dim=-1)[0], dim=-1)[0]

            print(torch.mean(pred))


            torchvision.utils.save_image(unnormalized_imgs, "/home/digipath2/projects/xai/contrast_diagnoses/out/tmp/masking_samples/img_"+str(i)+".jpg")





def pred_for_masked_input(original_activations, hyperplane_normal, hyperplane_bias, patches, model, other_class,
                   probability_function=lambda x: F.softmax(x, dim=1), do_prints=False, replacement_value=1.0):

    adjusted_preds = []
    max_preds = []
    all_masks = []

    masks = get_masks(original_activations, hyperplane_normal, hyperplane_bias)
    """x
    hyperplane_normal = hyperplane_normal.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)

    #print(hyperplane_normal)

    distance_from_hyperplane = original_activations * hyperplane_normal
    distance_from_hyperplane = torch.sum(distance_from_hyperplane, dim=[1]).unsqueeze(dim=1) + hyperplane_bias
    #print(distance_from_hyperplane)
    masks = create_masks(distance_from_hyperplane)
    """

    """
    for additional_bias in additional_bias_values:
        distance_from_hyperplane = original_activations * hyperplane_normal
        #print(distance_from_hyperplane.shape)
        #print(hyperplane_bias)
        #print(additional_bias)
        distance_from_hyperplane = torch.sum(distance_from_hyperplane, dim=[1]).unsqueeze(dim=1) + hyperplane_bias + additional_bias
        masks = torch.where(distance_from_hyperplane>0.0, 1.0, 0.0)
        all_masks.append(torch.mean(masks))

        #print(torch.mean(masks))
    """
    for mask in masks:
        mask = F.interpolate(mask, size=patches.shape[-1])
        images = patches*mask + replacement_value*(1.0-mask)

        with torch.no_grad():
            pred = probability_function(model(images))#[:, other_class]
            #print(pred.shape)
            #raise RuntimeError

            for_segmentation = (len(pred.shape) == 4)

            if for_segmentation:
                adjusted_preds.append(torch.mean(pred, dim=[0, 2, 3]))

                max_pred_per_image = torch.max(torch.max(pred, dim=-1)[0], dim=-1)[0]
                max_preds.append(torch.mean(max_pred_per_image, dim=0).cpu())
            else:
                pred = torch.mean(pred, dim=0)
                adjusted_preds.append(pred)
                max_preds.append(pred)



    adjusted_preds = torch.stack(adjusted_preds, dim=0)
    max_preds = torch.stack(max_preds, dim=0)
    if do_prints:
        #print("original_preds: {}".format(torch.mean(original_predictions)))

        print("adjusted_preds: {}".format(adjusted_preds))
        print("max adjusted_preds: {}".format(max_preds))
        #print("masks: {}".format(torch.mean(masks, dim=[1, 2, 3, 4])))

    return adjusted_preds, max_preds


def test_mask_input_wrapper(dataloader, model, layer, hyperplane_normal, hyperplane_bias,
                               class_idx, other_class,
                               do_prints=False, replacement_value=1.0):

    return test_mask_input(dataloader, model, layer, hyperplane_normal, hyperplane_bias,
                               other_class,
                               do_prints=do_prints, replacement_value=replacement_value)


def test_shifting_from_attribution_cutoff_only_wrapper(dataloader, model, wrapped_model, layer, hyperplane_normal, hyperplane_bias,
                               class_idx, other_class, offsets,
                               do_prints=False, replacement_value=1.0):

    return test_shifting_from_attribution_cutoff(dataloader, model, wrapped_model, layer, hyperplane_normal, class_idx, offsets, do_prints=do_prints)



def test_shifting_from_attribution_cutoff(dataloader, model, wrapped_model, layer, hyperplane_normal,
                               class_idx, offsets,
                               do_prints=False):

    attribution_masks = extract_attribution_cutoff_mask(dataloader, class_idx, wrapped_model, layer).to(DEVICE)
    activations = generate_activations(model, dataloader, layer).to(DEVICE)

    patches = collect_patches(dataloader).to(DEVICE)

    return get_adjusted_preds(offsets, activations, hyperplane_normal, patches, model, layer, None, masks=attribution_masks, do_prints=do_prints)




def test_mask_input(dataloader, model, layer, hyperplane_normal, hyperplane_bias,
                               other_class,
                               do_prints=False, replacement_value=1.0):

    #offsets = [0.0, -0.5, -1.0, -2.0, -10.0, -15.0, -20.0, -35.0, -50.0, -200.0]#, -500.0, -1000.0, -2000.0]
    #offsets = [-200.0, -2000.0]

    patches = collect_patches(dataloader).to(DEVICE)
    original_activations = generate_activations(model, dataloader, layer)

    hyperplane_normal = hyperplane_normal.to(DEVICE)

    adjusted_preds, max_preds = pred_for_masked_input(original_activations, hyperplane_normal, hyperplane_bias, patches,
                                                            model, other_class,
                                                            do_prints=do_prints,
                                                            replacement_value=replacement_value)

    return adjusted_preds, max_preds





def get_adjusted_preds(offsets, original_activations, hyperplane_normal, patches, model, layer, other_class,
                       probability_function=lambda x: F.softmax(x, dim=1), do_prints=False, use_relu=True,
                       masks=None):
    #print("masks shape: {}".format(masks.shape))
    if masks is None:
        masks = torch.ones_like(original_activations)
    adjusted_preds_thresholded = []
    max_preds_thresholds = []
    adjusted_preds = []
    max_preds = []
    for offset in offsets:
        #print("original_activations shape: {}".format(original_activations.shape))
        if len(original_activations.shape) == 4:
            # is a CNN
            hyperplane_normal_ = hyperplane_normal.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        elif len(original_activations.shape) == 3:
            # is a ViT
            hyperplane_normal_ = hyperplane_normal.unsqueeze(dim=0).unsqueeze(dim=-1)
        else:
            raise RuntimeError
        adjusted_activations = original_activations - masks*hyperplane_normal_*offset
        if use_relu:
            adjusted_activations = F.relu(adjusted_activations)

        #adjusted_activations = adjusted_activations*predictions_mask + original_activations*(1.0-predictions_mask)
        #adjusted_predictions = get_predictions_for_activations(patches, adjusted_activations, model, layer, target_channel=other_class).to(DEVICE)
        with torch.no_grad():
            adjusted_predictions = probability_function(get_pred_replace_activations([patches], model, layer, adjusted_activations))[:, other_class]

        for_segmentation = (len(adjusted_predictions.shape) == 4)

        if for_segmentation:
            adjusted_preds.append(torch.mean(adjusted_predictions, dim=[0, 2, 3]))

            max_pred_per_image = torch.max(torch.max(adjusted_predictions, dim=-1)[0], dim=-1)[0]
            max_preds.append(torch.mean(max_pred_per_image, dim=0).cpu())
        else:
            adjusted_predictions = torch.mean(adjusted_predictions, dim=0)
            adjusted_preds.append(adjusted_predictions)
            max_preds.append(adjusted_predictions)


        #adjusted_preds.append(torch.mean(adjusted_predictions))
        #max_pred_per_image = torch.max(torch.max(adjusted_predictions, dim=-1)[0], dim=-1)[0]
        #max_preds.append(torch.mean(max_pred_per_image).cpu())

    adjusted_preds = torch.stack(adjusted_preds, dim=0)
    max_preds = torch.stack(max_preds, dim=0)
    if do_prints:
        #print("original_preds: {}".format(torch.mean(original_predictions)))

        #print(adjusted_preds.shape)
        #print(max_preds.shape)

        print("adjusted_preds: {}, max: {}".format(adjusted_preds, max_preds))

    return adjusted_preds, max_preds



def test_move_along_hyperplane(patches, model, layer, hyperplane_normal, from_class,
                               other_class, offsets=[0.0, -0.5, -1.0, -2.0, -10.0, -15.0, -20.0, -35.0, -50.0, -200.0],
                               do_prints=False, use_relu=True):

    #offsets = [0.0, -0.5, -1.0, -2.0, -10.0, -15.0, -20.0, -35.0, -50.0, -200.0]#, -500.0, -1000.0, -2000.0]
    #offsets = [-200.0, -2000.0]

    #patches = collect_patches(dataloader).to(DEVICE)
    #original_activations = generate_activations(model, dataloader, layer)

    with torch.no_grad():
        original_activations = access_activations_forward_hook([patches], model, layer)#.cpu()

    hyperplane_normal = hyperplane_normal.to(DEVICE)

    adjusted_preds, max_adjusted_preds = get_adjusted_preds(offsets, original_activations, hyperplane_normal, patches, model, layer, other_class,
                                                            do_prints=do_prints, use_relu=use_relu)

    return adjusted_preds, max_adjusted_preds



    #all_adjusted_preds.append(adjusted_preds_thresholded)

    #print(adjusted_preds_thresholded)


