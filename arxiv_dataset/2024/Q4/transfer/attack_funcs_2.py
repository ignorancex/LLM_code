import os
import math
import random
import numpy as np
import torch
import torchvision
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from pytorch_msssim import ssim
from utils import GuidedDiffusion, dict2namespace, get_data_loaders_DB, get_data_loaders_midjourney, \
    get_data_loaders_nlb, log_progress, transform_image, transform_image_stablesign
from torch import nn
import yaml


def save_images_and_compute_metrics(
        images, encoded_images, attk_images, decoded_messages, decoded_messages_attk_var,
        messages, num, save_dir, encode_wm, smooth
):
    """Save images and compute per-image metrics."""
    # Normalize images to [0,1]
    images_to_save = (images.clone() + 1) / 2
    images_to_save = images_to_save.clamp_(0, 1)

    encoded_images_to_save = (encoded_images.clone() + 1) / 2
    encoded_images_to_save = encoded_images_to_save.clamp_(0, 1)

    attk_images_to_save = (attk_images.clone() + 1) / 2
    attk_images_to_save = attk_images_to_save.clamp_(0, 1)

    # Define directories
    if save_dir is None:
        save_base_dir = 'results/images/'
    else:
        save_base_dir = os.path.join(save_dir, 'images')

    unwatermarked_dir = os.path.join(save_base_dir, 'unwatermarked')
    watermarked_before_attack_dir = os.path.join(save_base_dir, 'watermarked_before_attack')
    watermarked_after_attack_dir = os.path.join(save_base_dir, 'watermarked_after_attack')

    # Create directories if they don't exist
    for dir_path in [unwatermarked_dir, watermarked_before_attack_dir, watermarked_after_attack_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Save images and compute per-image metrics
    batch_size = images.size(0)
    num_images_to_save = min(20, batch_size)  # Changed from 10 to 20

    per_image_metrics = []

    # Compute SSIM per image
    ssim_value = []
    for i in range(encoded_images.shape[0]):
        SSIM_metric = StructuralSimilarityIndexMeasure(data_range=attk_images[i].max() - attk_images[i].min()).to(
            encoded_images.device)
        ssim_val = SSIM_metric(encoded_images[i].unsqueeze(0), attk_images[i].unsqueeze(0)).cpu().numpy()
        if ssim_val == -math.inf or np.isnan(ssim_val):
            ssim_val = 0
        ssim_value.append(ssim_val)
    ssim_value = np.array(ssim_value)

    # Compute l2 and l-inf norms per image
    l_inf_values = torch.norm((encoded_images - attk_images).reshape(encoded_images.size(0), -1), p=float('inf'), dim=1)
    l2_values = torch.norm((encoded_images - attk_images).reshape(encoded_images.size(0), -1), p=2, dim=1)

    # Compute bitwise accuracy per image
    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    messages_np = messages.detach().cpu().numpy()

    bit_errors = np.sum(np.abs(decoded_rounded - messages_np), axis=1)
    bitwise_accuracy_per_image = 1 - bit_errors / messages.shape[1]

    # For attacked images
    if encode_wm and smooth:
        decoded_rounded_attk_all = decoded_messages_attk_var.detach().cpu().numpy().round().clip(0, 1)
        bitwise_accuracy_attk_per_image = []
        for i in range(decoded_rounded_attk_all.shape[0]):
            decoded_attk_i_all = decoded_rounded_attk_all[i]
            messages_i_expanded = messages_np[i][None, :]
            bit_errors_attk = np.sum(np.abs(decoded_attk_i_all - messages_i_expanded), axis=1)
            bit_errors_attk_median = np.median(bit_errors_attk)
            bitwise_accuracy_attk_i = 1 - bit_errors_attk_median / messages.shape[1]
            bitwise_accuracy_attk_per_image.append(bitwise_accuracy_attk_i)
        bitwise_accuracy_attk_per_image = np.array(bitwise_accuracy_attk_per_image)
    else:
        decoded_rounded_attk = decoded_messages_attk_var.detach().cpu().numpy().round().clip(0, 1)
        bit_errors_attk = np.sum(np.abs(decoded_rounded_attk - messages_np), axis=1)
        bitwise_accuracy_attk_per_image = 1 - bit_errors_attk / messages.shape[1]

    # Initialize LPIPS metric
    lpips_metric = lpips.LPIPS(net='alex').to(encoded_images.device)

    lpips_values = []

    for i in range(num_images_to_save):
        # Generate unique filename
        filename = f'image_batch{num}_idx{i}.png'
        # Save unwatermarked image
        torchvision.utils.save_image(images_to_save[i], os.path.join(unwatermarked_dir, filename))
        # Save watermarked image before attack
        torchvision.utils.save_image(encoded_images_to_save[i], os.path.join(watermarked_before_attack_dir, filename))
        # Save watermarked image after attack
        torchvision.utils.save_image(attk_images_to_save[i], os.path.join(watermarked_after_attack_dir, filename))

        # Compute per-image metrics
        l_inf_value = l_inf_values[i].item()
        l2_value = l2_values[i].item()

        ssim_value_i = ssim_value[i]

        bitwise_accuracy_i = bitwise_accuracy_per_image[i]

        bitwise_accuracy_attk_i = bitwise_accuracy_attk_per_image[i]

        # Compute LPIPS per image
        lpips_value_i = lpips_metric(encoded_images[i].unsqueeze(0), attk_images[i].unsqueeze(0)).item()
        lpips_values.append(lpips_value_i)

        # Format metrics to 5 significant digits
        per_image_metrics.append({
            'image_index': i,
            'l_inf': f"{l_inf_value:.5g}",
            'l2': f"{l2_value:.5g}",
            'ssim': f"{ssim_value_i:.5g}",
            'lpips': f"{lpips_value_i:.5g}",
            'bitwise_accuracy': f"{bitwise_accuracy_i:.5g}",
            'bitwise_accuracy_attk': f"{bitwise_accuracy_attk_i:.5g}"
        })

    return per_image_metrics


def project(param_data, backup, epsilon):
    """Project the perturbation back if it exceeds the upper bound."""
    r = param_data - backup
    r = epsilon * r
    return backup + r


def wevade_transfer_batch(all_watermarked_image, target_length, model_list, watermark_length, iteration, lr, r, epsilon,
                          num, name, train_type, model_type, batch_size, wm_method, target, white=False,
                          fixed_message=False, optimization=True, PA='mean',
                          budget=None, resnet_same_encoder=False, data_name='DB', visualization=False, no_cache=False):
    """Perform transfer attack by generating perturbations."""
    watermarked_image_cloned = all_watermarked_image.clone()
    criterion = nn.MSELoss(reduction='mean')

    dataset_name = 'DB' if 'DB' in data_name else 'midjourney'
    base_dir = f'./wevade_perturb_{wm_method}_to_{target}_{model_type}_{watermark_length}_to_{target_length}bits/{train_type}/{dataset_name}/{r}'

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    filename = (
        f'ensemble_model{len(model_list)}_{name}_{batch_size}_'
        f'{int((1 - epsilon) * 100)}_batch_{num}'
    )

    if fixed_message:
        filename += '_fixed'

    if white:
        filename += '_white'

    if not optimization:
        filename += '_no_optimization'
        if PA == 'mean':
            filename += '_mean'
        elif PA == 'median':
            filename += '_median'
    if resnet_same_encoder:
        filename += '_resnet_same_encoder'

    filename += '.pth'
    path = os.path.join(base_dir, filename)

    if not optimization:
        watermarks = []
        for idx, sur_model in enumerate(model_list):
            sur_model.encoder.eval()

            with torch.no_grad():
                sur_model.decoder.eval()
                decoded_messages = sur_model.decoder(all_watermarked_image)
                decoded_rounded = torch.clamp(torch.round(decoded_messages), 0, 1)
                target_watermark = 1 - decoded_rounded
                watermark = sur_model.encoder(all_watermarked_image, target_watermark) - all_watermarked_image
                watermarks.append(watermark)
        watermarks = torch.stack(watermarks)
        if PA == 'mean':
            all_perturbations = watermarks.mean(dim=0)
        elif PA == 'median':
            all_perturbations = watermarks.median(dim=0).values
        else:
            raise ValueError(f"Unsupported PA: {PA}")
    elif os.path.exists(path) and not visualization and not no_cache:
        all_perturbations = torch.load(path, map_location='cpu').to(all_watermarked_image.device)
    else:
        assert 0
        target_watermark_list = []

        with torch.no_grad():
            watermarked_image = watermarked_image_cloned.clone()
            all_perturbations = torch.zeros_like(watermarked_image)
            continue_processing_mask = torch.ones(watermarked_image.shape[0], dtype=torch.bool,
                                                  device=watermarked_image.device)

        for idx, sur_model in enumerate(model_list):
            with torch.no_grad():
                sur_model.decoder.eval()
                decoded_messages = sur_model.decoder(watermarked_image)
                decoded_rounded = torch.clamp(torch.round(decoded_messages), 0, 1)
                target_watermark = 1 - decoded_rounded
                target_watermark_list.append(target_watermark)
        target_watermark_list = torch.stack(target_watermark_list)

        for _ in tqdm(range(iteration)):
            if not continue_processing_mask.all():
                watermarked_image = all_watermarked_image[continue_processing_mask]
            watermarked_image = watermarked_image.requires_grad_(True)
            min_value, _ = torch.min(watermarked_image.reshape(watermarked_image.size(0), -1), dim=1, keepdim=True)
            max_value, _ = torch.max(watermarked_image.reshape(watermarked_image.size(0), -1), dim=1, keepdim=True)
            min_value = min_value.view(watermarked_image.size(0), 1, 1, 1)
            max_value = max_value.view(watermarked_image.size(0), 1, 1, 1)

            grads = 0
            idx_list = random.sample(range(0, len(model_list)), batch_size)
            for idx in idx_list:
                sur_model = model_list[idx]
                decoded_watermark = sur_model.decoder(watermarked_image)
                loss = criterion(decoded_watermark,
                                 target_watermark_list[idx][continue_processing_mask]) * watermarked_image.size(0)
                grads += torch.autograd.grad(loss, watermarked_image)[0]
                sur_model.decoder.zero_grad()
            grads /= len(idx_list)

            with torch.no_grad():
                watermarked_image = watermarked_image - lr * grads
                watermarked_image = torch.clamp(watermarked_image, min_value, max_value)

                # Projection.
                perturbation_norm = torch.norm(watermarked_image - watermarked_image_cloned[continue_processing_mask],
                                               p=float('inf'), dim=(1, 2, 3), keepdim=True)
                exceeding_indices = perturbation_norm > r
                c = torch.where(exceeding_indices, r / perturbation_norm, torch.ones_like(perturbation_norm))
                if torch.sum(exceeding_indices) > 0:
                    watermarked_image = project(watermarked_image, watermarked_image_cloned[continue_processing_mask],
                                                c)

                bit_acc_target = torch.zeros((len(model_list), len(watermarked_image))).to(watermarked_image.device)
                for idx, sur_model in enumerate(model_list):
                    sur_model.decoder.eval()
                    decoded_watermark = sur_model.decoder(watermarked_image)
                    rounded_decoded_watermark = decoded_watermark.detach().round().clamp(0, 1)
                    bit_acc_target[idx] = 1 - (torch.abs(
                        rounded_decoded_watermark - target_watermark_list[idx][continue_processing_mask]).sum(
                        dim=1) / watermark_length)
                bit_acc_target = torch.mean(bit_acc_target).item()

                perturbation = watermarked_image - watermarked_image_cloned[continue_processing_mask]
                all_perturbations[continue_processing_mask] = perturbation
                all_watermarked_image[continue_processing_mask] = watermarked_image
                ssim_value = ssim((watermarked_image + 1) / 2,
                                  (watermarked_image_cloned[continue_processing_mask] + 1) / 2, data_range=1,
                                  size_average=False)

                conditions_met = (ssim_value <= 0.9).view(-1) | (bit_acc_target >= 1 - epsilon)
                continue_processing_mask[continue_processing_mask.clone()] &= ~conditions_met

                # If all images in the batch have met the condition, stop processing
                if not continue_processing_mask.any():
                    break

        torch.save(all_perturbations, path)

    return all_perturbations


def get_val_data_loader(data_name, hidden_config, train_options, val_dataset, train=False):
    """Retrieve the validation data loader based on the dataset name."""
    if 'DB' in data_name:
        val_data = get_data_loaders_DB(hidden_config, train_options, dataset=val_dataset, train=train)
        data_type = 'batch_dict'
        max_batches = None
    elif 'midjourney' in data_name:
        val_data = get_data_loaders_midjourney(hidden_config, train_options, dataset=val_dataset, train=train)
        data_type = 'image_label'
        max_batches = None
    elif 'nlb' in data_name:
        val_data = get_data_loaders_nlb(hidden_config, train_options, dataset=val_dataset)
        data_type = 'image_label'
        max_batches = 1
    else:
        raise ValueError(f"Unknown data_name: {data_name}")
    return val_data, data_type, max_batches


class DiffPure:
    """Class for diffusion purification."""

    def __init__(self, steps=0.4, fname="base", device='cuda'):
        with open('DiffPure/configs/imagenet.yml', 'r') as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)
        self.device = device  # Ensure device is set
        self.config.device = self.device  # Set device in config if necessary
        self.runner = GuidedDiffusion(
            self.config,
            t=int(steps * int(self.config.model.timestep_respacing)),
            model_dir='DiffPure/pretrained/guided_diffusion',
            device=self.device  # Pass device to GuidedDiffusion
        )
        self.steps = steps
        self.cnt = 0
        self.fname = fname
        self.save_imgs = False  # Set default to False

    def __call__(self, img):
        # img is of shape (batch_size, channels, height, width), in [-1,1]
        img = img.to(self.device)
        # Convert from [-1,1] to [0,1]
        img = (img + 1) / 2
        # Resize input images to 256x256 before processing
        img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        # Now scale img to [-1,1] before feeding to DiffPure's image_editing_sample
        img_scaled = (img - 0.5) * 2
        img_pured, img_noisy = self.runner.image_editing_sample(img_scaled.to(self.device))
        # img_pured is in [-1,1]; scale back to [0,1]
        img_pured = (img_pured + 1) / 2  # Keep on device
        # Resize output images back to 128x128 after DiffPure processing
        img_pured = torch.nn.functional.interpolate(img_pured, size=(128, 128), mode='bilinear', align_corners=False)
        # Finally, scale back to [-1,1]
        img_pured = img_pured * 2 - 1
        return img_pured

    def __repr__(self):
        return self.__class__.__name__ + f'(steps={self.steps})'


def compute_tdr_avg(decoded_rounded, decoded_rounded_attk, messages, smooth, median_bitwise_errors=None):
    """Compute the true detection rate average."""
    threshold2_numerators = {
        20: 2,
        30: 5,
        32: 5,
        48: 11,
        64: 17,
        100: 31,
        256: 97,
        # <1e-4 false positive rate
    }

    message_length = messages.size(1)
    if message_length in threshold2_numerators:
        threshold2_numerator = threshold2_numerators[message_length]
        threshold1_numerator = message_length - threshold2_numerator
        threshold1 = threshold1_numerator / message_length
        threshold2 = threshold2_numerator / message_length
    else:
        raise ValueError(f"Unsupported message length: {message_length}")

    messages_numpy = messages.detach().cpu().numpy()
    decoded_numpy = decoded_rounded
    bit_errors = np.sum(np.abs(decoded_numpy - messages_numpy), axis=1)
    per_message_accuracy = 1 - bit_errors / message_length
    tdr_avg_1 = np.mean(per_message_accuracy > threshold1)
    tdr_avg_2 = np.mean(per_message_accuracy < threshold2)

    if smooth:
        if median_bitwise_errors is None:
            raise ValueError("median_bitwise_errors must be provided when smooth is True")
        per_message_accuracy_attk = 1 - median_bitwise_errors / message_length
    else:
        decoded_attk_numpy = decoded_rounded_attk
        bit_errors_attk = np.sum(np.abs(decoded_attk_numpy - messages_numpy), axis=1)
        per_message_accuracy_attk = 1 - bit_errors_attk / message_length

    tdr_avg_attk_1 = np.mean(per_message_accuracy_attk > threshold1)
    tdr_avg_attk_2 = np.mean(per_message_accuracy_attk < threshold2)

    return tdr_avg_1, tdr_avg_2, tdr_avg_attk_1, tdr_avg_attk_2
