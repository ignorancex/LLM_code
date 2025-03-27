import os
import csv
import logging
import time
from collections import defaultdict
import numpy as np
import torch
from average_meter import AverageMeter
from torchmetrics import AUROC

from tqdm import tqdm
from utils import GuidedDiffusion, dict2namespace, get_data_loaders_DB, get_data_loaders_midjourney, \
    get_data_loaders_nlb, log_progress, transform_image, transform_image_stablesign

from attack_funcs_2 import wevade_transfer_batch, save_images_and_compute_metrics, get_val_data_loader, DiffPure, \
    compute_tdr_avg


# torch.autograd.set_detect_anomaly(True)


def decode_messages_unwatermarked(model, images) -> torch.Tensor:
    """Decode messages from unwatermarked images and return rounded binary messages."""
    model.decoder.eval()
    with torch.no_grad():
        decoded_messages = model.decoder(images)
    decoded_rounded = torch.clamp(torch.round(decoded_messages), 0, 1).float()
    return decoded_rounded


def build_results_dir(r, wm_method, target, model_type, watermark_length, target_length, train_type, num_models,
                      fixed_message, data_name, optimization, PA, budget, normalization, model_index,
                      resnet_same_encoder, visualization, diffpure):
    """Build the directory string for saving results."""
    dir_string = (
            'results/' + str(r) + '/' +
            wm_method + '_to_' + target + '_' + model_type + '_' +
            str(watermark_length) + '_to_' + str(target_length) + 'bits' + train_type + '_' + str(num_models) + 'models'
    )

    if 'midjourney' in data_name:
        dir_string += '_midjourney'
    if diffpure > 0:
        dir_string += '_diffpure_' + str(diffpure)
    elif not optimization:
        dir_string += '_no_optimization'
        if PA == 'mean':
            dir_string += '_mean'
        elif PA == 'median':
            dir_string += '_median'
        if budget is not None:
            dir_string += f'_budget_{budget}'
            dir_string += f'_{normalization}'
        if not model_index == 1 and num_models == 1:
            dir_string += f'_model_{model_index}'

    # optional but no longer used
    if fixed_message:
        dir_string += '_fixed_message'

    if resnet_same_encoder:
        dir_string += '_resnet_same_encoder'

    if visualization:
        dir_string += '_visualization'

    return dir_string


def test_tfattk_hidden(
        model,
        model_list,
        device,
        hidden_config,
        train_options,
        val_dataset,
        train_type,
        model_type,
        data_name,
        wm_method,
        target,
        smooth,
        encode_wm=True,
        attk_param=None,
        pp=None,
        watermark_length=30,
        target_length=30,
        num_models=1,
        fixed_message=False,
        optimization=True,
        PA='mean',
        budget=None,
        resnet_same_encoder=False,
        normalization=None,
        debug=False,
        model_index=1,
        r=0.25,
        diffpure=0,
        visualization=False,
        attack_type='transfer',
        no_cache=False
):
    """Test transfer attack on Hidden watermarking method."""
    # Build directory string for saving results
    dir_string = build_results_dir(
        r, wm_method, target, model_type, watermark_length, target_length, train_type, num_models,
        fixed_message, data_name, optimization, PA, budget, normalization, model_index,
        resnet_same_encoder, visualization, diffpure
    )

    # Ensure the directory exists
    os.makedirs(dir_string, exist_ok=True)

    # Configure logging to save the log file inside dir_string
    log_file_path = os.path.join(dir_string, 'results.log')
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO)

    # Get the validation data loader and data_type
    val_data, data_type, max_batches = get_val_data_loader(data_name, hidden_config, train_options, val_dataset,
                                                           train=False)

    validation_losses = defaultdict(AverageMeter)
    logging.info('Running validation for transfer attack')
    num = 0

    # Initialize lists to store predictions and labels for AUROC
    all_decoded = []
    all_decoded_attk = []
    all_true = []

    # Initialize AUROC metric (binary classification)
    auroc_metric = AUROC(task='binary').to(device)

    all_decoded_ori = []

    per_image_metrics_all = []

    if diffpure > 0:
        diffpure_transform = DiffPure(steps=diffpure, device=device)

    total_elapsed_time = 0  # Initialize total elapsed time

    for data in tqdm(iter(val_data)):
        num += 1
        image, message = prepare_batch(data, data_type, hidden_config, device, num, fixed_message)

        # Determine attack type
        if diffpure > 0:
            attack_type = 'diffpure'
        else:
            attack_type = 'transfer'

        # Process the batch
        losses, batch_results, batch_elapsed_time = tfattk_validate_on_batch(
            model,
            [image, message],
            model_list,
            num,
            train_type,
            model_type,
            attk_param,
            pp,
            wm_method,
            target,
            smooth=smooth,
            fixed_message=fixed_message,
            optimization=optimization,
            PA=PA,
            budget=budget,
            resnet_same_encoder=resnet_same_encoder,
            normalization=normalization,
            data_name=data_name,
            r=r,
            save_dir=dir_string,
            is_first_batch=True,
            attack_type=attack_type,
            diffpure_transform=diffpure_transform if diffpure > 0 else None,
            visualization=visualization,  # Pass visualization parameter
            no_cache=no_cache
        )

        # Accumulate elapsed time
        total_elapsed_time += batch_elapsed_time

        # Unpack batch results
        encoded_images, attk_images, decoded_messages, decoded_messages_attk_var, per_image_metrics = batch_results

        # Accumulate per-image metrics from all batches
        per_image_metrics_all.extend(per_image_metrics)

        decoded_original = decode_messages_unwatermarked(model, image)
        all_decoded_ori.append(decoded_original.detach().cpu())

        update_validation_losses(validation_losses, losses)

        # Collect decoded messages and true messages for AUROC
        all_decoded.append(decoded_messages.detach().cpu())
        all_true.append(message.detach().cpu())

        if encode_wm:
            decoded_attk = process_decoded_attk(decoded_messages_attk_var, smooth)
            if decoded_attk is not None:
                all_decoded_attk.append(decoded_attk)

        if visualization and num >= 1:
            break

        if max_batches is not None and num >= max_batches:
            break
        # clear cache
        torch.cuda.empty_cache()

    process_and_log_results(
        all_decoded_ori,
        all_decoded,
        all_true,
        all_decoded_attk,
        hidden_config,
        validation_losses,
        auroc_metric,
        dir_string,
        per_image_metrics_all,
        total_elapsed_time  # Pass elapsed time
    )


def prepare_batch(data, data_type, hidden_config, device, num, fixed_message):
    """Prepare the batch data."""
    if data_type == 'batch_dict':
        image = data['image'].to(device)
    elif data_type == 'image_label':
        image, _ = data
        image = image.to(device)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    # Do not resize images here; they should remain at 128x128 unless using DiffPure

    if fixed_message:
        watermark_message = '0001101101000010001110111010101010001100010111001011100110011010'
        # Convert each bit character to a float (0.0 or 1.0)
        message_bits = [float(bit) for bit in watermark_message[:hidden_config.message_length]]
        message = torch.tensor(message_bits, dtype=torch.float).to(device)
        batch_size = image.shape[0]
        message = message.repeat(batch_size, 1)
    else:
        if hidden_config.message_length == 30:
            message_path = './message/' + str(hidden_config.message_length) + 'bits_message_' + str(num) + '.pth'
            message = torch.load(message_path, map_location=device).to(device)
        else:
            message = torch.randint(0, 2, (image.size(0), hidden_config.message_length)).float().to(device)

    return image, message


def process_decoded_attk(decoded_messages_attk_var, smooth):
    """Process the attacked decoded messages."""
    if smooth:
        if decoded_messages_attk_var is not None:
            decoded_attk = decoded_messages_attk_var.detach().cpu()
        else:
            decoded_attk = None
    else:
        if decoded_messages_attk_var is not None:
            decoded_attk = decoded_messages_attk_var.detach().cpu()
        else:
            decoded_attk = None
    return decoded_attk


def update_validation_losses(validation_losses, losses):
    """Update the validation losses."""
    for name, loss in losses.items():
        validation_losses[name].update(loss)


def process_and_log_results(
        all_decoded_ori,
        all_decoded,
        all_true,
        all_decoded_attk,
        hidden_config,
        validation_losses,
        auroc_metric,
        dir_string,
        per_image_metrics_all,
        elapsed_time
):
    """Process results and log metrics."""
    if len(all_decoded_ori) > 0:
        all_decoded_ori = torch.cat(all_decoded_ori, dim=0).view(-1, hidden_config.message_length)
    else:
        all_decoded_ori = None

    # Concatenate all collected data
    all_decoded = torch.cat(all_decoded, dim=0).view(-1, hidden_config.message_length)
    all_true = torch.cat(all_true, dim=0).view(-1, hidden_config.message_length)

    if len(all_decoded_attk) > 0:
        all_decoded_attk = torch.cat(all_decoded_attk, dim=0).view(-1, hidden_config.message_length)
    else:
        all_decoded_attk = None

    # Round all_decoded and all_decoded_attk to 0 and 1 in float
    all_decoded = torch.round(all_decoded)
    if all_decoded_attk is not None:
        all_decoded_attk = torch.clamp(torch.round(all_decoded_attk), 0, 1).float()

    ori_acc_image_wise = 1 - torch.sum(torch.abs(all_decoded_ori - all_true), dim=1) / hidden_config.message_length
    acc_image_wise = 1 - torch.sum(torch.abs(all_decoded - all_true), dim=1) / hidden_config.message_length
    if all_decoded_attk is not None:
        acc_image_wise_attk = 1 - torch.sum(torch.abs(all_decoded_attk - all_true),
                                            dim=1) / hidden_config.message_length

    # 0 for original and 1 for watermarked (attacked and not attacked)
    labels_ori = torch.zeros_like(ori_acc_image_wise)
    labels_wm = torch.ones_like(acc_image_wise)

    labels_ori_wm = torch.cat([labels_ori, labels_wm], dim=0)

    preds_ori_wm = torch.cat([ori_acc_image_wise, acc_image_wise], dim=0)
    preds_ori_wm_attk = torch.cat([ori_acc_image_wise, acc_image_wise_attk], dim=0)

    auroc_unattacked = auroc_metric(preds_ori_wm, labels_ori_wm)
    auroc = auroc_metric(preds_ori_wm_attk, labels_ori_wm)

    # Add AUROC to the validation losses
    validation_losses['auroc_unattacked'] = AverageMeter()
    validation_losses['auroc'] = AverageMeter()
    validation_losses['auroc_unattacked'].update(auroc_unattacked)
    validation_losses['auroc'].update(auroc)

    # Compute average LPIPS over all images
    lpips_values = [float(metric['lpips']) for metric in per_image_metrics_all if 'lpips' in metric]
    average_lpips = np.mean(lpips_values)

    # Update validation losses with the correct LPIPS value
    validation_losses['LPIPS'] = AverageMeter()
    validation_losses['LPIPS'].update(average_lpips)

    # Add elapsed time to validation losses
    validation_losses['elapsed_time'] = AverageMeter()
    validation_losses['elapsed_time'].update(elapsed_time)

    # Log all metrics including AUROC and elapsed time
    log_progress(validation_losses)

    # Save per-image metrics to CSV
    csv_file_path = os.path.join(dir_string, 'per_image_metrics.csv')
    with open(csv_file_path, mode='w', newline='') as csvfile:
        fieldnames = ['image_index', 'l_inf', 'l2', 'ssim', 'lpips', 'bitwise_accuracy', 'bitwise_accuracy_attk']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for metric in per_image_metrics_all:
            writer.writerow(metric)


def tfattk_validate_on_batch(
        model,
        batch: list,
        model_list: list,
        num: int,
        train_type: str,
        model_type: str,
        attk_param: float,
        pp: str,
        wm_method,
        target,
        encode_wm=True,
        white=False,
        smooth=False,
        fixed_message=False,
        optimization=True,
        PA='mean',
        budget=None,
        resnet_same_encoder=False,
        normalization='clamp',
        data_name='DB',
        r=0.25,
        save_dir=None,
        is_first_batch=False,
        attack_type='transfer',
        diffpure_transform=None,
        visualization=False,
        no_cache=False
):
    # Unpack the batch
    images, messages = batch
    batch_size = images.shape[0]

    if encode_wm:
        encoded_images = encode_watermark(model, images, messages)
    else:
        if model is not None:
            model.eval()
        encoded_images = images

    encoded_images = encoded_images.clamp(-1, 1)

    # Start timing here
    start_time = time.time()
    if attack_type == 'transfer':
        attk_images = perform_transfer_attack(
            model, encoded_images, messages, model_list, num, train_type, model_type, attk_param, pp, wm_method,
            target, encode_wm, white, fixed_message, optimization, PA, budget, resnet_same_encoder, normalization,
            data_name, r, visualization,
            no_cache=no_cache
        )
    elif attack_type == 'diffpure':
        attk_images = perform_diffpure_attack(encoded_images, diffpure_transform)
    else:
        raise ValueError(f"Unsupported attack_type: {attack_type}")
    # End timing here
    elapsed_time = time.time() - start_time

    # Decode messages
    decoded_messages, decoded_messages_attk_var = decode_messages(
        model, encoded_images, attk_images, encode_wm, smooth, images.device
    )

    # Save images and compute per-image metrics if it's the first batch
    per_image_metrics = []
    if is_first_batch:
        per_image_metrics = save_images_and_compute_metrics(
            images, encoded_images, attk_images, decoded_messages,
            decoded_messages_attk_var,
            messages, num, save_dir, encode_wm, smooth
        )

    # Compute losses
    losses = compute_losses(
        model, decoded_messages, decoded_messages_attk_var, messages, batch_size, encode_wm, smooth, per_image_metrics,
        encoded_images, attk_images  # Pass encoded and attacked images
    )

    return losses, (
        encoded_images,
        attk_images,
        decoded_messages,
        decoded_messages_attk_var,
        per_image_metrics
    ), elapsed_time


def encode_watermark(model, images, messages):
    """Encode the watermark into the images."""
    model.encoder.eval()
    model.decoder.eval()
    with torch.no_grad():
        # if messages.shape[-1] == 256:  # MBRS, scale to 256 by 256
        #    images_to_encode = torch.nn.functional.interpolate(images, size=(256, 256), mode='bilinear',
        # align_corners = False)
        #     encoded_images = model.encoder(images_to_encode, messages)
        # rescale to 128 by 128
        #     encoded_images = torch.nn.functional.interpolate(encoded_images, size=(128, 128), mode='bilinear',
        # align_corners = False)
        # else:
        encoded_images = model.encoder(images, messages)
    return encoded_images


def perform_transfer_attack(
        model, encoded_images, messages, model_list, num, train_type, model_type, attk_param, pp, wm_method, target,
        encode_wm, white, fixed_message, optimization, PA, budget, resnet_same_encoder, normalization, data_name, r,
        visualization=False,  # Added visualization parameter
        no_cache=False
):
    """Perform the transfer attack."""
    # Generate noise for attack
    noise = wevade_transfer_batch(
        encoded_images.clone(),
        messages.size(1),
        model_list,
        watermark_length=30,
        iteration=5000,
        lr=2,
        r=r,
        epsilon=0.2,
        num=num,
        name='flipratio_batch',
        train_type=train_type,
        model_type=model_type,
        batch_size=len(model_list),
        wm_method=wm_method,
        target=target,
        white=white,
        fixed_message=fixed_message,
        optimization=optimization,
        PA=PA,
        budget=budget,
        resnet_same_encoder=resnet_same_encoder,
        data_name=data_name,
        visualization=visualization,  # Pass visualization parameter,
        no_cache=no_cache
    )

    with torch.no_grad():
        attk_images = encoded_images + noise
        if budget:
            attk_images = adjust_attk_images_budget(attk_images, encoded_images, budget, normalization)

        # Do not resize images; keep them at 128x128

    return attk_images


def perform_diffpure_attack(encoded_images, diffpure_transform):
    """Apply the DiffPure attack."""
    with torch.no_grad():
        # Apply DiffPure to the encoded images
        attk_images = diffpure_transform(encoded_images)
        # Images are resized within DiffPure class
    return attk_images


def decode_messages(model, encoded_images, attk_images, encode_wm, smooth, device):
    """Decode messages from images."""
    with torch.no_grad():
        if encode_wm:
            attk_images = transform_image(attk_images, model.device)
            encoded_images = transform_image(encoded_images, model.device)

            decoded_messages = model.decoder(encoded_images)
            if smooth:
                decoded_messages_attk_all = []
                for image in attk_images:
                    noised_images = []
                    for _ in range(100):
                        gaussian_noise = torch.randn(image.shape).to(device)
                        noised_image = image + 0.0015 * gaussian_noise
                        noised_images.append(noised_image)
                    noised_images = torch.stack(noised_images)
                    decoded_messages_attk = model.decoder(noised_images)
                    decoded_messages_attk_all.append(decoded_messages_attk)
                decoded_messages_attk_all = torch.stack(decoded_messages_attk_all)
                decoded_messages_attk_var = decoded_messages_attk_all
            else:
                decoded_messages_attk = model.decoder(attk_images)
                decoded_messages_attk_var = decoded_messages_attk
        else:
            if model is not None:
                encoded_images_tdr = transform_image_stablesign(encoded_images, device)
                attk_images_tdr = transform_image_stablesign(attk_images, device)
                decoded_messages = model(encoded_images_tdr) + 0.5
                decoded_messages_attk = model(attk_images_tdr) + 0.5
                decoded_messages_attk_var = decoded_messages_attk
            else:
                decoded_messages = None
                decoded_messages_attk_var = None
    return decoded_messages, decoded_messages_attk_var


def adjust_attk_images_budget(attk_images, encoded_images, budget, normalization):
    """Adjust the attacked images based on budget constraints."""
    attk_images = attk_images.clamp(-1, 1)
    real_noise = attk_images - encoded_images
    l_inf = torch.norm(real_noise, p=float('inf'), dim=(1, 2, 3))
    l_inf = l_inf.reshape(-1, 1, 1, 1)
    if normalization == 'scale':
        attk_images = encoded_images + real_noise / l_inf * budget
    elif normalization == 'clamp':
        real_noise = torch.clamp(real_noise, -budget, budget)
        attk_images = encoded_images + real_noise
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")
    return attk_images


def compute_losses(model, decoded_messages, decoded_messages_attk_var, messages, batch_size, encode_wm, smooth,
                   per_image_metrics, encoded_images, attk_images):
    """Compute the losses for the batch."""
    if model is not None:
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        messages_np = messages.detach().cpu().numpy()
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages_np)) / (batch_size * messages.shape[1])

        if encode_wm and smooth:
            messages_expanded = messages_np[:, None, :]
            decoded_rounded_attk_all = decoded_messages_attk_var.detach().cpu().numpy().round().clip(0, 1)
            bitwise_errors = np.abs(decoded_rounded_attk_all - messages_expanded)
            bitwise_error_sums = np.sum(bitwise_errors, axis=2)
            median_bitwise_errors = np.median(bitwise_error_sums, axis=1)
            bitwise_avg_err_attk = np.mean(median_bitwise_errors) / messages.shape[1]
        elif encode_wm:
            decoded_rounded_attk = decoded_messages_attk_var.detach().cpu().numpy().round().clip(0, 1)
            bitwise_avg_err_attk = np.sum(np.abs(decoded_rounded_attk - messages_np)) / (
                    batch_size * messages.shape[1])
            median_bitwise_errors = None
        else:
            decoded_rounded_attk = decoded_messages_attk_var.detach().cpu().numpy().round().clip(0, 1)
            bitwise_avg_err_attk = np.sum(np.abs(decoded_rounded_attk - messages_np)) / (
                    batch_size * messages.shape[1])
            median_bitwise_errors = None

        tdr_avg_1, tdr_avg_2, tdr_avg_attk_1, tdr_avg_attk_2 = compute_tdr_avg(
            decoded_rounded,
            decoded_messages_attk_var.detach().cpu().numpy().round().clip(0,
                                                                          1) if encode_wm and not smooth else decoded_rounded_attk_all,
            messages,
            encode_wm and smooth,
            median_bitwise_errors
        )

        # Compute L-2 and L-inf norms between images
        l_inf_values = torch.norm((encoded_images - attk_images).reshape(encoded_images.size(0), -1), p=float('inf'),
                                  dim=1)
        l2_values = torch.norm((encoded_images - attk_images).reshape(encoded_images.size(0), -1), p=2, dim=1)

        losses = {
            'noise (L-infinity)  ': l_inf_values.mean().item(),
            'noise (L-2)         ': l2_values.mean().item(),
            'bitwise-acc         ': 1 - bitwise_avg_err,
            'bitwise-acc_attk    ': 1 - bitwise_avg_err_attk,
            'tdr                 ': tdr_avg_1 + tdr_avg_2,
            'tdr_attk            ': tdr_avg_attk_1 + tdr_avg_attk_2,
            'ssim                ': np.mean(
                [float(metric['ssim']) for metric in per_image_metrics]) if per_image_metrics else 0,
            # 'LPIPS               ': np.mean(
            #    [float(metric['lpips']) for metric in per_image_metrics]) if per_image_metrics else 0,
        }
    else:
        losses = {}
    return losses
