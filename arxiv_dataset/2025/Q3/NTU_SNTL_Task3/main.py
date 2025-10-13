"""
main.py

This is the entry point for the project. It orchestrates the training pipeline,
including data preparation, model training, and evaluation.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""
import os.path
import torch
import wandb
from parameters import params
from model import SELDModel
from loss import SELDLossADPIT, SELDLossSingleACCDOA
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from torch.utils.data import DataLoader
from extract_features import SELDFeatureExtractor
import utils
from rich.progress import Progress
import platform
import time
import training_utils as tu
import argparse
import numpy as np


def setup_model_and_loss(in_feat_shape, device):
    """
    Initializes the model, loss, and metrics.
    """
    model_name = params['model'].lower() # You can add additional models if you want
    if "seldnet" in model_name:
        model = SELDModel(params=params, in_feat_shape=in_feat_shape).to(device)
        print(f"Using SELDNet!")
    else:
        raise ValueError("Unknown model type specified.")

    import torchinfo
    model_profile = torchinfo.summary(model, input_size=in_feat_shape)
    print('MACC:\t \t %.3f' %  (model_profile.total_mult_adds/1e9), 'G')
    print('Memory:\t \t %.3f' %  (model_profile.total_params/1e6), 'M\n')

    # Checking for multiple GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Found {n_gpus} GPU(s)")
        model = torch.nn.DataParallel(model)  # replicates module across GPUs
        params['nb_workers'] = int(params['nb_workers'] * n_gpus)
        print(f"Using nn.DataParallel on {n_gpus} GPUs, using {params['nb_workers']} workers")

    # Instantiate loss based on the configuration.
    if params['multiACCDOA']:
        loss_fn = SELDLossADPIT(params=params).to(device)
    else:
        loss_fn = SELDLossSingleACCDOA(params=params).to(device)
    metrics = ComputeSELDResults(params=params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))
    print("Metrics are set up!")
    return model, loss_fn, metrics


def train_epoch(seld_model, dev_train_iterator, optimizer, seld_loss, step_scheduler=None):

    seld_model.train()
    batch_losses = []
    num_batches = len(dev_train_iterator)

    bs = params['batch_size']
    accum_bs = params.get('accum_batch', bs)
    if accum_bs % bs != 0: raise ValueError(f"accum_batch ({accum_bs}) must be a multiple of batch_size ({bs})")
    accum_steps = accum_bs // bs

    optimizer.zero_grad()
    with Progress(transient=True) as progress:
        task = progress.add_task("[red]Training:\t", total=num_batches)

        for batch_idx, (input_features, labels) in enumerate(dev_train_iterator):

            input_features, labels = input_features.to(device), labels.to(device)

            logits = seld_model(input_features)

            # compute loss and scale down by accum_steps
            loss = seld_loss(logits, labels) / accum_steps
            loss.backward()

            true_loss = loss.item() * accum_steps
            batch_losses.append(true_loss)

            # every accum_steps mini‑batches, do optimizer step + scheduler
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                optimizer.step()
                if step_scheduler is not None: step_scheduler.step()
                optimizer.zero_grad()

            progress.update(task, advance=1)

    # Return average loss
    return float(np.mean(batch_losses))


def val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, is_jackknife=False):

    seld_model.eval()
    val_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.

    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Validation:\t", total=len(dev_test_iterator))
        with torch.no_grad():
            for j, (input_features, labels) in enumerate(dev_test_iterator):
                input_features, labels = input_features.to(device), labels.to(device)

                # Forward pass
                logits = seld_model(input_features)

                # Compute loss
                loss = seld_loss(logits, labels)
                val_loss_per_epoch += loss.item()

                # save predictions to csv files for metric calculations
                start = j * params['val_batch_size']
                end = start + logits.size(0)
                utils.write_logits_to_dcase_format(logits, params, output_dir, dev_test_iterator.dataset.label_files[start: end])
                progress.update(task, advance=1)

    # Clear up memory
    del input_features, labels, logits
    torch.cuda.empty_cache()

    avg_val_loss = val_loss_per_epoch / len(dev_test_iterator)
    metric_scores = seld_metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'), is_jackknife=is_jackknife)

    return avg_val_loss, metric_scores


def main(device, args):

    # Convert and update the params dictionary
    arg_dict = {k: v for k,v in vars(args).items() if v is not None}
    params.update(arg_dict)
    params['net_type'] = args.exp # Get unique experiment name
    print("Experiment Parameters:")
    for key, value in params.items():
        print(f"\t{key}: {value}")

    # Adjust number of workers based on the system.
    if platform.system() == 'Linux': 
        params['nb_workers'] = 4

    # Set up directories for storing model checkpoints, predictions(output_dir), and create a summary writer
    checkpoints_folder, output_dir, summary_writer = utils.setup(params)
    print(f"Saving best model in: {checkpoints_folder}")

    # Feature extraction code.
    feat_folder = (f"mel{params['nb_mels']}" if params['nb_mels'] else "linspec") + ("_gamma" if params['gamma'] else "") + \
                  ("_ipd" if params['ipd'] else "") + ("_iv" if params['iv'] else "") + ("_slite" if params['slite'] else "") + \
                  ("_ms" if params['ms'] else "") + ("_dnorm" if params['dnorm'] else "") 
    params['feat_dir'] = os.path.join(params['root_dir'], feat_folder)

    # Set up feature extractor and preprocessing
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')
    feature_extractor.preprocess_features(split='dev')

    # Setup augmentations if enabled.
    n_aug_channels = 4 if params['ms'] else 2
    if params['cutout'] or params['itfm']:
        transforms = tu.CompositeCutoutTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels, use_itfm=params['itfm'])
    elif params['freqshift']:
        transforms = tu.RandomShiftUpDownTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    elif params['filtaug']:
        transforms = tu.FilterAugmentNormalized(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    elif params['compfreq']:
        transforms = tu.CompositeFrequencyTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    elif params['alltrans']:
        transforms = tu.CompositeEverythingTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels, use_itfm=params['itfm'])
    else:
        transforms = None

    # Set up dev_train and dev_test data iterator
    dev_train_dataset = DataGenerator(params=params, mode='dev_train', transform=transforms)
    dev_train_iterator = DataLoader(dataset=dev_train_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=params['shuffle'],
                                    drop_last=False, pin_memory=True)

    dev_test_dataset = DataGenerator(params=params, mode='dev_test')
    dev_test_iterator = DataLoader(dataset=dev_test_dataset, batch_size=params['val_batch_size'], num_workers=params['nb_workers'], shuffle=False, drop_last=False)

    # Getting the input feature shape
    first_batch = next(iter(dev_train_iterator))
    in_feat_shape, out_feat_shape = first_batch[0].shape, first_batch[1].shape
    print(f"Number of batches: {len(dev_train_iterator)}\nIn Shape: {in_feat_shape}\tOut Shape: {out_feat_shape}")

    # Start Training Loops
    start_epoch, best_epoch = 0, 0
    best_f_score = float('-inf')
    best_seld_err = 1.0

    # Set up model, loss, and metrics
    seld_model, seld_loss, seld_metrics = setup_model_and_loss(in_feat_shape=in_feat_shape, device=device)

    # Set up optimizer and scheduler
    if params['weight_decay'] != 0:
        no_decay = ["bias", "bn.weight", "bn.bias", "ln.weight", "ln.bias"]
        decay_params = []
        no_decay_params = []
        for name, param in seld_model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer_grouped_parameters = [{'params': decay_params,    'weight_decay': params['weight_decay']},
                                        {'params': no_decay_params, 'weight_decay': 0.0}]        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=params['learning_rate'])
        print(f"Using Adam with Weight Decay optimizer (Peak LR: {params['learning_rate']}, WD: {params['weight_decay']})")
    else:
        optimizer = torch.optim.Adam(params=seld_model.parameters(), lr=params['learning_rate'])
        print(f"Using vanilla Adam optimizer (Peak LR: {params['learning_rate']})")

    # Now we set the learning rate scheduler
    total_steps = params['nb_epochs'] * np.ceil(len(dev_train_iterator) / (int(params['accum_batch'] // params['batch_size'])))
    step_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=params['learning_rate'], total_steps=int(total_steps))
    print(f"OneCycleLR scheduler used (Total Number of Steps: {total_steps})")
    print(f"Distance Normalized: {params['dnorm']}\nITFM: {params['itfm']}\n"
          f"Cutout: {params['cutout']}\nFreqshift: {params['freqshift']}\n"
          f"FiltAug: {params['filtaug']}\nCompFreq: {params['compfreq']}\nAllAug: {params['alltrans']}\n")

    # Setup the W&B run
    wandb.init(project=params['project'], name=params['exp'], config=params, reinit=True)
    wandb.watch(seld_model, log='all', log_freq=100)

    try:
        for epoch in range(start_epoch, params['nb_epochs']):
            # ------------- Training -------------- #
            train_start_time = time.time()
            avg_train_loss = train_epoch(seld_model=seld_model, dev_train_iterator=dev_train_iterator, optimizer=optimizer, seld_loss=seld_loss, step_scheduler=step_scheduler)
            train_time = time.time() - train_start_time

            # Log training loss to W&B
            wandb.log({"Loss/Train": avg_train_loss, "LearningRate": optimizer.param_groups[0]['lr']}, step=epoch)

            # -------------  Validation -------------- #
            # Perform validation every `val_freq` epochs
            is_regular_validation_time = (epoch % params['val_freq'] == 0)

            # Always validate every epoch after 80% of total epochs
            is_final_phase = (epoch / params['nb_epochs']) > 0.8

            # If `full`, we validate each epoch
            is_full_training_midway = params.get('full', False)

            should_validate = (is_regular_validation_time or is_final_phase or is_full_training_midway)
            if should_validate:
                val_start_time = time.time()
                avg_val_loss, metric_scores = val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir)
                val_f, val_ang_error, val_dist_error, val_rel_dist_error, val_onscreen_acc, class_wise_scr = metric_scores
                val_seld_error = ((1 - val_f) + (val_ang_error / 180) + val_rel_dist_error)/3 
                val_time = time.time() - val_start_time

                # Log validation loss and key metrics
                wandb.log({"Loss/Validation": avg_val_loss,
                           "Metric/F1_Score": val_f,
                           "Metric/Angular_Error": val_ang_error,
                           "Metric/Distance_Error": val_dist_error,
                           "Metric/Relative_Distance_Error": val_rel_dist_error,
                           "Metric/Val_3D_SELD_Error": val_seld_error}, step=epoch)

            # ------------- Save model if validation f score improves -------------#
            indicator = (val_seld_error <= best_seld_err) if params['finetune'] else (should_validate and val_f >= best_f_score)  
            if indicator:
                best_f_score = val_f
                best_epoch = epoch
                best_seld_err = val_seld_error
                net_save = {'seld_model': seld_model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch, 'params': params,
                            'best_f_score': best_f_score, 'best_ang_err': val_ang_error, 'best_rel_dist_err': val_rel_dist_error}
                torch.save(net_save, checkpoints_folder + "/best_model.pth")

            # ------------- Log losses and metrics ------------- #
            print(
                f"Epoch {epoch + 1}/{params['nb_epochs']} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {train_time:.0f}/{val_time:.0f} | "
                f"Loss: {avg_train_loss:.4f} | "
                f"F1: {val_f * 100:.3f} | "
                f"LE: {val_ang_error:.2f} | "
                f"DE: {val_dist_error:.2f} | "
                f"RDE: {val_rel_dist_error:.3f} | "
                f"Best: {best_epoch + 1} ({best_f_score * 100:.2f})"
            )

    except KeyboardInterrupt:
        print("Training ended prematurely. Calculating results now.")

    # Evaluate the best model on dev-test.
    print(f"Loading best model checkpoint from: {os.path.join(checkpoints_folder, 'best_model.pth')}\n\tBest Epoch: {best_epoch + 1}\tBest F-score: {best_f_score * 100:.2f}")
    best_model_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'), map_location=device, weights_only=False)
    seld_model.load_state_dict(best_model_ckpt['seld_model'])
    use_jackknife = params['use_jackknife']
    test_loss, test_metric_scores = val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, is_jackknife=use_jackknife)
    test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
    utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)

    # Close the W&B instance
    wandb.finish()


if __name__ == '__main__':

    # Record the start time
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    parser = argparse.ArgumentParser(description='DCASE 2025 Task 3 argument parser')

    # Experiment Arguments
    parser.add_argument('--project', type=str, default='Task3', help='W&B Project Name')
    parser.add_argument('--exp', type=str, default='Baseline', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='seldnet', help='Model to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size to use')
    parser.add_argument('--accum_batch', type=int, default=64, help='True batch size to use')
    parser.add_argument('--val_batch', type=int, default=32, help='Batch size for the validation dataloader')
    parser.add_argument('--nb_epochs', type=int, default=100, help='Total number of epochs')
    parser.add_argument('--full', action='store_true', help='If want to validate each epoch')

    # Learning Arguments
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--scheduler', type=str, default="one")
    parser.add_argument('--val_freq', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.05)

    # Dataset and Augmentations
    parser.add_argument('--cutout', action='store_true', help='Use the Random Cutout (Time-Frequency Masking) augmentation')
    parser.add_argument('--freqshift', action='store_true', help='Use the Frequency Shifting augmentation')
    parser.add_argument('--filtaug', action='store_true', help='Use the FilterAugment augmentation')
    parser.add_argument('--compfreq', action='store_true', help='Use both Frequency Shifting and FilterAugment augmentation')
    parser.add_argument('--alltrans', action='store_true', help='Use the all the above augmentation')
    parser.add_argument('--itfm', action='store_true', help='Use the Inter-channel Level-Aware TFM variation')

    # Features and Labels
    parser.add_argument('--nb_mels', type=int, default=64)
    parser.add_argument('--max_freq', type=int, default=2000)
    parser.add_argument('--ms', action='store_true', help='Mid-Side spectrogram')
    parser.add_argument('--gamma', action='store_true', help='Magnitude Squared Coherence')
    parser.add_argument('--ipd', action='store_true', help='Inter-channel Phase Differences')
    parser.add_argument('--iv', action='store_true', help='Mid-Side Intensity Vector')
    parser.add_argument('--slite', action='store_true', help='SALSA-Lite (Normalized Inter-channel Phase Differences)')
    parser.add_argument('--dnorm', action='store_false', help='Distance Normalization')
    parser.add_argument('--multiACCDOA', action='store_false')
    parser.set_defaults(dnorm=True, multiACCDOA=True) # Default always use DNorm and Multi-ACCDOA format

    args = parser.parse_args()

    import sys
    try:
        sys.exit(main(device, args))
    except (ValueError, IOError) as e:
        # Handle exceptions and exit with the error
        sys.exit(e)
    finally:
        # Record the end time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        elapsed_time_str = utils.format_elapsed_time(elapsed_time)
        print(f"Execution time: {elapsed_time_str}")