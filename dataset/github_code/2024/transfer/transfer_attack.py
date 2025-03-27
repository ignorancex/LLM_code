import os
import argparse

import torch

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from options import TrainingOptions, HiDDenConfiguration
import utils
from attack_func import test_tfattk_hidden

from targets.MBRS.MBRS import MBRS
from targets.stegastamp.stega_stamp import StegaStampModel
from targets.RivaGAN_inference.RivaGAN_inference import RivaGAN


def main():
    """
    Main function to perform the transfer attack.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(description='Transfer Attack')
    parser.add_argument('--num_models', type=int, default=10, help='Number of surrogate models')
    parser.add_argument('--target', type=str, default='hidden', help='Target model (hidden, mbrs, stega, RivaGAN)')
    parser.add_argument('--target_length', type=int, default=-1, help='Target message length; defaults based on target')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID to use')
    parser.add_argument('--no_optimization', action='store_true', help='Disable optimization')
    parser.add_argument('--normalized', action='store_true', help='Normalize the watermark')
    parser.add_argument('--PA', type=str, default='mean', help='Parameter aggregation method for non-optimization')
    parser.add_argument('--model_type', type=str, default='cnn', help='Model type: cnn or resnet')
    parser.add_argument('--resnet_same_encoder', action='store_true', help='Use same encoder blocks for resnet')
    parser.add_argument('--normalization', type=str, default='clamp', help='Normalization method')
    parser.add_argument('--model_index', type=int, default=1, help='Model index when num_models=1')
    parser.add_argument('--data_name', type=str, default='DB', help='Data name for the model')
    parser.add_argument('--r', type=float, default=0.25, help="L-infinity norm used in the attack")
    parser.add_argument('--diffpure', type=float, default=0, help="diffpure strength")
    parser.add_argument('--visualization', action='store_true',
                        help='if true, only attack first batch and do not load cache for Hu et al., for time measurement')
    parser.add_argument('--no_cache', action='store_true',
                        help='if true, do not load cache for Hu et al., for time measurement')
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    # General parameters
    data_dir = ''
    batch_size = 100
    epochs = 200
    num_models = args.num_models
    name = '30bits_AT_DALLE_200epochs_50maintrain'
    size = 128
    train_dataset = 'large_random_10k'
    val_dataset = 'large_random_1k'
    tensorboard = False
    enable_fp16 = False
    noise = None
    train_type = 'AT'  # Adversarial Training
    data_name = args.data_name
    wm_method = 'hidden'
    model_type = args.model_type
    white = False
    smooth = False

    # Surrogate message length
    message_length = 30

    # Determine target message length if not specified
    target_length = args.target_length
    if target_length == -1:
        target_lengths = {'hidden': 30, 'mbrs': 64, 'stega': 100, 'RivaGAN': 32}
        target_length = target_lengths.get(args.target, 30)

    fixed_message = False
    target = args.target
    PA = args.PA
    optimization = not args.no_optimization

    if args.normalized:
        assert not optimization, "Normalization requires optimization to be disabled"
        budget = 0.25
    else:
        budget = None

    start_epoch = 1
    train_options = TrainingOptions(
        batch_size=batch_size,
        number_of_epochs=epochs,
        train_folder=os.path.join(data_dir, 'train'),
        validation_folder=os.path.join(data_dir, 'val'),
        runs_folder=os.path.join('.', 'runs'),
        start_epoch=start_epoch,
        experiment_name=name
    )

    # Configure noise
    noise_config = noise if noise is not None else []
    noiser = Noiser(noise_config, device)

    # Surrogate model configuration
    if wm_method == 'hidden':
        sur_config = HiDDenConfiguration(
            H=size, W=size,
            message_length=message_length,
            encoder_blocks=4, encoder_channels=64,
            decoder_blocks=7, decoder_channels=64,
            use_discriminator=True,
            use_vgg=False,
            discriminator_blocks=3, discriminator_channels=64,
            decoder_loss=1.0,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
            enable_fp16=enable_fp16
        )

    # Target model configuration
    target_config = None
    if target in ['hidden', 'mbrs', 'stega', 'RivaGAN']:
        encoder_blocks = 4 if (model_type == 'cnn' or args.resnet_same_encoder) else 7
        target_config = HiDDenConfiguration(
            H=size, W=size,
            message_length=target_length,
            encoder_blocks=encoder_blocks,
            encoder_channels=64,
            decoder_blocks=7, decoder_channels=64,
            use_discriminator=True,
            use_vgg=False,
            discriminator_blocks=3, discriminator_channels=64,
            decoder_loss=1.0,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
            enable_fp16=enable_fp16
        )

    # Initialize the target model
    if target == 'hidden':
        model = Hidden(target_config, device, noiser, model_type)
    elif target == 'mbrs':
        if target_length == 64:
            model = MBRS(H=size, W=size, message_length=target_length, device=device)
        else:
            model = MBRS(H=256, W=256, message_length=target_length, device=device)
    elif target == 'stega':
        model_path = '/scratch/qilong3/transferattack/targets/checkpoints/stegaStamp/stegastamp_pretrained'
        model = StegaStampModel(model_path, device=device)
    elif target == 'RivaGAN':
        model = RivaGAN(device=device)
    else:
        raise ValueError(f"Unknown target model: {target}")

    # Load the target model checkpoint if necessary
    if target == 'hidden':
        if 'DB' in data_name:
            target_cp_file = (
                f'/scratch/qilong3/transferattack/target_model/'
                f'{target}_{train_type}/{target_length}bits_{model_type}_{train_type}.pth'
            )
        elif 'midjourney' in data_name:
            target_cp_file = (
                f'/scratch/qilong3/transferattack/target_model/'
                f'{target_length}bits_{model_type}_AT_midjourney.pth'
            )
        else:
            raise ValueError(f"Unknown data_name: {data_name}")

        if args.resnet_same_encoder:
            target_cp_file = target_cp_file.replace('.pth', '_resnet_same_encoder.pth')

        # Load the checkpoint
        target_checkpoint = torch.load(target_cp_file, map_location='cpu')
        utils.model_from_checkpoint(model, target_checkpoint)

    # Set up surrogate models
    sur_cp_folder = f'/scratch/qilong3/transferattack/surrogate_model/{wm_method}/{train_type}/'
    sur_cp_list = []

    # Determine model indices
    if num_models == 1:
        indices = [args.model_index]
    else:
        indices = list(range(1, num_models + 1))

    for idx in indices:
        checkpoint_path = os.path.join(sur_cp_folder, f'model_{idx}.pth')
        sur_cp_list.append(checkpoint_path)

    sur_model_list = []
    for idx, sur_cp_path in enumerate(sur_cp_list):
        if wm_method == 'hidden':
            surrogate_model = Hidden(sur_config, device, noiser, 'cnn')
            if white:
                # Apply whitening transformation if required
                linear_layer = torch.nn.Linear(30, 30, bias=True)
                surrogate_model.decoder = torch.nn.Sequential(
                    surrogate_model.decoder, linear_layer.to(device)
                )
                checkpoint = torch.load(sur_cp_path.replace(".pth", "_whit.pth"), map_location='cpu')
            else:
                checkpoint = torch.load(sur_cp_path, map_location='cpu')
            utils.model_from_checkpoint(surrogate_model, checkpoint)
            sur_model_list.append(surrogate_model)
        else:
            raise NotImplementedError(f"Surrogate model for wm_method '{wm_method}' is not implemented")

    # Run the transfer attack
    test_tfattk_hidden(
        model=model,
        model_list=sur_model_list,
        device=device,
        hidden_config=target_config,
        train_options=train_options,
        val_dataset=val_dataset,
        train_type=train_type,
        model_type=model_type,
        data_name=data_name,
        wm_method=wm_method,
        target=target,
        smooth=smooth,
        target_length=target_length,
        num_models=num_models,
        fixed_message=fixed_message,
        optimization=optimization,
        PA=PA,
        budget=budget,
        resnet_same_encoder=args.resnet_same_encoder,
        normalization=args.normalization,
        r=args.r,
        model_index=args.model_index,
        diffpure=args.diffpure,
        visualization=args.visualization,
        no_cache=args.no_cache
    )


if __name__ == '__main__':
    utils.setup_seed(42)
    main()
