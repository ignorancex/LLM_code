import os
import torch
import numpy as np
import configparser
from TimeTransformer import Transformer2
from TimeTransformer.utils import RECDataset
from denoising_diffusion_pytorch import GaussianDiffusion1D, Trainer1D
from utils.data_preparation import process_models, extract_parameters, restore_parameters
from utils.config_utils import load_config, process_paths
from utils.logger_utils import setup_logging, log_config
from utils.merge_adapter_in import merge_pt_files

def main():
    config = load_config()
    config = process_paths(config)
    
    # set the logger
    logger = setup_logging(log_dir='Log_train_diffusion', log_filename=f"{config['DEFAULT']['model_name']}.log")
    log_config(logger, config)

    # Set GPU
    gpu = config.getint('DEFAULT', 'gpu')
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data preparation
    base_folder = config['PATHS']['base_folder']
    all_params, all_labels, scale = process_models(base_folder=base_folder)
    all_params = np.repeat(all_params, 30, axis=0)
    all_labels = np.repeat(all_labels, 30, axis=0)
    XEData = RECDataset(all_params, all_labels, all_labels)
    logger.info(f"Data prepared. Params shape: {all_params.shape}, Labels shape: {all_labels.shape}")

    # Model initialization
    d_model = config.getint('DEFAULT', 'modeldim')
    N = 4
    d_condition_emb_1 = all_labels.shape[1]
    d_condition_emb_2 = all_labels.shape[1]
    dropout = 0.1
    pe = 'original'
    d_input = all_params.shape[1]
    d_output = d_input
    layernum = all_params.shape[2]

    denoisingModel = Transformer2(d_input, d_model, d_output, d_condition_emb_1, d_condition_emb_2, N,
                                  layernum=layernum, dropout=dropout, pe=pe).to(device)
    logger.info("Denoising model initialized")

    diffusion = GaussianDiffusion1D(
        denoisingModel,
        seq_length=all_params.shape[2],
        timesteps=config.getint('DEFAULT', 'diffusionstep'),
        loss_type='l2',
        objective='pred_v',
        auto_normalize=False,
        beta_schedule='linear',
    ).to(device)
    logger.info("Diffusion model initialized")

    trainer = Trainer1D(
        diffusion,
        dataset=XEData,
        train_batch_size=config.getint('TRAINING', 'batch_size'),
        train_lr=config.getfloat('TRAINING', 'learning_rate'),
        train_num_steps=config.getint('DEFAULT', 'epochs'),
        gradient_accumulate_every=1,
        save_and_sample_every=10,
        ema_decay=0.995,
        amp=False,
        condition_emb_1=all_labels,
        condition_emb_2=all_labels,
        genTarget=all_params,
        targetDataset=config['DEFAULT']['targetDataset'],
        scale=1,
        sampleTimes=config.getint('TRAINING', 'sample_times'),
    )
    logger.info("Trainer initialized")

    # Train the model
    # logger.info("Starting training")
    # trainer.train()
    # logger.info("Training completed")

    # Generate and save parameters
    diffusion_model_path = config['PATHS']['diffusion_model_path'].format(**config['DEFAULT'])
    trainer.save_model(diffusion_model_path)
    logger.info(f"Diffusion Model saved to {diffusion_model_path}")

    original_model_path = config['PATHS']['original_model_path'].format(**config['DEFAULT'])
    param_info, _ = extract_parameters(original_model_path)

    reconstruct_dir = config['PATHS']['reconstruct_dir'].format(**config['DEFAULT'])
    os.makedirs(reconstruct_dir, exist_ok=True)

    params_list = [0]
    for i in range(0, 11):
        tag = np.array([[i * 0.1, 1 - i * 0.1]]).astype(np.float32)
        params = trainer.load_and_use_model(path=diffusion_model_path, condition_emb_1=tag, condition_emb_2=tag, num_samples=1)
        params_list.append(params)
        logger.info(f"Generated params for tag {tag[0]}")

    for i in range(0, 11):
        adapter_path = os.path.join(reconstruct_dir, f"{config['DEFAULT']['model_name']}_pretrain_and_finetune_{i}_acc_adapter.pt")
        output_path = os.path.join(reconstruct_dir, f"{config['DEFAULT']['model_name']}_pretrain_and_finetune_{i}_acc.pt")
        restore_parameters(params_list[int(i) + 1], param_info, scale, adapter_path)
        logger.info(f"Restored parameters saved to {adapter_path}")
        merge_pt_files(original_model_path , adapter_path , output_path)

    logger.info("ALL PARAMS HAVE BEEN SAVED!")

if __name__ == "__main__":
    main()