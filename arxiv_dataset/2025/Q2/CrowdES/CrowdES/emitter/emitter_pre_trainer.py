import os
import math
import json
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from CrowdES.emitter.emitter_pre_config import CrowdESEmitterPreConfig
from CrowdES.emitter.emitter_pre_model import CrowdESEmitterPreModel
from utils.dataloader.emitter_pre_dataloader import EmitterPreDataset
from utils.utils import reproducibility_settings


def main(config):
    # Reproducibility
    reproducibility_settings(seed=config.crowd_emitter.emitter_pre.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset_train = EmitterPreDataset(config, 'train')
    dataset_val = EmitterPreDataset(config, 'train')  # Do not use 'test' split during training
    loader_train = DataLoader(dataset_train, batch_size=config.crowd_emitter.emitter_pre.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=config.crowd_emitter.emitter_pre.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load pretrained model
    input_dim, output_dim = dataset_train.input_dim, dataset_train.output_dim
    num_unit_labels = config.crowd_emitter.emitter_pre.max_population
    model_config = CrowdESEmitterPreConfig.from_pretrained(config.crowd_emitter.emitter_pre.model_pretrained, num_channels=input_dim, num_labels=output_dim, num_unit_labels=num_unit_labels)
    model = CrowdESEmitterPreModel.from_pretrained(config.crowd_emitter.emitter_pre.model_pretrained, config=model_config, ignore_mismatched_sizes=True)
    model.to(device)
        
    # Optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=config.crowd_emitter.emitter_pre.learning_rate)
    gradient_accumulation_steps = config.crowd_emitter.emitter_pre.gradient_accumulation_steps

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(loader_train) / config.crowd_emitter.emitter_pre.gradient_accumulation_steps)
    num_train_epochs = config.crowd_emitter.emitter_pre.num_train_epochs
    max_train_steps = config.crowd_emitter.emitter_pre.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = config.crowd_emitter.emitter_pre.num_warmup_steps
    lr_scheduler_type = config.crowd_emitter.emitter_pre.lr_scheduler_type
    lr_scheduler = get_scheduler(name=lr_scheduler_type, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)

    # Checkpoints
    checkpoint_dir = config.crowd_emitter.emitter_pre.checkpoint_dir.format(config.dataset.dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    min_val_loss = float('inf')
    training_log = []

    progress_bar = tqdm(range(max_train_steps))

    for epoch in range(num_train_epochs):
        model.train()
        
        total_loss = 0
        progress_bar.set_description(f'Train Epoch {epoch}')
        for step, batch in enumerate(loader_train):
            input_batch = batch['input_data'].to(device)
            target_batch = batch['output_data'].to(device)
            target_unit_batch = batch['output_unit_data'].to(device)

            outputs = model(input_batch, target_batch, target_unit_batch)
            loss = outputs.loss
                
            total_loss += loss.detach().float().item()

            loss[torch.isnan(loss) | torch.isinf(loss)] = 0
            loss.backward()
            
            if step % gradient_accumulation_steps == 0 or step == len(loader_train) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update()
        
        model.eval()
        val_loss = []
        progress_bar.set_description(f'Valid Epoch {epoch}')
        with torch.no_grad():
            for step, batch in enumerate(loader_val):
                input_batch, target_batch = batch['input_data'].to(device), batch['output_data'].to(device)
                outputs = model(input_batch)
                
                pred_batch = torch.nn.functional.interpolate(outputs.logits, size=target_batch.shape[-2:], mode='bilinear', align_corners=False)
                pred_batch = pred_batch.sigmoid()
                pred_logit_batch = outputs.logits_unit.sigmoid().detach().cpu().numpy()
                population_probability_threshold = config.crowd_emitter.emitter_pre.population_probability_threshold
                pred_logit_batch[pred_logit_batch < population_probability_threshold] = 0

                error = (pred_batch - target_batch).abs().mean(axis=(1, 2, 3)).detach().cpu().tolist()  # MAE
                val_loss.extend(error)

        val_loss = np.mean(val_loss)
        eval_metrics = {'epoch': epoch, 'train_loss': total_loss, 'val_loss': val_loss}
        training_log.append(eval_metrics)
        progress_bar.set_postfix(eval_metrics)

        # Save best model
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            model.save_pretrained(checkpoint_dir, safe_serialization=False)
            
            # Save pred image sample
            os.makedirs(os.path.join(checkpoint_dir, 'pred_sample'), exist_ok=True)
            appearance_density_map = pred_batch[0, 0].detach().cpu().numpy()
            population_density_map = pred_batch[0, 1].detach().cpu().numpy()
            appearance_density_map = (appearance_density_map * 255).astype(np.uint8)
            population_density_map = (population_density_map * 255).astype(np.uint8)
            appearance_density_map = Image.fromarray(appearance_density_map)
            population_density_map = Image.fromarray(population_density_map)
            appearance_density_map.save(os.path.join(checkpoint_dir, 'pred_sample', f'appearance_density_map_{epoch}.png'))
            population_density_map.save(os.path.join(checkpoint_dir, 'pred_sample', f'population_density_map_{epoch}.png'))
            np.savetxt(os.path.join(checkpoint_dir, 'pred_sample', f'pred_logit_batch_{epoch}.txt'), [pred_logit_batch[0]], fmt='%.6f')

    with open(os.path.join(checkpoint_dir, 'all_results.json'), 'w') as f:
        json.dump(training_log, f, indent=2)


if __name__ == '__main__':
    import argparse
    from utils.config import get_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=int, default='./configs/model/CrowdES_gcs.yaml', help='Path to a model config file')
    args = parser.parse_args()
    config = get_config(args.model_config, args.dataset_config, args.trainer_config)

    main(config)
