import os
import math
import json
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers import get_scheduler as get_diffuser_scheduler

from CrowdES.emitter.emitter_model import CrowdESEmitterModel
from CrowdES.emitter.emitter_pipeline import CrowdESEmitterPipeline
from utils.dataloader.emitter_dataloader import EmitterDataset
from utils.utils import reproducibility_settings


def main(config):
    # Reproducibility
    reproducibility_settings(seed=config.crowd_emitter.emitter.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset_train = EmitterDataset(config, 'train')
    dataset_val = EmitterDataset(config, 'train')  # Do not use 'test' split during training
    loader_train = DataLoader(dataset_train, batch_size=config.crowd_emitter.emitter.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=config.crowd_emitter.emitter.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load pretrained model
    input_dim, output_dim = dataset_train.input_dim, dataset_train.output_dim
    image_size = dataset_train.image_size
    max_num_agents_pad = dataset_train.max_num_agents_pad
    vae_latent_dim = config.crowd_emitter.emitter.vae_latent_dim
    vae_hidden_dim = config.crowd_emitter.emitter.vae_hidden_dim
    latent_len_multiplier = config.crowd_emitter.emitter.latent_len_multiplier

    model = CrowdESEmitterModel(max_num_agents=max_num_agents_pad, num_classes=output_dim,
                                condition_size=image_size, condition_channels=input_dim, time_embedding_type='fourier',
                                vae_latent_dim=vae_latent_dim, vae_hidden_dim=vae_hidden_dim,
                                latent_len_multiplier=latent_len_multiplier)
    model.to(device)

    # Step 1: Initialize normalizer
    all_data = []
    for step, batch in enumerate(tqdm(loader_train, desc='Prepare mean and std')):
        target_pad_batch = batch['output_data']
        target_mask_batch = batch['output_mask']
        target_batch = target_pad_batch[target_mask_batch, :]  # use non-paded data
        target_batch = target_batch.detach().tolist()
        all_data.extend(target_batch)
    all_data = np.array(all_data)
    mean, std = np.mean(all_data, axis=0), np.std(all_data, axis=0)
    model.set_norm_mean_std(torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device))
    
    # Get noise scheduler
    diffusion_steps = config.crowd_emitter.emitter.diffusion_steps
    noise_scheduler = DDIMScheduler(num_train_timesteps=diffusion_steps, beta_schedule='linear')
    unguidance_prob = config.crowd_emitter.emitter.unguidance_prob
    do_classifier_free_guidance = unguidance_prob > 0
    leapfrog_steps = config.crowd_emitter.emitter.leapfrog_steps

    # Optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=config.crowd_emitter.emitter.learning_rate)
    gradient_accumulation_steps = config.crowd_emitter.emitter.gradient_accumulation_steps

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(loader_train) / config.crowd_emitter.emitter.gradient_accumulation_steps)
    num_train_epochs = config.crowd_emitter.emitter.num_train_epochs
    max_train_steps = config.crowd_emitter.emitter.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = config.crowd_emitter.emitter.num_warmup_steps
    lr_scheduler_type = config.crowd_emitter.emitter.lr_scheduler_type
    lr_scheduler = get_scheduler(name=lr_scheduler_type, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)

    # Checkpoints
    checkpoint_dir = config.crowd_emitter.emitter.checkpoint_dir.format(config.dataset.dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Step 2: Train the model
    min_val_loss = float('inf')
    training_log = []

    pred_crowd_color = np.random.rand(1000, 3) * 255

    progress_bar = tqdm(range(max_train_steps))

    for epoch in range(num_train_epochs):
        model.train()
        
        # start = time.time()
        total_loss = 0
        progress_bar.set_description(f'Train Epoch {epoch}')
        for step, batch in enumerate(loader_train):
            # print(f'Time taken for one step: {time.time() - start}')
            condition_batch = batch['input_data'].to(device)
            target_pad_batch = batch['output_data'].to(device)
            target_mask_batch = batch['output_mask'].to(device)
            batch_size = target_pad_batch.shape[0]
            num_crowd_batch = target_mask_batch.sum(dim=1)

            # noise scheduler
            timesteps = torch.randint(0, diffusion_steps, (batch_size,), device=device).long().to(device)

            ## The following is for latent diffusion model
            if True:
                target_latent_pad_batch = model.VAEEncoder(target_pad_batch).detach()
                noise = torch.randn_like(target_latent_pad_batch)
                noise.masked_fill_(~target_mask_batch.unsqueeze(-1), 0)
                noisy_target_latent_pad_batch = noise_scheduler.add_noise(target_latent_pad_batch, noise, timesteps)
                noisy_target_latent_pad_batch.masked_fill_(~target_mask_batch.unsqueeze(-1), 0)
                noisy_target_pad_batch = noisy_target_latent_pad_batch

                # Decide whether to drop the condition
                if do_classifier_free_guidance:
                    if torch.rand(1).item() < unguidance_prob:
                        condition_batch = torch.zeros_like(condition_batch)

                # Model forward
                noise_pred = model(noisy_target_pad_batch, mask=target_mask_batch, condition=condition_batch, timestep=timesteps, return_dict=False)[0]
                noise_pred.masked_fill_(~target_mask_batch.unsqueeze(-1), 0)
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
                # loss = torch.nn.functional.mse_loss(noise_pred, target_latent_pad_batch, reduction='none')  # x0 mode
                loss = loss[target_mask_batch, :].mean()  # use non-padded data

                total_loss += loss.detach().float().item()

                loss[torch.isnan(loss) | torch.isinf(loss)] = 0
                loss.backward()
                progress_bar.set_postfix({'loss': loss.item()})

            if step % gradient_accumulation_steps == 0 or step == len(loader_train) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update()
        
        model.eval()
        val_loss = []
        progress_bar.set_description(f'Valid Epoch {epoch}')
        with torch.no_grad():
            # Skip every k epochs
            if epoch % 5 != 0:
                continue

            pipeline = CrowdESEmitterPipeline(unet=model, scheduler=noise_scheduler)

            for step, batch in enumerate(loader_val):
                condition_batch = batch['input_data'].to(device)
                target_pad_batch = batch['output_data'].to(device)
                target_mask_batch = batch['output_mask'].to(device)
                batch_size = config.crowd_emitter.emitter.eval_batch_size

                # Maybe val loss to averaged preferred speed
                for i in range(batch_size):
                    sample_condition = condition_batch[[i]]
                    sample_target = target_pad_batch[[i]]
                    sample_mask = target_mask_batch[[i]]
                    num_crowd = i + 1
                    pred_crowd = pipeline(batch_size=1, condition=sample_condition, num_crowd=num_crowd, device=device, seed=i, num_inference_steps=diffusion_steps, leapfrog_steps=leapfrog_steps)

                    pred_crowd = pred_crowd['crowd_emission'].detach().cpu().numpy()[0]
                    appearance_density_map = condition_batch[i, -2].detach().cpu().numpy()
                    generated_sample = (appearance_density_map * 255).astype(np.uint8)
                    generated_sample = np.stack([generated_sample] * 3, axis=-1)
                    pred_crowd_origin = (pred_crowd[:, [3, 4]] * np.array(image_size)[None, [1, 0]]).astype(np.int32)
                    pred_crowd_goal = (pred_crowd[:, [5, 6]] * np.array(image_size)[None, [1, 0]]).astype(np.int32)
                    for j in range(num_crowd):
                        cv2.circle(generated_sample, tuple(pred_crowd_origin[j]), 3, pred_crowd_color[j], -1)
                        cv2.circle(generated_sample, tuple(pred_crowd_goal[j]), 2, pred_crowd_color[j], 1)

                    # Draw num_crowd digits on the top left corner
                    cv2.putText(generated_sample, str(num_crowd), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                    # Save pred_crwod to text file
                    os.makedirs(f'./output/emitter_trainer/{config.dataset.dataset_name}', exist_ok=True)
                    np.savetxt(f'./output/emitter_trainer/{config.dataset.dataset_name}/generated_crowd_{i}.txt', pred_crowd, fmt='%.6f')

                    # Save image
                    generated_sample = Image.fromarray(generated_sample)
                    generated_sample.save(f'./output/emitter_trainer/{config.dataset.dataset_name}/generated_sample_{i}.png')

                break

        val_loss = 0
        eval_metrics = {'epoch': epoch, 'train_loss': total_loss, 'val_loss': val_loss}
        training_log.append(eval_metrics)
        progress_bar.set_postfix(eval_metrics)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
        
        # Save the model
        model.save_pretrained(checkpoint_dir, safe_serialization=False)
            
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
