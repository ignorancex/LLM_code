import os
import math
import json
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from CrowdES.simulator.simulator_config import CrowdESSimulatorConfig
from CrowdES.simulator.simulator_model import CrowdESSimulatorModel
from utils.dataloader.simulator_dataloader import SimulatorDataset
from utils.utils import reproducibility_settings


def main(config):
    # Reproducibility
    reproducibility_settings(seed=config.crowd_simulator.simulator.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset_train = SimulatorDataset(config, 'train')
    dataset_val = SimulatorDataset(config, 'train')  # Do not use 'test' split during training
    loader_train = DataLoader(dataset_train, batch_size=config.crowd_simulator.simulator.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)  # 4
    loader_val = DataLoader(dataset_val, batch_size=config.crowd_simulator.simulator.eval_batch_size, shuffle=False, num_workers=8, pin_memory=True)  # 4

    # Load pretrained model
    env_dim = dataset_train.env_dim
    env_size = config.crowd_simulator.simulator.environment_size
    history_length = config.crowd_simulator.simulator.history_length
    future_length = config.crowd_simulator.simulator.future_length
    interaction_max_num_agents = config.crowd_simulator.simulator.interaction_max_num_agents
    latent_dim = config.crowd_simulator.simulator.latent_dim
    hidden_dim = config.crowd_simulator.simulator.hidden_dim


    model_config = CrowdESSimulatorConfig(history_length=history_length, future_length=future_length,
                                         env_size=env_size, env_dim=env_dim,
                                         neighbor_max_num=interaction_max_num_agents,
                                         latent_dim=latent_dim, hidden_dim=hidden_dim)
    model = CrowdESSimulatorModel(config=model_config)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=config.crowd_simulator.simulator.learning_rate)
    # optimizer = torch.optim.SGD(list(model.parameters()), lr=config.crowd_simulator.simulator.learning_rate)
    gradient_accumulation_steps = config.crowd_simulator.simulator.gradient_accumulation_steps

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(loader_train) / config.crowd_simulator.simulator.gradient_accumulation_steps)
    num_train_epochs = config.crowd_simulator.simulator.num_train_epochs
    max_train_steps = config.crowd_simulator.simulator.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = config.crowd_simulator.simulator.num_warmup_steps
    lr_scheduler_type = config.crowd_simulator.simulator.lr_scheduler_type
    lr_scheduler = get_scheduler(name=lr_scheduler_type, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)

    # Checkpoints
    checkpoint_dir = config.crowd_simulator.simulator.checkpoint_dir.format(config.dataset.dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    

    # Step 1: Condition clustering
    endpoints, controls = [], []
    for scene in dataset_train.scene_list:
        endpoints.append(dataset_train.traj_pred_norm_scene[scene][:, -1, :])
        controls.append(dataset_train.control_point_scene[scene])
    endpoint = torch.cat(endpoints, dim=0)
    control = torch.cat(controls, dim=0)
    endpoint = torch.cat([endpoint, endpoint * torch.tensor([-1, 1])], dim=0)  # Data augmentation
    control = torch.cat([control, control * torch.tensor([-1, 1])], dim=0)  # Data augmentation
    model.endpoint_cluster_center_generation(endpoint, control)


    # Step 2: Train the model
    min_val_loss = float('inf')
    training_log = []

    progress_bar = tqdm(range(max_train_steps))

    for epoch in range(num_train_epochs):
        model.train()
        
        total_loss = 0
        progress_bar.set_description(f'Train Epoch {epoch}')
        for step, batch in enumerate(loader_train):
            traj_hist = batch['traj_hist'].to(device)   # shape=(B, T_hist, 2)
            traj_fut = batch['traj_fut'].to(device)     # shape=(B, T_fut, 2)
            goal = batch['goal'].to(device)             # shape=(B, 2)
            attr = batch['attr'].to(device)             # shape=(B, 2)
            control = batch['control'].to(device)       # shape=(B, 2)
            neighbor = batch['neighbor'].to(device)     # shape=(B, N, T_hist, 2)
            env_data = batch['environment'].to(device)  # shape=(B, C, H, W (C=8))

            # Randomly drop traj_hist for robust simulation
            # 1. Randomly mask out traj_hist
            mask = torch.rand(traj_hist.shape[0]) < 0.1  # 10%
            traj_hist[mask] = 0

            # 2. Randomly mask out first 0~T_hist frames
            mask = torch.rand(traj_hist.shape[0]) < 0.1  # 10%
            mask_pos = torch.randint(0, traj_hist.shape[1], (traj_hist.shape[0],))
            
            for i in range(traj_hist.shape[0]):
                if mask[i]:
                    traj_hist[i, :mask_pos[i]] = 0

            outputs = model(traj_hist, traj_fut, goal, attr, control, neighbor, env_data)
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
        val_loss_ade = []
        val_loss_fde = []
        progress_bar.set_description(f'Valid Epoch {epoch}')
        with torch.no_grad():
            for step, batch in enumerate(loader_val):
                traj_hist = batch['traj_hist'].to(device)   # shape=(B, T_hist, 2)
                traj_fut = batch['traj_fut'].to(device)     # shape=(B, T_fut, 2)
                goal = batch['goal'].to(device)             # shape=(B, 2)
                attr = batch['attr'].to(device)             # shape=(B, 2)
                control = batch['control'].to(device)       # shape=(B, 2)
                neighbor = batch['neighbor'].to(device)     # shape=(B, N, T_hist, 2)
                env_data = batch['environment'].to(device)  # shape=(B, C, H, W (C=8))

                outputs = model(traj_hist, None, goal, attr, control, neighbor, env_data, sampling=False)
                traj_fut_pred = outputs.preds

                # Calculate ADE & FDE
                temp = torch.linalg.norm(traj_fut_pred - traj_fut, dim=-1)
                ade = temp.mean(dim=-1).detach().cpu().numpy()
                fde = temp[:, -1].detach().cpu().numpy()
                val_loss_ade.append(ade)
                val_loss_fde.append(fde)

            val_loss_ade = np.concatenate(val_loss_ade, axis=0).mean()
            val_loss_fde = np.concatenate(val_loss_fde, axis=0).mean()

        eval_metrics = {'epoch': epoch, 'train_loss': total_loss, 'val_ade': float(val_loss_ade), 'val_fde': float(val_loss_fde), 'lr': float(optimizer.param_groups[0]['lr']), 'min_val_loss': float(min_val_loss)}
        # val_loss = val_loss_ade  # use ADE for validation
        val_loss = val_loss_fde  # use FDE for validation
        # val_loss = val_loss_ade + val_loss_fde  # use ADE + FDE for validation
        training_log.append(eval_metrics)
        progress_bar.set_postfix(eval_metrics)
        print('')

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
        
        # save the model
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
