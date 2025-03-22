if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from PIL import Image as PILImage
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from VILP.policy.vilp_planning import VilpPlanning
OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainVilpPlanningWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: VilpPlanning = hydra.utils.instantiate(cfg.policy)

        self.ema_model: VilpPlanning = None
        if cfg.training.use_ema:
            print("Using EMA model")
            self.ema_model = copy.deepcopy(self.model)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.cfg = cfg

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                #print('saving model')
                #torch.save(self.model.model_high_level.state_dict(), '.pth')

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )
        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        if cfg.use_sim:
            # configure env
            env_runner: BaseImageRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                high_level_train_losses = list()

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        info = self.model.train_on_batch(batch)
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        # logging 
                        high_level_loss_cpu = info['high_level_loss']
                        tepoch.set_postfix(loss=high_level_loss_cpu, refresh=False)
                        high_level_train_losses.append(high_level_loss_cpu)
                        step_log = {
                            'high_level_train_loss': high_level_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average


                high_level_train_loss = np.mean(high_level_train_losses)
                step_log['high_level_train_loss'] = high_level_train_loss

                # ========= eval for this epoch ==========

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                if cfg.generated_views == 1:
                    if cfg.use_sim:
                        # run rollout
                        if (self.epoch % cfg.training.rollout_every) == 0:
                            runner_log = env_runner.run(policy)
                            # log all
                            step_log.update(runner_log)
                    else:
                        if (self.epoch % cfg.training.rollout_every) == 0:
                            start_image_buffer = []
                            episode_lengths = dataset.get_episode_lengths()
                            for i in range(episode_lengths):
                                is_image,image = dataset.get_val_episode_start(i,cfg.input_key)
                                if is_image:
                                    start_image_buffer.append(image)
                            # transfer all images to a batch and to tensor
                            start_image_batch = {
                                cfg.output_key: 
                                    torch.stack([torch.from_numpy(img) for img in start_image_buffer]).to(device).unsqueeze(1)
                                }
                            step_idx = 0
                            max_steps = cfg.max_generation_steps
                            result_image_buffer = [] 
                            pbar = tqdm.tqdm(total=max_steps, desc="Eval VideoPlanning", 
                                leave=False, mininterval=1) 
                            while step_idx < max_steps:
                                if step_idx == 0:
                                    with torch.no_grad():
                                        pred_img_seq = policy.predict_image(start_image_batch)
                                else:
                                    with torch.no_grad():
                                        pred_obs = {}
                                        pred_img_last = torch.from_numpy(np_pred_image_last).to(device=device)
                                        pred_obs[cfg.output_key] = pred_img_last.unsqueeze(1)
                                        pred_img_seq = policy.predict_image(pred_obs)

                                np_pred_image_seq = pred_img_seq.detach().to('cpu').numpy()
                                for i in range(np_pred_image_seq.shape[1]):
                                    result_image_buffer.append(np_pred_image_seq[:, i, :, :])
                                np_pred_image_last = np_pred_image_seq[:, -1, :, :]
                                step_idx += 1
                                pbar.update(1)
                            pbar.close()
                            batched_data = [[] for _ in range(result_image_buffer[0].shape[0])]  
                            for data in result_image_buffer:
                                for batch_index in range(data.shape[0]):
                                    batched_data[batch_index].append(data[batch_index])
                            videos = [np.stack(batch_images) for batch_images in batched_data]
                            for i, video in enumerate(videos):
                                key = f"video_{i}"  
                                video = (video*255).astype(np.uint8) 
                                step_log[key] = wandb.Video(video)  
                elif cfg.generated_views == 2:
                # When generated views >=1, we can't really evaluate the model during training
                # because we only train one view at a time.
                # This evaluation is only for monitoring if the model converges reasonably during training.
                    if (self.epoch % cfg.training.rollout_every) == 0:
                        start_image_buffer = []
                        episode_lengths = dataset.get_episode_lengths()
                        for i in range(episode_lengths):
                            is_image,image = dataset.get_val_episode_start(i,cfg.output_key)
                            if is_image:
                                start_image_buffer.append(image)
                        # transfer all images to a batch and to tensor
                        start_image_batch = {
                            cfg.output_key: 
                                torch.stack([torch.from_numpy(img) for img in start_image_buffer]).to(device).unsqueeze(1)
                            }
                        image_buffer_dict = {}
                        for key in cfg.input_keys:
                            episode_image_buffer = []
                            for i in range(episode_lengths):
                                is_image,image_seq = dataset.get_val_episode_full(i,key)
                                if is_image:
                                    episode_image_buffer.append(image_seq)
                            image_buffer_dict[key] = episode_image_buffer
                            start_image_batch[key] = torch.stack([torch.from_numpy(img_seq[0]) for img_seq in episode_image_buffer]).to(device).unsqueeze(1)
                        step_idx = 0
                        max_steps = cfg.max_generation_steps
                        result_image_buffer = [] 
                        pbar = tqdm.tqdm(total=max_steps, desc="Eval VideoPlanning", 
                            leave=False, mininterval=1) 
                        while step_idx < max_steps:
                            if step_idx == 0:
                                with torch.no_grad():
                                    pred_img_seq = policy.predict_image(start_image_batch)
                            else:
                                with torch.no_grad():
                                    pred_obs = {}
                                    pred_img_last = torch.from_numpy(np_pred_image_last).to(device=device)
                                    pred_obs[cfg.output_key] = pred_img_last.unsqueeze(1)
    
                                    for key in cfg.input_keys:
                                        pred_obs[key] = torch.stack([
                                            torch.from_numpy(img_seq[2*step_idx] if (2*step_idx) < len(img_seq) else img_seq[-1]) 
                                            for img_seq in image_buffer_dict[key]
                                        ]).to(device).unsqueeze(1)

                                    pred_img_seq = policy.predict_image(pred_obs)

                            np_pred_image_seq = pred_img_seq.detach().to('cpu').numpy()
                            for i in range(np_pred_image_seq.shape[1]):
                                result_image_buffer.append(np_pred_image_seq[:, i, :, :])
                            np_pred_image_last = np_pred_image_seq[:, -1, :, :]
                            step_idx += 1
                            pbar.update(1)
                        pbar.close()
                        batched_data = [[] for _ in range(result_image_buffer[0].shape[0])]  
                        for data in result_image_buffer:
                            for batch_index in range(data.shape[0]):
                                batched_data[batch_index].append(data[batch_index])
                        videos = [np.stack(batch_images) for batch_images in batched_data]
                        for i, video in enumerate(videos):
                            key = f"video_{i}"  
                            video = (video*255).astype(np.uint8) 
                            step_log[key] = wandb.Video(video)  

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():

                        high_level_val_losses = list()
                        image_input = None
                        image_rec = None
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                info = self.model.eval_on_batch(batch)

                                high_level_loss = info['high_level_loss']


                                high_level_val_losses.append(high_level_loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(high_level_val_losses) > 0:
                            # log epoch average validation loss

                            high_level_val_loss = np.mean(high_level_val_losses)
                            step_log['high_level_val_loss'] = high_level_val_loss

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainVilpPlanningWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
