# Copyright (c) 2025 Project Jamify
#               2025 Declare-Lab
#               2025 AMAAI Lab
#               2025 Renhang Liu (liurenhang0@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from safetensors.torch import save_file, load_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

from ema_pytorch import EMA

from jam.dataset import DiffusionWebDataset, enhance_webdataset_config

from jam.model.utils import exists, default

class BaseTrainer:
    """ class BaseTrainer is adapted from github repo:
        https://github.com/ASLP-lab/DiffRhythm.
    """
    def __init__(
        self,
        model,
        data_cfg,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        wandb_mode: str = "online",
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        reset_lr: bool = False,
        use_style_prompt: bool = False,
        grad_ckpt: bool = False,
        use_ema: bool = False,
        config = None,
    ):
        self.data_cfg = data_cfg

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, )

        logger = "wandb" if wandb.api.api_key else None
        
        if grad_accumulation_steps > 1:
            print('set gradient accumulation steps to', grad_accumulation_steps)
        self.accelerator = Accelerator(
            log_with=logger,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        if logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id, "mode": wandb_mode}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "mode": wandb_mode}}
            
            if config is not None:
                if OmegaConf.is_config(config):
                    config_dict = OmegaConf.to_container(config, resolve=True)
                else:
                    config_dict = config
            
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=config_dict,
            )
            if self.is_main:
                wb_tracker = self.accelerator.get_tracker("wandb")
                wb_run = wb_tracker.run
                wb_run.log_code(root=config.project_root)

        self.precision = self.accelerator.state.mixed_precision
        self.precision = self.precision.replace("no", "fp32")

        self.model = model

        self.use_ema = use_ema
        if self.is_main and self.use_ema:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)

            self.ema_model.to(self.accelerator.device)
            if self.accelerator.state.distributed_type in ["DEEPSPEED", "FSDP"]:
                self.ema_model.half()

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")

        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        self.reset_lr = reset_lr

        self.use_style_prompt = use_style_prompt
        
        self.grad_ckpt = grad_ckpt

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)

        if self.accelerator.state.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size
        
        self.get_dataloader()
        self.get_scheduler()

        self.model, self.optimizer, self.scheduler, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.train_dataloader)


class WebDatasetTrainer(BaseTrainer):
    def __init__(self, max_steps, log_every, wandb_cfg, resume_from_checkpoint=None, resume_from_safetensors=None, use_fsdp=False, use_constant_lr=False, *args, **kwargs):
        self.max_steps = max_steps
        self.log_every = log_every
        self.resume_from_checkpoint = resume_from_checkpoint
        self.resume_from_safetensors = resume_from_safetensors
        self.use_fsdp = use_fsdp
        self.use_constant_lr = use_constant_lr
        
        if self.resume_from_safetensors:
            model = kwargs["model"]
            model.load_state_dict(load_file(self.resume_from_safetensors), strict=False)

        kwargs |= {
            "wandb_project": wandb_cfg.project,
            "wandb_run_name": wandb_cfg.name,
            "wandb_resume_id": wandb_cfg.get("resume_id", None),
            "wandb_mode": wandb_cfg.get("mode", "online")
        }
        
        kwargs['epochs'] = 1  # Set dummy epochs for parent class
        super().__init__(*args, **kwargs)

    def get_dataloader(self):
        enhance_webdataset_config(self.data_cfg.train_dataset)
        train_dataset = DiffusionWebDataset(**self.data_cfg.train_dataset)
        self.train_dataloader = DataLoader(train_dataset, **self.data_cfg.train_dataloader, collate_fn=train_dataset.custom_collate_fn)

    def get_scheduler(self):
        if self.use_constant_lr:
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)
        else:
            # Step-based scheduler calculation
            warmup_steps = self.num_warmup_updates * self.accelerator.num_processes / self.grad_accumulation_steps
            total_steps = self.max_steps * self.accelerator.num_processes / self.grad_accumulation_steps
            decay_steps = total_steps - warmup_steps
            
            warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
            decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
            self.scheduler = SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
            )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        """Save checkpoint with clean directory structure"""
        if last:
            save_path = f"{self.checkpoint_path}/last"
        else:
            save_path = f"{self.checkpoint_path}/step_{step}"

        if self.is_main:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            os.makedirs(save_path, exist_ok=True)
        
        # Wait for everyone to finish before saving
        self.accelerator.wait_for_everyone()
        
        if self.is_main or self.use_fsdp:
            # Use Accelerate's save_state for model, optimizer, scheduler, RNG states
            self.accelerator.save_state(f'{save_path}/states')
            
        if self.is_main:
            # Save step number inside the checkpoint directory
            with open(f"{save_path}/step.txt", "w") as f:
                f.write(str(step))
            
            # Save EMA model using safetensors (more robust than pickle)
            if self.use_ema:
                ema_path = f"{save_path}/ema_model.safetensors"
                save_file(self.ema_model.state_dict(), ema_path)
            
            if last:
                print(f"Saved last checkpoint at step {step}")

        self.accelerator.wait_for_everyone()

    def load_checkpoint(self):
        """Load from specific checkpoint path with clean structure"""
        if not self.resume_from_checkpoint or not os.path.exists(self.resume_from_checkpoint):
            return 0

        self.accelerator.wait_for_everyone()
        
        # Load the main checkpoint (model, optimizer, scheduler, RNG states)
        self.accelerator.load_state(f'{self.resume_from_checkpoint}/states')
        
        # Load EMA model
        if self.is_main and self.use_ema:
            ema_safetensors_path = f"{self.resume_from_checkpoint}/ema_model.safetensors"
            ema_checkpoint = load_file(ema_safetensors_path)
            ema_dict = self.ema_model.state_dict()
            filtered_ema_dict = {
                k: v for k, v in ema_checkpoint.items()
                if k in ema_dict and ema_dict[k].shape == v.shape 
            }
            self.ema_model.load_state_dict(filtered_ema_dict, strict=False)
        
        # Load step number
        step_file = f"{self.resume_from_checkpoint}/step.txt"
        with open(step_file, "r") as f:
            step = int(f.read().strip())
        
        print(f"Checkpoint loaded from {self.resume_from_checkpoint} at step {step}")
        return step

    def train(self):
        start_step = self.load_checkpoint()
        global_step = start_step

        self.model.train()
        
        # Create progress bar for steps
        progress_bar = tqdm(
            range(start_step, self.max_steps),
            desc=f"Training",
            unit="step",
            disable=not self.accelerator.is_local_main_process,
            initial=start_step,
            total=self.max_steps,
            smoothing=0.15
        )

        for batch in self.train_dataloader:
            if global_step >= self.max_steps:
                break

            with self.accelerator.accumulate(self.model):
                text_inputs = batch["lrc"]
                mel_spec = batch["latent"].permute(0, 2, 1)
                mel_lengths = batch["latent_lengths"]
                style_prompt = batch["prompt"]
                start_time = batch["start_time"]
                duration_abs = batch["duration_abs"]
                duration_rel = batch["duration_rel"]

                loss, cond, pred = self.model(
                    mel_spec, text=text_inputs, lens=mel_lengths,
                    style_prompt=style_prompt,
                    start_time=start_time,
                    duration_abs=duration_abs, duration_rel=duration_rel
                )
                self.accelerator.backward(loss)

                if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self.is_main and self.use_ema:
                self.ema_model.update()

            global_step += 1

            if self.accelerator.is_local_main_process and global_step % self.log_every == 0:
                self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_step)

            progress_bar.set_postfix(step=str(global_step), loss=loss.item())
            progress_bar.update(1)

            if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                self.save_checkpoint(global_step)

            if global_step % self.last_per_steps == 0:
                self.save_checkpoint(global_step, last=True)

        progress_bar.close()
        self.save_checkpoint(global_step, last=True)
        self.accelerator.end_training()

