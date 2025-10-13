import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import trange

from src.augmentations import Augmenter
from src.nn import LAPO, ActionDecoder, Actor
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    DCSInMemoryDataset,
    DCSLAPOInMemoryDataset,
    create_env_from_df,
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
    unnormalize_img,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LAPOConfig:
    num_epochs: int = 100
    batch_size: int = 256
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = None
    latent_action_dim: int = 8
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_deep: bool = True
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"


@dataclass
class BCConfig:
    num_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    dropout: float = 0.0
    use_aug: bool = True
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 10
    eval_seed: int = 0


@dataclass
class DecoderConfig:
    total_updates: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    hidden_dim: int = 128
    use_aug: bool = True
    data_path: str = "data/test.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 10
    eval_seed: int = 0


@dataclass
class Config:
    project: str = "laom"
    group: str = "lapo"
    name: str = "lapo"
    seed: int = 0

    lapo: LAPOConfig = field(default_factory=LAPOConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


def train_lapo(config: LAPOConfig):
    dataset = DCSLAPOInMemoryDataset(
        config.data_path, max_offset=config.future_obs_offset, frame_stack=config.frame_stack, device=DEVICE
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    lapo = LAPO(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        latent_act_dim=config.latent_action_dim,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
    ).to(DEVICE)

    torchinfo.summary(
        lapo,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        ],
    )
    optim = torch.optim.Adam(params=get_optim_groups(lapo, config.weight_decay), lr=config.learning_rate, fused=True)

    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    probe_optim = torch.optim.Adam(linear_probe.parameters(), lr=config.learning_rate)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        lapo.train()
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, future_obs, actions, _ = [b.to(DEVICE) for b in batch]
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))
            future_obs = normalize_img(future_obs.permute((0, 3, 1, 2)))

            # update lapo
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_next_obs, latent_action = lapo(obs, future_obs)
                loss = F.mse_loss(pred_next_obs, next_obs)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(lapo.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()

            # update linear probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action = linear_probe(latent_action.detach())
                probe_loss = F.mse_loss(pred_action, actions)

            probe_optim.zero_grad(set_to_none=True)
            probe_loss.backward()
            probe_optim.step()

            wandb.log(
                {
                    "lapo/mse_loss": loss.item(),
                    "lapo/action_probe_mse_loss": probe_loss.item(),
                    "lapo/throughput": total_tokens / (time.time() - start_time),
                    "lapo/learning_rate": scheduler.get_last_lr()[0],
                    "lapo/grad_norm": get_grad_norm(lapo).item(),
                    "lapo/epoch": epoch,
                    "lapo/total_steps": total_tokens,
                }
            )

        # logging reconstruction of next state
        obs_example = [unnormalize_img(next_obs[0][i : i + 3]) for i in range(0, 3 * config.frame_stack, 3)]
        next_obs_example = [unnormalize_img(pred_next_obs[0][i : i + 3]) for i in range(0, 3 * config.frame_stack, 3)]
        reconstruction_img = make_grid(obs_example + next_obs_example, nrow=config.frame_stack, padding=1)
        reconstruction_img = reconstruction_img.permute((1, 2, 0))
        reconstruction_img = wandb.Image(reconstruction_img.cpu().numpy(), caption="Top: True, Bottom: Pred")
        wandb.log(
            {
                "lapo/next_obs_pred": reconstruction_img,
                "lapo/epoch": epoch,
                "lapo/total_steps": total_tokens,
            }
        )

    return lapo


@torch.no_grad()
def evaluate_bc(env, actor, num_episodes, seed=0, device="cpu", action_decoder=None):
    returns = []
    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        total_reward = 0.0
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_ = torch.tensor(obs.copy(), device=device)[None].permute(0, 3, 1, 2)
            obs_ = normalize_img(obs_)
            action, obs_emb = actor(obs_)
            if action_decoder is not None:
                if isinstance(action_decoder, ActionDecoder):
                    action = action_decoder(obs_emb, action)
                else:
                    action = action_decoder(action)

            obs, reward, terminated, truncated, info = env.step(action.squeeze().cpu().numpy())
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def train_bc(lam: LAPO, config: BCConfig):
    dataset = DCSInMemoryDataset(config.data_path, frame_stack=config.frame_stack, device=DEVICE)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_env = create_env_from_df(
        config.data_path,
        config.dcs_backgrounds_path,
        config.dcs_backgrounds_split,
        frame_stack=config.frame_stack,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    num_actions = lam.latent_act_dim
    for p in lam.parameters():
        p.requires_grad_(False)
    lam.eval()

    actor = Actor(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        num_actions=num_actions,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        dropout=config.dropout,
    ).to(DEVICE)

    optim = torch.optim.AdamW(params=get_optim_groups(actor, config.weight_decay), lr=config.learning_rate, fused=True)
    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    # for debug
    print("Latent action dim:", num_actions)
    act_decoder = nn.Sequential(
        nn.Linear(num_actions, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, dataset.act_dim)
    ).to(DEVICE)

    act_decoder_optim = torch.optim.AdamW(params=act_decoder.parameters(), lr=config.learning_rate, fused=True)
    act_decoder_scheduler = linear_annealing_with_warmup(act_decoder_optim, warmup_updates, total_updates)

    torchinfo.summary(actor, input_size=(1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw))
    if config.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        actor.train()
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, true_actions = [b.to(DEVICE) for b in batch]
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))

            # label with lapo latent actions
            target_actions = lam.label(obs, next_obs)

            # augment obs only for bc to make action labels determenistic
            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_actions, _ = actor(obs)
                loss = F.mse_loss(pred_actions, target_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            # optimizing the probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_true_actions = act_decoder(pred_actions.detach())
                decoder_loss = F.mse_loss(pred_true_actions, true_actions)

            act_decoder_optim.zero_grad(set_to_none=True)
            decoder_loss.backward()
            act_decoder_optim.step()
            act_decoder_scheduler.step()

            wandb.log(
                {
                    "bc/mse_loss": loss.item(),
                    "bc/throughput": total_tokens / (time.time() - start_time),
                    "bc/learning_rate": scheduler.get_last_lr()[0],
                    "bc/act_decoder_probe_mse_loss": decoder_loss.item(),
                    "bc/epoch": epoch,
                    "bc/total_steps": total_steps,
                }
            )

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=act_decoder,
    )
    wandb.log(
        {
            "bc/eval_returns_mean": eval_returns.mean(),
            "bc/eval_returns_std": eval_returns.std(),
            "bc/epoch": epoch,
            "bc/total_steps": total_steps,
        }
    )

    return actor


def train_act_decoder(actor: Actor, config: DecoderConfig, bc_config: BCConfig):
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    dataset = DCSInMemoryDataset(config.data_path, frame_stack=bc_config.frame_stack, device=DEVICE)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    # to make equal number of updates for all labeled datasets which vary in size
    num_epochs = config.total_updates // len(dataloader)

    action_decoder = ActionDecoder(
        obs_emb_dim=math.prod(actor.final_encoder_shape),
        latent_act_dim=actor.num_actions,
        true_act_dim=dataset.act_dim,
        hidden_dim=config.hidden_dim,
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        params=get_optim_groups(action_decoder, config.weight_decay), lr=config.learning_rate, fused=True
    )
    eval_env = create_env_from_df(
        config.data_path,
        config.dcs_backgrounds_path,
        config.dcs_backgrounds_split,
        frame_stack=bc_config.frame_stack,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    # scheduler setup
    total_updates = len(dataloader) * num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    if config.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0

    for epoch in trange(num_epochs, desc="Epochs"):
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, _, true_actions = [b.to(DEVICE) for b in batch]
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs.permute((0, 3, 1, 2)))

            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                with torch.no_grad():
                    latent_actions, obs_emb = actor(obs)
                pred_actions = action_decoder(obs_emb, latent_actions)

                loss = F.mse_loss(pred_actions, true_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            wandb.log(
                {
                    "decoder/mse_loss": loss.item(),
                    "decoder/throughput": total_tokens / (time.time() - start_time),
                    "decoder/learning_rate": scheduler.get_last_lr()[0],
                    "decoder/epoch": epoch,
                    "decoder/total_steps": total_steps,
                }
            )

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=action_decoder,
    )
    wandb.log(
        {
            "decoder/eval_returns_mean": eval_returns.mean(),
            "decoder/eval_returns_std": eval_returns.std(),
            "decoder/epoch": epoch,
            "decoder/total_steps": total_steps,
        }
    )

    return action_decoder


@pyrallis.wrap()
def train(config: Config):
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    set_seed(config.seed)
    # stage 1: pretraining lapo on unlabeled dataset
    lapo = train_lapo(config=config.lapo)
    # stage 2: pretraining bc on latent actions
    actor = train_bc(lam=lapo, config=config.bc)
    # stage 3: finetune on labeles ground-truth actions
    action_decoder = train_act_decoder(actor=actor, config=config.decoder, bc_config=config.bc)

    run.finish()
    return lapo, actor, action_decoder


if __name__ == "__main__":
    train()
