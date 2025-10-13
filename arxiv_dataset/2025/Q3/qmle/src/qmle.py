import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from src.buffers import PrioritizedReplayBufferWithArgmax
from src.gym_dmc_env import make_gym_dmc_env
from src.model import QMLENetwork


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="quadruped_run",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(5e6),
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=200,
        help="the train steps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--epsilon", type=float, default=0.1,
        help="the epsilon for exploration")
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--num-action-samples-target", type=int, default=100,
        help="number of action samples for approximate target maximization")
    parser.add_argument("--num-action-samples-greedy", type=int, default=1000,
        help="number of action samples for greedy action approximation")
    parser.add_argument("--num-action-bins-per-dim", type=int, default=3,
        help="the support size per action dimension for discrete action samples")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        assert "_" in env_id, f"Expected DeepMind Control envs like 'walker_walk', got '{env_id}'"
        env = make_gym_dmc_env(env_id, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])

    # Model setup
    online_net = QMLENetwork(envs, num_categories=args.num_action_bins_per_dim).to(device)
    target_net = QMLENetwork(envs, num_categories=args.num_action_bins_per_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())

    # Optimizer setup
    optimizer = optim.Adam(online_net.parameters(), lr=args.learning_rate)

    # Replay buffer setup
    replay_eps = 1e-6
    beta = 0.2
    replay_buffer = PrioritizedReplayBufferWithArgmax(args.buffer_size, alpha=0.6)

    start_time = time.time()
    obs = envs.reset()
    for step in range(args.total_timesteps):
        # Action selection
        if random.random() < args.epsilon:
            action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_vals, act_samples = online_net(Tensor(obs).to(device), num_samples=args.num_action_samples_greedy)
            _, max_indices = torch.max(q_vals, dim=1)
            action = act_samples[np.arange(act_samples.shape[0]), max_indices].cpu().numpy()

        next_obs, reward, done, info = envs.step(action)

        for i in info:
            if "episode" in i:
                print(f"step={step}, episodic_return={i['episode']['r']}")
                writer.add_scalar("charts/episodic_return", i["episode"]["r"], step)
                break

        true_next_obs = next_obs.copy()
        for i, d in enumerate(done):
            if d:
                true_next_obs[i] = info[i]["terminal_observation"]

        done[:] = False  # Override for continuing DMC tasks; remove for envs with terminal states

        replay_buffer.add(obs, action, reward, true_next_obs, done, argmax=action.copy())
        obs = next_obs

        # Training
        if step > args.learning_starts and step % args.train_frequency == 0:
            batch = replay_buffer.sample(args.batch_size, beta=beta)
            obs_b, act_b, rew_b, next_obs_b, done_b, argmax_b, weight_b, idx_b = batch
            obs_b = torch.from_numpy(obs_b).squeeze(1).to(device)
            act_b = torch.from_numpy(act_b).squeeze(1).to(device)
            rew_b = torch.from_numpy(rew_b).float().to(device)
            next_obs_b = torch.from_numpy(next_obs_b).squeeze(1).to(device)
            done_b = torch.from_numpy(done_b).float().to(device)
            argmax_b = torch.from_numpy(argmax_b).squeeze(1).to(device)
            weight_b = torch.from_numpy(weight_b).float().to(device)

            q_old = online_net(obs_b, act=act_b).squeeze()
            with torch.no_grad():
                tgt_q, next_act_samples = target_net(
                    next_obs_b,
                    num_samples=args.num_action_samples_target,
                    prior_argmax=argmax_b,
                )
                tgt_q_max, tgt_argmax_idx = tgt_q.max(dim=1)
                td_target = rew_b.flatten() + args.gamma * tgt_q_max * (1 - done_b.flatten())
                best_next_actions = (
                    next_act_samples[np.arange(next_act_samples.shape[0]), tgt_argmax_idx].cpu().numpy().copy()
                )
                replay_buffer.update_argmaxes(idx_b, best_next_actions)

            # Loss: Q-learning
            loss_q_no_reduce = F.mse_loss(q_old, td_target, reduction="none")
            loss_q = (loss_q_no_reduce * weight_b).mean()

            # Update priorities
            with torch.no_grad():
                new_priorities = torch.abs(q_old - td_target).cpu().numpy() + replay_eps
                replay_buffer.update_priorities(idx_b, new_priorities)

            # Loss: categorical
            with torch.no_grad():
                argmax_as_cat_tgt = online_net.map_continuous_to_bins(argmax_b)
            cat_outputs_b = online_net.get_categorical_predictor_outputs(next_obs_b)
            loss_cat = torch.mean(
                torch.stack(
                    [F.cross_entropy(out, argmax_as_cat_tgt[:, i]) for i, out in enumerate(cat_outputs_b)], dim=0
                )
            )

            # Loss: delta
            delta_outputs_b = online_net.get_delta_predictor_outputs(next_obs_b)
            loss_delta = F.mse_loss(delta_outputs_b, argmax_b.detach())

            # Combine and optimize
            total_loss = loss_q + loss_cat + loss_delta
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            if step % 100 == 0:
                writer.add_scalar("losses/loss_q", loss_q, step)
                writer.add_scalar("losses/loss_cat", loss_cat, step)
                writer.add_scalar("losses/loss_delta", loss_delta, step)
                writer.add_scalar("losses/q_values", q_old.mean().item(), step)
                sps = int(step / (time.time() - start_time))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, step)

            # Update target network
            if step % args.target_network_frequency == 0:
                target_net.load_state_dict(online_net.state_dict())

    envs.close()
    writer.close()
