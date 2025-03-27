# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from po_dynamics.modules import metrics


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "po-dynamics"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    activation_fn: str = "tanh"
    """the activation function"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        if args.activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
        elif args.activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError("Not implemented")

        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            self.activation_fn,
            layer_init(nn.Linear(64, 64)),
            self.activation_fn,
        )
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            self.activation_fn,
            layer_init(nn.Linear(64, 64)),
            self.activation_fn,
        )
        self.actor_mean = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)
        self.actor_logstd = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)

    def get_value(self, x):
        return self.critic(self.critic_trunk(x))

    def get_action_and_value(self, x, action=None):
        hidden_actor = self.actor_trunk(x)
        hidden_critic = self.critic_trunk(x)
        action_mean = self.actor_mean(hidden_actor)
        action_logstd = self.actor_logstd(hidden_actor)
        action_std = torch.exp(action_logstd.clamp(min=np.log(1e-9), max=np.log(1e9)))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            action_std,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(hidden_critic),
            hidden_actor,
            hidden_critic,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=False,
            dir=Path("outputs").absolute(),
        )
    writer = SummaryWriter(f"outputs/cleanrl_runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    hiddens_actor = torch.zeros((args.num_steps, args.num_envs, 64)).to(device)
    hiddens_critic = torch.zeros((args.num_steps, args.num_envs, 64)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    pbar = trange(1, args.num_iterations + 1)
    desc = {"perf": "", "fps": ""}
    for iteration in pbar:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        fps = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, action_std, logprob, _, value, hidden_actor, hidden_critic = agent.get_action_and_value(
                    next_obs
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            hiddens_actor[step] = hidden_actor
            hiddens_critic[step] = hidden_critic

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                returns = []
                final_timesteps = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        returns.append(info["episode"]["r"])
                        final_timesteps.append(info["episode"]["l"])
                if len(returns) > 0:
                    returns = np.array(returns)
                    final_timesteps = np.array(final_timesteps)
                    desc["perf"] = f"global_step={global_step} | episodic_return={returns.mean()}"
                    writer.add_scalar("batch/std/min", action_std.min())
                    writer.add_scalar("batch/std/max", action_std.max())
                    writer.add_scalar("batch/std/mean", action_std.mean())
                    writer.add_scalar("batch/perf/avg_return_raw", returns.mean(), global_step)
                    writer.add_scalar("batch/perf/max_return_raw", returns.max(), global_step)
                    writer.add_scalar("batch/perf/max_episode_timestep", final_timesteps.max(), global_step)
                    writer.add_scalar("batch/perf/avg_episode_timestep", final_timesteps.mean(), global_step)

        fps = int(args.num_envs * args.num_steps / (time.time() - fps))
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_hiddens_actor = hiddens_actor.reshape((-1, 64))
        b_hiddens_critic = hiddens_critic.reshape((-1, 64))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue, _, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]

                # Avoid
                ratio = logratio.clamp(min=np.log(1e-9), max=np.log(1e9)).exp()
                ratio_clipped = logratio.clamp(min=np.log(1 - args.clip_coef), max=np.log(1 + args.clip_coef)).exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)

                pg_loss, clip_frac_minibatch = torch.stack([pg_loss1, pg_loss2], dim=-1).max(dim=-1)
                pg_loss = pg_loss.mean()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [clip_frac_minibatch.detach().float().mean().item()]

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if iteration % 1 == 0:
            # Compute effective ranks:
            ranks = metrics.compute_effective_ranks(
                [{"features_policy": b_hiddens_actor, "features_value": b_hiddens_critic}],
                ["batch"],
                [["features_policy", "features_value"]],
            )
            for rank_k, rank_v in ranks.items():
                writer.add_scalar(f"batch/start/{rank_k}", rank_v, global_step)
            writer.add_scalar(
                "batch/last_epoch/last_minibatch/learning_rate/policy", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar(
                "batch/last_epoch/last_minibatch/learning_rate/value", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("batch/last_epoch/last_minibatch/std/min", action_std.min())
            writer.add_scalar("batch/last_epoch/last_minibatch/std/max", action_std.max())
            writer.add_scalar("batch/last_epoch/last_minibatch/std/mean", action_std.mean())
            writer.add_scalar(
                "batch/last_epoch/last_minibatch/loss/loss_critic", args.vf_coef * v_loss.item(), global_step
            )
            writer.add_scalar("batch/last_epoch/last_minibatch/loss/loss_policy", pg_loss.item(), global_step)
            writer.add_scalar(
                "batch/last_epoch/last_minibatch/loss/loss_entropy", -args.ent_coef * entropy_loss.item(), global_step
            )
            writer.add_scalar("batch/last_epoch/last_minibatch/loss/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("batch/last_epoch/last_minibatch/loss/kl_sample_naive", old_approx_kl.item(), global_step)
            writer.add_scalar("batch/last_epoch/last_minibatch/loss/kl_sample_schulman", approx_kl.item(), global_step)
            writer.add_scalar("batch/last_epoch/mean_minibatch/loss/clipped_fraction", np.mean(clipfracs), global_step)
            writer.add_scalar("batch/last_epoch/last_minibatch/loss/clipped_fraction", clipfracs[-1], global_step)
            # writer.add_scalar("losses/explained_variance", explained_var, global_step)
            desc["fps"] = f"SPS: {int(global_step / (time.time() - start_time))} | FPS: {int(fps)}"
            writer.add_scalar("timers/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("timers/fps_collector", fps, global_step)
        pbar.set_description(f"{desc['fps']} | {desc['perf']} ")

    envs.close()
    writer.close()
