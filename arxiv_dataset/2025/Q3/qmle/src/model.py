import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.categorical import Categorical


def perturb_actions(actions: Tensor, num_samples: int, std: float) -> Tensor:
    noise = torch.randn(num_samples, *actions.shape[1:], dtype=actions.dtype, device=actions.device)
    perturbed = actions.unsqueeze(1) + std * noise
    return perturbed.squeeze(2)


class ResidualLayernormWrapper(nn.Module):
    def __init__(self, layer: nn.Module, hidden_dim: int):
        super().__init__()
        self.layer = layer
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor):
        return self.layer_norm(self.layer(x) + x)


class QMLENetwork(nn.Module):
    def __init__(
        self,
        env,
        obs_hidden_dim=128,
        act_hidden_dim=128,
        mle_hidden_dim=128,
        ratio_uniform=0.9,
        ratio_delta=0.01,
        ratio_categorical=0.09,
        num_categories=3,
    ):
        super().__init__()

        self.ratio_delta = ratio_delta
        self.ratio_categorical = ratio_categorical
        self.ratio_uniform_discrete = ratio_uniform / 2
        self.ratio_uniform_continuous = ratio_uniform / 2

        self.num_categories = num_categories

        if self.ratio_uniform_discrete > 0:
            self.uniform_discrete_support = list(np.linspace(0, 1, num_categories, dtype=np.float32))

        self.env = env
        act_low_np = env.single_action_space.low.astype(np.float32)
        act_high_np = env.single_action_space.high.astype(np.float32)

        # Register action space parameters for device transfer and checkpointing
        self.register_buffer("act_low", torch.tensor(act_low_np))
        self.register_buffer("act_high", torch.tensor(act_high_np))
        self.register_buffer("act_width", torch.tensor(act_high_np - act_low_np))
        self.register_buffer("action_scale", torch.tensor((act_high_np - act_low_np) / 2.0))
        self.register_buffer("action_bias", torch.tensor((act_high_np + act_low_np) / 2.0))

        # Observation encoder
        obs_resnet = nn.Sequential(
            nn.Linear(obs_hidden_dim, obs_hidden_dim),
            nn.ReLU(),
            nn.Linear(obs_hidden_dim, obs_hidden_dim),
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(np.prod(env.single_observation_space.shape), obs_hidden_dim),
            ResidualLayernormWrapper(obs_resnet, obs_hidden_dim),
            nn.ELU(),
        )

        # Action encoder
        self.act_encoder = nn.Sequential(
            nn.Linear(env.single_action_space.shape[0], act_hidden_dim),
            nn.LayerNorm(act_hidden_dim),
            nn.ELU(),
        )

        # Q-value predictor
        obs_act_resnet = nn.Sequential(
            nn.Linear(obs_hidden_dim + act_hidden_dim, obs_hidden_dim + act_hidden_dim),
            nn.ReLU(),
            nn.Linear(obs_hidden_dim + act_hidden_dim, obs_hidden_dim + act_hidden_dim),
        )
        self._obs_act_encoder = nn.Sequential(
            ResidualLayernormWrapper(obs_act_resnet, obs_hidden_dim + act_hidden_dim),
            nn.ELU(),
            nn.Linear(obs_hidden_dim + act_hidden_dim, 1),
        )

        # Delta argmax predictor
        self.delta_predictor = nn.Sequential(
            nn.Linear(obs_hidden_dim, mle_hidden_dim),
            nn.ReLU(),
            nn.Linear(mle_hidden_dim, np.prod(env.single_action_space.shape)),
            nn.Tanh(),
        )

        # Categorical argmax predictor
        self.category_to_action_map = [
            np.linspace(low, high, num_categories, dtype=np.float32) for low, high in zip(act_low_np, act_high_np)
        ]
        self.categorical_torso = nn.Sequential(
            nn.Linear(obs_hidden_dim, mle_hidden_dim),
            nn.ReLU(),
        )
        self.categorical_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mle_hidden_dim, num_categories),
                nn.Softmax(dim=-1),
            )
            for _ in range(env.single_action_space.shape[0])
        ])

    def forward(self, obs, act=None, num_samples=None, prior_argmax=None):
        obs_latent = self.obs_encoder(obs)

        if act is not None:
            # No maximization needed: predict Q-values for given (obs, act) pairs
            act_latent = self.act_encoder(act)
            obs_act_latent = torch.cat([obs_latent, act_latent], dim=-1)
            return self._obs_act_encoder(obs_act_latent)

        combined_action_samples = []

        # Sample uniformly from continuous space in [0, 1]
        if self.ratio_uniform_continuous > 0:
            num_uniform_continuous_samples = int(self.ratio_uniform_continuous * num_samples)
            uniform_continuous_samples = torch.rand(
                obs.shape[0],
                num_uniform_continuous_samples,
                self.env.single_action_space.shape[0],
            )
            combined_action_samples.append(uniform_continuous_samples)

        # Sample uniformly from discrete grid in [0, 1]
        if self.ratio_uniform_discrete > 0:
            num_discrete_samples = int(self.ratio_uniform_discrete * num_samples)
            uniform_discrete_samples = np.random.choice(
                self.uniform_discrete_support,
                (
                    obs.shape[0],
                    num_discrete_samples,
                    self.env.single_action_space.shape[0],
                ),
            )
            uniform_discrete_samples = torch.from_numpy(uniform_discrete_samples)
            combined_action_samples.append(uniform_discrete_samples)

        # Project uniform samples into action space range
        if combined_action_samples:
            combined_action_samples = torch.cat(combined_action_samples, dim=1).to(obs.device)
            combined_action_samples *= self.act_width
            combined_action_samples += self.act_low

        # Sample from the categorical argmax predictor
        if self.ratio_categorical > 0:
            num_categorical_samples = int(self.ratio_categorical * num_samples)
            categorical_samples = self._get_actions_categorical(obs_latent, num_categorical_samples)
            if isinstance(combined_action_samples, Tensor):
                combined_action_samples = torch.cat([combined_action_samples, categorical_samples], dim=1)
            else:
                combined_action_samples = categorical_samples

        # Sample from the delta argmax predictor with noise
        if self.ratio_delta > 0:
            num_delta_samples = int(self.ratio_delta * num_samples)
            delta_samples = self._get_actions_delta(obs_latent, num_delta_samples)
            if isinstance(combined_action_samples, Tensor):
                combined_action_samples = torch.cat([combined_action_samples, delta_samples], dim=1)
            else:
                combined_action_samples = delta_samples
                assert isinstance(combined_action_samples, Tensor)

        # Append prior argmax approximation if provided
        if prior_argmax is not None:
            combined_action_samples = torch.cat([combined_action_samples, prior_argmax.unsqueeze(1)], dim=1)

        obs_latent = obs_latent.unsqueeze(1).repeat(1, num_samples + (0 if prior_argmax is None else 1), 1)
        act_latent_sample = self.act_encoder(combined_action_samples)
        obs_act_latent_sample = torch.cat([obs_latent, act_latent_sample], dim=-1)
        q_values = self._obs_act_encoder(obs_act_latent_sample).squeeze(-1)
        return q_values, combined_action_samples

    def _get_actions_categorical(self, obs_latent, num_samples):
        with torch.no_grad():
            obs_latent_categorical = self.categorical_torso(obs_latent)
            probs = [categorical_head(obs_latent_categorical) for categorical_head in self.categorical_predictor]
            probs = torch.stack(probs, dim=1)
            idxes = Categorical(probs=probs).sample(sample_shape=(num_samples,)).permute(1, 0, 2)

            actions = []
            for act_dim in range(self.env.single_action_space.shape[0]):
                subactions_dim = torch.from_numpy(
                    self.category_to_action_map[act_dim][idxes[:, :, act_dim].cpu().numpy()]
                ).to(obs_latent.device).unsqueeze(2)
                actions.append(subactions_dim)
            return torch.concat(actions, dim=2)

    def _get_actions_delta(self, obs_latent, num_actions=1, std=0.1):
        with torch.no_grad():
            delta_params = self.delta_predictor(obs_latent).unsqueeze(1)
            num_actions_perturbed = num_actions - 1  # one action is always based on the delta parameters without noise
            if num_actions_perturbed > 0:
                delta_params_perturbed = perturb_actions(delta_params, num_actions_perturbed, std)
                actions_no_map = torch.cat((delta_params, delta_params_perturbed), dim=1)
            else:
                actions_no_map = delta_params
            actions_no_clamp = actions_no_map * self.action_scale + self.action_bias
            return torch.clamp(actions_no_clamp, self.act_low, self.act_high)

    def get_categorical_predictor_outputs(self, obs):
        with torch.no_grad():
            obs_latent = self.obs_encoder(obs)
        obs_latent_categorical = self.categorical_torso(obs_latent)
        return [categorical_head(obs_latent_categorical) for categorical_head in self.categorical_predictor]

    def get_delta_predictor_outputs(self, obs):
        with torch.no_grad():
            obs_latent = self.obs_encoder(obs)
        return self.delta_predictor(obs_latent)

    def map_continuous_to_bins(self, actions: torch.Tensor) -> torch.Tensor:
        norm = (actions - self.act_low) / self.act_width
        return torch.round(norm * (self.num_categories - 1)).to(torch.long)
