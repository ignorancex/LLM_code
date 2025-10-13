import tqdm
import torch
import numpy as np
import gymnasium as gym
from typing import Optional, Union, Dict
from icon.utils.gym_utils.multistep_wrapper import MultiStepWrapper
from icon.utils.gym_utils.video_recording_wrapper import VideoRecordingWrapper
from icon.utils.pytorch_utils import to


class EnvRunner:

    def __init__(
        self,
        env: gym.Env,
        obs_horizon: int,
        action_horizon: int,
        max_episode_steps: Optional[int] = 200,
        num_episodes: Optional[int] = 50,
        initial_seed: Optional[int] = 10000,
        enable_temporal_ensemble: Optional[bool] = True,
        video_save_dir: Union[str, None] = None
    ) -> None:
        env = VideoRecordingWrapper(
            env=env,
            video_save_dir=video_save_dir
        )
        env = MultiStepWrapper(
            env=env,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            max_episode_steps=max_episode_steps,
            enable_temporal_ensemble=enable_temporal_ensemble
        )
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes
        self.initial_seed = initial_seed
    
    def _process_obs(self, raw_obs: Dict) -> Dict:
        """
        Process observations from gym environments.
        """
        obs = dict()
        images = dict()
        obs['low_dims'] = torch.from_numpy(raw_obs['low_dims']).float().unsqueeze(0)
        for key, val in raw_obs.items():
            if key.endswith('images'):
                images[key.replace('_images', '')] = torch.from_numpy(val).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        obs['images'] = images
        return obs
    
    def run(self, policy, device: torch.device) -> None:
        success = 0
        for i in range(self.num_episodes):
            seed = self.initial_seed + i
            obs = self.env.reset(seed=seed)
            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Trial {i + 1}/{self.num_episodes}", 
                leave=False,
                mininterval=5.0
            )
            done = False
            while not done:
                obs = self._process_obs(obs)
                to(obs, device)
                with torch.no_grad():
                    action = policy.predict_action(obs)['actions']
                action = action.detach().to('cpu').squeeze(0).numpy()
                obs, reward, done, _ = self.env.step(action)
                self.env.render()
                done = np.all(done)
                if reward:
                    success += 1
                if self.env.enable_temporal_ensemble:
                    pbar.update(1)
                else:
                    pbar.update(action.shape[1])
            pbar.close()
        print(f"Success rate: {success / self.num_episodes}")
        self.env.reset()