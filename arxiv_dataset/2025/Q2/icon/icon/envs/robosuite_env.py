import robosuite
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Union, Tuple, Literal, List


def task_to_env_name(task: str) -> str:
    if task == 'lift_cube':
        return 'Lift'
    if task == 'stack_cube':
        return 'Stack'
    elif task == 'open_door':
        return 'Door'
    elif task == 'assemble_round_nut':
        return 'NutAssemblyRound'
    elif task == 'pick_place_cereal':
        return 'PickPlaceCereal'


class RobosuiteEnv(gym.Env):

    def __init__(
        self,
        task: str,
        cameras: List,
        shape_meta: Dict,
        robot: Literal['Panda', 'Sawyer', 'UR5e', 'Kinova3', 'Jaco', 'IIWA'] = 'Panda',
        controller: Literal['OSC_POSE', 'JOINT_POSITION', 'JOINT_VELOCITY'] = 'OSC_POSE',
        render_mode: Literal['human', 'rgb_array'] = 'rgb_array',
        render_camera: Optional[str] = 'frontview',
        gpu_id: Union[int, None] = None
    ) -> None:
        env_kwargs = dict(
            env_name=task_to_env_name(task),
            camera_heights=shape_meta['images'],
            camera_widths=shape_meta['images'],
            robots=robot,
            controller_configs=robosuite.load_controller_config(default_controller=controller),
            has_offscreen_renderer=True,
            render_camera=render_camera,
            use_object_obs=False,
            use_camera_obs=True,
            camera_depths=False,
            ignore_done=True,   
            control_freq=20
        )
        env_kwargs['camera_names'] = cameras
        if render_mode == 'human':
            env_kwargs['has_renderer'] = True
        else:
            if render_camera not in cameras:
                env_kwargs['camera_names'] = [*cameras, render_camera]
        if gpu_id is not None:
            env_kwargs['render_gpu_device_id'] = gpu_id
        self.robosuite_env = robosuite.make(**env_kwargs)

        observation_space = dict()
        for camera in cameras:
            observation_space[f'{camera}_images'] = spaces.Box(
                low=0,
                high=255,
                shape=(shape_meta['images'], shape_meta['images'], 3), 
                dtype=np.uint8
            )
        observation_space['low_dims'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(shape_meta['low_dims'],),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict(observation_space)

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(shape_meta['actions'],),
            dtype=np.float32
        )

        self.cameras = cameras
        self.render_mode = render_mode
        self.render_camera = render_camera
        self.render_cache = None

    def _extract_obs(self, raw_obs: Dict) -> Dict:
        if self.render_mode == 'rgb_array':
            self.render_cache = np.flipud(raw_obs[f'{self.render_camera}_image'])

        obs = dict()
        ee_pose = np.concatenate([raw_obs['robot0_eef_pos'], raw_obs['robot0_eef_quat']])
        gripper_qpos = raw_obs['robot0_gripper_qpos']
        low_dims = np.concatenate([ee_pose, gripper_qpos], dtype=np.float32)
        obs['low_dims'] = low_dims
        obs.update({f'{camera}_images': np.flipud(raw_obs[f'{camera}_image']) for camera in self.cameras})
        return obs
    
    def render(self) -> Union[np.ndarray, None]:
        if self.render_mode == 'rgb_array':
            if self.render_cache is None:
                raise RuntimeError('Run reset() or step() before rendering!')
            return self.render_cache
        else:
            self.robosuite_env.render()
            return None
        
    def reset(self, seed: Union[int, None] = None, options: Union[Dict, None] = None) -> Dict:
        super().reset(seed=seed, options=options)
        np.random.seed(seed=seed)
        obs = self.robosuite_env.reset()
        return self._extract_obs(obs)

    def step(self, action: np.ndarray) -> Tuple:
        obs, reward, done, info = self.robosuite_env.step(action)
        done = bool(reward) or done
        return self._extract_obs(obs), reward, done, info