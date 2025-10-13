import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import JointPosition, JointVelocity, EndEffectorPoseViaIK
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.utils import name_to_task_class
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Union, Tuple, Literal, List


def name_to_action_mode(name: str):
    absolute_mode = True
    if name.startswith("delta"):
        absolute_mode = False
    if name.endswith("joint_pos"):
        return JointPosition(absolute_mode)
    elif name.endswith("joint_vel"):
        return JointVelocity(absolute_mode)
    elif name.endswith("ee_pose"):
        return EndEffectorPoseViaIK(absolute_mode)

# Adapted from https://github.com/stepjam/RLBench/blob/master/rlbench/gym.py
class RLBenchEnv(gym.Env):

    def __init__(
        self, 
        task: str,
        cameras: List,
        shape_meta: Dict,
        robot: Literal['panda', 'jaco', 'mico', 'sawyer', 'ur5'] = 'panda',
        action_mode: Literal['joint_pos', 'joint_vel', 'ee_pose', 'delta_joint_pos', 'delta_joint_vel', 'delta_ee_pose'] = 'delta_ee_pose',
        render_mode: Literal['human', 'rgb_array', None] = None,
        render_size: Optional[int] = 512
    ) -> None:
        all_cameras = {
            'left_shoulder_camera',
            'right_shoulder_camera',
            'overhead_camera',
            'wrist_camera',
            'front_camera'
        }
        del_cameras = list(all_cameras - set(cameras))
        camera_configs = {
            camera: CameraConfig(
                rgb=True,
                depth=False,
                point_cloud=False,
                mask=False,
                image_size=(shape_meta['images'], shape_meta['images'])
            )
            for camera in cameras
        }
        del_camera_configs = {
            camera: CameraConfig(
                rgb=False,
                depth=False,
                point_cloud=False,
                mask=False
            )
            for camera in del_cameras
        }
        obs_config = ObservationConfig(**camera_configs, **del_camera_configs)
        
        self.action_mode = action_mode
        action_mode = MoveArmThenGripper(
            arm_action_mode=name_to_action_mode(action_mode),
            gripper_action_mode=Discrete()
        )

        self.rlbench_env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=True,
            robot_setup=robot
        )
        self.rlbench_env.launch()
        self.rlbench_task_env = self.rlbench_env.get_task(name_to_task_class(task))
        
        if render_mode is not None:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self.gym_cam = VisionSensor.create([render_size, render_size])
            self.gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == "human":
                self.gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self.gym_cam.set_render_mode(RenderMode.OPENGL3)

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

    def _extract_obs(self, raw_obs) -> Dict:
        obs = dict()
        qpos = raw_obs.joint_positions
        ee_pose = raw_obs.gripper_pose
        gripper_open = np.array([raw_obs.gripper_open], dtype=np.float32)
        ee_pose = np.concatenate([
            ee_pose[:3],
            R.from_quat(ee_pose[3:]).as_euler('xyz')
        ])
        low_dims = np.concatenate([qpos, ee_pose, gripper_open], dtype=np.float32)
        obs['low_dims'] = low_dims
        obs.update({f'{camera}_images': getattr(raw_obs, camera.replace('camera', 'rgb')) for camera in self.cameras})
        return obs

    def render(self) -> Union[np.ndarray, None]:
        if self.render_mode == 'rgb_array':
            frame = self.gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame
        else:
            return None

    def reset(self, seed: Union[int, None] = None, options: Union[Dict, None] = None) -> Dict:
        super().reset(seed=seed, options=options)
        np.random.seed(seed=seed)
        _, obs = self.rlbench_task_env.reset()
        return self._extract_obs(obs)

    def step(self, action: np.ndarray) -> Tuple:
        if self.action_mode.endswith("ee_pose"):
            translation = action[:3]
            rotation = R.from_euler('xyz', action[3: 6]).as_quat()
            gripper_state = action[6][np.newaxis]
            action = np.concatenate([translation, rotation, gripper_state])
        obs, reward, terminated = self.rlbench_task_env.step(action)
        return self._extract_obs(obs), reward, terminated, {}

    def close(self) -> None:
        self.rlbench_env.shutdown()