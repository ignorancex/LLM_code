from collections import OrderedDict

import gym
import numpy as np
import pyglet
from dm_control import suite
from dm_env.specs import BoundedArray
from gym.spaces import Box, Dict

RENDER_KWARGS = {"height": 480, "width": 640, "camera_id": 0, "overlays": (), "scene_option": None}
RENDER_MODES = {
    "human": {"show": True, "return_pixel": False, "render_kwargs": RENDER_KWARGS},
    "rgb_array": {"show": False, "return_pixel": True, "render_kwargs": RENDER_KWARGS},
    "human_rgb_array": {"show": True, "return_pixel": True, "render_kwargs": RENDER_KWARGS},
}


class DMCViewer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pitch = width * -3
        self.window = pyglet.window.Window(width=width, height=height)

    def update(self, pixel):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        img = pyglet.image.ImageData(self.width, self.height, "RGB", pixel.tobytes(), pitch=self.pitch)
        img.blit(0, 0)
        self.window.flip()

    def close(self):
        self.window.close()


class EnvSpec:
    def __init__(self, id):
        self.id = id


class DMCEnv(gym.Env):
    def __init__(self, domain_name, task_name, task_kwargs=None, seed=None, visualize_reward=False):
        if seed is not None:
            task_kwargs = task_kwargs or {}
            task_kwargs["random"] = seed

        if not hasattr(self, "metadata"):
            self.metadata = {}
        self.metadata["render.modes"] = list(RENDER_MODES.keys())
        self.viewer = {key: None for key in RENDER_MODES}
        self.dmc_env = suite.load(domain_name, task_name, task_kwargs=task_kwargs, visualize_reward=visualize_reward)

        self.physics = self.dmc_env.physics

        obs_spec = self.dmc_env.observation_spec()
        assert isinstance(obs_spec, OrderedDict), obs_spec
        self.observation_space = Dict(
            {k: Box(-np.inf, np.inf, shape=v.shape, dtype="float32") for k, v in obs_spec.items()}
        )

        action_spec = self.dmc_env.action_spec()
        assert isinstance(action_spec, BoundedArray), action_spec
        self.action_space = Box(
            low=np.full(action_spec.shape, action_spec.minimum, dtype=np.float32),
            high=np.full(action_spec.shape, action_spec.maximum, dtype=np.float32),
            dtype=np.float32,
        )

        self.spec = EnvSpec(f"{domain_name}_{task_name}")
        self.aux_env = suite.load(domain_name="hopper", task_name="stand", visualize_reward=True)

    def reset(self):
        self.timestep = self.dmc_env.reset()
        return self.timestep.observation

    def step(self, action):
        info = {"true_state": self.physics.get_state().copy()}
        self.timestep = self.dmc_env.step(action)
        info["next_true_state"] = self.physics.get_state().copy()
        info["step_type"] = self.timestep.step_type

        done = self.timestep.last()
        if not done:
            assert self.timestep.step_type == 1, f"Unexpected step type: {self.timestep.step_type}"

        return self.timestep.observation, self.timestep.reward, done, info

    def render(self, mode="human", close=False):
        self.aux_env.physics.render(width=10, height=10, camera_id=0)
        self.pixels = self.dmc_env.physics.render(**RENDER_MODES[mode]["render_kwargs"])

        if close:
            if self.viewer[mode]:
                self.viewer[mode].close()
                self.viewer[mode] = None
            return

        if RENDER_MODES[mode]["show"]:
            self._get_viewer(mode).update(self.pixels)

        if RENDER_MODES[mode]["return_pixel"]:
            return self.pixels

    def _get_viewer(self, mode):
        if self.viewer[mode] is None:
            self.viewer[mode] = DMCViewer(self.pixels.shape[1], self.pixels.shape[0])
        return self.viewer[mode]

    def close(self):
        self.dmc_env.close()
        for viewer in self.viewer.values():
            if viewer:
                viewer.close()

    def set_state(self, state):
        self.physics.set_state(state)


class FlattenedObsEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, Dict)
        flat_dim = sum(int(np.prod(space.shape)) for space in env.observation_space.spaces.values())
        self.observation_space = Box(-np.inf, np.inf, shape=(flat_dim,), dtype="float32")

    def observation(self, observation):
        return np.concatenate([np.ravel(v) for v in observation.values()])


def make_gym_dmc_env(env_id, seed=None):
    domain = next((d for d in suite.TASKS_BY_DOMAIN if d in env_id), None)
    if not domain:
        raise ValueError(f"Domain not found in env_id: {env_id}")
    if "humanoid_CMU" in env_id:
        domain = "humanoid_CMU"

    task = env_id.split(domain + "_", 1)[-1]
    env = DMCEnv(domain, task, seed=seed)
    return FlattenedObsEnv(env)
