import os
import numpy as np
from typing import Callable, Optional
import gymnasium as gym
from gymnasium import logger
import imageio

def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordMarioVideo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper records rollouts as videos.
    Allows intermittent recording of videos based on number of weights evaluted by specifying ``weight_trigger``.
    To increased weight_number, call `env.reset(options={"weights": w, "step":s})` at the beginning of each evaluation. 
    If weight trigger is activated, the video recorded file name will include the  current step `s` and evaluated weight `w` as a suffix. 
    `w` must be a numpy array and `s` must be an integer.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        weight_trigger: Callable[[int], bool] = None,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
        fps: int = 30,
    ):
        """Wrapper records rollouts as videos.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the videos will be stored
            weight_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this weight evaluation
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            disable_logger (bool): Whether to disable logger or not.
            fps (int): Frames per second for the video recording.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi", "ansi_list"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )

        if episode_trigger is None and step_trigger is None and weight_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger, weight_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.weight_trigger = weight_trigger
        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length
        self.fps = env.metadata.get("render_fps", fps)

        self.recording = False
        self.terminated = False
        self.truncated = False
        self.video_writer = None
        self.recorded_frames = 0
        self.episode_id = 0

        # Custom multi-objective attributes
        self.weight_id = -1
        self.current_weight = None
        self.current_step = 0

        try:
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.is_vector_env = False

    def reset(self, **kwargs):
        """Reset the environment, set multi-objective weights if provided, and start video recording if enabled."""
        # Check for multi-objective weights in kwargs
        options = kwargs.get("options", {})
        if options and "weights" in options and "step" in options:
            assert isinstance(options["weights"], np.ndarray)
            assert isinstance(options["step"], int)
            self.current_weight = np.array2string(options["weights"], precision=2, separator=',')
            self.weight_id += 1
            self.current_step = options["step"]
        
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        if self.recording:
            assert self.video_writer is not None
            self._capture_frame()
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_writer()
        if self._video_enabled():
            self.start_video_recording()

        return observations

    def start_video_recording(self):
        """Initialize video recording."""
        self.close_video_writer()

        video_name = f"{self.name_prefix}-step-{self.step_id}.mp4"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}.mp4"
        elif self.weight_trigger:
            video_name = f"{self.name_prefix}-step{self.current_step}-weight-{self.current_weight}.mp4"
        video_path = os.path.join(self.video_folder, video_name)

        self.video_writer = imageio.get_writer(video_path, fps=self.fps, format='mp4')

        self._capture_frame()
        self.recording = True
    
    def _capture_frame(self):
        """Capture a frame from the environment and add it to the video."""
        frame = self.env.render()
        self.video_writer.append_data(frame)
        self.recorded_frames += 1

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        elif self.episode_trigger:
            return self.episode_trigger(self.episode_id)
        elif self.weight_trigger:
            return self.weight_trigger(self.weight_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            self.step_id += 1
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_writer is not None
                self._capture_frame()
                if self.video_length > 0:
                    if self.recorded_frames >= self.video_length:
                        self.close_video_writer()
                else:
                    if not self.is_vector_env:
                        if terminateds or truncateds:
                            self.close_video_writer()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_writer()
            elif self._video_enabled():
                self.start_video_recording()

        return observations, rewards, terminateds, truncateds, infos

    def close_video_writer(self):
        """Close the video writer if it is open."""
        if self.recording:
            assert self.video_writer is not None
            self.video_writer.close()
            self.recording = False
        self.recording = False
        self.recorded_frames = 1

    def close(self):
        """Closes the wrapper and saves any ongoing video recording."""
        super().close()
        self.close_video_writer()
