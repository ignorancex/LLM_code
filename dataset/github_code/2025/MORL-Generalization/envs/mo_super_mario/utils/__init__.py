from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from envs.mo_super_mario.utils.mario_video_wrapper import RecordMarioVideo

def wrap_mario(env, gym_id="", algo_name="", seed=0, record_video=False, record_video_w_freq=None, record_video_ep_freq=None):
    from gymnasium.wrappers import (
        FrameStackObservation,
        GrayscaleObservation,
        ResizeObservation,
        TimeLimit,
    )
    from mo_gymnasium.envs.mario.joypad_space import JoypadSpace
    from mo_gymnasium.wrappers import MOMaxAndSkipObservation

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = TimeLimit(env, max_episode_steps=2000) # this must come before video recording else truncation will not be captured
    if record_video:
        if record_video_w_freq:
            env = RecordMarioVideo(
                env, 
                f"videos/{algo_name}/seed{seed}/{gym_id}/", 
                weight_trigger=lambda t: t % record_video_w_freq == 0,
                disable_logger=True
            )
        elif record_video_ep_freq:
            env = RecordMarioVideo(
                env, 
                f"videos/{algo_name}/seed{seed}/{gym_id}/", 
                episode_trigger=lambda t: t % record_video_ep_freq == 0,
                disable_logger=True
            )
    env = MOMaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, 4)
    return env