from envs.mo_lava_grid.mo_lava_grid_test_envs import register_lava_grid
from envs.mo_lunar_lander.lunarlander_test_envs import register_lunar_lander
from envs.mo_mujoco.mo_mujoco_test_envs import register_mujoco
from envs.mo_super_mario.super_mario_test_envs import register_mario

def register_envs():
    try:
        register_lava_grid()
        register_lunar_lander()
        register_mujoco()
        register_mario()
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
