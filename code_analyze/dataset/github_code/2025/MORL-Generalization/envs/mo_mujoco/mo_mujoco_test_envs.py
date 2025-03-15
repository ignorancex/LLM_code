from envs.mo_mujoco.mo_hopper_randomized import MOHopperDR
from envs.mo_mujoco.mo_halfcheetah_randomized import MOHalfCheetahDR
from envs.mo_mujoco.mo_humanoid_randomized import MOHumanoidDR
import gymnasium as gym
import numpy as np

# ============================ Hopper ============================
class MOHopperLight(MOHopperDR):
    def __init__(self, **kwargs):
        masses = np.array([0.5, 0.5, 0.3, 0.7])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task, **kwargs)

class MOHopperHeavy(MOHopperDR):
    def __init__(self, **kwargs):
        masses = np.array([9.0, 9.0, 8.5, 10.0])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task, **kwargs)

class MOHopperSlippery(MOHopperDR):
    def __init__(self, **kwargs):
        masses = np.array([3.7, 4.0, 2.8, 5.3])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([0.1])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task, **kwargs)

class MOHopperLowDamping(MOHopperDR):
    def __init__(self, **kwargs):
        masses = np.array([3.7, 4.0, 2.8, 5.3])
        damping = np.array([0.1, 0.1, 0.1])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task, **kwargs)

class MOHopperHard(MOHopperDR):
    def __init__(self, **kwargs):
        # light torso, heavy (thigh, shin), light foot
        masses = np.array([0.1, 9.0, 9.0, 0.1])
        damping = np.array([0.1, 0.1, 0.1])
        friction = np.array([0.1])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task, **kwargs)

# ============================ Cheetah ============================
class MOHalfCheetahLight(MOHalfCheetahDR):
    def __init__(self, **kwargs):
        masses = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        friction = np.array([0.4])
        task = np.concatenate([masses, friction])
        super().__init__(task=task, **kwargs)

class MOHalfCheetahHeavy(MOHalfCheetahDR):
    def __init__(self, **kwargs):
        masses = np.array([10.0, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5])
        friction = np.array([0.4])
        task = np.concatenate([masses, friction])
        super().__init__(task=task, **kwargs)

class MOHalfCheetahSlippery(MOHalfCheetahDR):
    def __init__(self, **kwargs):
        masses = np.array([6.25020921, 1.54351464, 1.5874477, 1.09539749, 1.43807531, 1.20083682, 0.88451883])
        friction = np.array([0.1])
        task = np.concatenate([masses, friction])
        super().__init__(task=task, **kwargs)

class MOHalfCheetahHard(MOHalfCheetahDR):
    def __init__(self, **kwargs):
        # default torso, heavy back (thigh, shin, foot), light front (thigh, shin, foot)
        masses = np.array([6.25020921, 9.5, 9.5, 9.5, 0.1, 0.1, 0.1])
        friction = np.array([0.1])
        task = np.concatenate([masses, friction])
        super().__init__(task=task, **kwargs)


# ============================ Humanoid ============================
class MOHumanoidLight(MOHumanoidDR):
    def __init__(self, **kwargs):
        masses = np.array([1.7, 0.5, 1.3, 0.7, 0.6, 0.5, 0.7, 0.5, 0.3, 0.3, 0.1, 0.3, 0.1])
        damping = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        task = np.concatenate([masses, damping])
        super().__init__(task=task, **kwargs)

class MOHumanoidHeavy(MOHumanoidDR):
    def __init__(self, **kwargs):
        masses = np.array([10.0, 7.0, 9.0, 8.0, 7.0, 6.0, 8.0, 7.0, 6.0, 6.0, 5.5, 6.0, 5.5])
        damping = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        task = np.concatenate([masses, damping])
        super().__init__(task=task, **kwargs)

class MOHumanoidLowDamping(MOHumanoidDR):
    def __init__(self, **kwargs):
        masses = np.array([8.90746237, 2.26194671, 6.61619413, 4.75175093, 2.75569617, 1.76714587, 4.75175093, 2.75569617, 1.76714587, 1.66108048, 1.22954019, 1.66108048, 1.22954019])
        damping = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        task = np.concatenate([masses, damping])
        super().__init__(task=task, **kwargs)

class MOHumanoidHard(MOHumanoidDR):
    def __init__(self, **kwargs):
        # default (torso, waist, pelvis), light right leg , heavy left leg, light right arm, heavy left arm
        masses = np.array([8.90746237, 2.26194671, 6.61619413, 0.7, 0.6, 0.5, 8.0, 7.0, 6.0, 0.1, 0.1, 5.0, 5.0])
        damping = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        task = np.concatenate([masses, damping])
        super().__init__(task=task, **kwargs)

def register_mujoco():
    # HalfCheetah
    try:
        gym.envs.register(
            id="MOHalfCheetahDR-v5",
            entry_point="envs.mo_mujoco.mo_halfcheetah_randomized:MOHalfCheetahDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
        
    try:
        gym.envs.register(
            id="MOHalfCheetahDefault-v5", # copy of the dr environment but renamed for clarity
            entry_point="envs.mo_mujoco.mo_halfcheetah_randomized:MOHalfCheetahDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHalfCheetahLight-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHalfCheetahLight",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHalfCheetahHeavy-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHalfCheetahHeavy",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    
    try:
        gym.envs.register(
            id="MOHalfCheetahSlippery-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHalfCheetahSlippery",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")


    try:
        gym.envs.register(
            id="MOHalfCheetahHard-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHalfCheetahHard",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")


    # Hopper
    try:
        gym.envs.register(
            id="MOHopperDR-v5",
            entry_point="envs.mo_mujoco.mo_hopper_randomized:MOHopperDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperDefault-v5", # copy of the dr environment but renamed for clarity
            entry_point="envs.mo_mujoco.mo_hopper_randomized:MOHopperDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperLight-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperLight",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperHeavy-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperHeavy",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperSlippery-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperSlippery",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperLowDamping-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperLowDamping",
            max_episode_steps=1000,
        )

    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    
    try:
        gym.envs.register(
            id="MOHopperHard-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperHard",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    
    # Humanoid
    try:
        gym.envs.register(
            id="MOHumanoidDR-v5",
            entry_point="envs.mo_mujoco.mo_humanoid_randomized:MOHumanoidDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHumanoidDefault-v5", # copy of the dr environment but renamed for clarity
            entry_point="envs.mo_mujoco.mo_humanoid_randomized:MOHumanoidDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHumanoidLight-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHumanoidLight",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
    
    try:
        gym.envs.register(
            id="MOHumanoidHeavy-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHumanoidHeavy",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
    
    try:
        gym.envs.register(
            id="MOHumanoidLowDamping-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHumanoidLowDamping",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
    
    try:
        gym.envs.register(
            id="MOHumanoidHard-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHumanoidHard",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
