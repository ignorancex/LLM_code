from envs.mo_lunar_lander.mo_lunar_lander_randomized import MOLunarLanderDR
import gymnasium as gym

class MOLunarLanderHighGravity(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        gravity = -13.0
        super().__init__(gravity=gravity, continuous=continuous, **kwargs)

class MOLunarLanderLowGravity(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        gravity = -3.0
        super().__init__(gravity=gravity, continuous=continuous, **kwargs)

class MOLunarLanderWindy(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        wind_power = 20.0
        super().__init__(wind_power=wind_power, continuous=continuous, **kwargs)

class MOLunarLanderTurbulent(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        turbulence_power = 3.5
        super().__init__(turbulence_power=turbulence_power, continuous=continuous, **kwargs)

class MOLunarLanderLowMainEngine(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        main_engine_power = 10.0
        super().__init__(main_engine_power=main_engine_power, continuous=continuous, **kwargs)

class MOLunarLanderLowSideEngine(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        side_engine_power = 0.3
        initial_x_coeff = 0.4
        super().__init__(side_engine_power=side_engine_power, initial_x_coeff=initial_x_coeff, continuous=continuous, **kwargs)

class MOLunarLanderStartRight(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        initial_x_coeff = 0.75
        super().__init__(initial_x_coeff=initial_x_coeff, continuous=continuous, **kwargs)

class MOLunarLanderStartLow(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        initial_y_coeff = 0.7
        super().__init__(initial_y_coeff=initial_y_coeff, continuous=continuous, **kwargs)

class MOLunarLanderHard(MOLunarLanderDR):
    def __init__(self, continuous=False, **kwargs):
        gravity = -12.0
        wind_power = 17.0
        turbulence_power = 2.5
        main_engine_power = 12.0
        side_engine_power = 0.4
        initial_x_coeff = 0.4
        super().__init__(
            gravity=gravity, 
            wind_power=wind_power, 
            turbulence_power=turbulence_power,
            main_engine_power=main_engine_power,
            side_engine_power=side_engine_power, 
            initial_x_coeff=initial_x_coeff,
            continuous=continuous, 
            **kwargs
        )

        
def register_lunar_lander():
    try:
        gym.envs.register(
            id="MOLunarLanderDR-v0",
            entry_point="envs.mo_lunar_lander.mo_lunar_lander_randomized:MOLunarLanderDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousDR-v0",
            entry_point="envs.mo_lunar_lander.mo_lunar_lander_randomized:MOLunarLanderDR",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderDefault-v0", # copy of the dr environment but renamed for clarity
            entry_point="envs.mo_lunar_lander.mo_lunar_lander_randomized:MOLunarLanderDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousDefault-v0", # copy of the dr environment but renamed for clarity
            entry_point="envs.mo_lunar_lander.mo_lunar_lander_randomized:MOLunarLanderDR",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderHighGravity-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderHighGravity",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousHighGravity-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderHighGravity",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderLowGravity-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderLowGravity",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousLowGravity-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderLowGravity",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderWindy-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderWindy",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousWindy-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderWindy",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderTurbulent-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderTurbulent",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousTurbulent-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderTurbulent",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
    
    try:
        gym.envs.register(
            id="MOLunarLanderLowMainEngine-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderLowMainEngine",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
    
    try:
        gym.envs.register(
            id="MOLunarLanderContinuousLowMainEngine-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderLowMainEngine",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
    
    try:
        gym.envs.register(
            id="MOLunarLanderLowSideEngine-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderLowSideEngine",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousLowSideEngine-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderLowSideEngine",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderStartRight-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderStartRight",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousStartRight-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderStartRight",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderStartLow-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderStartLow",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousStartLow-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderStartLow",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderHard-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderHard",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLunarLanderContinuousHard-v0",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:MOLunarLanderHard",
            max_episode_steps=1000,
            kwargs={"continuous": True},
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
