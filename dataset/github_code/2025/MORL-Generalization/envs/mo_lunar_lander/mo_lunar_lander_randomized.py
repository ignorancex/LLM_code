import math

import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import (
    FPS,
    LEG_DOWN,
    MAIN_ENGINE_Y_LOCATION,
    SCALE,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    VIEWPORT_H,
    VIEWPORT_W,
)
from envs.mo_lunar_lander.utils.lunar_lander_custom import LunarLander
from morl_generalization.algos.dr import DREnv

class MOLunarLanderDR(LunarLander, DREnv):
    """
    ## Description
    Multi-objective version of the LunarLander environment with 5 environment parameters
    - gravity: The gravity of the environment
    - wind_power: The power of the wind
    - turbulence_power: The power of the turbulence
    - main_engine_power: The power of the main engine
    - side_engine_power: The power of the side engine

    ## Reward Space
    The reward is 3-dimensional:
    - 1: Shaping reward and approach to the landing pad
    - 2: Fuel cost (main engine)
    - 3: Fuel cost (side engine)
    100 is added/subtracted from each of the 3 components if the lander lands successfully/crashes.
    """

    param_info = {
        'names': ['gravity', 'wind_power', 'turbulence_power', 'main_engine_power', 'side_engine_power', 'initial_x_coeff', 'initial_y_coeff'],
        'param_max': [-7.0, 20.0, 3.5, 16.0, 0.9, 0.75, 1.0],
        'param_min': [-13.0, 0.0, 0.0, 10.0, 0.3, 0.25, 0.7]
    }
    DEFAULT_PARAMS = [-10.0, 15.0, 1.5, 13.0, 0.6, 0.5, 1.0]

    def __init__(
        self, 
        gravity: float = -10.0,
        enable_wind: bool = True,
        continuous: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        main_engine_power: float = 13.0,
        side_engine_power: float = 0.6,
        *args, 
        **kwargs
    ):
        DREnv.__init__(self)
        LunarLander.__init__(
            self,
            continuous=continuous,
            gravity=gravity, 
            wind_power=wind_power, 
            turbulence_power=turbulence_power,
            enable_wind=enable_wind,
            *args, 
            **kwargs
        )

        self.main_engine_power = main_engine_power
        self.side_engine_power = side_engine_power

        # shaping reward, main engine cost, side engine cost
        self.reward_space = spaces.Box(
            low=np.array([-100, -np.inf, -1, -1]),
            high=np.array([100, np.inf, 0, 0]),
            shape=(4,),
            dtype=np.float32,
        )
        self.reward_dim = 4

    def reset_random(self):
        """
        Reset the environment with a new parameters. Please call self.reset() after this.
        """
        params_max = self.param_info['param_max']
        params_min = self.param_info['param_min']
        new_params = [
            np.random.uniform(params_min[0], params_max[0]), # gravity
            np.random.uniform(params_min[1], params_max[1]), # wind_power
            np.random.uniform(params_min[2], params_max[2]), # turbulence_power
            np.random.uniform(params_min[3], params_max[3]), # main_engine_power
            np.random.uniform(params_min[4], params_max[4]), # side_engine_power
            np.random.uniform(params_min[5], params_max[5]), # initial_x_coeff
            np.random.uniform(params_min[6], params_max[6]), # initial_y_coeff
        ]
        self._update_params(*new_params)
    
    def get_task(self):
        return np.array([
            self.gravity, 
            self.wind_power,
            self.turbulence_power,
            self.main_engine_power,
            self.side_engine_power,
            self.initial_x_coeff,
            self.initial_y_coeff
        ])

    def _update_params(self, gravity, wind_power, turbulence_power, main_engine_power, side_engine_power, initial_x_coeff, initial_y_coeff):
        self.wind_power = wind_power
        self.gravity = gravity
        self.turbulence_power = turbulence_power
        self.main_engine_power = main_engine_power
        self.side_engine_power = side_engine_power
        self.initial_x_coeff = initial_x_coeff
        self.initial_y_coeff = initial_y_coeff

    def step(self, action):
        assert self.lander is not None

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = math.tanh(math.sin(0.02 * self.wind_idx) + (math.sin(math.pi * 0.01 * self.wind_idx))) * self.wind_power
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = (
                math.tanh(math.sin(0.02 * self.torque_idx) + (math.sin(math.pi * 0.01 * self.torque_idx)))
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(
                torque_mag,
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * self.main_engine_power * m_power,
                        oy * self.main_engine_power * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * self.main_engine_power * m_power, -oy * self.main_engine_power * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * self.side_engine_power * s_power,
                        oy * self.side_engine_power * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * self.side_engine_power * s_power, -oy * self.side_engine_power * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        vector_reward = np.zeros(4, dtype=np.float32)
        shaping = (
            - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
            vector_reward[1] = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heuristic landing
        vector_reward[2] = -m_power
        reward -= s_power * 0.03
        vector_reward[3] = -s_power

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
            vector_reward[0] -= 100.0
        if not self.lander.awake:
            terminated = True
            reward = +100
            vector_reward[0] += 100.0

        if self.render_mode == "human":
            self.render()

        return np.array(state, dtype=np.float32), vector_reward, terminated, False, {"original_scalar_reward": reward}
    

if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium.envs.registration import register
    import matplotlib.pyplot as plt
    register(
        id="LunarLander-v0",
        entry_point="envs.mo_lunar_lander.mo_lunar_lander_randomized:MOLunarLanderDR",
    )
    env = gym.make("LunarLander-v0", render_mode="human")

    obs = env.reset()
    done = False
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if done:
            env.reset_random()
            obs = env.reset()