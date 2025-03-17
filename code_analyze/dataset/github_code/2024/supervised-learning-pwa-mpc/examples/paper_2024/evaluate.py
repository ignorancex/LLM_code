import pickle
from typing import Any, Callable, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.wrappers import TimeLimit
from model import Model
from mpc_mld import ThisMpcMld
from mpcrl.wrappers.envs import MonitorEpisodes

from slpwampc.agents.parc_agent import ParcAgent
from slpwampc.core.systems import PwaSystem


class PwaEnv(gymnasium.Env):
    """Environement simulating the PWA system."""

    def __init__(
        self,
        system: PwaSystem,
        viable_first_state_function: Callable | None = None,
    ) -> None:
        """Initialize the environment.

        Parameters
        ----------
        viable_first_state_function : function, optional
            A function to validate if the first state is viable, by default None
        system : PwaSystem
            The system."""
        self.system = system
        self.viable_first_state_function = viable_first_state_function

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            The seed for the environment, by default None
        options : dict[str, Any], optional
            Options for the environment, by default None

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The initial state and options."""
        super().reset(seed=seed, options=options)
        self.x = self.np_random.uniform(-10, 10, size=(2, 1))
        if self.viable_first_state_function is not None:
            while not self.viable_first_state_function(self.x):
                self.x = self.np_random.uniform(-10, 10, size=(2, 1))
        else:
            while not (self.system.D @ self.x <= self.system.E).all():
                self.x = self.np_random.uniform(-10, 10, size=(2, 1))
        return self.x, {}

    def get_stage_cost(self, state: np.ndarray, action: np.ndarray) -> np.floating[Any]:
        return np.linalg.norm(state, 1) + np.linalg.norm(action, 1)

    def step(
        self, action: np.ndarray
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment.

        Parameters
        ----------
        action : np.ndarray
            The control input.

        Returns
        -------
        tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]
            The next state, the cost, whether the episode is terminated, whether the episode is truncated, and info dict.
        """
        r = self.get_stage_cost(self.x, action)
        for i in range(len(self.system.S)):
            if (
                self.system.S[i] @ self.x + self.system.R[i] @ action
                <= self.system.T[i]
            ):
                x_new = (
                    self.system.A[i] @ self.x
                    + self.system.B[i] @ action
                    + self.system.c[i]
                )
        self.x = x_new
        terminated = np.linalg.norm(self.x) < 0.01
        return x_new, r, False, terminated, {}


np.random.seed(0)
np_random = np.random.default_rng(0)
SAVE = True

num_episodes = 1000
sim_length = 100
N = 12
use_learned_policy = True

nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()

mpc = ThisMpcMld(system_dict, N, nx, nu, X_f=Model.X_f, verbose=False)
agent = ParcAgent(
    system,
    mpc,
    N,
    learn_infeasible_regions=True,
)
agent.load(f"examples/paper_2024/results/parc_agent_N_{N}")

func = lambda x: agent.parc.predict(x.T)[0] != -1 and (system.D @ x <= system.E).all()
env = MonitorEpisodes(
    TimeLimit(PwaEnv(system, viable_first_state_function=func), sim_length)
)
f, x, t = agent.evaluate(
    env,
    num_episodes=num_episodes,
    seed=0,
    use_learned_policy=use_learned_policy,
    K_term=Model.K_term,
)

X = [env.observations[i].squeeze() for i in range(num_episodes)]
U = [env.actions[i].squeeze() for i in range(num_episodes)]
R = [env.rewards[i].squeeze() for i in range(num_episodes)]

if SAVE:
    with open(
        f"examples/paper_2024/results/evaluate_closed_loop_N_{N}_learned_{use_learned_policy}.pkl",
        "wb",
    ) as file:
        pickle.dump({"X": X, "U": U, "R": R, "t": t}, file)
