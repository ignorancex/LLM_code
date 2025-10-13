import argparse
import sys
import time

import cvxpy as cp

SOLVER_KWARGS = {"solver": cp.CVXOPT, "abstol": 1e-5, "reltol": 1e-4, "feastol": 1e-5}
SOLVER_KWARGS_2 = {"solver": cp.SCS, "eps": 1e-5}
SOLVER_KWARGS_3 = {"solver": cp.CLARABEL}


def eprint(*args, **kwargs) -> None:
    print(*args, **kwargs, file=sys.stderr)


def add_env_args(parser: argparse.ArgumentParser):
    parser.add_argument("--env", choices=["distancing", "congestion2"], required=True, help="Type of environment")


def add_general_train_args(parser: argparse.ArgumentParser):
    # lr = 0.002 (Ding et al.), 0.0001 (Leonardos et al.)
    parser.add_argument("--lr", type=float, required=False, default=0.0003, help="Learning rate for the PGA algorithm")
    # gamma = 0.99 (Ding et al.), 0.9 (Rank et al.)
    parser.add_argument("--gamma", type=float, required=False, default=0.99, help="Environment reward discount factor")

    parser.add_argument("--n_rounds", type=int, required=False, default=5000, help="Number of retraining rounds performed")
    parser.add_argument("--n_episodes", type=int, required=False, default=10, help="Number of episodes to perform per round (per retraining) / trajectories per round")
    parser.add_argument("--seed", type=int, required=False, default=1, help="Random seed")


def add_performative_args(parser: argparse.ArgumentParser):
    parser.add_argument("--beta", type=float, required=False, default=1.0, help="Performativity strength - temperature for the softmax over Q-values for the intervening agent")
    parser.add_argument("--alpha", type=float, required=False, default=0.1, help="Performativity probability - probability for activating intervening agent")


def add_performative2_args(parser: argparse.ArgumentParser):
    parser.add_argument("--omega_r", type=float, required=False, default=0.0, help="Performativity strength on the reward")
    parser.add_argument("--omega_p", type=float, required=False, default=0.0, help="Performativity strength on the transition probability")


def add_experiment_replication_args(parser: argparse.ArgumentParser, default: int = 5):
    parser.add_argument(
        "--n_experiment_replications", type=int, required=False, default=default, help="Number of environments to run in parallel / number of experiment replications"
    )


def add_pga_train_args(parser: argparse.ArgumentParser):
    add_general_train_args(parser)
    parser.add_argument("--ding", action="store_true", default=False, help="Use Ding et al. (2022) method (does not use the state visitation distribution multiplier)")


def add_log_save_args(parser: argparse.ArgumentParser):
    parser.add_argument("--logging", type=str, choices=["wandb", "none"], default="wandb", help="Logging type")
    parser.add_argument("--log_interval", type=int, required=False, default=100, help="Interval between how many rounds logging should be performed")
    parser.add_argument("--save_interval", type=int, required=False, default=10000, help="Interval between rounds for saving the policy")


def get_filename(
    method: str, env: str, config, round_id: int | None = None, replication_id: int | None = None, n_rounds: int | None = None, n_experiment_replications: int | None = None
):
    filename = f"{env}_{method}_"
    filename += f"lr{config.lr}_gamma{config.gamma}_seed{config.seed}"

    if env.endswith("2"):
        filename += f"_omegar{config.omega_r}_omegap{config.omega_p}"
    else:
        filename += f"_beta{config.beta}_alpha{config.alpha}"

    if "log_barrier_reg" in config and config.log_barrier_reg != 0:
        filename += f"_reg{config.log_barrier_reg}"

    if replication_id is None:
        if n_experiment_replications is not None:
            filename += f"_{n_experiment_replications}repls"
        else:
            filename += f"_repl---"
    else:
        filename += f"_repl{replication_id:03d}"
    if round_id is None:
        if n_rounds is not None:
            filename += f"_{n_rounds}rounds"
        else:
            filename += "allrounds"
    else:
        filename += f"round{round_id:04d}"

    return filename


class Stopwatch:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.last_time = self.start_time

    def lap(self) -> float:
        if self.last_time is None:
            return -1

        new_time = time.time()
        lap_time = new_time - self.last_time
        self.last_time = new_time
        return lap_time

    def stop(self) -> float:
        if self.start_time is None:
            return -1

        end_time = time.time()
        total_time = end_time - self.start_time
        self.start_time = None
        self.last_time = None

        return total_time


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore
