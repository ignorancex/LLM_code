import os
import time
from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from util.util import (
    DotDict,
    add_env_args,
    add_experiment_replication_args,
    add_performative2_args,
    add_performative_args,
    add_pga_train_args,
    get_filename,
)

plt.rcParams.update({"font.size": 14})
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Latin Modern Math"


def load_and_compute_policy_dist(config, num_take_last: int = 10, take: int | None = None):
    method = config.method
    n_agents = 8 if config.env.startswith("distancing") else 4

    filename = get_filename(method, config.env, config, n_rounds=config.n_rounds, n_experiment_replications=config.n_experiment_replications)
    policies = np.load(f"data/{filename}.npy")

    pi_changes_last = []

    for repl in range(config.n_experiment_replications):
        pis_env_np = np.array(policies[repl])
        pis_env_np = np.where(np.isnan(pis_env_np), 0, pis_env_np)
        pi_ref_last = np.mean(pis_env_np[-num_take_last:], axis=0)

        pi_changes_last += [np.sum(np.linalg.norm(pis_env_np - pi_ref_last, 2, axis=(2, 3)), axis=1)]

    pi_changes_last = np.array(pi_changes_last) * (1 / n_agents)

    pi_plottable = pi_changes_last
    take = pi_plottable.shape[1] if take is None else take

    pi_change_mean = pi_plottable.mean(axis=0)[:take]
    pi_change_std = pi_plottable.std(axis=0)[:take]

    x_plot = range(pi_change_mean.shape[0])

    return x_plot, pi_change_mean, pi_change_std


def set_style():
    plt.title("")
    plt.gcf().set_dpi(150)
    plt.gca().set_box_aspect(1)
    plt.xlabel("Round 𝑡")
    plt.ylabel(r"$\frac{1}{N} \sum_i^N \|\pi_i^t - \pi_i^{\text{last}}\|_2$")
    plt.grid(alpha=0.15)
    plt.legend(loc="upper right")


def get_compares(compare_list: List[str], env: str) -> List[Dict[str, float | str]]:
    compares = []
    for item in compare_list:
        compare = {}
        items = item.split("_")

        compare["method"] = items.pop(0)

        if "_reg" in item:
            reg = items.pop(0)
            compare["log_barrier_reg"] = float(reg.removeprefix("reg"))

        lr = items.pop(0)
        compare["lr"] = float(lr.removeprefix("lr"))

        if env == "distancing":
            alpha = items.pop(0)
            compare["alpha"] = float(alpha.removeprefix("alpha"))
        elif env == "congestion2":
            omegar = items.pop(0)
            compare["omega_r"] = float(omegar.removeprefix("omegar"))
            omegap = items.pop(0)
            compare["omega_p"] = float(omegap.removeprefix("omegap"))
        else:
            raise NotImplementedError()
        assert len(items) == 0

        compares.append(DotDict(compare))
    return compares


def run(config):
    config = DotDict(vars(config))
    print(config)

    compare_configs = get_compares(config.compare, config.env)

    if config.labels is not None:
        labels = config.labels
        assert len(labels) == len(compare_configs)
    else:
        labels = config.compare

    if config.colors is not None:
        colors = config.colors
        assert len(colors) == len(compare_configs)
    else:
        colors = list(mcolors.TABLEAU_COLORS.keys())[: len(compare_configs)]

    for extra_config, color, label in zip(compare_configs, colors, labels):

        config_use = DotDict({**config, **extra_config})

        x_plot, pi_change_mean, pi_change_std = load_and_compute_policy_dist(config_use)

        dont_show = 50  # how many last rounds not to show
        every = 3  # dilation to make the figure load faster

        plt.plot(x_plot[:-dont_show][::every], pi_change_mean[:-dont_show][::every], label=label, linewidth=1, color=color)
        plt.fill_between(
            x_plot[:-dont_show][::every],
            pi_change_mean[:-dont_show][::every] - pi_change_std[:-dont_show][::every],
            pi_change_mean[:-dont_show][::every] + pi_change_std[:-dont_show][::every],
            alpha=0.1 if color == "#fa6f1e" else 0.15,
            color=color,
            edgecolor="#ffffff00",
        )

    set_style()

    os.makedirs("figs", exist_ok=True)
    figure_file = f"figs/{config.out}.svg"
    plt.savefig(figure_file)
    print(f"Saved to '{figure_file}'.")


def list_str(values: str) -> List[str]:
    return values.split(",")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_env_args(parser)
    add_pga_train_args(parser)
    add_performative_args(parser)
    add_performative2_args(parser)
    add_experiment_replication_args(parser)
    parser.add_argument(
        "--compare",
        type=list_str,
        required=True,
        help="A comma separated list of things to compare. Each compare item must be of the format METHOD_regREG_lrLR_alphaALPHA or METHOD_lrLR_alphaALPHA for the distancing environment, or METHOD_regREG_lrLR_omegarOMEGAR_omegapOMEGAP or METHOD_lrLR_omegarOMEGAR_omegapOMEGAP for the congestion2 environment, where METHOD is one of leo,ding,inpg, REG is the regularization strength (optional), LR is the value of the learning rate, ALPHA is the alpha parameter value, OMEGAR and OMEGAP are the omega_r and omega_p values respectively",
    )
    parser.add_argument(
        "--labels",
        type=list_str,
        required=False,
        default=None,
        help="List of labels for the methods under comparison. Must match the lenght of `--compare`. By default takes strings from `--compare`.",
    )
    parser.add_argument(
        "--colors",
        type=list_str,
        required=False,
        default=None,
        help="List of colors for the methods under comparison. Must match the lenght of `--compare`. By default takes matplotlib default colors.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output figure name")
    args = parser.parse_args()

    time_start = time.time()
    run(args)
    time_end = time.time()

    print(f"Total runtime: {time_end - time_start:.2f} s")
