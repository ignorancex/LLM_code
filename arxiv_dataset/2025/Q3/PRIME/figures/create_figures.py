import os

import matplotlib.pyplot as plt

from figures.fig_feasibility import fig_feasibility
from figures.fig_meas_noise import fig_meas_noise
from figures.fig_nonconvex_toy import fig_nonconvex_toy
from figures.fig_unicorn import fig_unicorn
from figures.fig_unicorn_noise import fig_unicorn_noise

FIGURES = [fig_feasibility, fig_meas_noise, fig_unicorn, fig_unicorn_noise, fig_nonconvex_toy]
SAVE_PATH = "./figures/output"


def create_figures(overwrite=True):
    print("creating figures ...")

    # create save folder
    if not os.path.exists(SAVE_PATH):
        try:
            os.mkdir(SAVE_PATH)
        except Exception as e:
            print(f"An error occurred during directory creation: {e}")
            return

    # iteratively create figures
    for figure in FIGURES:
        path = f"{SAVE_PATH}/{figure.__name__}.pdf"

        # set plot parameters
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.size": 13,
                "legend.fontsize": 10,
                "lines.linewidth": 0.8,
            }
        )

        if overwrite or not os.path.exists(path):
            print(path)
            figure()
            plt.savefig(path, transparent=True)


if __name__ == "__main__":
    create_figures(overwrite=False)
