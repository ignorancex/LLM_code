from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def show_velocity_model(data: NDArray, vmin: Union[float, None] = None, vmax: Union[float, None] = None, title: str = "velocity model", cmap: str = "jet"):
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X [km]")
    plt.ylabel("Depth [km]")
    plt.show()


def show_minimum_velocity_model(data: np.ndarray, vmin: Union[float, None] = None, vmax: Union[float, None] = None, cmap: str = "jet"):
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    raw_figsize = (data.shape[1], data.shape[0])
    scale = 10.0 / max(raw_figsize[0], raw_figsize[1])
    image_aspect = (raw_figsize[0] * scale, raw_figsize[1] * scale)

    fig, ax = plt.subplots(figsize=image_aspect)
    ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")

    x_phys_max = 0.68
    y_phys_max = 0.21

    x_ticks_in_real = np.arange(0, x_phys_max + 0.1, 0.1)
    x_ticks_in_real = x_ticks_in_real[x_ticks_in_real <= x_phys_max]  # 0.68 以下に制限

    y_ticks_in_real = np.arange(0, y_phys_max + 0.1, 0.1)
    y_ticks_in_real = y_ticks_in_real[y_ticks_in_real <= y_phys_max]  # 0.21 以下に制限

    x_ticks = x_ticks_in_real / x_phys_max * (data.shape[1] - 1)
    y_ticks = y_ticks_in_real / y_phys_max * (data.shape[0] - 1)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{val:.1f}" for val in x_ticks_in_real], fontsize=22)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{val:.1f}" for val in y_ticks_in_real], fontsize=22)

    plt.xlabel("X [km]", fontsize=24)
    plt.ylabel("Depth [km]", fontsize=24)

    plt.axis("tight")
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.995, bottom=0.26, top=0.96)
    plt.show()
