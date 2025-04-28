from pathlib import Path
from typing import Literal, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from numpy.typing import NDArray
from tqdm import tqdm


def save_array3d_as_gif(
    array3d: NDArray,
    save_path: Path,
    animation_duration: float = 0.02,
    disable_progress_display: bool = False,
    normalize: Union[Normalize | Literal["auto"] | None] = None,
    image_aspect: Union[Tuple[float, float] | Literal["auto"] | None] = None,
):
    if normalize == "auto":
        min_value = np.min(array3d)
        max_value = np.max(array3d)
        mid_value = (max_value + min_value) / 2.0
        normalize = TwoSlopeNorm(vmin=min_value, vcenter=mid_value, vmax=max_value)

    if image_aspect == "auto":
        raw_figsize = (array3d.shape[2], array3d.shape[1])
        scale = 10.0 / max(raw_figsize[0], raw_figsize[1])
        image_aspect = (raw_figsize[0] * scale, raw_figsize[1] * scale)

    n = array3d.shape[0]
    with imageio.get_writer(save_path, mode="I", duration=animation_duration) as writer:
        for i in tqdm(range(n), disable=disable_progress_display):
            fig, ax = plt.subplots(figsize=image_aspect)
            ax.imshow(array3d[i], cmap="seismic", interpolation="nearest", norm=normalize)
            plt.axis("tight")
            plt.axis("off")
            fig.tight_layout()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            writer.append_data(image[..., :3])
            plt.close(fig)
