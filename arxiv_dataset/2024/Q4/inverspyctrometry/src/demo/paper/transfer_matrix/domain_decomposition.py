from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType
from src.demo.paper.transfer_matrix.utils import SamplingOptionsSchema, dct_orthogonalize


def main():
    options_schema = {
        "experiment_title": "fp_0_low_r",
        "device": {
            "type": InterferometerType.FABRY_PEROT,
            "reflectance_scalar": 0.2,
            "opds": {
                "num": 51,
                "step": 0.2,
            },
        },
        "spectral_range": {
            "min": 1.2,
            "max": 2.5,
            "override_harmonic_order": None,
        },
    }

    options = SamplingOptionsSchema(**options_schema)
    experiment = options.create_experiment()
    transfer_matrix = experiment.transfer_matrix()
    transfer_matrix_decomposition = experiment.transfer_matrix_decomposition()

    transfer_matrix_ortho = dct_orthogonalize(
        transfer_matrix=transfer_matrix,
        device_type=experiment.device_type,
        reflectance=options.device.reflectance_scalar,
    )
    transfer_matrix_decomposition_ortho = [
        dct_orthogonalize(
            transfer_matrix=replace(component, data=component.data + 2 * (1 - options.device.reflectance_scalar)),
            device_type=InterferometerType.MICHELSON,
            reflectance=options.device.reflectance_scalar,
        )
        for component in transfer_matrix_decomposition
    ]

    fig, axs = plt.subplots(2, 1, squeeze=False, tight_layout=True)
    transfer_matrix.visualize(fig=fig, axs=axs[0, 0], aspect="auto", title="Full matrix")
    transfer_matrix_ortho.visualize_singular_values(axs=axs[1, 0], title="Full matrix")

    harmonic_order = len(transfer_matrix_decomposition)
    fig, axes = plt.subplots(2, harmonic_order - 1, squeeze=False, tight_layout=True)
    for idx in range(1, harmonic_order):
        transfer_matrix_decomposition[idx].visualize(fig=fig, axs=axes[0, idx - 1], aspect="auto", title=f"n = {idx}")
        transfer_matrix_decomposition_ortho[idx].visualize_singular_values(axs=axes[1, idx - 1], title=f"n = {idx}")

    # fig, axs = plt.subplots(2, 1, squeeze=False)
    # transfer_matrix_decomposition_array = np.array([component.data for component in transfer_matrix_decomposition])
    # transfer_matrix_sum = replace(transfer_matrix, data=np.sum(transfer_matrix_decomposition_array, axis=0))
    # transfer_matrix_sum.visualize(fig=fig, axs=axs[0, 0], aspect="auto", title="Components sum")
    # singvals_sum = np.zeros_like(transfer_matrix_decomposition_ortho[0].singular_values())
    # for idx in range(1, harmonic_order):
    #     singvals_sum += transfer_matrix_decomposition_ortho[idx].singular_values()
    # axs[1, 0].plot(singvals_sum)
    # axs[1, 0].set_title("Components sum")

    plt.show()


def visualize_differences(options_schema: dict):
    options = SamplingOptionsSchema(**options_schema)
    experiment = options.create_experiment()
    transfer_matrix = experiment.transfer_matrix()
    transfer_matrix_decomposition = experiment.transfer_matrix_decomposition()
    transfer_matrix_decomposition_array = np.array([component.data for component in transfer_matrix_decomposition])
    transfer_matrix_sum = np.sum(transfer_matrix_decomposition_array, axis=0)
    transfer_matrix_diff = transfer_matrix.data - transfer_matrix_sum

    rmse = np.sqrt(np.mean(transfer_matrix_diff ** 2))
    print(rmse)

    fig, axs = plt.subplots(1, 3, squeeze=False)
    axs[0, 0].imshow(transfer_matrix.data, aspect="auto")
    axs[0, 1].imshow(transfer_matrix_sum, aspect="auto")
    axs[0, 2].imshow(transfer_matrix_diff, aspect="auto")
    plt.show()


if __name__ == "__main__":
    main()
