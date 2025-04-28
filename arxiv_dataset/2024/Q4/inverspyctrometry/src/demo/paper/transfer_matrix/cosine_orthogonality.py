import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.transmittance_response import TransmittanceResponse
from src.demo.paper.transfer_matrix.utils import dct_orthogonalize


def main():
    row_num = 51  # opds
    col_num = row_num * 1  # wavenumbers
    alpha = 0.6
    col_idx_start = int((1 - alpha) * col_num)

    row_idxs = np.arange(row_num)
    col_idxs = np.arange(start=col_idx_start, stop=col_num)

    matrix = 2 * np.cos(np.pi / (2 * col_num) * row_idxs[..., None] * (2 * col_idxs[None] + 1))
    transfer_matrix = TransmittanceResponse(data=matrix, wavenumbers=col_idxs, opds=row_idxs)
    transfer_matrix_ortho = dct_orthogonalize(
        transfer_matrix=transfer_matrix,
        device_type=None,
        reflectance=-1,
    )

    fig, axs = plt.subplots(1, 1, squeeze=False, tight_layout=True, figsize=(6.4, 4.8))
    transfer_matrix.visualize(fig=fig, axs=axs[0, 0], aspect="auto")

    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, tight_layout=True, figsize=(12.8, 4.8))
    transfer_matrix.visualize_singular_values(axs=axs[0, 0], title="SVs (Original)")
    transfer_matrix_ortho.visualize_singular_values(axs=axs[0, 1], title="SVs (Compensated)")
    v_line_position = int(np.round(alpha * np.min(transfer_matrix.data.shape)))
    axs[0, 0].axvline(x=v_line_position, color='r', linestyle='--', label=f'SV Drop')
    axs[0, 1].axvline(x=v_line_position, color='r', linestyle='--', label=f'SV Drop')

    plt.show()


if __name__ == "__main__":
    main()
