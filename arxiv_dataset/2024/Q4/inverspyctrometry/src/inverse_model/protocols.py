from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft
from tqdm import tqdm

from src.common_utils.custom_vars import InversionProtocolType
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.utils import generate_wavenumbers_from_opds
from src.inverse_model.loris_verhoeven_utils import LorisVerhoevenIteration
from src.inverse_model.operators import ProximalOperator, LinearOperator


def inversion_protocol_factory(option: InversionProtocolType, parameters: dict):
    if option == InversionProtocolType.IDCT:
        return IDCT()

    elif option == InversionProtocolType.PSEUDO_INVERSE:
        return PseudoInverse()

    elif option == InversionProtocolType.TSVD:
        return TSVD(penalization_ratio=parameters["penalization_ratio"])

    elif option == InversionProtocolType.RIDGE_REGRESSION:
        return RidgeRegression(penalization=parameters["penalization"])

    elif option == InversionProtocolType.LORIS_VERHOEVEN:
        if "is_compute_and_save_cost" not in parameters:
            parameters["is_compute_and_save_cost"] = False
        if "experiment_id" not in parameters:
            parameters["experiment_id"] = -1
        return LorisVerhoeven(
            regularization_parameter=parameters["regularization_parameter"],
            prox_functional=parameters["prox_functional"],
            domain_transform=parameters["domain_transform"],
            nb_iters=parameters["nb_iters"],
            is_compute_and_save_cost=parameters["is_compute_and_save_cost"],
            experiment_id=parameters["experiment_id"],
        )

    elif option == InversionProtocolType.ADMM:
        return ADMM()

    else:
        raise ValueError(f"Inversion Protocol option '{option}' is not supported")


@dataclass(frozen=True)
class InversionProtocol(ABC):

    @abstractmethod
    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        pass


@dataclass(frozen=True)
class IDCT(InversionProtocol):
    """
    Inverse Discrete Cosine Transform
    """
    is_mean_center: bool = True

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        if self.is_mean_center:
            interferogram = interferogram.center(new_mean=0, axis=-2)
        spectrum = fft.idct(interferogram.data, axis=-2, norm="ortho")
        wavenumbers = generate_wavenumbers_from_opds(
            wavenumbers_num=spectrum.shape[-2],
            del_opd=np.mean(np.diff(interferogram.opds))
        )
        spectrum_obj = Spectrum(data=spectrum, wavenumbers=wavenumbers)
        spectrum_obj = spectrum_obj.interpolate(
            wavenumbers=transmittance_response.wavenumbers,
            kind="linear",
            fill_value="extrapolate",
        )
        return spectrum_obj


@dataclass(frozen=True)
class PseudoInverse(InversionProtocol):

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        tr_pinv = np.linalg.pinv(transmittance_response.data)
        spectrum = tr_pinv @ interferogram.data
        return Spectrum(
            data=spectrum,
            wavenumbers=transmittance_response.wavenumbers,
        )


@dataclass(frozen=True)
class TSVD(InversionProtocol):
    """
    Truncated Singular Value Decomposition
    """
    penalization_ratio: float

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        lsv, sv, rsv = np.linalg.svd(a=transmittance_response.data, full_matrices=False, compute_uv=True)
        nb_sv = int(sv.size * self.penalization_ratio)
        sv_penalized = 1 / sv[:nb_sv]
        spectrum = (rsv[:nb_sv].T * sv_penalized) @ lsv[:, :nb_sv].T @ interferogram.data
        return Spectrum(data=spectrum, wavenumbers=transmittance_response.wavenumbers)


@dataclass(frozen=True)
class RidgeRegression(InversionProtocol):
    penalization: float

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        lsv, sv, rsv = np.linalg.svd(a=transmittance_response.data, full_matrices=False, compute_uv=True)
        sv_penalized = sv / (sv ** 2 + self.penalization ** 2)
        spectrum = (rsv.T * sv_penalized) @ lsv.T @ interferogram.data
        return Spectrum(data=spectrum, wavenumbers=transmittance_response.wavenumbers)


@dataclass(frozen=True)
class LorisVerhoeven(InversionProtocol):
    """
    Solve problems of the form:
      x = argmin_{x} 1/2 ||Ax - y||^2 + g(Lx),
      proposed in the following paper:
    [1] Loris, Ignace, and Caroline Verhoeven. "On a generalization of the iterative soft-thresholding algorithm for the
      case of non-separable penalty." Inverse Problems 27.12 (2011): 125007.

    The current implementation includes an additional over-relaxation step, for which we refer to the following paper:
    [2] Condat, Laurent, et al. "Proximal splitting algorithms for convex optimization: A tour of recent advances, with
      new twists." SIAM Review 65.2 (2023): 375-435.

    Note: In [2], the original Loris-Verhoeven algorithm in [1] is generalized to solve problems of the form:
      x = argmin_{x} h(x) + g(Lx),
      where h(x) is convex, differentiable, and beta-Lipschitz (beta>0).
      The special case is when h(x) = 1/2 ||Ax - y||^2 (data fidelity least-squares), and so Del_h(x) = A^T(Ax - y).
    """
    regularization_parameter: float
    prox_functional: ProximalOperator
    domain_transform: LinearOperator
    nb_iters: int
    is_compute_and_save_cost: bool = False
    experiment_id: int = -1

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        transfer_matrix = LinearOperator.from_matrix(matrix=transmittance_response.data)

        lv_iter = LorisVerhoevenIteration(
            transfer_matrix=transfer_matrix,
            domain_transform=self.domain_transform,
            prox_functional=self.prox_functional,
            regularization_parameter=self.regularization_parameter,
            interferogram=interferogram.data,
            is_compute_cost=self.is_compute_and_save_cost,
        )
        prim = transfer_matrix.adjoint(interferogram.data)
        dual = self.domain_transform.direct(prim)
        cost_progress = np.zeros(shape=self.nb_iters)
        for q in tqdm(range(self.nb_iters)):
            prim, dual, cost_progress[q] = lv_iter.update(prim=prim, dual=dual)

        if self.is_compute_and_save_cost:
            project_dir = Path(__file__).resolve().parents[2]
            folder_subdir = f"reports/experiment_{self.experiment_id}/loris_verhoeven_cost/lambdaa_{self.regularization_parameter:07.4f}"
            folder_dir = project_dir / folder_subdir
            if not folder_dir.exists():
                folder_dir.mkdir(parents=True, exist_ok=True)
            np.save(file=folder_dir / "cost_progress", arr=cost_progress)

        return Spectrum(data=prim, wavenumbers=transmittance_response.wavenumbers)


@dataclass(frozen=True)
class ADMM(InversionProtocol):
    """
    Alternating Direction Method of Multipliers
    Solve problems of the form:
      x = argmin_{x} f(x) + g(x)
    """

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        pass
