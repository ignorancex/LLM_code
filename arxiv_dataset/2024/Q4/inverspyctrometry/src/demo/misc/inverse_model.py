from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType, InversionProtocolType, NormOperatorType, \
    LinearOperatorMethod
from src.common_utils.interferogram import Interferogram
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.utils import generate_sampled_opds, generate_wavenumbers_from_opds
from src.direct_model.interferometer import interferometer_factory
from src.inverse_model.analytical_inverter import MichelsonInverter
from src.inverse_model.operators import CTVOperator, NormOperator, LinearOperator
from src.inverse_model.protocols import inversion_protocol_factory, InversionProtocol


@dataclass(frozen=True)
class InversionProtocolDemo:
    """
    Basic demo for inversion protocols
    """
    inverter: InversionProtocol

    def test(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
            acq_ind: int,
    ):
        spectrum = self.inverter.reconstruct_spectrum(
            interferogram=interferogram,
            transmittance_response=transmittance_response,
        )
        fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
        interferogram.visualize(axs=axs[0, 0], acq_ind=acq_ind)
        spectrum.visualize(axs=axs[0, 1], acq_ind=acq_ind)


def generate_transmittance_response(kwargs):
    interferometer = interferometer_factory(
        option=kwargs["type"],
        transmittance_coefficients=kwargs["transmittance_coefficients"],
        opds=kwargs["opds"],
        reflectance_coefficients=kwargs["reflectance_coefficients"],
        phase_shift=kwargs["phase_shift"],
        order=kwargs["order"],
    )
    transmittance_response = interferometer.transmittance_response(wavenumbers=kwargs["wavenumbers"])
    return transmittance_response


def main():
    nb_opd, del_opd = 320, 0.175
    opds = generate_sampled_opds(nb_opd=nb_opd, opd_step=del_opd)

    # For controlled experiment: Reference interferometer and spectrum assumptions
    reflectance_cst = 0.13
    transmittance_cst = 1 - reflectance_cst
    nb_wn = nb_opd

    # Observation
    acq_ind = 4
    observation = (transmittance_cst * nb_wn) * np.eye(N=opds.size, M=5)
    interferogram = Interferogram(data=observation, opds=opds)

    # Transmittance response information
    wavenumbers = generate_wavenumbers_from_opds(wavenumbers_num=nb_wn, del_opd=del_opd)
    reflectance_coefficients = 0.13 * np.ones(shape=(opds.size, 1))
    transmittance_coefficients = 1 - reflectance_coefficients
    phase_shift = np.zeros_like(a=opds)

    # Generate or load a transmittance response (Generated using Interferometer or Loaded from disk)
    transmittance_response = generate_transmittance_response(
        kwargs={
            "type": InterferometerType.MICHELSON,
            "opds": opds,
            "transmittance_coefficients": transmittance_coefficients,
            "reflectance_coefficients": reflectance_coefficients,
            "order": 0,
            "wavenumbers": wavenumbers,
            "phase_shift": phase_shift,
        }
    )

    # Michelson reconstruction using IDCT
    inverspectrometer_michelson = MichelsonInverter(transmittance=np.array([transmittance_cst]))
    spectrum_mich = inverspectrometer_michelson.reconstruct_spectrum(interferogram=interferogram)
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
    interferogram.visualize(axs=axs[0, 0], acq_ind=acq_ind)
    spectrum_mich.visualize(axs=axs[0, 1], acq_ind=acq_ind)

    # Plain IDCT
    idct_inverter = inversion_protocol_factory(option=InversionProtocolType.IDCT, parameters={})
    demo_idct = InversionProtocolDemo(inverter=idct_inverter)
    demo_idct.test(interferogram=interferogram, transmittance_response=transmittance_response, acq_ind=acq_ind)

    # Pseudo-inverse
    pinv_inverter = inversion_protocol_factory(option=InversionProtocolType.PSEUDO_INVERSE, parameters={})
    demo_pinv = InversionProtocolDemo(inverter=pinv_inverter)
    demo_pinv.test(interferogram=interferogram, transmittance_response=transmittance_response, acq_ind=acq_ind)

    plt.show()

    # Truncated SVD
    tsvd_inverter = inversion_protocol_factory(option=InversionProtocolType.TSVD, parameters={"penalization_ratio": 0.9})
    demo_tsvd = InversionProtocolDemo(inverter=tsvd_inverter)
    demo_tsvd.test(interferogram=interferogram, transmittance_response=transmittance_response, acq_ind=acq_ind)

    # Ridge Regression
    rr_inverter = inversion_protocol_factory(option=InversionProtocolType.RIDGE_REGRESSION, parameters={"penalization": 10})
    demo_rr = InversionProtocolDemo(inverter=rr_inverter)
    demo_rr.test(interferogram=interferogram, transmittance_response=transmittance_response, acq_ind=acq_ind)

    # Loris-Verhoeven Primal-Dual
    lv_inverter = inversion_protocol_factory(
        option=InversionProtocolType.LORIS_VERHOEVEN,
        parameters={
            # Sparsity on the spectrum = L1 norm + Identity
            "regularization_parameter": 3e2,
            "prox_functional": CTVOperator(norm=NormOperator.from_norm(norm=NormOperatorType.L1O)),  # TODO: Load directly from a given norm, e.g., CTVOperator.from_norm(etc, etc), or from an Enum, or from a factory function using an Enum
            "domain_transform": LinearOperator.from_method(method=LinearOperatorMethod.IDENTITY),  # TODO: Load directly from a given Enum, or from a factory function using an Enum
            "nb_iters": int(3e3),
        }
    )
    demo_lv = InversionProtocolDemo(inverter=lv_inverter)
    demo_lv.test(interferogram=interferogram, transmittance_response=transmittance_response, acq_ind=acq_ind)

    plt.show()


if __name__ == "__main__":
    main()
