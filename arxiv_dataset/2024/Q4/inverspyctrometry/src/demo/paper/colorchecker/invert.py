from __future__ import annotations

from dataclasses import dataclass, replace

import matplotlib.pyplot as plt

from src.database.experiments import ExperimentSchema
from src.direct_model.interferometer import simulate_interferogram
from src.interface.configuration import load_config
from src.inverse_model.protocols import inversion_protocol_factory


@dataclass(frozen=True)
class ExperimentSimple:
    exp_id: int
    ds_id: int
    char_id: int
    nl_idx: int
    inv_ids: list[int]

    @classmethod
    def from_experiment_schema(cls, experiment_schema: ExperimentSchema) -> ExperimentSimple:
        return ExperimentSimple(
            exp_id=experiment_schema.id,
            ds_id=experiment_schema.dataset_ids[0],
            char_id=experiment_schema.interferometer_ids[0],
            nl_idx=experiment_schema.noise_level_indices[0],
            inv_ids=experiment_schema.inversion_protocol_ids,
        )


@dataclass(frozen=True)
class ExtrapolationOptions:
    case: int
    description: str
    opds_resampler: str
    ifm_kind: str
    ifm_fill_value: any


def main():
    exp_id = 8
    extrap_opts = ExtrapolationOptions(
        case=0,
        description="Concatenate missing OPDs and extrapolate the interferogram using Fourier Series.",
        opds_resampler="concatenate_missing",
        ifm_kind="fourier",
        ifm_fill_value="extrapolate",
    )
    acq_idx = 0

    config = load_config()
    db = config.database()

    exp = ExperimentSimple.from_experiment_schema(experiment_schema=db.experiments[exp_id])

    spectrum_ref = db.dataset_spectrum(ds_id=2)
    spectrum_ref = replace(spectrum_ref, data=spectrum_ref.data[:, 13:14])

    interferogram_ref = db.dataset_interferogram(ds_id=exp.ds_id)
    interferogram_ref = interferogram_ref.extrapolate(
        support_resampler=extrap_opts.opds_resampler,
        kind=extrap_opts.ifm_kind,
        fill_value=extrap_opts.ifm_fill_value,
    )
    interferogram_ref = interferogram_ref.rescale(new_max=1., axis=-2)

    characterization = db.characterization(characterization_id=exp.char_id)
    characterization = characterization.extrapolate_opds(support_resampler=extrap_opts.opds_resampler)
    # wavenumbers = db.characterization_wavenumbers(char_id=exp.char_id)
    wavenumbers = spectrum_ref.wavenumbers
    transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers)
    transfer_matrix = transfer_matrix.rescale(new_max=1., axis=None)

    interferogram_sim = simulate_interferogram(transmittance_response=transfer_matrix, spectrum=spectrum_ref)
    interferogram_sim = interferogram_sim.rescale(new_max=1., axis=-2)

    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(10, 7))
    interferogram_ref.visualize(axs=axs[0, 0], acq_ind=acq_idx, color="C0", is_sort_opds=True, label="Reference")
    interferogram_sim.visualize(axs=axs[0, 0], acq_ind=acq_idx, color="C1", is_sort_opds=True, label="Simulated")
    plt.show()

    for inv_id in exp.inv_ids:
        lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=inv_id)
        for lambdaa in lambdaas:
            inverter = db.inversion_protocol(inv_protocol_id=inv_id, lambdaa=lambdaa)
            spectrum_rec = inverter.reconstruct_spectrum(
                interferogram=interferogram_ref,
                transmittance_response=transfer_matrix,
            )

            spectrum_rec, spectrum_ref = spectrum_rec.match_stats(
                reference=spectrum_ref, axis=-2, is_rescale_reference=True,
            )

            fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(20, 4))
            interferogram_ref.visualize(axs=axs[0, 0], acq_ind=acq_idx, color="C0", is_sort_opds=True)
            interferogram_sim.visualize(axs=axs[0, 0], acq_ind=acq_idx, color="C1", is_sort_opds=True)
            transfer_matrix.visualize(fig=fig, axs=axs[0, 1])
            spectrum_ref.visualize(axs=axs[0, 2], acq_ind=acq_idx, label="Reference", color="C0")
            spectrum_rec.visualize(axs=axs[0, 2], acq_ind=acq_idx, label="Reconstructed", color="C1")
            plt.suptitle(t=f"{db.inversion_protocols[inv_id].title.upper()} - " + r"$\lambda$" + f" = {lambdaa:.4f}")
        plt.show()


if __name__ == "__main__":
    main()
