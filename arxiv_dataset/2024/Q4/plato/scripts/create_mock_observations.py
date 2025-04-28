import argparse
from typing import Optional

import pandas as pd

from plato.planets.observation import ObservationModel
from plato.stars import filter_valid_targets
from plato.utils import get_abspath


def creat_and_save_mocks(
    num_embryos: int,
    metallicity_limit: Optional[float] = None,
) -> None:
    # Create and save mock observations
    LOPS2 = pd.read_csv(get_abspath() + "data/processed/LOPS2_targets.csv")
    LOPN1 = pd.read_csv(get_abspath() + "data/processed/LOPN1_targets.csv")

    fields = pd.concat([LOPS2, LOPN1])
    fields = filter_valid_targets(fields)
    fields = fields[
        [
            "Radius",
            "Mass",
            "[Fe/H]",
            "u1",
            "u2",
            "gaiaV",
            "n_cameras",
            "Population",
        ]
    ]

    fields["cos_i"] = 0
    fields["sigma_star"] = 10e-6
    fields = fields.rename(
        columns={
            "Radius": "R_star",
            "Mass": "M_star",
            "gaiaV": "Magnitude_V",
        }
    )

    num_mocks = 300
    obs_model = ObservationModel(load_mocks=False)

    if metallicity_limit is None:
        print(f"Creating mocks for {num_embryos} embryos with no metallicity limit.")
        obs_model.save_mocks(
            fields,
            num_embryos=num_embryos,
            num_mocks=num_mocks,
        )
    else:
        print(
            f"Creating mocks for {num_embryos} embryos with metallicity "
            f"limit of {metallicity_limit}."
        )
        obs_model.save_mocks(
            fields,
            num_embryos=num_embryos,
            metallicity_limit=metallicity_limit,
            num_mocks=num_mocks,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run observation model with specified parameters."
    )
    parser.add_argument(
        "num_embryos",
        type=int,
        help="The number of embryos.",
    )
    parser.add_argument(
        "--metallicity_limit",
        type=float,
        default=None,
        help="Float specifying the metallicity limit, or None for no limit.",
    )

    args = parser.parse_args()
    creat_and_save_mocks(args.num_embryos, args.metallicity_limit)
