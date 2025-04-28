# -*- coding: utf-8 -*-
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import yaml

from ..loader import load_datasets
from ..log import logging

_logger = logging.getLogger(__name__)


class Prefit:
    def __init__(self, config):
        self.datasets = load_datasets(
            config["data_path"],
            config["datasets"],
            config["coefficients"],
            config["use_quad"],
            config["use_theory_covmat"],
            config["use_t0"],
            False,
            config.get("default_order", "LO"),
            config.get("theory_path", None),
            config.get("rot_to_fit_basis", None),
            config.get("uv_couplings", False),
        )

    def chi2_sm(self):
        """Prints the SM chi2 per datapoint per dataset."""

        diff_sm = self.datasets.Commondata - self.datasets.SMTheory
        covmat_diff_sm = self.datasets.InvCovMat @ diff_sm

        chi2_sm = []
        cnt = 0
        for ndat_exp in self.datasets.NdataExp:
            chi2_sm.append(
                np.dot(
                    diff_sm[cnt : cnt + ndat_exp],
                    covmat_diff_sm[cnt : cnt + ndat_exp],
                )
            )
            cnt += ndat_exp

        df = pd.DataFrame(
            {
                "ndat": self.datasets.NdataExp,
                "chi2_sm/ndat": np.array(chi2_sm) / self.datasets.NdataExp,
            },
            index=self.datasets.ExpNames,
        )

        _logger.info(f"Chi2 average : {df.to_string()}")
