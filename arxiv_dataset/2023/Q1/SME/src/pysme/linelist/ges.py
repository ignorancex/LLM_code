# -*- coding: utf-8 -*-
"""
Module for handling linelist data from the VALD3 database (http://vald.astro.uu.se/).


"""
import logging
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

from .linelist import LineList, LineListError

logger = logging.getLogger(__name__)


class GesError(LineListError):
    """Vald Data File Error"""


class GesFile(LineList):
    """Atomic data for a list of spectral lines."""

    def __init__(self, filename, medium=None):
        self.filename = filename
        linelist = self.loads(filename)

        super().__init__(linelist, lineformat=self.lineformat, medium=self.medium)
        # Convert to desired medium
        if medium is not None:
            self.medium = medium

    @staticmethod
    def load(filename):
        """
        Read line data file from the VALD extract stellar service

        Parameters
        ----------
        filename : str
            Name of the VALD linelist file to read

        Returns
        -------
        vald : ValdFile
            Parsed vald file
        """
        return GesFile(filename)

    def loads(self, filename):
        logger.info("Loading GES file %s", filename)

        hdu = fits.open(filename)
        data = hdu[1].data

        linedata = {
            "species": data["name"][:, 0] + " " + data["ion"].astype(str),
            "atom_number": np.zeros(len(data)),
            "ionization": data["ion"],
            "wlcent": data["lambda"],
            "excit": data["e_low"],
            "gflog": data["log_gf"],
            "gflog_err": data["log_gf_err"],
            "gamrad": data["rad_damp"],
            "gamqst": data["stark_damp"],
            "gamvw": data["vdw_damp"],
            "lande": data["lande_mean"],
            "depth": data["depth"],
            "reference": data["lambda_ref"],
            "lande_lower": data["lande_low"],
            "lande_upper": data["lande_up"],
            "j_lo": data["j_low"],
            "e_upp": data["e_up"],
            "j_up": data["j_up"],
            "term_lower": data["label_low"],
            "term_upper": data["label_up"],
        }

        sort = np.argsort(linedata["wlcent"])

        # We need to make sure all data is in the system defined byteorder
        # Otherwise pandas will not work properly
        for key, value in linedata.items():
            if (value.dtype.byteorder == ">" and sys.byteorder == "little") or (
                value.dtype.byteorder == "<" and sys.byteorder == "big"
            ):
                value = value.byteswap().newbyteorder()
            linedata[key] = value[sort]

        linelist = pd.DataFrame.from_dict(linedata)
        self.lineformat = "long"
        self.unit = "Angstrom"
        self._medium = "air"

        # self.citation_info += self.parse_references(refdata, fmt)

        return linelist
