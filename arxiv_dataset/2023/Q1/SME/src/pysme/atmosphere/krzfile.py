# -*- coding: utf-8 -*-
import re
from os.path import basename

import numpy as np

from ..abund import Abund
from .atmosphere import Atmosphere


class KrzFile(Atmosphere):
    """Read .krz atmosphere files"""

    def __init__(self, filename, source=None):
        super().__init__()
        if source is None:
            self.source = basename(filename)
        else:
            self.source = source
        self.method = "embedded"
        self.citation_info = r"""
            @MISC{2017ascl.soft10017K,
                author = {{Kurucz}, Robert L.},
                title = "{ATLAS9: Model atmosphere program with opacity distribution functions}",
                keywords = {Software},
                year = "2017",
                month = "Oct",
                eid = {ascl:1710.017},
                pages = {ascl:1710.017},
                archivePrefix = {ascl},
                eprint = {1710.017},
                adsurl = {https://ui.adsabs.harvard.edu/abs/2017ascl.soft10017K},
                adsnote = {Provided by the SAO/NASA Astrophysics Data System}}
        """
        self.load(filename)

    def load(self, filename):
        """
        Load data from disk

        Parameters
        ----------
        filename : str
            name of the file to load
        """
        # TODO: this only works for some krz files
        # 1..2 lines header
        # 3 line opacity
        # 4..13 elemntal abundances
        # 14.. Table data for each layer
        #    Rhox Temp XNE XNA RHO

        with open(filename, "r") as file:
            header1 = file.readline()
            header2 = file.readline()
            opacity = file.readline()
            abund = [file.readline() for _ in range(10)]
            table = file.readlines()

        # Combine the first two lines
        header = header1 + header2
        # Parse header
        # vturb

        try:
            self.vturb = float(re.findall(r"VTURB=?\s*(\d)", header, flags=re.I)[0])
        except IndexError:
            self.vturb = 0

        try:
            self.lonh = float(re.findall(r"L/H=?\s*(\d+.?\d*)", header, flags=re.I)[0])
        except IndexError:
            self.lonh = 0

        self.teff = float(re.findall(r"T ?EFF=?\s*(\d+.?\d*)", header, flags=re.I)[0])
        self.logg = float(
            re.findall(r"GRAV(ITY)?=?\s*(\d+.?\d*)", header, flags=re.I)[0][1]
        )

        model_type = re.findall(r"MODEL TYPE=?\s*(\d)", header, flags=re.I)[0]
        self.model_type = int(model_type)

        model_type_key = {0: "rhox", 1: "tau", 3: "sph"}
        self.depth = model_type_key[self.model_type]
        self.geom = "pp"

        self.wlstd = float(re.findall(r"WLSTD=?\s*(\d+.?\d*)", header, flags=re.I)[0])
        # parse opacity
        i = opacity.find("-")
        opacity = opacity[:i].split()
        self.opflag = np.array([int(k) for k in opacity])

        # parse abundance
        pattern = np.genfromtxt(abund).flatten()[:-1]
        pattern[1] = 10 ** pattern[1]
        self.abund = Abund(monh=0, pattern=pattern, type="sme")

        # parse table
        self.table = np.genfromtxt(table, delimiter=",", usecols=(0, 1, 2, 3, 4))
        self.rhox = self.table[:, 0]
        self.temp = self.table[:, 1]
        self.xne = self.table[:, 2]
        self.xna = self.table[:, 3]
        self.rho = self.table[:, 4]
