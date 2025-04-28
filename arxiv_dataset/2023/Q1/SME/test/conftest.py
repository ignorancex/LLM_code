# -*- coding: utf-8 -*-
from os.path import dirname, join

import numpy as np
import pytest
from scipy.constants import speed_of_light

from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure as SME_Struct
from pysme.synthesize import synthesize_spectrum

# TODO create various kinds of default sme structures
# then run test on all of the relevant ones


@pytest.fixture
def sme_empty():
    sme = SME_Struct()
    return sme


@pytest.fixture
def testcase1():
    c_light = speed_of_light * 1e-3

    # TODO get better test case for this
    cwd = dirname(__file__)
    fname = join(cwd, "testcase1.inp")
    sme = SME_Struct.load(fname)
    sme = synthesize_spectrum(sme)

    # Add a second segment (just for testing)
    sme.spec = [sme.spec[0], sme.spec[0]]
    sme.wave = [sme.wave[0], sme.wave[0]]
    sme.synth = [sme.synth[0], sme.synth[0]]
    sme.mask = [sme.mask[0], sme.mask[0]]
    sme.uncs = [sme.uncs[0], sme.uncs[0]]
    # sme.wran = [sme.wran[0], sme.wran[0]]

    rv = 10
    x_syn = sme.wave[0] * (1 - rv / c_light)
    y_syn = sme.synth[0]

    x_syn = np.array([x_syn, x_syn])
    y_syn = np.array([y_syn, y_syn])

    return sme, x_syn, y_syn, rv


@pytest.fixture
def sme_2segments():
    cwd = dirname(__file__)

    sme = SME_Struct()
    sme.teff = 5000
    sme.logg = 4.4
    sme.vmic = 1
    sme.vmac = 1
    sme.vsini = 1
    sme.abund = Abund(monh=0, pattern="asplund2009")
    sme.linelist = ValdFile("{}/testcase1.lin".format(cwd))
    sme.atmo.source = "marcs2012p_t2.0.sav"
    sme.atmo.method = "grid"

    sme.wran = [[6550, 6560], [6560, 6574]]

    sme.vrad_flag = "none"
    sme.cscale_flag = "none"
    return sme
