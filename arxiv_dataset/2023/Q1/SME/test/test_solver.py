# -*- coding: utf-8 -*-
from os.path import dirname

import numpy as np
import pytest

from pysme.sme import SME_Structure as SME_Struct
from pysme.solve import solve

cwd = dirname(__file__)
filename = "{}/testcase1.inp".format(cwd)


def test_simple():
    sme = SME_Struct.load(filename)
    sme2 = solve(sme, ["teff"])

    assert sme2.synth is not None
    assert sme2.fitresults is not None
    assert sme2.fitresults.covariance is not None
    assert isinstance(sme2.fitresults.covariance, np.ndarray)
    assert np.all(sme2.fitresults.covariance != 0)

    assert isinstance(sme2.fitresults.uncertainties, np.ndarray)
    assert len(sme2.fitresults.uncertainties) == 1
    assert sme2.fitresults.parameters[0] == "teff"
    assert sme2.fitresults.uncertainties[0] != 0

    assert np.array_equal(sme2.fitresults.covariance.shape, [1, 1])
    assert sme2.fitresults.covariance.ndim == 2

    assert sme2.fitresults.chisq is not None
    assert sme2.fitresults.chisq != 0
