# -*- coding: utf-8 -*-
# TODO implement continuum and radial velocity tests

from itertools import product

import numpy as np
import pytest

from pysme.continuum_and_radial_velocity import (
    apply_radial_velocity_and_continuum,
    match_rv_continuum,
)


def test_match_both(testcase1):
    sme, x_syn, y_syn, rv = testcase1

    # Fix random results of the MCMC
    np.random.seed(0)

    vrad_options = ["none", "fix", "each", "whole"]
    cscale_options = ["none", "fix", "constant", "linear"]  # quadratic
    cscale_types = [
        "mask",
        "match",
        "match+mask",
        "mcmc",
        "spline",
        "spline+mask",
    ]

    # vrad_options = ["whole"]
    # cscale_options = ["linear"]  # quadratic
    # cscale_types = ["spline", "spline+mask"]
    segments = [[0], range(sme.nseg)]

    for voption, coption, ctype, segment in product(
        vrad_options, cscale_options, cscale_types, segments
    ):
        sme.vrad_flag = voption
        sme.cscale_flag = coption
        sme.cscale_type = ctype
        sme.vrad = None
        sme.cscale = None

        # Determine the radial velocity and the continuum
        cscale, cunc, vrad, vunc = match_rv_continuum(
            sme, segment, x_syn[segment], y_syn[segment]
        )

        # check that it can be applied
        smod = apply_radial_velocity_and_continuum(
            sme.wave, x_syn, y_syn, vrad, cscale, ctype, segment, copy=True
        )

        assert vrad is not None
        assert vunc is not None
        assert cscale is not None
        assert cunc is not None

        assert vrad.ndim == 1
        assert vrad.shape[0] == sme.nseg
        assert vunc.ndim == 2
        assert vunc.shape[0] == sme.nseg
        assert vunc.shape[1] == 2

        assert cscale.ndim == 2
        assert cscale.shape[0] == sme.nseg
        if ctype in ["mask", "match", "match+mask", "mcmc"]:
            assert cscale.shape[1] == sme.cscale_degree + 1
        elif ctype in ["spline", "spline+mask"]:
            assert np.allclose(cscale.shape[1], sme.cscale_degree)
        else:
            pass

        if ctype in ["mask", "match", "match+mask", "mcmc"]:
            assert cunc.ndim == 3
            assert cunc.shape[0] == sme.nseg
            assert cunc.shape[1] == sme.cscale_degree + 1
            assert cunc.shape[2] == 2
        elif ctype in ["spline", "spline+mask"]:
            assert cunc.ndim == 2
            assert cunc.shape[0] == sme.nseg
            assert np.allclose(cunc.shape[1], sme.cscale_degree)
        else:
            pass

        if voption in ["none", "fix"]:
            assert np.all(vrad == 0)
        else:
            for seg in range(sme.nseg):
                if seg in segment:
                    assert np.allclose(vrad[seg], rv, atol=1)
                else:
                    assert vrad[seg] == 0

        if ctype in ["mask", "match", "match+mask", "mcmc"]:
            if coption in ["none", "fix"]:
                assert np.all(cscale[:, -1] == 1)
                assert np.all(cscale[:, :-1] == 0)
            else:
                assert np.allclose(cscale[:, -1], 1, atol=1e-1)
                assert np.allclose(cscale[:, :-1], 0, atol=1e-1)
        elif ctype in ["spline", "spline+mask"]:
            if coption in ["none", "fix"]:
                assert np.all(cscale == 1)
            else:
                pass
        else:
            pass


# def test_nomask(testcase1):
#     sme, x_syn, y_syn, rv = testcase1
#     sme.cscale_flag = "constant"
#     sme.vrad_flag = "each"
#     sme.cscale_type = "mask"

#     sme.mask = 0
#     with pytest.warns(UserWarning):
#         rvel, vunc, cscale, cunc = match_rv_continuum(sme, 0, x_syn, y_syn)
