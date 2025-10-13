import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.wcs import Sip
from dfreproject.sip import get_sip_coeffs  # adjust import as needed

class TestGetSipCoeffs:

    def test_no_sip_returns_none(self):
        """Should return None when WCS has no SIP"""
        wcs = WCS(naxis=2)
        assert get_sip_coeffs(wcs) is None

    def test_sip_with_forward_only(self):
        """Should extract forward SIP coefficients when no inverse present"""
        a_order, b_order = 2, 2
        a = np.zeros((a_order + 1, a_order + 1))
        b = np.zeros((b_order + 1, b_order + 1))
        a[1, 0], b[0, 1] = 1e-5, 1e-5  # simple non-zero terms

        sip = Sip(a, b, None, None, [100, 100])
        wcs = WCS(naxis=2)
        wcs.sip = sip

        coeffs = get_sip_coeffs(wcs)
        assert coeffs["a_order"] == a_order
        assert coeffs["b_order"] == b_order
        np.testing.assert_array_equal(coeffs["a"], a)
        np.testing.assert_array_equal(coeffs["b"], b)
        assert coeffs["ap_order"] == 0
        assert coeffs["bp_order"] == 0

    def test_sip_with_full_coeffs(self):
        """Should extract both forward and inverse SIP coefficients"""
        a_order, b_order = 2, 2
        ap_order, bp_order = 1, 1

        a = np.zeros((a_order + 1, a_order + 1))
        b = np.zeros((b_order + 1, b_order + 1))
        ap = np.zeros((ap_order + 1, ap_order + 1))
        bp = np.zeros((bp_order + 1, bp_order + 1))

        a[1, 0], b[0, 1] = 1e-5, 1e-5
        ap[1, 0], bp[0, 1] = -1e-5, -1e-5

        sip = Sip(a, b, ap, bp,[100, 100])
        wcs = WCS(naxis=2)
        wcs.sip = sip

        coeffs = get_sip_coeffs(wcs)
        assert coeffs["a_order"] == a_order
        assert coeffs["b_order"] == b_order
        assert coeffs["ap_order"] == ap_order
        assert coeffs["bp_order"] == bp_order
        np.testing.assert_array_equal(coeffs["a"], a)
        np.testing.assert_array_equal(coeffs["b"], b)
        np.testing.assert_array_equal(coeffs["ap"], ap)
        np.testing.assert_array_equal(coeffs["bp"], bp)

    def test_sip_with_inverse_order_zero(self):
        """Should handle inverse SIP with order 0 correctly"""
        a = np.zeros((3, 3))  # a_order = 2
        b = np.zeros((3, 3))  # b_order = 2

        # No inverse coefficients
        ap = None
        bp = None

        sip = Sip(a, b, ap, bp, [100, 100])
        wcs = WCS(naxis=2)
        wcs.sip = sip

        coeffs = get_sip_coeffs(wcs)

        assert coeffs["a_order"] == 2
        assert coeffs["b_order"] == 2
        assert coeffs["ap_order"] == 0  # verify it was interpreted as zero
        assert coeffs["bp_order"] == 0
        assert "ap" not in coeffs
        assert "bp" not in coeffs