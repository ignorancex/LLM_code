#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: May 2021
Author: A. P. Naik
"""
from scipy.constants import pi, G, parsec as pc
Mpc = pc * 1e6

def calc_rho_mean(h, omega_m):
    H0 = h * 100.0 * 1000.0 / Mpc
    rhocrit = 3.0 * H0**2 / (8.0 * pi * G)
    rho_mean = omega_m * rhocrit
    return rho_mean
