#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical constants, including cosmological parameters, all in SI units.

Created: Dec 2023
Author: Bradley March
"""
from math import pi

# Physical constants
c = 299792458  # speed of light
G = 6.6743e-11  # gravitational constant

# Distance measures
pc = 3.08567758e+16  # parsec
kpc, Mpc = pc * 1e3, pc * 1e6

# Mass measures
M_sun = 1.9885e+30  # solar mass

# Cosmological parameters
h = 0.7  # Hubble factor
H0 = h * 100.0 * 1000.0 / Mpc  # Hubble constant
omega_m = 0.3  # matter density parameter
omega_L = 1 - omega_m  # cosmological constant density parameter
rho_c = 3.0 * H0**2 / (8.0 * pi * G)  # critical density
rho_m = rho_c * omega_m  # mean cosmic matter density
rho_L = rho_c * omega_L  # cosmological constant density
R0 = 3 * omega_m * H0**2 * \
    (1 + 4 * (1 - omega_m) / omega_m) / c**2  # background curvature
