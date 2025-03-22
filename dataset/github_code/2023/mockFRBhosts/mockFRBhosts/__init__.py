#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make some functions importable from mockFRBhosts.

Created on Tue May  9 19:38:59 2023
"""
from .generate_FRBs import generate_frbs, plot_population
from .observable import get_file_zs, get_file_galaxies, draw_galaxies, observed_bands
from .observable import observing_time, observing_time_spectrum, draw_DM