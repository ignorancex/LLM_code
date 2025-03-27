#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two simple, yet handy functions to calculate survey overlaps.

These Functions were used to create table 4 in Jahns et al. 2023.

Created on Wed Aug 24 15:17:14 2022
"""
import numpy as np


def ang(lat):
    """Convert latitude to radiants."""
    return (lat+90)/180*np.pi


def sky_area(high_lat=90, low_lat=-90):
    """Calculate the sky area between two latitudes in square degrees."""
    return 2*np.abs(np.cos(ang(low_lat))-np.cos(ang(high_lat)))*np.pi*(180/np.pi)**2


if __name__ == '__main__':
    print(f"The overlap fraction of ASKAP's observable sky with LSST is "
          f"{18000 / sky_area(high_lat=48, low_lat=-90)}")
