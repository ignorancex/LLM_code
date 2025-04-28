#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to help calculate screening, namely: 
    gradient, laplacian, volume_elements, and mass_enclosed.

Created: Dec 2023
Author: Bradley March
"""
import numpy as np
from Solvers.fR2D import fR2DSolver


def gradient(grid: fR2DSolver, f: np.ndarray,
             fout: float, magnitude: bool = True) -> np.ndarray:
    """
    Calculates the gradient of f, with outer BC fout, assuming zero 
    gradient at the centre and periodicity over theta.

    Parameters:
    grid (fR2DSolver): The grid object.
    f (numpy array): The values to calculate the gradient of.
    fout (float): The outer boundary condition.
    magnitude (bool): Whether to return the magnitude of the gradient.

    Returns:
    numpy array: The gradient of f.    
    """
    hx, hy = grid.hq, grid.hct
    r = grid.r
    sinth = np.sin(grid.theta)

    # Extend f via BC's
    f = np.vstack((f, fout * np.ones(grid.N_th)))   # outer BC f(inf) = fout
    f = np.vstack((f[0], f))                        # inner BC f'(0) = 0
    f = np.hstack((f, f[:, -1][:, None]))           # theta = 0 BC (periodic)
    f = np.hstack((f[:, 0][:, None], f))            # theta = pi BC (periodic)

    # Calculate the r and th vector components of the gradient
    r_term = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * r * hx)
    th_term = - (f[1:-1, 2:] - f[1:-1, :-2]) * sinth / (2 * r * hy)

    if magnitude is True:
        grad_mag = np.sqrt(r_term * r_term + th_term * th_term)
        return grad_mag
    else:
        return r_term, th_term


def laplacian(grid: fR2DSolver, f: np.ndarray, fout: float) -> np.ndarray:
    """
    Calculates the Laplacian of f, with outer BC fout, assuming zero 
    gradient at the centre and periodicity over theta.

    Parameters:
    grid (fR2DSolver): The grid object.
    f (numpy array): The values to calculate the Laplacian of.
    fout (float): The outer boundary condition.

    Returns:
    numpy array: The Laplacian of f.
    """
    hx, hy = grid.hq, grid.hct
    rin, rout = grid.rin, grid.rout
    sin2, sout2 = grid.stin2, grid.stout2
    r = grid.r

    # Extend f via BC's
    f = np.vstack((f, fout * np.ones(grid.N_th)))   # outer BC f(inf) = fout
    f = np.vstack((f[0], f))                        # inner BC f'(0) = 0
    f = np.hstack((f, f[:, -1][:, None]))           # theta = 0 BC (periodic)
    f = np.hstack((f[:, 0][:, None], f))            # theta = pi BC (periodic)

    # calculate terms in the discretised Laplacian
    prefac1 = 1 / (hx*hx * r*r*r)
    prefac2 = 1 / (hy*hy * r*r)
    D11 = f[:-2, 1:-1] * rin
    D12 = f[2:, 1:-1] * rout
    D13 = - f[1:-1, 1:-1] * (rin + rout)
    D21 = f[1:-1, :-2] * sin2
    D22 = f[1:-1, 2:] * sout2
    D23 = - f[1:-1, 1:-1] * (sin2 + sout2)

    D = prefac1 * (D11 + D12 + D13) + prefac2 * (D21 + D22 + D23)

    return D


def volume_elements(grid: fR2DSolver) -> np.ndarray:
    """
    Calculates the volume of each grid element.

    Parameters:
    grid (fR2DSolver): The grid object.

    Returns:
    numpy array: The volume of each grid element.
    """
    # get the inner and outer sin(theta) edges
    sth_in = np.sqrt(grid.stin2)
    sth_out = np.sqrt(grid.stout2)

    # and calculate inner and outer theta edges
    th_in = np.append(np.arcsin(sth_in[:, :grid.disc_idx+1]),
                      np.pi - np.arcsin(sth_in[:, grid.disc_idx+1:]),
                      axis=1)
    th_out = np.append(np.arcsin(sth_out[:, :grid.disc_idx]),
                       np.pi - np.arcsin(sth_out[:, grid.disc_idx:]),
                       axis=1)

    # calculate differences between coordinates
    dr = grid.rout - grid.rin
    dth = th_out - th_in

    # calculate volume element for each grid cell
    vol = 2 * np.pi * grid.r**2 * np.sin(grid.theta) * dr * dth

    return vol


def mass_enclosed(grid: fR2DSolver, rho: np.ndarray, rs: np.ndarray) -> float:
    """
    Calculates the fraction of mass within the screening radius/surface, rs.

    Parameters:
    grid (fR2DSolver): The grid object.
    rho (numpy array): The density field.
    rs (numpy array or float): The screening surface or radius.

    Returns:
    float: The fraction of mass within the screening radius/surface.
    """
    # Pass fully screened solutions through
    if isinstance(rs, (float, int)) and (rs == -1 or rs == -2 or rs is np.inf):
        return rs

    # else calculate % mass screened
    mass = rho * volume_elements(grid)
    pms = np.sum(mass[grid.r <= rs]) / np.sum(mass)

    return pms
