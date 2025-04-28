# -*- coding: utf-8 -*-
"""
Functions to perform various utilities and analysis of the f(R) theory. 

Created: June 2022
Author: Bradley March
"""

# %% Python Preamble

# Import relevant modules
import os
import numpy as np
import h5py

# import user defined modules
from Solvers.fR2D import fR2DSolver  # 2D f(R) solver
import Packages.galaxy_relations as galf
from constants import M_sun, R0, G, c, rho_m

# get current working directory
cwd = os.getcwd()

# check if folders to store solutions exist, if not create them
if not os.path.exists(os.path.join(cwd, 'solutions', 'fR')):
    os.makedirs(os.path.join(cwd, 'solutions', 'fR'))
if not os.path.exists(os.path.join(cwd, 'solutions', 'fR_DM')):
    os.makedirs(os.path.join(cwd, 'solutions', 'fR_DM'))

# get grid min and max values
r_min = galf.r_min
r_max = galf.r_max

# %% define h5py save/load functions


def get_filename(logfR0: float, logMvir: float, N_r: int, N_th: int,
                 cwd=cwd, SD=True) -> str:
    """
    Generates the filename for given input parameters.

    Parameters:
    logfR0 (float): The log value background fR scalar field.
    logMvir (float): The log value of the virial mass.
    N_r (int): The number of radial grid coordinates.
    N_th (int): The number of angular grid coordinates.
    cwd (str, optional): The working directory, 
        if different to default directory.
    SD (bool, optional): A flag indicating whether the density profile 
        includes the stellar disc. Defaults to True.

    Returns:
    str: The generated filename.
    """
    # round to 5 decimals to prevent floating point error when saving/loading
    logMvir = np.round(float(logMvir), 5)
    logfR0 = np.round(float(logfR0), 5)
    filename = f"{logMvir}_{logfR0}_{N_r}_{N_th}.hdf5"
    if SD:
        subfolder = "fR"
    else:
        subfolder = "fR_DM"
    return os.path.join(cwd, 'solutions', subfolder, filename)


def save_solution(logfR0: float, logMvir: float, N_r: int, N_th: int,
                  fR: np.ndarray,
                  cwd: str = cwd, SD: bool = True) -> None:
    """
    Saves the fR field profile, along with its associated parameters.

    Parameters:
    logfR0 (float): The log value background fR scalar field.
    logMvir (float): The log value of the virial mass.
    N_r (int): The number of radial grid coordinates.
    N_th (int): The number of angular grid coordinates.
    fR (np.ndarray): The fR scalar field profile.
    cwd (str, optional): The working directory, 
        if different to default directory.
    SD (bool, optional): A flag indicating whether the density profile 
        includes the stellar disc. Defaults to True.
    """
    # create file
    filename = get_filename(logfR0, logMvir, N_r, N_th, cwd=cwd, SD=SD)
    file = h5py.File(filename, 'w')

    # set up header group
    header = file.create_group("Header")
    header.attrs['logfR0'] = logfR0
    header.attrs['logMvir'] = logMvir
    header.attrs['N_r'] = N_r
    header.attrs['N_th'] = N_th

    # save scalar field solution
    file.create_dataset('fR', data=fR)

    file.close()
    return


def load_solution(logfR0: float, logMvir: float, N_r: int, N_th: int,
                  cwd: str = cwd, SD: bool = True) -> np.ndarray:
    """
    Loads the fR field profile, for the associated input parameters.

    Parameters:
    logfR0 (float): The log value background fR scalar field.
    logMvir (float): The log value of the virial mass.
    N_r (int): The number of radial grid coordinates.
    N_th (int): The number of angular grid coordinates.
    cwd (str, optional): The working directory, 
        if different to default directory.
    SD (bool, optional): A flag indicating whether the density profile 
        includes the stellar disc. Defaults to True.

    Returns:
    numpy.ndarray: The fR scalar field profile.

    Raises:
    FileNotFoundError: If no solution found with the given parameters.
    """
    filename = get_filename(logfR0, logMvir, N_r, N_th, cwd=cwd, SD=SD)
    # check if solution not yet saved
    if os.path.exists(filename) is False:
        err = "No solution saved with parameters: logfR0={logfR0}, "
        err += f"logMvir={logMvir}, N_r={N_r}, N_th={N_th}, SD={SD}"
        raise FileNotFoundError(err)

    file = h5py.File(filename, 'r')
    fR = file['fR'][:]
    file.close()
    return fR


# %% solver function


def solve_field(logfR0: float, logMvir: float, N_r: int, N_th: int,
                cwd: str = cwd, SD: bool = True) -> None:
    """
    Solve and save the fR field for given input variables.

    Parameters:
    logfR0 (float): The log value background fR scalar field.
    logMvir (float): The log value of the virial mass.
    N_r (int): The number of radial grid coordinates.
    N_th (int): The number of angular grid coordinates.
    cwd (str, optional): The working directory, 
        if different to default directory.
    SD (bool, optional): A flag indicating whether the density profile 
        includes the stellar disc. Defaults to True.
    """
    # define filename...
    filename = get_filename(logfR0, logMvir, N_r, N_th, cwd=cwd, SD=SD)
    # ... and check if solution already exists
    if os.path.exists(filename):
        print('fR solution already exists!')
        return

    # derive model parameters
    fR0 = -10**logfR0
    Mvir = 10**logMvir * M_sun

    # Setting up the grid structures
    fR_grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

    # Set up the density
    rhos = galf.get_densities(Mvir, fR_grid.r, fR_grid.theta,
                              splashback_cutoff=True, total=True)
    if SD:
        drho = rhos['total']
    else:
        drho = rhos['DM']
    rho = rho_m + drho
    fR_grid.set_density(rho)

    # Running the solver
    fR_grid.solve(fR0=fR0, verbose=True, tol=1e-14, imin=100, imax=1000000)

    # calculate fR
    fR = fR_grid.usq * fR0

    save_solution(logfR0, logMvir, N_r, N_th, fR, SD=SD)

    print(f"Saved: logfR0={logfR0}, logMvir={logMvir}, "
          + f"N_r={N_r}, N_th={N_th}, SD={SD}")


# %% define functions for each fR EoM term


def delta_R(fR: np.ndarray, fR0: float) -> np.ndarray:
    """
    Calculate delta R term in the fR EoM.

    Parameters:
    fR (numpy array): The values of the fR scalar field.
    fR0 (float): The background value of the fR scalar field.

    Returns:
    numpy array: The delta R term, with the same shape as fR.
    """
    return R0 * (np.sqrt(fR0 / fR) - 1)


def delta_rho_term(drho: np.ndarray) -> np.ndarray:
    """
    Calculate delta rho term in the fR EoM.

    Parameters:
    drho (numpy array): The density perturbation.

    Returns:
    numpy array: The delta rho term, with the same shape as drho
    """
    return drho * 8 * np.pi * G / c**2


def get_curvature_density_ratio(fR: np.ndarray, fR0: float, drho: np.ndarray
                                ) -> np.ndarray:
    """
    Calculate the ratio of the two terms in the fR EoM. Expect this ratio 
    to be 1 in the screened region and zero in the unscreened region.

    Parameters:
    fR (numpy array): The values of the fR scalar field.
    fR0 (float): The background value of the fR scalar field.
    drho (numpy array): The density perturbation.

    Returns:
    numpy array: The ratio of the curvature and density terms, 
        with the same shape as fR & drho.
    """
    dR = delta_R(fR, fR0)
    drho_term = delta_rho_term(drho)
    # calculate ratio, masking invalid numbers from division by zero
    curvature_density_ratio = np.ma.divide(dR, drho_term)
    return curvature_density_ratio


# %% Calculate fR screening radius


def calc_rs(fR: np.ndarray, fR0: float, drho: np.ndarray,
            grid: fR2DSolver = None,
            threshold: float = 0.9,
            unscrthreshold: float = 1e-3) -> np.ndarray:
    """
    Calculate the position of the screening surface using the ratio of 
    the two EoM terms. Unscreened solutions are determined by a 
    threshold on the central field value.

    Parameters:
    fR (numpy array): The values of the fR scalar field.
    fR0 (float): The background value of the fR scalar field.
    drho (numpy array): The density perturbation.
    grid (fR2DSolver, optional): The grid structure for the fR and 
        density field.
    threshold (float, optional): The threshold for the 
        curvature/density ratio screening radius calculation.
    unscrthreshold (float, optional): The threshold for the central 
        field value to determine unscreened solutions.

    Returns:
    numpy array: The screening radius. Note that -1 indicates a fully
        unscreened solution.
    """
    # check if fully unscreened
    if all(fR[0, :] / fR0 > unscrthreshold):
        rs = -1
        return rs

    if grid is None:
        N_r, N_th = fR.shape
        grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

    # determine the screened region (where lap(fR) = 0 --> dR/drho = 1)
    curvature_density_ratio = get_curvature_density_ratio(fR, fR0, drho)
    # find values above a threshold
    inds = np.argmin(curvature_density_ratio >= threshold, axis=0)
    # get screening radius
    rs = grid.r[inds, 0]
    return rs


def get_rs(logfR0: float, logMvir: float, N_r: int, N_th: int,
           threshold: float = 0.9,
           unscrthreshold: float = 1e-3,
           SD: bool = True) -> np.ndarray:
    """
    Calculate the screening surfaces for given input parameters.

    Parameters:
    logfR0 (float): The log value background fR scalar field.
    logMvir (float): The log value of the virial mass.
    N_r (int): The number of radial grid coordinates.
    N_th (int): The number of angular grid coordinates.
    threshold (float, optional): The threshold for the 
        curvature/density ratio screening radius calculation.
    unscrthreshold (float, optional): The threshold for the central 
        field value to determine unscreened solutions.
    SD (bool, optional): A flag indicating whether the density profile 
        includes the stellar disc. Defaults to True.

    Returns:
    numpy array: The screening radius. Note that -1 indicates a fully 
        unscreened solution.    
    """
    # load field profile
    fR = load_solution(logfR0, logMvir, N_r, N_th, SD=SD)

    # calculate model parameters
    fR0 = -10**logfR0
    Mvir = M_sun * 10**logMvir

    # set up grid structure
    grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

    # calculate density
    rhos = galf.get_densities(Mvir, grid.r, grid.theta,
                              splashback_cutoff=True, total=True)

    # calculate screening radius
    if SD:
        return calc_rs(fR, fR0, rhos['total'], grid, threshold, unscrthreshold)
    else:
        return calc_rs(fR, fR0, rhos['DM'], grid, threshold, unscrthreshold)
