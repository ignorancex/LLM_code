# -*- coding: utf-8 -*-
"""
Running all field solutions required to plot the figures in plot_figures.py

Created: April 2024
Author: Bradley March
"""
# %% Python Preamble
import numpy as np
from time import time
from Packages.fR_functions import solve_field as solve_fR


def multi_fR_solver(logMvirAr: np.ndarray, logfR0Ar: np.ndarray,
                    N_r: int = 512, N_th: int = 101,
                    SD: bool = True) -> None:
    """
    Function to call solve field function for an arrays of parameters, 
    saving the results.

    Parameters:
    logMvirAr (numpy array): Array of log10 virial masses.
    logfR0Ar (numpy array): Array of log10 fR0 values.
    N_r (int): Number of radial grid points.
    N_th (int): Number of angular grid points.
    SD (bool): Flag to include stellar disc in density profile.
    """
    for logMvir in logMvirAr:
        for logfR0 in logfR0Ar:
            solve_fR(logfR0=logfR0, logMvir=logMvir,
                     N_r=N_r, N_th=N_th, SD=SD)


# %% Figure 1 & 2 solutions
starttime = time()

# defining parameters
logMvir = np.log10(1.5e12)
logfR0 = -6
N_r, N_th = 512, 101

# run solver
solve_fR(logfR0=logfR0, logMvir=logMvir, N_r=N_r, N_th=N_th, SD=True)

endtime = time()
print(f"Figure 1 & 2 solution took {endtime-starttime:.2f}")

# %% Figure 3 & 4 solutions
starttime = time()


# define arrays of parameters
dfR0 = 0.1
dMvir = 0.1
logfR0_Ar = np.arange(-8, -5+dfR0/2, dfR0)
logMvir_Ar = np.arange(10, 13.5+dMvir/2, dMvir)
N_r, N_th = 512, 101

# run solver for DM + SD solutions
multi_fR_solver(logMvir_Ar, logfR0_Ar, N_r, N_th, SD=True)

# also need DM only solutions for figure 4
multi_fR_solver(logMvir_Ar, logfR0_Ar, N_r, N_th, SD=False)

endtime = time()
print("Figure 3 & 4 solutions took {endtime-starttime:.2f}")
