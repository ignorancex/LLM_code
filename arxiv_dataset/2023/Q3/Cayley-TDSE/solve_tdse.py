# python code for solving the TDSE using Cayley's operator
# Copyright (C) 2022  Ankit Kumar
# Email: akvyas1995@gmail.com
#-----------------------------------------------------------------------
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#-----------------------------------------------------------------------


import numpy as np
from math import pi
from scipy.integrate import simps
import matplotlib.pyplot

from calc_diff import *

# specify the discretisation type as 'tridiag' or 'pentadiag'
#-----------------------------------------------------------------------
disc_type = 'tridiag'

# defining the system
#-----------------------------------------------------------------------
hb = 1                                  #Planck's constant, hbar
m =	1                                   #mass of particle, m

xmin,xmax = 0,+10                     #x-limits of simulation box
dx = 0.01                               #grid size, dx 
x = np.arange(xmin,xmax+dx,dx)          #defining the position grid
J = len(x)                              #dimension of position grid

# defining the potential V (for example, HO potential)
#-----------------------------------------------------------------------
w = 0.1
V = 1/2*m*w**2*x**2
V = np.zeros(J,int)

# defining the initial wave packet (for example, Gaussian)
#-----------------------------------------------------------------------
psi = np.sqrt(2/L)*np.sin(pi*x/L)



# time limits for the simulation
#-----------------------------------------------------------------------
tmax = 100                           #simulation time limit
dt = 0.01                               #time step, dt
plot_steps = 10                         #time steps b/w two successive print statements

# setting up the numerical structure
#-----------------------------------------------------------------------
hb2m = hb**2/(2*m)                                          #value of hbar^2/2m
if disc_type == 'tridiag':
    from discretise import lhs_lumatrix_tridiag,zeta_tridiag
    lhs_lumatrix = locals()['lhs_lumatrix_tridiag']
    zeta = locals()['zeta_tridiag']
elif disc_type == 'pentadiag':
    from discretise import lhs_lumatrix_pentadiag,zeta_pentadiag
    lhs_lumatrix = locals()['lhs_lumatrix_pentadiag']
    zeta = locals()['zeta_pentadiag']
else:
    raise Exception('Specify disc_type type as tridiag/pentadiag')
lhs_lu = lhs_lumatrix(J,dx,dt,V,hb,hb2m)	#LU decomposition for the LHS matrix

# solving the TDSE
#-----------------------------------------------------------------------
t = 0
while t < tmax:
    ex = simps(x*np.abs(psi)**2,dx=dx)                      #<x>
    dpsi = diff_5pt(psi,dx,J)                               #d(psi)/dx
    ep = (-1j*hb*simps(np.conj(psi)*dpsi,dx=dx)).real       #<p>

    for j in range(plot_steps):                             #evolve plot_steps times
        psi = lhs_lu.solve(zeta(J,psi,dx,dt,V,hb,hb2m))   
    t = t + plot_steps*dt

    plt.
