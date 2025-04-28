from calc_gGR_gNT_grids import *
import numpy as np
import time, datetime

# XTE J1550-564

# Output HDF5 file
fout = '../data/GR/gGR_gNT_J1550.h5'

# System parameters
inc  = 74.7  # [deg]
m    = 9.10  # [Msun]
d    = 4.38  # [kpc]
fcol = 1.6   # [-]
l    = 0.12  # [L_Edd]

# Grids: r [Rg], i [deg], E [keV]
r_min  = 1.08  # <-- CAREFUL: kerrbb's a_max = 0.9999 (r_min = 1.08)
r_max  = 9.0
r_step = 0.01
i_min  = 0.0
i_max  = 85.0
i_step = 0.1
E_min  = 1e-3
E_max  = 1e2
NE     = 500

# Remaining inputs for calc_gGR() and calc_gNT()
lflag   = 1.0
eta     = 0.0
rflag   = 1.0
norm    = 1.0
r_out   = 1e6
NrIgral = 600
chat    = 0
Nproc   = 15

# Calculate the gGR(r,i)-grid
tic_gGR = time.time()
gGR, r_grid, i_grid, a_grid = \
    calc_gGR(m=m, d=d, fcol=fcol, l=l, lflag=lflag, eta=eta, rflag=rflag, norm=norm, chat=chat,
             E_min=E_min, E_max=E_max, NE=NE, r_out=r_out, NrIgral=NrIgral,
             r_min=r_min, r_max=r_max, r_step=r_step, i_min=i_min, i_max=i_max, i_step=i_step, Nproc=Nproc)
toc_gGR = time.time()

# Calculate the gNT(r)-grid
tic_gNT = time.time()
gNT, r_grid, a_grid = \
    calc_gNT(inc=inc, m=m, d=d, fcol=fcol, l=l, lflag=lflag,
             E_min=E_min, E_max=E_max, NE=NE, r_out=r_out, NrIgral=NrIgral,
             r_min=r_min, r_max=r_max, r_step=r_step, Nproc=Nproc)
toc_gNT = time.time()

# Write out the results
write_gGR_gNT(fout=fout,
              gGR=gGR, gNT=gNT, r_grid=r_grid, i_grid=i_grid, a_grid=a_grid,
              inc=inc, m=m, d=d, fcol=fcol, l=l, lflag=lflag, eta=eta, rflag=rflag, norm=norm,
              E_min=E_min, E_max=E_max, NE=NE, r_out=r_out, NrIgral=NrIgral)
              
# Calculate the total run times for gGR and gNT
tictoc_gGR = toc_gGR - tic_gGR
tictoc_gNT = toc_gNT - tic_gNT

print("\n")
print("gGR RUN TIME (HH:MM:SS) = ", str(datetime.timedelta(seconds=tictoc_gGR)))
print("gNT RUN TIME (HH:MM:SS) = ", str(datetime.timedelta(seconds=tictoc_gNT)))
print("\n")
