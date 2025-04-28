import kerrbb  # <-- XSPEC wants to be imported first for some reason
import disksd
import numpy as np
import h5py
from pdf_trans import spin2risco, risco2spin
from multiprocessing import Pool
from itertools import product

#====================================================================================================
# Calculate grids of the relativistic disk flux correction factors gGR(r,i) and gNT(r).

#----------------------------------------------------------------------------------------------------
# Calculate gGR(r,i) on an evenly-spaced (r,i)-grid
def calc_gGR(m, d, fcol, l, lflag=1.0, eta=0.0, rflag=1.0, norm=1.0, chat=0,
             E_min=1e-2, E_max=1e3, NE=500, r_out=1e6, NrIgral=600,
             r_min=1.0, r_max=9.0, r_step=0.1, i_min=0.0, i_max=85.0, i_step=1.0, Nproc=1):
    '''
    m       - Black hole mass [Msun]
    d       - Distance [kpc]
    fcol    - Color correction factor [fcol]
    l       - Eddington-scaled disk luminosity [l=L/L_Edd]
    lflag   - kerrbb limb-darkening flag
    eta     - kerrbb torque on inner disk edge
    rflag   - kerrbb returning radiation flag
    norm    - kerrbb normalization flag
    chat    - Chatter level for what XSPEC prints to the terminal
    E_min   - Minimum photon energy [keV]
    E_max   - Maximum photon energy [keV]
    NE      - Number of energy bins for the disk continuum
    r_out   - Outer disk radius [Rg] upper bound for the disksd r-integral <-- kerrbb uses r_out = 10^6 Rg
    NrIgral - Number of bins for the r-integral calculated by disksd
    r_min, r_max, r_step - Minimum, Maximum, grid spacing <-- inner disk radius
    i_min, i_max, i_step - Minimum, Maximum, grid spacing <-- inner disk inclination
    Nproc - Number of processors to use for calculating the integrated disk spectra
    '''
    # Check that (r_min,r_max) and (i_min,i_max) are compatible with the kerrbb limits
    if (r_min < spin2risco(0.9999)): print("\nERROR: r_min < 1  not allowed by kerrbb.\n"); quit()
    if (r_max > 9.0):                print("\nERROR: r_max > 9  not allowed by kerrbb.\n"); quit()
    if (i_min < 0.0):                print("\nERROR: i_min < 0  not allowed by kerrbb.\n"); quit()
    if (i_max > 85.0):               print("\nERROR: i_max > 85 not allowed by kerrbb.\n"); quit()
    
    # Grid of inner disk radii [Rg] <-- kerrbb requires: 1.08 < r <= 9
    Nr     = int((r_max - r_min) / r_step + 1)
    r_grid = np.linspace(r_min, r_max, Nr)
    
    # Grid of disk inclination angles [deg] <-- kerrbb requires: 0 <= i <= 85
    Ni     = int((i_max - i_min) / i_step + 1)
    i_grid = np.linspace(i_min, i_max, Ni)

    # Convert r_grid --> a_grid
    a_grid = np.zeros(Nr)
    for i in range(Nr):
        a_grid[i] = risco2spin(r_grid[i])

    # Run (in parallel): kerrbb, diskNT
    global task_kerrbb, task_diskNT  # <-- HACK! Some weird thing about Pool and pickling
    def task_kerrbb(a, inc):  # kerrbb model
        return kerrbb.spectrum(a, inc, m, d, fcol, l, lflag, eta, rflag, norm, E_min, E_max, NE, chat)
    def task_diskNT(a, inc):  # Novikov & Thorne Teff profile
        return disksd.spectrum(a, inc, m, d, fcol, l, lflag, E_min, E_max, NE, r_out, NrIgral, NTSS='NT')
    with Pool(Nproc) as p:
        print("\nCALCULATING: kerrbb")
        results_kerrbb = p.starmap(task_kerrbb, product(a_grid,i_grid))
        print("\nCALCULATING: diskNT")
        results_diskNT = p.starmap(task_diskNT, product(a_grid,i_grid))

    # Collect the results
    print("\nCOLLECTING: kerrbb, diskNT")
    kerrbb_F_disk = np.zeros([Nr, Ni])
    diskNT_F_disk = np.zeros([Nr, Ni])
    k = 0
    for i in range(Nr):
        for j in range(Ni):
            kerrbb_F_disk[i,j] = results_kerrbb[k]['F_disk']
            diskNT_F_disk[i,j] = results_diskNT[k]['F_disk']
            k += 1

    # Calculate gGR on an (r,i)-grid
    print("\nCALCULATING: gGR")
    gGR = kerrbb_F_disk / diskNT_F_disk
    return gGR, r_grid, i_grid, a_grid

#----------------------------------------------------------------------------------------------------
# Calculate gNT(r) on an evenly-spaced r-grid
def calc_gNT(inc, m, d, fcol, l, lflag,
             E_min=1e-3, E_max=1e2, NE=500, r_out=1e6, NrIgral=600,
             r_min=1.0, r_max=9.0, r_step=0.1, Nproc=1):
    '''
    inc - Inner disk inclination [deg]
    *** All other inputs are the same as those in calc_gGR() above ***
    NOTE: Unlike for calc_gGR(), there is no restriction on (r_min,r_max)
    '''
    # Grid of inner disk radii [Rg] <-- kerrbb requires: 1 < r <= 9
    Nr     = int((r_max - r_min) / r_step + 1)
    r_grid = np.linspace(r_min, r_max, Nr)

    # Convert r_grid --> a_grid
    a_grid = np.zeros(Nr)
    for i in range(Nr):
        a_grid[i] = risco2spin(r_grid[i])

    # Run (in parallel): diskNT, diskSS
    global task_diskNT, task_diskSS  # <-- HACK! Some weird thing about Pool and pickling
    def task_diskNT(a):  # Novikov & Thorne Teff profile
        return disksd.spectrum(a, inc, m, d, fcol, l, lflag, E_min, E_max, NE, r_out, NrIgral, NTSS='NT')
    def task_diskSS(a):  # Shakura & Sunyaev Teff profile
        return disksd.spectrum(a, inc, m, d, fcol, l, lflag, E_min, E_max, NE, r_out, NrIgral, NTSS='SS')
    with Pool(Nproc) as p:
        print("\nCALCULATING: diskNT")
        results_diskNT = p.map(task_diskNT, a_grid)
        print("\nCALCULATING: diskSS")
        results_diskSS = p.map(task_diskSS, a_grid)

    # Collect the results
    print("\nCOLLECTING: diskNT, diskSS")
    diskNT_F_disk = np.zeros(Nr)
    diskSS_F_disk = np.zeros(Nr)
    k = 0
    for i in range(Nr):
        diskNT_F_disk[i] = results_diskNT[k]['F_disk']
        diskSS_F_disk[i] = results_diskSS[k]['F_disk']
        k += 1
        
    # Calculate gNT on an r-grid
    print("\nCALCULATING: gNT")
    gNT = diskNT_F_disk / diskSS_F_disk
    return gNT, r_grid, a_grid

#----------------------------------------------------------------------------------------------------
# Write out the results to an output HDF5 file
def write_gGR_gNT(fout,
                  gGR, gNT, r_grid, i_grid, a_grid,
                  inc, m, d, fcol, l, lflag, eta, rflag, norm,
                  E_min, E_max, NE, r_out, NrIgral):
    f = h5py.File(fout, 'w')
    # Inputs
    G_inputs = f.create_group('inputs')
    G_inputs.create_dataset('inc',     data=inc)      # [deg]
    G_inputs.create_dataset('m',       data=m)        # [Msun]
    G_inputs.create_dataset('d',       data=d)        # [kpc]
    G_inputs.create_dataset('fcol',    data=fcol)     # [-]
    G_inputs.create_dataset('l',       data=l)        # [LEdd]
    G_inputs.create_dataset('lflag',   data=lflag)    # [-]
    G_inputs.create_dataset('eta',     data=eta)      # [-]
    G_inputs.create_dataset('rflag',   data=rflag)    # [-]
    G_inputs.create_dataset('norm',    data=norm)     # [-]
    G_inputs.create_dataset('E_min',   data=E_min)    # [keV]
    G_inputs.create_dataset('E_max',   data=E_max)    # [keV]
    G_inputs.create_dataset('NE',      data=NE)       # [-]
    G_inputs.create_dataset('r_out',   data=r_out)    # [Rg]
    G_inputs.create_dataset('NrIgral', data=NrIgral)  # [-]
    # gGR(r,i), gNT(r)
    f.create_dataset('gGR',    data=gGR)     # ([Rg],[deg])
    f.create_dataset('gNT',    data=gNT)     # ([Rg],[deg])
    f.create_dataset('r_grid', data=r_grid)  # [Rg]
    f.create_dataset('i_grid', data=i_grid)  # [deg]
    f.create_dataset('a_grid', data=a_grid)  # [-]
    f.close()

#====================================================================================================
