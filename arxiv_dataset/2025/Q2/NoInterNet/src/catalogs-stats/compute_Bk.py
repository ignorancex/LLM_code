import numpy as np
import argparse
import os

import MAS_library as MASL
from pyspectrum import pyspectrum as pySpec

import compute_Pk as cPk

def paint_density_field(pos_h, grid=256, BoxSize=1000., MAS='CIC', verbose=True):

    # define 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    
    # construct 3D density field
    MASL.MA(pos_h, delta, BoxSize, MAS, verbose=verbose)

    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0

    return delta

def compute_Bk(pos_h_T, BoxSize=1000, grid=256, step=3, Ncut=3, Nmax=26, fft='pyfftw', nthreads=1, silent=False):
    kf = 2 * np.pi / BoxSize
    
    Bks_all = pySpec.Bk_periodic(pos_h_T, Lbox=BoxSize, Ngrid=grid, step=step, Ncut=Ncut, Nmax=Nmax, fft=fft, nthreads=nthreads, silent=silent)
    Bks = np.array([Bks_all['i_k1'], 
                    Bks_all['i_k2'],
                    Bks_all['i_k3'],
                    Bks_all['p0k1'],
                    Bks_all['p0k2'],
                    Bks_all['p0k3'],
                    Bks_all['b123'],
                    Bks_all['b123_sn'],
                    Bks_all['counts']]).T

    return Bks, kf

def analyse_FoF(pos_h, output, snapdir, snapnum, grid=256, BoxSize=1000.0, dz=0., name="Bk_pyspectrum.dat", step=3, Ncut=3, Nmax=26, fft='pyfftw', nthreads=1, silent=False):

    Bks, kf = compute_Bk(pos_h.T, BoxSize, grid, step, Ncut, Nmax, fft, nthreads, silent)

    f_name = os.path.join(output, name)
    header = f'''Bk of {snapdir} group 00{snapnum:d} and displacemente Deltaz = {dz:.2f}. \n Grid {grid}, Boxsize {BoxSize} Mpc/h, kf={kf:1.4e} \n bin centers in units of kF \n k1 k2 k3 P(k1) P(k2) P(k3) B(k1,k2,k3) SN_B number of triangles in bin'''
    np.savetxt(f_name, Bks, header=header)

def main(ns):
    grid=256
    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        output_complete = os.path.join(ns.output, f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos_h = cPk.read_halo(snapdir, ns.snapnum)
        analyse_FoF(pos_h, output_complete, snapdir, ns.snapnum, grid, ns.boxsize, name=ns.name, step=ns.step, Ncut=ns.Ncut, Nmax=ns.Nmax)

    print(f"Bk without displacement in real space computed and saved in {ns.output}! Have fun!")

if __name__ == '__main__':
    desc = "Compute and save the Bk of the box without displacement"
    parser = argparse.ArgumentParser(description=desc)  

    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/FoF/', help=h)
    
    h = 'folder to save the Bk'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/Pk/no-dz/', help=h)  

    h = 'number of catalogues'
    parser.add_argument('--number', type=int, default=500, help=h)    

    h = 'first catalogue to analyse'
    parser.add_argument('--first', type=int, default=0, help=h)    

    h = 'snapshot number'
    parser.add_argument('--snapnum', type=int, default=2, choices=[0,1,2,3,4], help=h)   

    h = 'physical size of the box in Mpc/h'
    parser.add_argument('--boxsize', type=float, default=1000.0, help=h)

    h = 'Bk file name'
    parser.add_argument('--name', type=str, default='Bk_pyspectrum.dat', help=h)

    h = 'step of the k-grid (k_fund units)'
    parser.add_argument('--step', type=int, default=3, help=h)

    h = 'minimum k (k_fund units), Ncut'
    parser.add_argument('--Ncut', type=int, default=3, help=h)

    h = 'number of differnt k in the grid, Nmax'
    parser.add_argument('--Nmax', type=int, default=26, help=h)

    # and go!
    main(parser.parse_args())
