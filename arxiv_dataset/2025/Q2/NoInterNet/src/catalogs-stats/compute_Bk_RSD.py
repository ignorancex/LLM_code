import numpy as np
import argparse
import os
import time

import compute_Pk_RSD as cPk_RSD
import compute_Bk as cBk

def main(ns):
    grid=256
    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        output_complete = os.path.join(ns.output, f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos_h, vel_h = cPk_RSD.read_halo(snapdir, ns.snapnum)
        pos_rsd = cPk_RSD.apply_rsd(pos_h, vel_h, ns.boxsize, ns.Om, cPk_RSD.redshift[str(ns.snapnum)], ns.axis)

        cBk.analyse_FoF(pos_rsd, output_complete, snapdir, ns.snapnum, grid, ns.boxsize,  name=ns.name, step=ns.step, Ncut=ns.Ncut, Nmax=ns.Nmax)

    print(f"Bk without displacement in redshift space computed and saved in {ns.output}! Have fun!")

if __name__ == '__main__':
    desc = "Compute and save the Bk of the box without displacement in redshift space"
    parser = argparse.ArgumentParser(description=desc)  

    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/FoF/', help=h)
    
    h = 'folder to save the Bk'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/Pk/RSD/no-dz/', help=h)  

    h = 'number of catalogues'
    parser.add_argument('--number', type=int, default=500, help=h)    

    h = 'first catalogue to analyse'
    parser.add_argument('--first', type=int, default=0, help=h)    

    h = 'snapshot number'
    parser.add_argument('--snapnum', type=int, default=2, choices=[0,1,2,3,4], help=h)   

    h = 'physical size of the box in Mpc/h'
    parser.add_argument('--boxsize', type=float, default=1000.0, help=h)

    h = 'axis for RSD and Bk computation'
    parser.add_argument('-axis', type=int, default=2, choices=[0,1,2], help=h)
    
    h = 'Om_0'
    parser.add_argument('--Om', type=float, default=0.3175, help=h)

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