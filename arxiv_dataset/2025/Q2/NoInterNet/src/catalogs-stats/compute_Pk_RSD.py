import numpy as np
import argparse
import os
import readfof
import redshift_space_library as RSL

import compute_Pk as cPk

redshift = {'0': 3., '1': 2., '2': 1., '3': 0.5, '4': 0.}

def read_halo(snapdir, snapnum=2):

    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                          swap=False, SFR=False, read_IDs=False)
    pos_h = FoF.GroupPos/1e3                          #Halo positions in Mpc/h
    vel_h = FoF.GroupVel*(1.0+redshift[str(snapnum)]) #Halo peculiar velocities in km/s

    return pos_h, vel_h

def apply_rsd(pos_h, vel_h, BoxSize=1000, Omega_m=0.3175, redshift=1, axis=2):
    '''Apply RSD to halo catalog'''
    Omega_l = 1. - Omega_m
    Hubble = 100. * np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)
    pos_rsd = np.copy(pos_h)
    RSL.pos_redshift_space(pos_rsd, vel_h, BoxSize, Hubble, redshift, axis)

    return pos_rsd

def main(ns):
    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        output_complete = os.path.join(ns.output, f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos, vel = read_halo(snapdir, ns.snapnum)
        pos_rsd = apply_rsd(pos, vel, ns.boxsize, ns.Om, redshift[str(ns.snapnum)], ns.axis)
        cPk.analyse_FoF(pos_rsd, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize, axis=ns.axis)

    print(f"Pk in redshift space without displacement computed and saved in {ns.input}! Have fun!")

if __name__ == '__main__':
    desc = "Compute and save the Pk of the box without displacement in redshift space"
    parser = argparse.ArgumentParser(description=desc)  

    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/FoF/', help=h)
    
    h = 'folder to save the Pk'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/Pk/RSD/no-dz/', help=h)  

    h = 'grid dimension'
    parser.add_argument('--grid', type=int, default=512, help=h)  

    h = 'number of catalogues'
    parser.add_argument('--number', type=int, default=500, help=h)    

    h = 'first catalogue to analyse'
    parser.add_argument('--first', type=int, default=0, help=h)    

    h = 'snapshot number'
    parser.add_argument('--snapnum', type=int, default=2, choices=[0,1,2,3,4], help=h)   

    h = 'physical size of the box in Mpc/h'
    parser.add_argument('--boxsize', type=float, default=1000.0, help=h)

    h = 'axis for RSD and Pk computation'
    parser.add_argument('-axis', type=int, default=2, choices=[0,1,2], help=h)
    
    h = 'Om_0'
    parser.add_argument('--Om', type=float, default=0.3175, help=h)

    # and go!
    main(parser.parse_args())

