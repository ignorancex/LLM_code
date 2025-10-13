import numpy as np
import argparse
import os

import readfof
import Pk_library as PKL
import MAS_library as MASL

def read_halo(snapdir, snapnum=2):

    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                          swap=False, SFR=False, read_IDs=False)
    pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h

    return pos_h

def compute_Pk(pos, grid, BoxSize=1000.0, MAS='CIC', axis=0, verbose=True):

    # define 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    
    # construct 3D density field
    MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)
    
    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0

    # compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis=axis, MAS=MAS, threads=1, verbose=verbose)
    
    # 3D P(k)
    k       = Pk.k3D
    Pk0     = Pk.Pk[:,0] #monopole
    Pk2     = Pk.Pk[:,1] #quadrupole
    Pk4     = Pk.Pk[:,2] #hexadecapole
    Nmodes  = Pk.Nmodes3D


    return k, Pk0, Pk2, Pk4, Nmodes

def analyse_FoF(pos, output, snapdir, snapnum, grid=512, BoxSize=1000.0, MAS='CIC', axis=0, dz=0., name="Pk_pylians.dat"):
    
    k, Pk0, Pk2, Pk4, Nmodes = compute_Pk(pos, grid, BoxSize, MAS, axis, False)
    Pk = np.vstack([k, Pk0, Pk2, Pk4, Nmodes]).T

    f_name = os.path.join(output, name)
    np.savetxt(f_name, Pk, header=f'''Pk of {snapdir} group 00{snapnum:d} and displacemente Deltaz = {dz:.2f} \n k [h/Mpc] Pk0 Pk2 Pk4 Nmodes''')

def main(ns):
    
    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        output_complete = os.path.join(ns.output, f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos = read_halo(snapdir, ns.snapnum)
        analyse_FoF(pos, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize)

    print(f"Pk without displacement computed and saved in {ns.input}! Have fun!")

if __name__ == '__main__':
    desc = "Compute and save the Pk of the box without displacement"
    parser = argparse.ArgumentParser(description=desc)  

    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/FoF/', help=h)
    
    h = 'folder to save the Pk'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/Pk/no-dz/', help=h)  

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
    
    # and go!
    main(parser.parse_args())

