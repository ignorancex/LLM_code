import numpy as np
import argparse
import os
import readfof

import compute_Pk as cPk
import compute_Pk_RSD as cPk_RSD
import displace
import displace_diff_z as ddz
import vary_fraction_cosmoLH as vLH
import compute_Bk as cBk

#the Big Sobol SeQuence has a different redshift dictionary:
redshift = {'0': 6., '1': 5., '2': 4., '3': 3., '4': 2., '5': 1.5, '6': 1., '7': 0.7, '8': 0.5, '9': 0.2, '10': 0.}

def read_halo(snapdir, snapnum=2):

    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                          swap=False, SFR=False, read_IDs=False)
    pos_h = FoF.GroupPos/1e3                          #Halo positions in Mpc/h
    vel_h = FoF.GroupVel*(1.0+redshift[str(snapnum)]) #Halo peculiar velocities in km/s

    return pos_h, vel_h

def main_real(ns):
    print("Real space diplacement ON")

    fractions = np.loadtxt(ns.f)
    Om, _, h, _, _  = np.loadtxt(ns.cosmo_file, unpack=True)

    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        
        f = fractions[i]
        if ns.inbox:
            output_complete = os.path.join(ns.output, f'{ns.dz:.2f}', 'f-BSQ', f'{i:d}')
        else:
            output_complete = os.path.join(ns.output, f'{ns.dz:.2f}-iz{ns.snapnum_dz}', 'f-BSQ', f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos = cPk.read_halo(snapdir, ns.snapnum)
        
        if not ns.inbox:
            pos_iz = cPk.read_halo(snapdir, ns.snapnum_dz)
            #rotate different z
            pos_iz = ddz.rotatexyz(pos_iz, ns.thetax, ns.thetay, ns.thetaz)
            
        seed = i * ns.seed_m + ns.seed_q #to control the displacement realisations
        if ns.inbox:
            pos_t, pos_i = displace.displace(pos, f, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)
        else:
            pos_t, pos_i = ddz.bringin_from_diffz(pos, pos_iz, f, separation=ns.separation, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)
        
        pos_d = np.vstack((pos_t, pos_i))

        #AP time
        pos_d = vLH.AP_distortion(pos_d, Om[i], h[i], redshift[str(ns.snapnum)], ns.boxsize)
        pos_t = vLH.AP_distortion(pos_t, Om[i], h[i], redshift[str(ns.snapnum)], ns.boxsize)

        Pk_name_t = 'Pk_pylians-no-dz.dat'
        Pk_name = f'Pk_pylians_f_{f:.2f}.dat'

        cPk.analyse_FoF(pos_d, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize, dz=ns.dz, name=Pk_name) #contaminated PK
        cPk.analyse_FoF(pos_t, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize, name=Pk_name_t) #target Pk

        if ns.Bk:
            grid = 256

            Bk_name_t = 'Bk_pyspectrum-no-dz.dat'
            Bk_name = f'Bk_pyspectrum_f_{f:.2f}.dat'

            cBk.analyse_FoF(pos_d, output_complete, snapdir, ns.snapnum, grid, ns.boxsize, dz=ns.dz, name=Bk_name)
            cBk.analyse_FoF(pos_t, output_complete, snapdir, ns.snapnum, grid, ns.boxsize, name=Bk_name_t)



    print("Real space displaced Pk computation completed! Have fun!")


def main_RSD(ns):
    print("Redshift space displacemet ON")

    fractions = np.loadtxt(ns.f)
    Om, _, h, _, _  = np.loadtxt(ns.cosmo_file, unpack=True)

    first, last = ns.first, ns.first + ns.number
    print(first, last)
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        
        f = fractions[i]
        if ns.inbox:
            output_complete = os.path.join(ns.output, f'{ns.dz:.2f}', 'f-BSQ', f'{i:d}') 
        else:
            output_complete = os.path.join(ns.output, f'{ns.dz:.2f}-iz{ns.snapnum_dz}', 'f-BSQ', f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos, vel = read_halo(snapdir, ns.snapnum)
        
        if not ns.inbox:
            pos_iz, vel_iz = read_halo(snapdir, ns.snapnum_dz)
            #rotate different z
            pos_iz = ddz.rotatexyz(pos_iz, ns.thetax, ns.thetay, ns.thetaz)
            vel_iz = ddz.rotatexyz(vel_iz, ns.thetax, ns.thetay, ns.thetaz, center=np.array([0,0,0], dtype=np.float32))

        seed = i * ns.seed_m + ns.seed_q #to control the displacement realisations
        if ns.inbox:
            pos_t, pos_i, vel_t, vel_i = displace.displace_with_vel(pos, vel, f, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)
            #apply RSD (in correct cosmology)
            pos_rsd_t = cPk_RSD.apply_rsd(pos_t, vel_t, ns.boxsize, Om[i], redshift[str(ns.snapnum)], ns.axis)
            pos_rsd_i = cPk_RSD.apply_rsd(pos_i, vel_i, ns.boxsize, Om[i], redshift[str(ns.snapnum)], ns.axis)
            #concatenate
            pos_rsd = np.vstack((pos_rsd_t, pos_rsd_i))        
        else: #outbox
            pos_t, pos_i, vel_t, vel_i = ddz.bringin_from_diffz_with_vel(pos, pos_iz, vel, vel_iz, f, separation=ns.separation, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)            
            #apply RSD (in correct cosmology)
            pos_rsd_t = cPk_RSD.apply_rsd(pos_t, vel_t, ns.boxsize, Om[i], redshift[str(ns.snapnum)],    ns.axis)
            pos_rsd_i = cPk_RSD.apply_rsd(pos_i, vel_i, ns.boxsize, Om[i], redshift[str(ns.snapnum_dz)], ns.axis)
            #concatenate
            pos_rsd = np.vstack((pos_rsd_t, pos_rsd_i))
        
        Pk_name_t = 'Pk_pylians-no-dz.dat'
        Pk_name = f'Pk_pylians_f_{f:.2f}.dat'

        #AP time
        pos_rsd   = vLH.AP_distortion(pos_rsd,   Om[i], h[i], redshift[str(ns.snapnum)], ns.boxsize)
        pos_rsd_t = vLH.AP_distortion(pos_rsd_t, Om[i], h[i], redshift[str(ns.snapnum)], ns.boxsize)

        cPk.analyse_FoF(pos_rsd, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize, dz=ns.dz, name=Pk_name) #contaminated Pk
        if ns.target:
            cPk.analyse_FoF(pos_rsd_t, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize, name=Pk_name_t) #target Pk

        if ns.Bk:
            grid = 256

            Bk_name_t = 'Bk_pyspectrum-no-dz.dat'
            Bk_name = f'Bk_pyspectrum_f_{f:.2f}.dat'

            cBk.analyse_FoF(pos_rsd, output_complete, snapdir, ns.snapnum, grid, ns.boxsize, dz=ns.dz, name=Bk_name, 
                        step=ns.step, Ncut=ns.Ncut, Nmax=ns.Nmax)
            if ns.target:
                cBk.analyse_FoF(pos_rsd_t, output_complete, snapdir, ns.snapnum, grid, ns.boxsize, name=Bk_name_t, 
                            step=ns.step, Ncut=ns.Ncut, Nmax=ns.Nmax)
    


    print("Redshift space displaced Pk computation completed! Have fun!")


if __name__ == "__main__":
    desc = "Compute and save the Pk (and Bk) of the box with displacemente with or without RSD and varying fraction for the BSQ"
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')    

    h = 'the dispalcement along z'
    group.add_argument('--dz', type=float, help=h, required=True)

    h = 'file containing the interloper fraction'
    group.add_argument('--f', type=str, help=h, required=True)

    # default argument
    h = 'separation of the boxes in Mpc/h'
    parser.add_argument('--separation', type=float, default=0, help=h)

    h = 'file containing the LH cosmologies'
    parser.add_argument('--cosmo-file', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_BSQ/BSQ_params.txt', help=h)

    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_BSQ/FoF/', help=h)
    
    h = 'folder to save the Pk'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_BSQ/Pk/RSD/dz/', help=h)

    h = 'grid dimension'
    parser.add_argument('--grid', type=int, default=512, help=h)  

    h = 'number of catalogues'
    parser.add_argument('--number', type=int, default=100, help=h)

    h = 'first catalogue to analyse'
    parser.add_argument('--first', type=int, default=0, help=h)

    h = 'rotation angle around x-axis [deg]'
    parser.add_argument('--thetax', type=float, default=90, help=h)

    h = 'rotation angle around y-axis [deg]'
    parser.add_argument('--thetay', type=float, default=90, help=h)
    
    h = 'rotation angle around z-axis [deg]'
    parser.add_argument('--thetaz', type=float, default=0, help=h)

    h = 'snapshot number'
    parser.add_argument('--snapnum', type=int, default=6, choices=[0,1,2,3,4,5,6,7,8,9,10], help=h)

    h = 'snapshot number from which halos are taken'
    parser.add_argument('--snapnum-dz', type=int, default=4, choices=[0,1,2,3,4,5,6,7,8,9,10], help=h)

    h = 'physical size of the box in Mpc/h'
    parser.add_argument('--boxsize', type=float, default=1000.0, help=h)

    h = 'slope of the seed line'
    parser.add_argument('--seed-m', type=int, default=2, help=h)

    h = 'intercept of the seed line'
    parser.add_argument('--seed-q', type=int, default=7, help=h)

    h = 'flag inbox displace'
    parser.add_argument('--inbox', action='store_true', help=h)

    h = 'axis for RSD and Pk computation'
    parser.add_argument('-axis', type=int, default=2, choices=[0,1,2], help=h)

    h = 'flag for RSD space computation'
    parser.add_argument('--RSD', action='store_true', help=h)

    h = 'flag to compute target Pk'
    parser.add_argument('--target', action='store_true', help=h)

    ###if we also compute Bk
    h = 'flag to cpmpute Bk'
    parser.add_argument('--Bk', action='store_true', help=h)

    h = 'step of the k-grid (k_fund units)'
    parser.add_argument('--step', type=int, default=6, help=h)

    h = 'minimum k (k_fund units), Ncut'
    parser.add_argument('--Ncut', type=int, default=3, help=h)

    h = 'number of differnt k in the grid, Nmax'
    parser.add_argument('--Nmax', type=int, default=26, help=h)

    # and go!
    if parser.parse_args().RSD:
        main_RSD(parser.parse_args())
    else:
        main_real(parser.parse_args())