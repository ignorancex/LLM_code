import numpy as np
import argparse
import os

import compute_Pk as cPk
import compute_Pk_RSD as cPk_RSD
import displace
import displace_diff_z as ddz

redshift = {'0': 3., '1': 2., '2': 1., '3': 0.5, '4': 0.}

def main(ns):

    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        if ns.inbox:
            output_complete = os.path.join(ns.output, f'{ns.dz:.2f}', f'f{ns.f:.2f}', f'{i:d}')
            name_t = 'Pk_pylians-no-dz.dat'
        else:
            output_complete = os.path.join(ns.output, f'{ns.dz:.2f}-iz{ns.snapnum_dz}', f'f{ns.f:.2f}', f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos, vel = cPk_RSD.read_halo(snapdir, ns.snapnum)
        
        if not ns.inbox:
            pos_iz, vel_iz = cPk_RSD.read_halo(snapdir, ns.snapnum_dz)
            #rotate different z
            pos_iz = ddz.rotatexyz(pos_iz, ns.thetax, ns.thetay, ns.thetaz)
            vel_iz = ddz.rotatexyz(vel_iz, ns.thetax, ns.thetay, ns.thetaz, center=np.array([0,0,0], dtype=np.float32)) 

        seed = i * ns.seed_m + ns.seed_q #to control the displacement realisations
        if ns.inbox:
            pos_t, pos_i, vel_t, vel_i = displace.displace_with_vel(pos, vel, ns.f, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)
            pos_d = np.vstack((pos_t, pos_i))
            vel_d = np.vstack((vel_t, vel_i))
            #apply RSD
            pos_rsd_t = cPk_RSD.apply_rsd(pos_t, vel_t, ns.boxsize, ns.Om, redshift[str(ns.snapnum)], ns.axis)
            pos_rsd = cPk_RSD.apply_rsd(pos_d, vel_d, ns.boxsize, ns.Om, redshift[str(ns.snapnum)], ns.axis)
        else:
            pos_t, pos_i, vel_t, vel_i = ddz.bringin_from_diffz_with_vel(pos, pos_iz, vel, vel_iz, ns.f, separation=ns.separation, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)

            #apply RSD
            pos_rsd_t = cPk_RSD.apply_rsd(pos_t, vel_t, ns.boxsize, ns.Om, redshift[str(ns.snapnum)], ns.axis)
            pos_rsd_i = cPk_RSD.apply_rsd(pos_i, vel_i, ns.boxsize, ns.Om, redshift[str(ns.snapnum_dz)], ns.axis)
            pos_rsd = np.vstack((pos_rsd_t, pos_rsd_i))

        cPk.analyse_FoF(pos_rsd, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize, dz=ns.dz)
        if ns.inbox:
            cPk.analyse_FoF(pos_rsd_t, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize, name=name_t) 

    print(f"Pk with dz={ns.dz:.2f} computed and saved in {ns.input}! Have fun!")

if __name__ == '__main__':

    desc = "Compute and save the Pk of the box with displacemente with RSD"
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')    

    h = 'the dispalcement along z'
    group.add_argument('--dz', type=float, help=h, required=True)

    h = 'interloper fraction'
    group.add_argument('--f', type=displace.fraction_type, help=h, required=True)

    # default argument
    h = 'separation of the boxes in Mpc/h'
    parser.add_argument('--separation', type=float, default=0, help=h)

    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/FoF/', help=h)
    
    h = 'folder to save the Pk'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/Pk/RSD/dz/', help=h)

    h = 'grid dimension'
    parser.add_argument('--grid', type=int, default=512, help=h)  

    h = 'number of catalogues'
    parser.add_argument('--number', type=int, default=500, help=h)

    h = 'first catalogue to analyse'
    parser.add_argument('--first', type=int, default=0, help=h)

    h = 'rotation angle around x-axis [deg]'
    parser.add_argument('--thetax', type=float, default=90, help=h)

    h = 'rotation angle around y-axis [deg]'
    parser.add_argument('--thetay', type=float, default=90, help=h)
    
    h = 'rotation angle around z-axis [deg]'
    parser.add_argument('--thetaz', type=float, default=0, help=h)

    h = 'snapshot number'
    parser.add_argument('--snapnum', type=int, default=2, choices=[0,1,2,3,4], help=h)

    h = 'snapshot number from which halos are taken'
    parser.add_argument('--snapnum-dz', type=int, default=1, choices=[0,1,2,3,4], help=h)

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
    
    h = 'Om_0'
    parser.add_argument('--Om', type=float, default=0.3175, help=h)
    
    # and go!
    main(parser.parse_args())