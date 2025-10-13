import numpy as np
import argparse
import os

import readfof
import vary_fraction_catalog as vfc

def main(ns):
    fractions = np.loadtxt(ns.f)

    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        
        f = fractions[i]
        if ns.inbox:
            output_complete = os.path.join(ns.output, f'inbox', 'latin_hypercube', f'{i:d}') 
        else:
            output_complete = os.path.join(ns.output, f'outbox', 'latin_hypercube', f'{i:d}')
        
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)

        f_save = os.path.join(output_complete, 'fractions.txt')
        if os.path.isfile(f_save):
            with open(f_save, 'a') as fout:
                print(f, file=fout)
        else:
            with open(f_save, 'w') as fout:
                print(f, file=fout)

        output_complete = os.path.join(output_complete, f'{ns.frealization:d}', f'groups_00{ns.snapnum:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)

        FoF = vfc.read_halo(snapdir, ns.snapnum)
        
        if not ns.inbox:
            FoF_iz = vfc.read_halo(snapdir, ns.snapnum_dz)
            
        seed = i * ns.seed_m + ns.seed_q #to control the displacement realisations
        if ns.inbox:
            FoF_c = vfc.displace_inbox(FoF, f, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)  
        else: #outbox
            FoF_c = vfc.displace_outbox(FoF, FoF_iz, f, separation=ns.separation, thetax=ns.thetax, thetay=ns.thetay, thetaz=ns.thetaz, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)

        FoF_name = f'group_tab_00{ns.snapnum}.0'
        
        readfof.writeFoFCatalog(FoF_c, os.path.join(output_complete, FoF_name))
        
if __name__ == "__main__":
    desc = "Produce and save the contaminated catalogues in cosmology Latin hypercube case"
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')    

    h = 'the dispalcement along z'
    group.add_argument('--dz', type=float, help=h, required=True)

    h = 'file containing the interloper fraction'
    group.add_argument('--f', type=str, help=h, required=True)

    h = 'fraction realization'
    group.add_argument('--frealization', type=int, help=h, required=True)

    # default argument
    h = 'separation of the boxes in Mpc/h'
    parser.add_argument('--separation', type=float, default=0, help=h)

    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_LH/FoF/', help=h)
    
    h = 'folder to save the contaminated FoF'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_LH/FoF_contaminated/', help=h)

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

    # and go!
    main(parser.parse_args())
