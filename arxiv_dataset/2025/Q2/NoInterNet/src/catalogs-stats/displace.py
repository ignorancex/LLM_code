import numpy as np
import argparse
import os

import compute_Pk as cPk

def fraction_type(arg):
    """ Type function for argparse - a float within [0,1] """
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0. or f > 1.:
        raise argparse.ArgumentTypeError("Argument must be <= " + str(1) + "and >= " + str(0))
    return f

def displace(pos, f, axis=2, dz=100.0, BoxSize=1000.0, seed=7):

    np.random.seed(seed)
    p = np.copy(pos)
    ind = np.random.randint(0, len(p) - 1, int(len(p) * f))
    print(len(ind))

    pt = np.delete(p, ind, axis=0)
    pi = p[ind,:]

    pi[:, axis] += dz
    if dz > 0:
        packman = pi[:,axis] > BoxSize
        pi[packman,axis] -= BoxSize
    elif dz < 0:
        packman = pi[:,axis] < 0
        pi[packman,axis] += BoxSize
    to_keep = (pi[:,2] < BoxSize) & (pi[:,2] > 0.) #this is needed if dz > BoxSize, we are removing object that remains out after packman
    pi = pi[to_keep]
    return pt, pi

def displace_with_vel(pos, vel, f, axis=2, dz=100.0, BoxSize=1000.0, seed=7):

    np.random.seed(seed)
    p = np.copy(pos)
    ind = np.random.randint(0, len(p) - 1, int(len(p) * f))
    print(len(ind))

    pt = np.delete(p, ind, axis=0)
    pi = p[ind,:]
    vt = np.delete(vel, ind, axis=0)
    vi = vel[ind,:]

    pi[:, axis] += dz
    if dz > 0:
        packman = pi[:,axis] > BoxSize
        pi[packman,axis] -= BoxSize
    elif dz < 0:
        packman = pi[:,axis] < 0
        pi[packman,axis] += BoxSize
    to_keep = (pi[:,2] < BoxSize) & (pi[:,2] > 0.) #this is needed if dz > BoxSize, we are removing object that remains out after packman
    pi = pi[to_keep]
    vi = vi[to_keep]
    return pt, pi, vt, vi

def main(ns):

    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        output_complete = os.path.join(ns.output, f'{ns.dz:.2f}', f'f{ns.f:.2f}', f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos = cPk.read_halo(snapdir, ns.snapnum)

        seed = i * ns.seed_m + ns.seed_q #to control the displacement realisations
        pos_t, pos_i = displace(pos, ns.f, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)

        pos_d = np.vstack((pos_t, pos_i))

        cPk.analyse_FoF(pos_d, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize)

    print(f"Bk with dz={ns.dz:.2f} computed and saved in {ns.input}! Have fun!")

if __name__ == '__main__':

    desc = "Compute and save the Bk of the box"
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')    

    h = 'the dispalcement along z'
    group.add_argument('--dz', type=float, help=h, required=True)

    h = 'interloper fraction'
    group.add_argument('--f', type=fraction_type, help=h, required=True)

    # default argument
    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/FoF/', help=h)
    
    h = 'folder to save the Pk'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/Pk/dz/', help=h)

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

    h = 'slope of the seed line'
    parser.add_argument('--seed-m', type=int, default=2, help=h)

    h = 'intercept of the seed line'
    parser.add_argument('--seed-q', type=int, default=7, help=h)
    
    # and go!
    main(parser.parse_args())
    