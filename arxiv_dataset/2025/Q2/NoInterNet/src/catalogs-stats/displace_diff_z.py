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

def rotatexyz(pos, thetax, thetay, thetaz, center=np.array([500,500,500], dtype=np.float32)):
    c = np.pi / 180.
    
    Rx = np.array([[1, 0, 0], [0, np.cos(thetax*c), -np.sin(thetax * c)], [0, np.sin(thetax * c), np.cos(thetax * c)]], dtype=np.float32)
    Ry = np.array([[np.cos(thetay * c), 0, np.sin(thetay * c)], [0, 1,  0], [-np.sin(thetay*c), 0, np.cos(thetay * c)]], dtype=np.float32)
    Rz = np.array([[np.cos(thetaz*c), -np.sin(thetaz * c), 0], [np.sin(thetaz * c), np.cos(thetaz * c), 0], [0, 0, 1]], dtype=np.float32)

    print(pos.shape)

    pos_center = pos - center
    pos_center = pos_center.T
    print(pos_center.shape)
    pos_center = np.dot(Rx, pos_center) #rotation around x
    pos_center = np.dot(Ry, pos_center) #rotation around y
    pos_center = np.dot(Rz, pos_center) #rotation around z

    return pos_center.T + center

def bringin_from_diffz(posz1, posz2, f, separation, axis=2, dz=1000., BoxSize=1000., seed=7):

    np.random.seed(seed)

    pz1 = np.copy(posz1)
    pz2 = np.copy(posz2)

    pz2[:,axis] += separation
    ind = np.random.randint(0, len(pz2) - 1, int(len(pz2) * f))
    print(len(ind))
    p = pz2[ind]
    p[:,axis] += dz

    #no packman in this case they are displaced inside the box or discarded
    to_keep = (p[:,axis] < BoxSize) & (p[:,axis] > 0.)
    p = p[to_keep]

    #pz1 = np.vstack((pz1, p))
    return pz1, p

def bringin_from_diffz_with_vel(posz1, posz2, velz1, velz2, f, separation, axis=2, dz=1000., BoxSize=1000., seed=7):

    np.random.seed(seed)

    pz1 = np.copy(posz1)
    pz2 = np.copy(posz2)

    pz2[:,axis] += separation
    N1, N2 = len(pz1), len(pz2)
    fr = f * N1 / (1. - f) / N2 #in this way the fraction of interlopers in the final sample is f
    ind = np.random.randint(0, len(pz2) - 1, int(len(pz2) * fr))
    print(len(ind))
    p = pz2[ind]
    v = velz2[ind]
    p[:,axis] += dz

    #no packman in this case they are displaced inside the box or discarded
    to_keep = (p[:,axis] < BoxSize) & (p[:,axis] > 0.)
    p = p[to_keep]
    v = v[to_keep]

    return pz1, p, velz1, v

def main(ns):

    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        output_complete = os.path.join(ns.output, f'{ns.dz:.2f}-iz{ns.snapnum_dz}', f'f{ns.f:.2f}', f'{i:d}')
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)
        pos = cPk.read_halo(snapdir, ns.snapnum)
        pos_iz = cPk.read_halo(snapdir, ns.snapnum_dz)

        #rotate different z
        pos_iz = rotatexyz(pos_iz, ns.thetax, ns.thetay, ns.thetaz)

        seed = i * ns.seed_m + ns.seed_q #to control the displacement realisations
        pos_t, pos_i = bringin_from_diffz(pos, pos_iz, ns.f, separation=ns.separation, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)

        pos_d = np.vstack((pos_t, pos_i))

        cPk.analyse_FoF(pos_d, output_complete, snapdir, ns.snapnum, ns.grid, ns.boxsize, dz=ns.dz)

    print(f"Pk with dz={ns.dz:.2f} computed and saved in {ns.input}! Have fun!")

if __name__ == '__main__':

    desc = "Compute and save the Pk of the box without displacement"
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')    

    h = 'the dispalcement along z'
    group.add_argument('--dz', type=float, help=h, required=True)

    h = 'interloper fraction'
    group.add_argument('--f', type=fraction_type, help=h, required=True)

    h = 'separation of the boxes in Mpc/h'
    group.add_argument('--separation', type=float, help=h, required=True)

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
    
    # and go!
    main(parser.parse_args())
    