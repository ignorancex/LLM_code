from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import os

from nbodykit.source.mesh.bigfile import BigFileMesh
from nbodykit import CurrentMPIComm
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.source.catalog.uniform import RandomCatalog
from nbodykit.source.catalog import ArrayCatalog
from pmesh.pm import ParticleMesh
from nbodykit.source.mesh.array import ArrayMesh
#import pmesh
from nbodykit.utils import GatherArray

from read_utils import readout_mesh_at_cat_pos
from lsstools.nbkit03_utils import apply_smoothing


def main():
    """
    Convert grids4plots 3d grids to 2d slices.
    """

    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    ap = ArgumentParser()

    ap.add_argument('--inbasepath',
                    type=str,
                    default='$SCRATCH/perr/grids4plots/',
                    help='Input base path.')

    ap.add_argument('--outbasepath',
                    type=str,
                    default='$SCRATCH/perr/grids4plots/',
                    help='Output base path.')

    ap.add_argument('--inpath',
                    type=str,
                    default='main_calc_Perr_2020_Sep_22_18:44:31_time1600800271.dill', # laptop
                    #default='main_calc_Perr_2020_Aug_26_02:49:57_time1598410197.dill', # cluster
                    help='Input path.')

    ap.add_argument('--Rsmooth', type=float, default=2.0,
    	help='3D Gaussian smoothing applied to field.')

    # min and max index included in output. inclusive.
    ap.add_argument('--ixmin', type=int, default=5, help='xmin of output. must be between 0 and Ngrid.')
    ap.add_argument('--ixmax', type=int, default=5, help='xmax of output')
    ap.add_argument('--iymin', type=int, default=0, help='ymin of output')
    ap.add_argument('--iymax', type=int, default=-1, help='ymax of output')
    ap.add_argument('--izmin', type=int, default=0, help='zmin of output')
    ap.add_argument('--izmax', type=int, default=-1, help='zmax of output')

    cmd_args = ap.parse_args()

    verbose = True

    #####################################
    # START PROGRAM
    #####################################
    comm = CurrentMPIComm.get()
    rank = comm.rank

    path = os.path.join(os.path.expandvars(cmd_args.inbasepath), cmd_args.inpath)
    if rank==0:
        print('path: ', path)

    initialized_slicecat = False
    for fname in os.listdir(path):
        if fname.startswith('SLICE'):
            continue

        full_fname = os.path.join(path, fname)
        print('%d Reading %s' % (rank, full_fname))

        inmesh = BigFileMesh(full_fname, dataset='tmp4storage', header='header')
        Ngrid = inmesh.attrs['Ngrid']
        boxsize = inmesh.attrs['boxsize']

        # apply smoothing
        mesh = apply_smoothing(inmesh, mode='Gaussian', R=cmd_args.Rsmooth)
        del inmesh


        # convert indices to modulo ngrid
        ixmin = cmd_args.ixmin % Ngrid
        ixmax = cmd_args.ixmax % Ngrid
        iymin = cmd_args.iymin % Ngrid
        iymax = cmd_args.iymax % Ngrid
        izmin = cmd_args.izmin % Ngrid
        izmax = cmd_args.izmax % Ngrid

        # convert to boxsize units (Mpc/h)
        xmin = float(ixmin)/float(Ngrid) * boxsize
        xmax = float(ixmax+1)/float(Ngrid) * boxsize
        ymin = float(iymin)/float(Ngrid) * boxsize
        ymax = float(iymax+1)/float(Ngrid) * boxsize
        zmin = float(izmin)/float(Ngrid) * boxsize
        zmax = float(izmax+1)/float(Ngrid) * boxsize

        if not initialized_slicecat:
            # Generate catalog with positions of slice points, to readout mesh there.
            # First generate all 3D points. Then keep only subset in slice.
            # THen readout mesh at those points.


            # use pmesh generate_uniform_particle_grid
            # http://rainwoodman.github.io/pmesh/pmesh.pm.html?highlight=
            # readout#pmesh.pm.ParticleMesh.generate_uniform_particle_grid
            partmesh = ParticleMesh(BoxSize=boxsize,
                                 Nmesh=[Ngrid, Ngrid, Ngrid])
            ptcles = partmesh.generate_uniform_particle_grid(shift=0.0, dtype='f8')
            #print("type ptcles", type(ptcles), ptcles.shape)
            #print("head ptcles:", ptcles[:5,:])

            dtype = np.dtype([('Position', ('f8', 3))])

            # number of rows is given by number of ptcles on this rank
            uni_cat_array = np.empty((ptcles.shape[0],), dtype=dtype)
            uni_cat_array['Position'] = ptcles

            uni_cat = ArrayCatalog(uni_cat_array,
                                   comm=None,
                                   BoxSize=boxsize * np.ones(3),
                                   Nmesh=[Ngrid, Ngrid, Ngrid])

            del ptcles
            del uni_cat_array

            print("%d: Before cut: local Nptcles=%d, global Nptcles=%d" %
                  (comm.rank, uni_cat.size, uni_cat.csize))

            # only keep points in the slice
            uni_cat = uni_cat[ 
                  (uni_cat['Position'][:,0]>=xmin)
                & (uni_cat['Position'][:,0]<xmax)
                & (uni_cat['Position'][:,1]>=ymin)
                & (uni_cat['Position'][:,1]<ymax)
                & (uni_cat['Position'][:,2]>=zmin)
                & (uni_cat['Position'][:,2]<zmax)
                ]

            print("%d: After cut: local Nptcles=%d, global Nptcles=%d" %
                  (comm.rank, uni_cat.size, uni_cat.csize))

            initialized_slicecat = True

        # read out full 3D mesh at catalog positions. this is a numpy array
        slicecat = readout_mesh_at_cat_pos(mesh=mesh, cat=uni_cat,
            readout_window='nearest')

        if rank == 0:
            print('slicecat type:', type(slicecat))

        slicecat = GatherArray(slicecat, comm, root=0)


        if rank == 0:
            if not slicecat.shape == ((ixmax-ixmin+1)*(iymax-iymin+1)*(izmax-izmin+1),):
                raise Exception('Unexpected shape of particles read out on slice: %s' % str(
                    slicecat.shape))

            slicecat = slicecat.reshape((ixmax-ixmin+1, iymax-iymin+1, izmax-izmin+1))

            print('slicecat shape:', slicecat.shape)
            if verbose:
                print('slicecat:', slicecat)


        # convert to a mesh. assume full numpy array sits on rank 0.
        Lx = xmax-xmin
        Ly = ymax-ymin
        Lz = zmax-zmin
        if Lx == 0.: Lx = boxsize/float(Ngrid)
        if Ly == 0.: Ly = boxsize/float(Ngrid)
        if Lz == 0.: Lz = boxsize/float(Ngrid)
        BoxSize_slice = np.array([Lx, Ly, Lz])
        slicemesh = ArrayMesh(slicecat, BoxSize=BoxSize_slice, root=0)

        outshape = slicemesh.compute(mode='real').shape
        if verbose:
            print('slicemesh: ', slicemesh.compute(mode='real'))


        # write to disk
        outpath = os.path.join(
            os.path.expandvars(cmd_args.outbasepath), 
            cmd_args.inpath, 
            'SLICE_R%g_%d-%d_%d-%d_%d-%d/' % (
                cmd_args.Rsmooth, ixmin, ixmax, iymin, iymax, izmin, izmax))
        if rank == 0:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
        full_outfname = os.path.join(outpath, fname)
        if rank == 0:
            print('Writing %s' % full_outfname)
        slicemesh.save(full_outfname)
        if rank == 0:
            print('Wrote %s' % full_outfname)



if __name__ == '__main__':
    main()
