from __future__ import print_function, division

from nbodykit.lab import *
from lsstools.nbkit03_utils import get_cstats_string

# TODO: move to lsstools?

def read_delta_from_rho_bigfile(fname):
    """Compute delta from input file."""
    # Compute delta from input file containing rho(x).
    outfield = BigFileMesh(fname, 'Field').compute(mode='real')
    cmean = outfield.cmean()
    print('cmean before getting delta: ', cmean)
    # compute delta = rho/rhobar - 1
    outfield = outfield/cmean - 1.0
    return FieldMesh(outfield)

def read_delta_from_1plusdelta_bigfile(fname):
    """Compute delta from input file."""
    # Compute delta from input file containing 1+delta(x).
    outfield = BigFileMesh(fname, 'Field').compute(mode='real')
    cmean = outfield.cmean()
    print('cmean before getting delta: ', cmean)
    print('subtracting 1')
    # compute delta
    outfield = outfield - 1.0
    return FieldMesh(outfield)

def read_delta_from_2plusdelta_bigfile(fname):
    """Compute delta from input file."""
    # Compute delta from input file containing 2+delta(x).
    outfield = BigFileMesh(fname, 'Field').compute(mode='real')
    cmean = outfield.cmean()
    print('cmean before getting delta: ', cmean)
    print('subtracting 2')
    # compute delta
    outfield = outfield - 2.0
    return FieldMesh(outfield)



def read_vel_from_bigfile(fname):
    """Read rho in the file, don't divide by mean or subtract mean."""
    return BigFileMesh(fname, 'Field')


def readout_mesh_at_cat_pos(mesh=None, cat=None, readout_window='cic'):
    # Readout field at catalog positions.
    # mesh must be a meshsource object.

    layout = mesh.pm.decompose(cat['Position'], smoothing=readout_window)
    mesh_at_cat_pos = mesh.compute(mode='real').readout(
        cat['Position'], resampler=readout_window, layout=layout)

    return mesh_at_cat_pos

