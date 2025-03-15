from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
from nbodykit.lab import *
import numpy as np
import os

def main():
    """
    resample bigfile grid.

    Note 29/12/2019: not working b/c IC files are in old bigfile format.
    need to use nbk01 code in psirec folder.

    # 13 March 2020: Does not seems to run on sns cluster, issue with BigFileCatalog
    # in nbk01 and nbk03.
    # INSTEAD: Use main_create_LinearMesh.py


    Example usage
    -------------
    conda activate nbodykit-0.3.7-env
    srun -n 32 -t 1:00:00 python main_resample_BigFileGrid_nbk03.py 256
    """

    # command line args
    ap = ArgumentParser()

    ap.add_argument('out_Nmesh',
                    type=int,
                    default=1024,
                    help='Nmesh for output density')

    # optional arguments
    ap.add_argument('--in_Nmesh',
                    type=int,
                    default=1536,
                    help='Nmesh for input grid')

    ap.add_argument('--SimSeed',
                    type=int,
                    default=400,
                    help='Simulation seed to load.')

    # copy args
    cmd_args = ap.parse_args()
    opts = OrderedDict()
    opts['out_Nmesh'] = cmd_args.out_Nmesh
    opts['in_Nmesh'] = cmd_args.in_Nmesh
    sim_seed = cmd_args.SimSeed


    #####################################
    # MORE OPTIONS (not on command line)
    #####################################

    # input grid
    opts['in'] = {}       
    if False:
        # jerryousims
        opts['in_path'] = os.path.expandvars(
            '$SCRATCH/lssbisp2013/jerryou/baoshift/run5//00000%d-120-0%d-500.0-wig/IC/' % (sim_seed, opts['in_Nmesh']))
        opts['in_dataset'] = 'LinearDensityLowRes'
    else:
        # ms_gadget sims (does not seem to work, possibly b/c uses new bigfile format)
        opts['in_path'] = os.path.expandvars(
            '$SCRATCH/lss/ms_gadget/run4/00000%d-0%d-500.0-wig/IC/' % (sim_seed, opts['in_Nmesh']))
        opts['in_dataset'] = '1/ICDensity'
        

    # output grid
    opts['out_path'] = opts['in_path']
    opts['out_dataset'] = '%s_Ng%d' % (opts['in_dataset'], opts['out_Nmesh'])

    
    
    # ################################################################################
    # Resample grid using nbodykit 0.3.x
    # ################################################################################

    print('Read %s' % opts['in_path'])

    bfmesh = BigFileMesh(opts['in_path'], opts['in_dataset'])
    if bfmesh.comm.rank == 0:
        print("Successfully read %s" % opts['in_path'])
    rfield = bfmesh.paint(mode='real', Nmesh=opts['out_Nmesh'])

    out_meshsource = FieldMesh(rfield)

    if out_meshsource.comm.rank == 0:
        print("Writing %s ..." % opts['out_path'])

    out_meshsource.save(opts['out_path'], dataset=opts['out_dataset'], mode='real')

    if out_meshsource.comm.rank == 0:
        print("Wrote %s" % opts['out_path'])





if __name__ == '__main__':
    main()




    


    
