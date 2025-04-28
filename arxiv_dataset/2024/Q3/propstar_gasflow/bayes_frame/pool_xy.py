"""
Instead of MPI parallelisation for each pixel, it is [citation needed] faster
to simply distribute pixels to different processes via pooling. The speedup is
not about the actual sampling, but the overheads are only executed once...
"""
from __future__ import division
import sys
import time
# the os.niceness will be inherited by child processes
# behave, kids!
import os
os.nice(19)
import multiprocessing
from collections import OrderedDict
import numpy as np
from pyspecnest import pool_multinest
from opencube import make_cube
from config import file_Ks, sampler_script_file, logs_dir
import resource

cut_up = 6

pool_multinest.Config.log_dir = logs_dir


def try_get_args(n, fallback, forcetype=str):
    try:
        # sys.argv[0] is some env executable path...
        arg = forcetype(sys.argv[n+1])
    except IndexError:
        arg = fallback

    return arg


def batch_sample_xy():
    # NOTE: normal dict would mess up the order of the arguments
    default_args = OrderedDict([('npeaks', 1), ('method', 'snr'), ('cut', 5),
                                ('n_cpu', 7)])

    runtime_args = {}
    for i, (argname, argval) in enumerate(default_args.items()):
        runtime_args[argname] = try_get_args(i, argval, type(argval))

    method = runtime_args.pop('method')
    n_cpu = runtime_args.pop('n_cpu')
    npeaks = runtime_args['npeaks']
    cut = runtime_args['cut']

    if method == 'snr':
        spc = make_cube() # comes with pregen snr attributes...
        sort_array = spc.snrmap
        # Trick: if you just want to go from an upper cut to a lower cut, add:
        sort_array[np.where(sort_array>cut_up)] = np.nan
    elif method == 'Bfactor':
        from astropy.io import fits
        sort_array = fits.getdata(file_Ks)[npeaks-2]
    elif method == 'chisq':
        raise NotImplementedError

    order = pool_multinest.get_xy_sorted(
        sort_array, np.indices(sort_array.shape), cut=cut)

    tasks = pool_multinest.get_tasks(n_cpu, xy_order=order, npeaks=npeaks,
                                     script=sampler_script_file)

    pool = multiprocessing.Pool(processes=n_cpu)
    # NOTE: map vs imap? imap has better ordering... see more here:
    #       [https://stackoverflow.com/questions/26520781]
    # NOTE 2: imap won't work here...
    # NOTE 3 (teresa): imap needs a "finalizer" (somewhere to give its "return") to work
    # so we added list() as a trick
    list(pool.imap(pool_multinest.work, tasks))


if __name__ == '__main__':
    st = time.time()
    batch_sample_xy()
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time/3600, ' hours')
