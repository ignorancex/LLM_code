from __future__ import print_function, division

import glob
import os
import random

def get_densities_needed_for_trf_fcns(trf_specs, include_target=True):
    """Get list of all densities actually needed for trf fcns. Makes sure that
    we only load the densities from disk that we need.

    Parameters
    ----------
    trf_specs : list of ``TrfSpec`` objects.
        Transfer functions specs for all models to be used.

    Returns
    -------
    densities_needed_for_trf_fcns : list of strings
        Names of fields needed for the transfer functions.
    """
    needed = set()
    for ts in trf_specs:
        # linear sources
        if hasattr(ts, 'linear_sources'):
            if ts.linear_sources is not None:
                needed.update(set(ts.linear_sources))

        # fixed linear sources
        if hasattr(ts, 'fixed_linear_sources'):
            needed.update(set(ts.fixed_linear_sources))

        # field_to_smoothen_and_square for quadratic fields, and target field
        for f in [ts.field_to_smoothen_and_square,
                  ts.field_to_smoothen_and_square2
                  ]:
            if f is not None:
                needed.add(f)

        # fields contributing to target
        if include_target:
            if ts.target_field is not None:
                needed.add(ts.target_field)
            if hasattr(ts, 'target_spec'):
                if hasattr(ts.target_spec, 'linear_target_contris'):
                    needed.update(set(ts.target_spec.linear_target_contris))

    return list(needed)


def make_cache_path(cache_base_path, comm):
    """
    Unique id for cached files so we can run multiple instances simultaneously.
    Rank 0 gets the cache id and then broadcasts it to other ranks
    """
    cacheid = None
    if comm.rank == 0:
        file_exists = True
        while file_exists:
            cacheid = ('CACHE%06x' % random.randrange(16**6)).upper()
            cache_path = os.path.join(cache_base_path, cacheid)
            file_exists = (len(glob.glob(cache_path)) > 0)
        # create cache path
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
    # broadcast cacheid to all ranks
    cacheid = comm.bcast(cacheid, root=0)
    # get cache path on all ranks
    cache_path = os.path.join(cache_base_path, cacheid)
    return cache_path
