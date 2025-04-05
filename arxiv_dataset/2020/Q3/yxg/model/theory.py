import numpy as np
from .power_spectrum import hm_ang_power_spectrum


def get_theory(p, dm, cosmo, return_separated=False,
               include_1h=True, include_2h=True,
               selection=None,
               hm_correction=None, **kwargs):
    """Computes the theory prediction used in the MCMC.

    Args:
        p (:obj:`ParamRun`): parameters for this run.
        dm (:obj:`DataManager`): data manager for this set of
            correlations.
        cosmo (:obj:`ccl.Cosmology`): cosmology object.
        return_separated (bool): return output power spectra
            in separate arrays.
        hm_correction(:obj:`HalomodCorrection`): halo model correction
            factor.
        selection (function): selection function in (M,z) to include
            in the calculation. Pass None if you don't want to select
            a subset of the M-z plane.
        **kwargs: model parameters
    """
    # q = np.array([1.75099144e-03, 6.52856132e-04, 3.23515507e-04, 1.92588001e-04,
    #               1.27727957e-04, 9.11227607e-05, 6.70197694e-05, 4.97078576e-05,
    #               3.72736687e-05, 2.84091847e-05, 2.20225491e-05, 1.74933640e-05,
    #               1.43154558e-05, 1.19448115e-05, 1.01280765e-05])

    nz_default = p.get('mcmc')['nz_points_g']
    use_zlog = p.get('mcmc')['z_log_sampling']

    cls_out = []
    for tr, ls, bms in zip(dm.tracers, dm.ells, dm.beams):
        profiles = (tr[0].profile, tr[1].profile)
        if tr[0].name == tr[1].name:
            zrange = tr[0].z_range
            zpoints = nz_default
        else:
            # At least one of them is g
            if tr[0].type == 'g' or tr[1].type == 'g':
                if tr[0].type != tr[1].type:  # Only one is g
                    # Pick which one is g.
                    # That one governs the redshift slicing
                    t = tr[0] if tr[0].type == 'g' else tr[1]
                    zrange = t.z_range
                    zpoints = nz_default
                else:  # Both are g, but different samples
                    # Get a range that encompasses both N(z) curves
                    zrange = [min(tr[0].z_range[0], tr[1].z_range[0]),
                              max(tr[0].z_range[1], tr[1].z_range[1])]
                    # Get the minimum sampling rate of both curves
                    dz = min((tr[0].z_range[1]-tr[0].z_range[0])/nz_default,
                             (tr[1].z_range[1]-tr[1].z_range[0])/nz_default)
                    # Calculate the point preserving that sampling rate
                    zpoints = int((zrange[1]-zrange[0])/dz)
            else:  # Only other option right now is for both of them to be y
                zrange = tr[0].z_range
                zpoints = nz_default

        cl = hm_ang_power_spectrum(cosmo, ls, profiles,
                                   zrange=zrange, zpoints=zpoints,
                                   zlog=use_zlog, hm_correction=hm_correction,
                                   include_1h=include_1h,
                                   include_2h=include_2h,
                                   selection=selection,
                                   **kwargs)
        # cl = q
        if cl is None:
            return None

        cl *= bms  # Multiply by beams
        if return_separated:
            cls_out.append(cl)
        else:
            cls_out += cl.tolist()

    cls_out = np.array(cls_out)
    return cls_out
