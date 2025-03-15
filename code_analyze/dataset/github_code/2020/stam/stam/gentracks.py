import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.table import Table
from matplotlib.path import Path
from .getmodels import colname


def get_isotrack(models, vals, params=("mass", "mh"),
                 mass_res=0.007, absmag_res=0.01, age_res=0.1, log_age_res=0.05, mh_res=0.05, stage=1, sort_by=None,
                 mass_min=0, mass_max=np.inf, absmag_min=-np.inf, absmag_max=np.inf, age_min=0, age_max=np.inf, mh_min=-np.inf, mh_max=np.inf,
                 stage_min=0, stage_max=np.inf, color_filter1="G_BPmag", color_filter2="G_RPmag",
                 mag_filter="Gmag", return_idx=False, return_table=False):
    """
    get_isotrack(models, vals, params=("mass", "mh"),
                 mass_res=0.007, absmag_res=0.01, age_res=0.1, log_age_res=0.05, mh_res=0.05, stage=1, sort_by=None,
                 mass_min=0, mass_max=np.inf, absmag_min=-np.inf, absmag_max=np.inf, age_min=0, age_max=np.inf, mh_min=-np.inf, mh_max=np.inf,
                 stage_min=0, stage_max=np.inf, color_filter1="G_BPmag", color_filter2="G_RPmag",
                 mag_filter="Gmag", return_idx=False, return_table=False)

    Get a specific stellar evolution track, with two out of four parameters fixed (mass, age, metallicity, absmag).
    NOTE: This function doesn't interpolate between tracks, just return all tracks within the mass/age/metallicity/absmag bins.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    vals : array_like (of length 2)
        Values of the fixed parameters (units: [mass] = Msun, [age] = Gyr, [mh] = dex).
    params : tuple (of length 2), optional
        Fixed parameters names (default: ("mass", "mh")).
    mass_res : float, optional
        Mass resolution, in Msun (default: 0.007 Msun). If negative - treat as fractional.
    absmag_res : float, optional
        Absolute magnitude resolution, in mag (default: 0.01 mag). If negative - treat as fractional.
    age_res : float, optional
        Age resolution, in Gyr (default: 0.1 Gyr). If negative - treat as fractional.
    log_age_res : float, optional
        log(age/yr) resolution, in dex, if using `log_age` as one of the `params` (default: 0.05 dex).
    mh_res : float, optional
        Metallicity resolution, in dex (default: 0.05 dex)
    stage : int, optional
        Stellar evolution stage label (None = don't select a specific stage, 0 = pre-MS, 1 = MS, etc.; default: 1).
    mass_min : float, optional
        Minimum mass to consider, in Msun (if no fixed mass was chosen; default: 0 Msun).
    mass_max : float, optional
        Maximum mass to consider, in Msun (if no fixed mass was chosen; default: 1 Msun).
    absmag_min : float, optional
        Minimum absolute magnitude to consider, in mag (if no fixed absolute magnitude was chosen; default: `-np.inf`).
    absmag_max : float, optional
        Maximum absolute magnitude to consider, in mag (if no fixed absolute magnitude was chosen; default: `np.inf`).
    age_min : float, optional
        Minimum age to consider, in Gyr (if no fixed age was chosen; default: 0 Gyr).
    age_max : float, optional
        Maximum age to consider, in Gyr (if no fixed age was chosen; default: `np.inf` Gyr).
    mh_min : float, optional
        Minimum [M/H] to consider, in dex (if no fixed metallicity was chosen; default: `-np.inf`).
    mh_max : float, optional
        Maximum [M/H] to consider, in dex (if no fixed metallicity was chosen; default: `np.inf`).
    stage_min : int, optional
        Minimum stellar evolution stage label to consider (if no fixed stage was chosen; default: 0).
    stage_max : int, optional
        Maximum stellar evolution stage label to consider (if no fixed stage was chosen; default: `np.inf`).
    color_filter1 : str, optional
        Bluer band to use for the color calculation (default: `G_BPmag`).
    color_filter2 : str, optional
        Redder band to use for the color calculation (default: `G_RPmag`).
    mag_filter : str, optional
        Which band to use for the magnitude axis (default: `Gmag`).
    return_idx : bool, optional
        If true, return also the indices of the relevant rows in `models` (default: False).
    return_table : bool, optional
        If true, return tracks in `astropy` `Table` format (default: False).

    Returns
    -------
    color1 : array_like
        Chosen stellar evolution track `color_filter1` magnitude
    color2 : array_like
        Chosen stellar evolution track `color_filter2` magnitude
    absmag : array_like
        Chosen stellar evolution track `mag_filter` absolute magnitude
    mass : array_like
        Chosen stellar evolution track mass, in Msun
    mh : array_like
        Chosen stellar evolution track metallicity ([M/H]), in dex
    age : array_like
        Chosen stellar evolution track age, in Gyr
    idx : array_like
        The indices of the relevant rows in `models` (if `return_idx=True`).
    """

    if "mass" in params:
        mass = vals[params.index("mass")]
        if mass_res is None:
            # take nearest track
            masses = np.unique(models[colname("m0")])
            i = np.argmin(np.abs(mass - masses))
            mass_idx = models[colname("m0")] == masses[i]
        elif mass_res < 0:
            # treat as fractional
            mass_res = -mass_res
            mass_idx = ((mass*(1 - mass_res)) <= models[colname("m0")]) & (models[colname("m0")] < (mass*(1 + mass_res)))
        else:
            mass_idx = ((mass - mass_res) <= models[colname("m0")]) & (models[colname("m0")] < (mass + mass_res))
        if ~np.any(mass_idx):
            raise Exception(f"No tracks found for mass {mass}+/-{mass_res} Msun!")
    else:
        if mass_res < 0:
            # treat as fractional
            mass_res = -mass_res
            mass_idx = (mass_min*(1 - mass_res) <= models[colname("m0")]) & (models[colname("m0")] <= mass_max*(1 + mass_res))
        else:
            mass_idx = (mass_min - mass_res <= models[colname("m0")]) & (models[colname("m0")] <= mass_max + mass_res)
        if ~np.any(mass_idx):
            raise Exception(f"No tracks found in mass range {mass_min}-{mass_max} Msun!")

    if "absmag" in params:
        absmag = vals[params.index("absmag")]
        if absmag_res is None:
            # take nearest track
            absmags = np.unique(models[colname(mag_filter)])
            i = np.argmin(np.abs(absmag - absmags))
            absmag_idx = models[colname(mag_filter)] == absmags[i]
        elif absmag_res < 0:
            # treat as fractional
            absmag_res = -absmag_res
            absmag_idx = ((absmag*(1 - absmag_res)) <= models[colname(mag_filter)]) & (models[colname(mag_filter)] < (absmag*(1 + absmag_res)))
        else:
            absmag_idx = ((absmag - absmag_res) <= models[colname(mag_filter)]) & (models[colname(mag_filter)] < (absmag + absmag_res))
        if ~np.any(mass_idx):
            raise Exception(f"No tracks found for absolute magnitude {absmag}+/-{absmag_res} Msun!")
    else:
        if absmag_res < 0:
            # treat as fractional
            absmag_res = -absmag_res
            absmag_idx = (absmag_min*(1 - absmag_res) <= models[colname(mag_filter)]) & (models[colname(mag_filter)] <= absmag_max*(1 + absmag_res))
        else:
            absmag_idx = (absmag_min - absmag_res <= models[colname(mag_filter)]) & (models[colname(mag_filter)] <= absmag_max + absmag_res)
        if ~np.any(mass_idx):
            raise Exception(f"No tracks found in absolute magnitude range {absmag_min}-{absmag_max} Msun!")

    if "age" in params:
        age = vals[params.index("age")]
        if age_res is None:
            # take nearest track
            ages = 10**np.unique(models[colname("log_age")])*1e-9
            i = np.argmin(np.abs(age - ages))
            age_idx = 10**models[colname("log_age")]*1e-9 == ages[i]
        elif age_res < 0:
            # treat as fractional
            age_res = -age_res
            age_idx = (np.log10((np.maximum(age*(1 - age_res), 0)) * 1e9) <= models[colname("log_age")]) & \
                      (models[colname("log_age")] < np.log10((age*(1 + age_res)) * 1e9))
            if ~np.any(age_idx):
                raise Exception(f"No tracks found for age {age}Gyr+/-{age_res*100}%!")
        else:
            age_idx = (np.log10((np.maximum(age - age_res, 0)) * 1e9) <= models[colname("log_age")]) & \
                      (models[colname("log_age")] < np.log10((age + age_res) * 1e9))
            if ~np.any(age_idx):
                raise Exception(f"No tracks found for age {age}+/-{age_res} Gyr!")
    elif "log_age" in params:
        log_age = vals[params.index("log_age")]
        if age_res is None:
            # take nearest track
            log_ages = np.unique(models[colname("log_age")])
            i = np.argmin(np.abs(log_age - log_ages))
            age_idx = models[colname("log_age")] == log_ages[i]
        else:
            age_idx = ((np.maximum(log_age - log_age_res, 0)) <= models[colname("log_age")]) & \
                      (models[colname("log_age")] < (log_age + log_age_res))
        if ~np.any(age_idx):
            raise Exception(f"No tracks found for log(age) {log_age}+/-{log_age_res}!")
    else:
        if age_res < 0:
            # treat as fractional
            age_res = -age_res
            age_idx = (np.log10((age_min*(1 - age_res)) * 1e9) <= models[colname("log_age")]) & \
                      (models[colname("log_age")] <= np.log10((age_max*(1 + age_res)) * 1e9))
        else:
            if age_min > age_res:
                age_idx = (np.log10((age_min - age_res) * 1e9) <= models[colname("log_age")]) & \
                          (models[colname("log_age")] <= np.log10((age_max + age_res) * 1e9))
            else:
                # usually relevant to the pre-MS stage
                age_idx = (np.log10(age_min * 1e9) <= models[colname("log_age")]) & \
                          (models[colname("log_age")] <= np.log10((age_max + age_res) * 1e9))
        if ~np.any(age_idx):
            raise Exception(f"No tracks found in age range {age_min}-{age_max} Gyr!")

    if "mh" in params:
        mh = vals[params.index("mh")]
        if mh_res is None:
            # take nearest track
            mhs = np.unique(models[colname("mh")])
            i = np.argmin(np.abs(mh - mhs))
            mh_idx = models[colname("mh")] == mhs[i]
        else:
            mh_idx = ((mh - mh_res) <= models[colname("mh")]) & (models[colname("mh")] < (mh + mh_res))
        if ~np.any(mh_idx):
            raise Exception(f"No tracks found for metallicity {mh}+/-{mh_res}!")
    else:
        mh_idx = (mh_min - mh_res <= models[colname("mh")]) & (models[colname("mh")] <= mh_max + mh_res)
        if ~np.any(mh_idx):
            raise Exception(f"No tracks found in metallicity range {mh_min}-{mh_max}!")

    if stage is not None:
        stage_idx = models[colname("phase")] == stage  # 1 = main sequence
        if ~np.any(stage_idx):
            raise Exception(f"No tracks found for stage {stage}!")
    else:
        stage_idx = (stage_min <= models[colname("phase")]) & (models[colname("phase")] <= stage_max)
        if ~np.any(stage_idx):
            raise Exception(f"No tracks found in stage range {stage_min}-{stage_max}!")

    idx = mass_idx & absmag_idx & age_idx & mh_idx & stage_idx

    color1 = models[idx][colname(color_filter1)]
    color2 = models[idx][colname(color_filter2)]
    absmag = models[idx][colname(mag_filter)]
    mass = models[idx][colname("m0")]
    mh = models[idx][colname("mh")]
    age = 10 ** models[idx][colname("log_age")] * 1e-9  # [Gyr]
    stage = models[idx][colname("phase")]

    if sort_by is None:
        if "mass" not in params:
            sort_idx = np.argsort(mass)
        elif "age" not in params:
            sort_idx = np.argsort(age)
        elif "mh" not in params:
            sort_idx = np.argsort(mh)
    else:
        if sort_by == "mass":
            sort_idx = np.argsort(mass)
        elif sort_by == "age":
            sort_idx = np.argsort(age)
        elif sort_by == "mh":
            sort_idx = np.argsort(mh)
    color1 = color1[sort_idx]
    color2 = color2[sort_idx]
    absmag = absmag[sort_idx]
    mass = mass[sort_idx]
    mh = mh[sort_idx]
    age = age[sort_idx]
    stage = stage[sort_idx]

    if return_table:
        output = Table([mass, color1 - color2, absmag, mh, stage, age], names=('mass', 'color', 'absmag', 'mh', 'stage', 'age'))
    else:
        output = [color1, color2, absmag, mass, mh, age, stage]

    if return_idx:
        idx = np.where(idx)[0]
        idx = idx[sort_idx]
        output.append(idx)

    return output


def get_pre_ms_isomass(models, mass, mh, is_smooth=True, smooth_sigma=3, **kwargs):
    """
    get_pre_ms_isomass(models, mass, mh, is_smooth=True, smooth_sigma=3, **kwargs)

    Get a specific stellar evolution pre-MS track, with fixed mass and metallicity.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    mass : float
        Stellar track mass, in Msun
    mh : float
        Stellar track metallicity ([M/H]), in dex
    is_smooth : bool, optional
        Smooth the stellar track? (default: True).
    smooth_sigma : float, optional
        Smoothing Gaussian sigma (default: 3).
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.tracks.get_isotrack`.

    Returns
    -------
    bp_rp : array_like
        Chosen stellar evolution track Gaia Gbp-Grp color.
    mg : array_like
        Chosen stellar evolution track Gaia G magnitude.
    age : array_like
        Chosen stellar evolution track age, in Gyr.
    """

    bp, rp, g, _, _, age, stage = get_isotrack(models, [mass, mh], params=("mass", "mh"), stage=0, **kwargs)

    if is_smooth:
        bp_rp = gaussian_filter1d(np.array(bp - rp), smooth_sigma)
        mg = gaussian_filter1d(np.array(g), smooth_sigma)

        # remove transition to MS part
        d = np.diff(gaussian_filter1d(np.diff(bp_rp), 10))
        idx = np.argmax(d)
        if np.abs(mass - 0.6) < 0.01:  # fix plot wobble
            idx -= 30
        bp_rp = bp_rp[:idx]
        mg = mg[:idx]
        age = age[:idx]
        stage = stage[:idx]
    else:
        bp_rp = np.array(bp - rp)
        mg = np.array(g)

    return bp_rp, mg, age, stage


def get_combined_isomass(models, mass, age, mh_pre_ms=0.7, is_smooth=True, smooth_sigma=3, **kwargs):
    """
    get_combined_isomass(models, mass, age, mh_pre_ms=0.7, is_smooth=True, smooth_sigma=3, **kwargs)

    Get a specific stellar evolution track, combining a pre-MS track with fixed mass and metallicity, and an MS track
    with fixed mass and age.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    mass : float
        Stellar track mass, in Msun.
    age : float
        MS stellar track age, in Gyr.
    mh_pre_ms : float, optional
        Pre-MS stellar track metallicity ([M/H]), in dex (default: 0.7 dex).
    is_smooth : bool, optional
        Smooth the stellar track? (default: True).
    smooth_sigma : float, optional
        Smoothing Gaussian sigma (default: 3).
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.tracks.get_isotrack`.

    Returns
    -------
    bp_rp : array_like
        Chosen stellar evolution track Gaia Gbp-Grp color.
    mg : array_like
        Chosen stellar evolution track Gaia G magnitude.
    mh : array_like
        Chosen stellar evolution track metallicity, in dex.
    age_vec : array_like
        Chosen stellar evolution track age, in Gyr.
    ms_idx : array_like (bool)
        Indices of MS points on track.
    """

    # get MS track (fixed mass and age)
    bp, rp, g, _, mh, _, stage = get_isotrack(models, [mass, age], params=("mass", "age"), stage=1, **kwargs)

    # sort by metallicity
    sort_idx = np.argsort(mh)
    bp = bp[sort_idx]
    rp = rp[sort_idx]
    g = g[sort_idx]
    mh = mh[sort_idx]
    stage = stage[sort_idx]

    # get pre-MS track (fixed mass and metallicity)
    bp_rp0, mg0, age0, stage0 = get_pre_ms_isomass(models, mass, mh_pre_ms, is_smooth=is_smooth,
                                                   smooth_sigma=smooth_sigma,
                                                   **kwargs)

    # combine MS and pre-MS tracks
    bp_rp = np.append(bp - rp, np.flipud(bp_rp0))
    mg = np.append(g, np.flipud(mg0))
    mh = np.append(mh, mh_pre_ms * np.ones(len(mg0)))
    age_vec = np.append(age * np.ones(len(bp)), np.flipud(age0))
    stage = np.append(stage, np.flipud(stage0))
    # ms_idx = np.ones_like(bp_rp, dtype=bool)  # main sequence
    # ms_idx[len(bp):] = False

    if is_smooth & (mass != 0.15):
        bp_rp = gaussian_filter1d(bp_rp, smooth_sigma)
        mg = gaussian_filter1d(mg, smooth_sigma)

    return bp_rp, mg, mh, age_vec, stage


def get_isomasses(models, mass=np.arange(0.1, 1.2, 0.1), age=5, **kwargs):
    """
    get_isomasses(models, mass=np.arange(0.1, 1.2, 0.1), age=5, **kwargs)

    Get fixed-age stellar evolution tracks, for a range of masses.
    If the `stage` keyword is not specified (under `kwargs`), it assumes `stage=1` (MS phase).

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    mass : array_like, optional
        Stellar track mass array, in Msun (default: `np.arange(0.1, 1.2, 0.1)` Msun).
    age : float, optional
        Stellar track age, in Gyr (default: 5 Gyr).
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.gentracks.get_isotrack`.

    Returns
    -------
    tracks : Table
        Stellar evolution tracks, for the given age and range of masses, with columns `mass`, `bp_rp`, `mg`, and `mh`.
    """

    m_vec = np.array([])
    bp_rp = np.array([])
    mg = np.array([])
    mh = np.array([])
    stage = np.array([])
    for m in mass:
        bp, rp, g, _, curr_mh, _, curr_stage = get_isotrack(models, [m, age], params=("mass", "age"), **kwargs)
        bp_rp = np.append(bp_rp, np.array([bp - rp]))
        mg = np.append(mg, np.array(g))
        mh = np.append(mh, np.array(curr_mh))
        stage = np.append(stage, np.array(curr_stage))
        m_vec = np.append(m_vec, m * np.ones(len(g)))
    age_vec = age * np.ones(len(m_vec))

    tracks = Table([m_vec, bp_rp, mg, mh, stage, age_vec], names=('mass', 'bp_rp', 'mg', 'mh', 'stage', 'age'))

    return tracks


def get_combined_isomasses(models, mass=np.arange(0.1, 1.2, 0.1), age=5, mh_pre_ms=0.7, is_smooth=True, smooth_sigma=3,
                           exclude_pre_ms_masses=[], **kwargs):
    """
    get_combined_isomasses(models, mass=np.arange(0.1, 1.2, 0.1), age=5, mh_pre_ms=0.7, is_smooth=True, smooth_sigma=3,
                           **kwargs)

    Get combined stellar evolution tracks, for a range of masses. Each track combines a pre-MS track with fixed mass
    and metallicity, and an MS track with fixed mass and age.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    mass : array_like, optional
        Stellar track mass array, in Msun (default: `np.arange(0.1, 1.2, 0.1)` Msun).
    age : float, optional
        Stellar track age, in Gyr (default: 5 Gyr).
    mh_pre_ms : float, optional
        Pre-MS stellar track metallicity ([M/H]), in dex (default: 0.7 dex).
    is_smooth : bool, optional
        Smooth the stellar track? (default: True).
    smooth_sigma : float, optional
        Smoothing Gaussian sigma (default: 3).
    exclude_pre_ms_masses : array_like, optional
        List of pre-main-sequence mass tracks to exclude (to avoid interpolation artifacts; default: []).
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.gentracks.get_isotrack`.

    Returns
    -------
    tracks : Table
        Stellar evolution tracks, for the given age and range of masses, with columns `mass`, `bp_rp`, `mg`, and `mh`.

    """

    m_vec = np.array([])
    age_vec = np.array([])
    bp_rp = np.array([])
    mg = np.array([])
    mh = np.array([])
    stage = np.array([])
    for m in mass:
        if ~np.any(np.abs(m - exclude_pre_ms_masses) < 1e-4):
            curr_bp_rp, curr_mg, curr_mh, curr_age, curr_stage = get_combined_isomass(models, m, age,
                                                                                      mh_pre_ms=mh_pre_ms,
                                                                                      is_smooth=is_smooth,
                                                                                      smooth_sigma=smooth_sigma,
                                                                                      **kwargs)
            curr_age = np.array(curr_age)
        else:
            # don't take the pre-MS track for this mass
            curr_bp, curr_rp, curr_mg, _, curr_mh, _, curr_stage = get_isotrack(models, [m, age],
                                                                                params=("mass", "age"), **kwargs)
            curr_bp_rp = curr_bp - curr_rp
            curr_age = age * np.ones(len(curr_mg))

        bp_rp = np.append(bp_rp, np.array([curr_bp_rp]))
        mg = np.append(mg, np.array(curr_mg))
        mh = np.append(mh, np.array(curr_mh))
        stage = np.append(stage, np.array(curr_stage))
        m_vec = np.append(m_vec, m * np.ones(len(curr_mg)))
        age_vec = np.append(age_vec, curr_age)

    tracks = Table([m_vec, bp_rp, mg, mh, stage, age_vec], names=('mass', 'bp_rp', 'mg', 'mh', 'stage', 'age'))

    return tracks


def get_isochrone_polygon(models, age1, mh1, age2, mh2, age_res=0.001, log_age_res=0.05, mh_res=0.05, mass_res=0.007, mass_max=1.2,
                          stage1=1, stage2=1, bp_rp_min=-np.inf, bp_rp_max=np.inf, bp_rp_shift1=0, bp_rp_shift2=0,
                          mg_shift1=0, mg_shift2=0):
    """
    get_isochrone_polygon(models, age1, mh1, age2, mh2, age_res=0.001, mh_res=0.05, mass_res=0.007, mass_max=1.2,
                          stage1=1, stage2=1, bp_rp_min=-np.inf, bp_rp_max=np.inf, bp_rp_shift1=0, bp_rp_shift2=0,
                          mg_shift1=0, mg_shift2=0)

    Get the polygon enclosed by two isochrones.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    age1 : float, optional
        Stellar track age of the first track, in Gyr.
    mh1 : float
        Stellar track metallicity ([M/H]) of the first track, in dex.
    age2 : float, optional
        Stellar track age of the second track, in Gyr.
    mh2 : float
        Stellar track metallicity ([M/H]) of the second track, in dex.
    age_res : float, optional
        Age resolution, in Gyr (default: 0.001 Gyr). If negative - treat as fractional.
    log_age_res : float, optional
        log(age/yr) resolution, in dex, if using `log_age` as one of the `params` (default: 0.05 dex).
    mh_res : float, optional
        Metallicity resolution, in dex (default: 0.05 dex).
    mass_res : float, optional
        Mass resolution, in Msun (default: 0.007 Msun).
    mass_max : float, optional
        Maximum mass to consider, in Msun (if no fixed mass was chosen; default: 1.2 Msun).
    stage1 : int, optional
        Stellar evolution stage label of the first track (0 = pre-MS, 1 = MS, etc.; default: 1).
    stage2 : int, optional
        Stellar evolution stage label of the second track (0 = pre-MS, 1 = MS, etc.; default: 1).
    bp_rp_min : float, optional
        Minimal Gaia Gbp-Grp color to consider (default: -np.inf).
    bp_rp_max : float, optional
        Maximal Gaia Gbp-Grp color to consider (default: np.inf).
    bp_rp_shift1 : float, optional
        How much to shift the Gaia Gbp-Grp color of the first track (default: 0).
    bp_rp_shift2 : float, optional
        How much to shift the Gaia Gbp-Grp color of the second track (default: 0).
    mg_shift1 : float, optional
        How much to shift the Gaia G-band of the first track (default: 0).
    mg_shift2 : float, optional
        How much to shift the Gaia G-band of the second track (default: 0).

    Returns
    -------
    polygon : Path object
        The polygon enclosed by the two stellar evolutionary tracks.

    """
    BP1, RP1, G1, mass1 = get_isotrack(models, [age1, mh1], params=("age", "mh"),
                                       mass_max=mass_max, age_res=age_res, log_age_res=log_age_res,
                                       mh_res=mh_res, mass_res=mass_res, stage=stage1)[:4]
    BP2, RP2, G2, mass2 = get_isotrack(models, [age2, mh2], params=("age", "mh"),
                                       mass_max=mass_max, age_res=age_res, log_age_res=log_age_res,
                                       mh_res=mh_res, mass_res=mass_res, stage=stage2)[:4]

    BP_RP1 = BP1 - RP1 + bp_rp_shift1
    idx = (bp_rp_min <= BP_RP1) & (BP_RP1 <= bp_rp_max)
    BP_RP1 = BP_RP1[idx]
    G1 = G1[idx] + mg_shift1
    mass1 = mass1[idx]

    BP_RP2 = BP2 - RP2 + bp_rp_shift2
    idx = (bp_rp_min <= BP_RP2) & (BP_RP2 <= bp_rp_max)
    BP_RP2 = BP_RP2[idx]
    G2 = G2[idx] + mg_shift2
    mass2 = mass2[idx]

    vertices = np.vstack((np.array([BP_RP1, G1]).T, np.flipud(np.array([BP_RP2, G2]).T)))

    polygon = Path(vertices)

    return polygon, BP_RP1, G1, mass1, BP_RP2, G2, mass2


def get_isochrone_side(models, vals, params=("age", "mh"), side="blue", age_res=0.001, log_age_res=0.05, mh_res=0.05, mass_res=0.007, mass_min=0, mass_max=1.2,
                       stage=1, stage_min=0, stage_max=np.inf, bp_rp_min=-10, bp_rp_max=10, bp_rp_shift=0, mg_shift=0,
                       is_extrapolate=True, color_filter1="G_BPmag", color_filter2="G_RPmag", mag_filter="Gmag"):
    """
    get_isochrone_side(models, age, mh, side="blue", age_res=0.001, mh_res=0.05, mass_res=0.007, mass_max=1.2,
                       stage=1, bp_rp_min=-np.inf, bp_rp_max=np.inf, bp_rp_shift=0, mg_shift=0)

    Get the polygon enclosed by one side of an evolutionary track.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    vals : array_like (of length 2)
        Values of the fixed parameters (units: [mass] = Msun, [age] = Gyr, [mh] = dex).
    params : tuple (of length 2), optional
        Fixed parameters names (default: ("age", "mh")).
    side : str, optional (default: "blue")
        Which side of the track to include ("blue"/"red").
    age_res : float, optional
        Age resolution, in Gyr (default: 0.001 Gyr). If negative - treat as fractional.
    log_age_res : float, optional
        log(age/yr) resolution, in dex, if using `log_age` as one of the `params` (default: 0.05 dex).
    mh_res : float, optional
        Metallicity resolution, in dex (default: 0.05 dex).
    mass_res : float, optional
        Mass resolution, in Msun (default: 0.007 Msun).
    mass_max : float, optional
        Maximum mass to consider, in Msun (if no fixed mass was chosen; default: 1.2 Msun).
    stage : int, optional
        Stellar evolution stage label of the track (0 = pre-MS, 1 = MS, etc.; default: 1).
    stage_min : int, optional
        Minimum stellar evolution stage label to consider (if no fixed stage was chosen; default: 0).
    stage_max : int, optional
        Maximum stellar evolution stage label to consider (if no fixed stage was chosen; default: `np.inf`).
    bp_rp_min : float, optional
        Minimal Gaia Gbp-Grp color to consider (default: -np.inf).
    bp_rp_max : float, optional
        Maximal Gaia Gbp-Grp color to consider (default: np.inf).
    bp_rp_shift : float, optional
        How much to shift the Gaia Gbp-Grp color of the track (default: 0).
    mg_shift : float, optional
        How much to shift the Gaia G-band of the track (default: 0).
    is_extrapolate : bool, optional
        Whether to extrapolate the massive part of the isochrone (default: True).
    color_filter1 : str, optional
        Bluer band to use for the color calculation (default: `G_BPmag`).
    color_filter2 : str, optional
        Redder band to use for the color calculation (default: `G_RPmag`).
    mag_filter : str, optional
        Which band to use for the magnitude axis (default: `Gmag`).


    Returns
    -------
    polygon : Path object
        The polygon enclosed by one side of an evolutionary track.

    """
    BP, RP, G, mass = get_isotrack(models, vals, params=params,
                                   mass_min=mass_min, mass_max=mass_max, age_res=age_res, log_age_res=log_age_res,
                                   mh_res=mh_res, mass_res=mass_res,
                                   stage=stage, stage_min=stage_min, stage_max=stage_max,
                                   color_filter1=color_filter1, color_filter2=color_filter2, mag_filter=mag_filter)[:4]

    BP_RP = BP - RP + bp_rp_shift
    idx = (bp_rp_min <= BP_RP) & (BP_RP <= bp_rp_max)
    BP_RP = BP_RP[idx]
    G = G[idx] + mg_shift
    mass = mass[idx]

    if is_extrapolate:
        idx = np.argsort(G[-2:])
        p = np.polyfit(G[-2:][idx], BP_RP[-2:][idx], 1)
        p = np.poly1d(p)
        extra_point = p(np.array([-1]))
        BP_RP = np.concatenate([BP_RP, extra_point])
        G = np.concatenate([G, np.array([-1])])

    if side.lower() == "blue":
        vertices = np.vstack((np.array([BP_RP, G]).T, np.array([[-10, -10], [np.min(G), np.max(G)]]).T))
    elif side.lower() == "red":
        vertices = np.vstack((np.array([BP_RP, G]).T, np.array([[10, 10], [np.min(G), np.max(G)]]).T))
    else:
        print(f"Unimplemented side {side}!")

    polygon = Path(vertices)

    return polygon, BP_RP, G, mass
