import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import os
import argparse
import logging, traceback
import pandas as pd
from copy import copy

# import ..config

from ..analysis_seeds.bkg_rate_estimation import get_avg_lin_cub_rate_quad_obs
from ..config import (
    quad_dicts,
    EBINS0,
    EBINS1,
    solid_angle_dpi_fname,
    bright_source_table_fname,
)
from ..lib.sqlite_funcs import write_rate_fits_from_obj, get_conn
from ..lib.dbread_funcs import get_info_tab, guess_dbfname, get_files_tab
from ..lib.event2dpi_funcs import filter_evdata
from ..models.models import (
    Bkg_Model_wFlatA,
    CompoundModel,
    Point_Source_Model_Binned_Rates,
)
from ..llh_analysis.LLH import LLH_webins
from ..llh_analysis.minimizers import NLLH_ScipyMinimize, NLLH_ScipyMinimize_Wjacob
from ..response.ray_trace_funcs import RayTraces
from ..lib.coord_conv_funcs import convert_radec2imxy
from ..lib.gti_funcs import add_bti2gti, bti2gti, gti2bti, union_gtis
from ..lib.wcs_funcs import world2val


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument("--job_id", type=int, help="Job ID", default=0)
    parser.add_argument("--Njobs", type=int, help="Number of jobs", default=1)
    parser.add_argument(
        "--twind",
        type=float,
        help="Number of seconds to go +/- from the trigtime",
        default=20,
    )
    parser.add_argument("--bkg_dur", type=float, help="bkg duration", default=40.0)
    parser.add_argument(
        "--bkg_nopost",
        help="Don't use time after signal window for bkg",
        action="store_true",
    )
    parser.add_argument(
        "--bkg_nopre",
        help="Don't use time before signal window for bkg",
        action="store_true",
    )
    parser.add_argument(
        "--pcfname", type=str, help="partial coding file name", default="pc_2.img"
    )

    args = parser.parse_args()
    return args


def ang_sep(ra0, dec0, ra1, dec1):
    dcos = np.cos(np.radians(np.abs(ra0 - ra1)))
    angsep = np.arccos(
        np.cos(np.radians(90 - dec0)) * np.cos(np.radians(90 - dec1))
        + np.sin(np.radians(90 - dec0)) * np.sin(np.radians(90 - dec1)) * dcos
    )
    return np.rad2deg(angsep)


def im_dist(imx0, imy0, imx1, imy1):
    return np.hypot((imx1 - imx0), (imy1 - imy0))


def add_imxy2src_tab(src_tab, attfile, t0):
    att_ind = np.argmin(np.abs(attfile["TIME"] - t0))
    att_quat = attfile["QPARAM"][att_ind]
    pnt_ra, pnt_dec = attfile["POINTING"][att_ind, :2]
    imxs = np.zeros(len(src_tab))
    imys = np.zeros(len(src_tab))
    src_tab["PntSep"] = ang_sep(pnt_ra, pnt_dec, src_tab["RAJ2000"], src_tab["DEJ2000"])
    for i in range(len(imxs)):
        if src_tab["PntSep"][i] > 80.0:
            imxs[i], imys[i] = np.nan, np.nan
            continue
        imxs[i], imys[i] = convert_radec2imxy(
            src_tab["RAJ2000"][i], src_tab["DEJ2000"][i], att_quat
        )
    src_tab["imx"] = imxs
    src_tab["imy"] = imys
    return src_tab


def get_srcs_infov(attfile, t0, pcfname=None, pcmin=5e-2):
    brt_src_tab = Table.read(bright_source_table_fname)
    add_imxy2src_tab(brt_src_tab, attfile, t0)
    bl_infov = (np.abs(brt_src_tab["imy"]) < 0.95) & (np.abs(brt_src_tab["imx"]) < 1.75)
    if pcfname is not None:
        PC = fits.open(pcfname)[0]
        pc = PC.data
        w_t = WCS(PC.header, key="T")
        pcvals = world2val(w_t, pc, brt_src_tab["imx"], brt_src_tab["imy"])
        bl_infov = bl_infov & (pcvals >= pcmin)
    N_infov = np.sum(bl_infov)
    return brt_src_tab[bl_infov]


def bkg_withPS_fit(
    PS_tab, model, llh_obj, t0s, t1s, dimxy=2e-3, im_steps=5, test_null=False
):
    Nps = len(PS_tab)
    imax = np.linspace(-dimxy, dimxy, im_steps)
    if im_steps == 3:
        imax = np.linspace(-dimxy / 2.0, dimxy / 2.0, im_steps)
    elif im_steps == 2:
        imax = np.linspace(-dimxy / 2.0, dimxy / 2.0, im_steps)
    elif im_steps == 1:
        imax = np.array([0.0])

    imlist = []
    for i in range(Nps):
        imlist += [imax, imax]
    imgs = np.meshgrid(*imlist)
    Npnts = imgs[0].size

    bkg_miner = NLLH_ScipyMinimize_Wjacob("")
    bkg_miner.set_llh(llh_obj)
    llh_obj.set_time(t0s, t1s)

    bf_params_list = []
    bkg_nllhs = np.zeros(Npnts)

    nebins = model.nebins

    for i in range(Npnts):
        bf_params = {}
        im_names = []
        for j in range(Nps):
            row = PS_tab[j]
            psname = row["Name"]
            bf_params[psname + "_imx"] = imgs[2 * j].ravel()[i] + row["imx"]
            bf_params[psname + "_imy"] = imgs[2 * j + 1].ravel()[i] + row["imy"]
            im_names = [psname + "_imx", psname + "_imy"]
        im_vals = [bf_params[nm] for nm in im_names]

        for e0 in range(nebins):
            bkg_miner.set_fixed_params(bkg_miner.param_names)
            bkg_miner.set_fixed_params(im_names, im_vals)
            e0_pnames = []
            for pname in bkg_miner.param_names:
                try:
                    if int(pname[-1]) == e0:
                        e0_pnames.append(pname)
                except:
                    pass
            bkg_miner.set_fixed_params(e0_pnames, fixed=False)
            llh_obj.set_ebin(e0)

            bf_vals, bkg_nllh, res = bkg_miner.minimize()
            bkg_nllhs[i] += bkg_nllh[0]
            for ii, pname in enumerate(e0_pnames):
                bf_params[pname] = bf_vals[0][ii]
        bf_params_list.append(bf_params)

    bf_ind = np.argmin(bkg_nllhs)
    bf_params = bf_params_list[bf_ind]
    bf_nllh = bkg_nllhs[bf_ind]

    if test_null:
        TS_nulls = {}
        for i in range(Nps):
            params_ = copy(bf_params)
            row = PS_tab[i]
            psname = row["Name"]
            for j in range(nebins):
                params_[psname + "_rate_" + str(j)] = 0.0
            llh_obj.set_ebin(-1)
            nllh_null = -llh_obj.get_logprob(params_)
            TS_nulls[psname] = np.sqrt(2.0 * (nllh_null - bf_nllh))
            if np.isnan(TS_nulls[psname]):
                TS_nulls[psname] = 0.0
        return bf_nllh, bf_params, TS_nulls

    return bf_nllh, bf_params


def do_init_bkg_wPSs(bkg_mod, llh_obj, src_tab, rt_obj, GTI, sig_twind):
    gti_bkg = add_bti2gti(sig_twind, GTI)
    bkg_t0s = gti_bkg["START"]
    bkg_t1s = gti_bkg["STOP"]

    Nsrcs = len(src_tab)
    nebins = bkg_mod.nebins

    for ii in range(Nsrcs):
        mod_list = [bkg_mod]
        im_steps = 5
        if Nsrcs >= 3:
            im_steps = 3
        if Nsrcs >= 5:
            im_steps = 2
        if Nsrcs >= 7:
            im_steps = 1

        ps_mods = []
        for i in range(Nsrcs):
            row = src_tab[i]
            mod = Point_Source_Model_Binned_Rates(
                row["imx"],
                row["imy"],
                0.1,
                [llh_obj.ebins0, llh_obj.ebins1],
                rt_obj,
                llh_obj.bl_dmask,
                use_deriv=True,
                name=row["Name"],
            )
            ps_mods.append(mod)

        mod_list += ps_mods
        comp_mod = CompoundModel(mod_list)

        llh_obj.set_model(comp_mod)

        bf_nllh, bf_params, TS_nulls = bkg_withPS_fit(
            src_tab,
            comp_mod,
            llh_obj,
            bkg_t0s,
            bkg_t1s,
            test_null=True,
            im_steps=im_steps,
        )

        logging.debug("TS_nulls: ")
        logging.debug(TS_nulls)

        bkg_rates = np.array(
            [bf_params["Background" + "_bkg_rate_" + str(j)] for j in range(nebins)]
        )
        min_rate = 1e-1 * bkg_rates
        logging.debug("min_rate: ")
        logging.debug(min_rate)
        PSs2keep = []
        for name, TS in TS_nulls.items():
            if TS < 8.0:
                ps_rates = np.array(
                    [bf_params[name + "_rate_" + str(j)] for j in range(nebins)]
                )
                logging.debug(name + " rates: ")
                logging.debug(ps_rates)
                # print ps_rates
                if np.all(ps_rates < min_rate):
                    continue
            PSs2keep.append(name)

        if len(PSs2keep) == len(src_tab):
            break
        if len(PSs2keep) == 0:
            Nsrcs = 0
            src_tab = src_tab[np.zeros(len(src_tab), dtype=bool)]
            break
        bl = np.array([src_tab["Name"][i] in PSs2keep for i in range(Nsrcs)])
        src_tab = src_tab[bl]
        Nsrcs = len(src_tab)
        logging.debug("src_tab: ")
        logging.debug(src_tab)

    return bf_params, src_tab


def get_info_mat_around_min(llh_obj, mod, params_, ebin):
    params = copy(params_)
    dt = llh_obj.dt
    mod_cnts = llh_obj.model.get_rate_dpi(params, ebin) * dt
    data_cnts = llh_obj.data_dpis[ebin]

    dR_dparams = mod.get_dr_dp(params, ebin)

    cov_ndim = len(dR_dparams)

    info_mat = np.zeros((cov_ndim, cov_ndim))

    for i in range(cov_ndim):
        for j in range(cov_ndim):
            info_mat[i, j] = np.sum(
                ((dR_dparams[j] * dt) * (dR_dparams[i] * dt) * data_cnts)
                / np.square(mod_cnts)
            )
    #             cov_mat[i,j] = np.sum(np.square(mod_cnts)/\
    #                         ((dR_dparams[j]*dt)*(dR_dparams[i]*dt)*data_cnts))

    return info_mat


def get_errs_corrs(llh_obj, model, params, e0, pnames2skip=[]):
    imat = get_info_mat_around_min(llh_obj, model, copy(params), e0)
    cov_mat = np.linalg.inv(imat)
    e0_pnames = []
    for pname in model.param_names:
        try:
            if int(pname[-1]) == e0 and not (pname in pnames2skip):
                e0_pnames.append(pname)
        except:
            pass

    err_dict = {}
    corr_dict = {}
    errs = np.sqrt(np.diag(cov_mat))
    for i, pname in enumerate(e0_pnames):
        k = "err_" + pname
        err_dict[k] = errs[i]
    Npars = len(e0_pnames)
    for i in range(Npars - 1):
        pname0 = e0_pnames[i]
        for j in range(i + 1, Npars):
            pname1 = e0_pnames[j]
            k = "corr_" + pname0 + "_" + pname1
            corr_dict[k] = cov_mat[i, j] / (errs[i] * errs[j])

    return err_dict, corr_dict


def bkg_withPS_fit_fiximxy(
    PS_tab, model, llh_obj, t0s, t1s, params_, fixed_pnames=None
):
    Nps = len(PS_tab)
    params = copy(params_)

    llh_obj.set_model(model)
    bkg_miner = NLLH_ScipyMinimize_Wjacob("")
    bkg_miner.set_llh(llh_obj)
    llh_obj.set_time(t0s, t1s)

    nllh = 0.0
    #     bf_params = {fixed_pars[i]:fixed_vals[i] for i in range(len(fixed_pars))}
    bf_params = copy(params)
    fixed_vals = [bf_params[pname] for pname in fixed_pnames]
    errs_dict = {}
    corrs_dict = {}

    for e0 in range(llh_obj.nebins):
        bkg_miner.set_fixed_params(bkg_miner.param_names)
        if fixed_pnames is not None:
            bkg_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
        e0_pnames = []
        for pname in bkg_miner.param_names:
            try:
                if int(pname[-1]) == e0 and not (pname in fixed_pnames):
                    e0_pnames.append(pname)
            except:
                pass
        bkg_miner.set_fixed_params(e0_pnames, fixed=False)
        llh_obj.set_ebin(e0)

        bf_vals, bkg_nllh, res = bkg_miner.minimize()
        nllh += bkg_nllh[0]
        for ii, pname in enumerate(e0_pnames):
            bf_params[pname] = bf_vals[0][ii]

        print(bf_params)

        err_dict, corr_dict = get_errs_corrs(
            llh_obj, model, copy(bf_params), e0, pnames2skip=fixed_pnames
        )
        for k, val in err_dict.items():
            errs_dict[k] = val
        for k, val in corr_dict.items():
            corrs_dict[k] = val

    return nllh, bf_params, errs_dict, corrs_dict


def main(args):
    logging.basicConfig(
        filename="bkg_rate_estimation_wPSs.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    if args.bkg_nopost and args.bkg_nopre:
        raise Exception("Can't have no pre and no post")

    if args.dbfname is None:
        db_fname = guess_dbfname()
        if isinstance(db_fname, list):
            db_fname = db_fname[0]
    else:
        db_fname = args.dbfname

    logging.info("Connecting to DB")
    conn = get_conn(db_fname)

    info_tab = get_info_tab(conn)
    logging.info("Got info table")

    files_tab = get_files_tab(conn)
    logging.info("Got files table")

    trigtime = info_tab["trigtimeMET"][0]
    tstart = trigtime - args.twind
    tstop = trigtime + args.twind

    evfname = files_tab["evfname"][0]
    dmfname = files_tab["detmask"][0]
    attfname = files_tab["attfname"][0]
    ev_data = fits.open(evfname)[1].data
    try:
        GTI = Table.read(evfname, hdu="GTI_POINTING")
    except:
        GTI = Table.read(evfname, hdu="GTI")
    dmask = fits.open(dmfname)[0].data
    bl_dmask = dmask == 0.0
    attfile = fits.open(attfname)[1].data
    logging.debug("Opened up event, detmask, and att files")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    nebins = len(ebins0)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    solid_angle_dpi = np.load(solid_angle_dpi_fname)

    src_tab = get_srcs_infov(attfile, trigtime, pcfname=args.pcfname)
    Nsrcs = len(src_tab)
    logging.info("src_tab: ")
    logging.info(src_tab)

    bkg_mod = Bkg_Model_wFlatA(bl_dmask, solid_angle_dpi, nebins, use_deriv=True)

    llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)

    # add in stuff later for if there's no srcs

    if Nsrcs > 0:
        rt_dir = files_tab["rtDir"][0]
        rt_obj = RayTraces(rt_dir)

        sig_dtwind = (-10 * 1.024, 20 * 1.024)
        sig_twind = (trigtime + sig_dtwind[0], trigtime + sig_dtwind[1])

        init_bf_params, src_tab = do_init_bkg_wPSs(
            bkg_mod, llh_obj, src_tab, rt_obj, GTI, sig_twind
        )

        Nsrcs = len(src_tab)

        logging.debug("Final src_tab:")
        logging.debug(src_tab)

        # Now need to do each time, with these PSs and these imxys

    else:
        init_bf_params = {k: bkg_mod.param_dict[k]["val"] for k in bkg_mod.param_names}

    if Nsrcs > 0:
        fixed_pars = [
            pname
            for pname in list(init_bf_params.keys())
            if "_flat_" in pname or "_imx" in pname or "_imy" in pname
        ]

        mod_list = [bkg_mod]
        ps_mods = []
        for i in range(Nsrcs):
            row = src_tab[i]
            mod = Point_Source_Model_Binned_Rates(
                row["imx"],
                row["imy"],
                0.1,
                [ebins0, ebins1],
                rt_obj,
                bl_dmask,
                use_deriv=True,
                name=row["Name"],
            )
            ps_mods.append(mod)

        mod_list += ps_mods
        mod = CompoundModel(mod_list)

    else:
        init_bf_params = {k: bkg_mod.param_dict[k]["val"] for k in bkg_mod.param_names}
        mod = bkg_mod
        fixed_pars = []

    bkg_dur = args.bkg_dur * 1.024
    twind = args.twind * 1.024
    bkg_tstep = 2 * 1.024
    dt_ax = np.arange(-twind, twind + 1, bkg_tstep)
    t_ax = dt_ax + trigtime
    Ntpnts = len(dt_ax)
    logging.info("Ntpnts: %d" % (Ntpnts))
    logging.info("min(dt_ax): %.3f" % (np.min(dt_ax)))
    logging.info("max(dt_ax): %.3f" % (np.max(dt_ax)))
    sig_wind = 10 * 1.024
    # sig_twind = (trigger_time + sig_dtwind[0], trigger_time + sig_dtwind[1])

    bkg_bf_dicts = []

    for i in range(Ntpnts):
        tmid = t_ax[i]
        sig_twind = (-sig_wind / 2.0 + tmid, sig_wind / 2.0 + tmid)
        gti_ = add_bti2gti(sig_twind, GTI)
        bkg_t0 = tmid - sig_wind / 2.0 - bkg_dur / 2.0
        bkg_t1 = tmid + sig_wind / 2.0 + bkg_dur / 2.0
        bkg_bti = Table(
            data=([-np.inf, bkg_t1], [bkg_t0, np.inf]), names=("START", "STOP")
        )
        gti_ = add_bti2gti(bkg_bti, gti_)
        print(tmid - trigtime)
        print(gti_)
        t0s = gti_["START"]
        t1s = gti_["STOP"]

        nllh, params, errs_dict, corrs_dict = bkg_withPS_fit_fiximxy(
            src_tab,
            mod,
            llh_obj,
            t0s,
            t1s,
            copy(init_bf_params),
            fixed_pnames=fixed_pars,
        )

        params.update(errs_dict)
        params.update(corrs_dict)
        params["nllh"] = nllh
        params["time"] = tmid
        params["dt"] = tmid - trigtime
        bkg_bf_dicts.append(params)
        params["exp"] = llh_obj.dt

    bkg_df = pd.DataFrame(bkg_bf_dicts)

    save_fname = "bkg_estimation.csv"
    logging.info("Saving results in a DataFrame to file: ")
    logging.info(save_fname)
    bkg_df.to_csv(save_fname, index=False)


if __name__ == "__main__":
    args = cli()

    main(args)
