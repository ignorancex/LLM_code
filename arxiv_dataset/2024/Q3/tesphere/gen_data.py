from turb_sphere import tes, utils
import numpy as np
import pickle
from pathlib import Path
import sys


def run_TES():
    pindex_list = np.array(sys.argv[1:], dtype=float)
    for pindex in pindex_list:
        velocity_dispersion = np.logspace(-1, 1, 100)
        rcrit, mcrit, ucrit, rsonic, rfwhm = [], [], [], [], []
        for sigma in velocity_dispersion:
            print(f"[run_TES] Generate data for p = {pindex}, sigma = {sigma}")
            rs = tes.TES.find_sonic_radius(pindex, sigma)
            ts = tes.TES(pindex=pindex, rsonic=rs)
            rcrit.append(ts.rcrit)
            mcrit.append(ts.mcrit)
            ucrit.append(ts.ucrit)
            rsonic.append(ts.rsonic)
            rfwhm.append(utils.fwhm(ts.density, ts.rcrit))
        rcrit = np.array(rcrit)
        mcrit = np.array(mcrit)
        ucrit = np.array(ucrit)
        rsonic = np.array(rsonic)
        rfwhm = np.array(rfwhm)

        res = dict(ucrit=ucrit,
                   rcrit=rcrit,
                   mcrit=mcrit,
                   rsonic=rsonic,
                   sigma=velocity_dispersion,
                   rfwhm=rfwhm)
        fp = Path(__file__).parent / f"data/tsc.p{pindex}.p"
        with open(fp, "wb") as handle:
            pickle.dump(res, handle)


def run_TESe():
    pindex_list = np.array(sys.argv[1:], dtype=float)
    for pindex in pindex_list:
        velocity_dispersion = np.logspace(-1, np.log10(50), 100)
        rcrit, mcrit, ucrit, rsonic = [], [], [], []
        for sigma in velocity_dispersion:
            print(f"[run_TESe] Generate data for p = {pindex}, sigma = {sigma}")
            rs = tes.TESe.find_sonic_radius(pindex, sigma)
            ts = tes.TESe(pindex=pindex, rsonic=rs)
            rcrit.append(ts.rcrit)
            mcrit.append(ts.mcrit)
            ucrit.append(ts.ucrit)
            rsonic.append(ts.rsonic)
        rcrit = np.array(rcrit)
        mcrit = np.array(mcrit)
        ucrit = np.array(ucrit)
        rsonic = np.array(rsonic)

        res = dict(ucrit=ucrit,
                   rcrit=rcrit,
                   mcrit=mcrit,
                   rsonic=rsonic,
                   sigma=velocity_dispersion)
        fp = Path(__file__).parent / f"data/tse.p{pindex}.p"
        with open(fp, "wb") as handle:
            pickle.dump(res, handle)


def manuscript_table1():
    pindex = 0.5
    rs_min = tes.TES.minimum_sonic_radius(pindex=pindex)
    sonic_radius = np.logspace(np.log10(rs_min), 3, 20000)
    f = open("tab1.txt", "w")
    for rsonic in sonic_radius:
        ts = tes.TES(pindex=pindex, rsonic=rsonic)
        f.write(f"{rsonic:.5e}, ")
        f.write(f"{ts.rcrit:.5e}, ")
        f.write(f"{ts.mcrit:.5e}, ")
        f.write(f"{ts.ucrit:.5e}, ")
        f.write(f"{ts.sigma:.5e}\n")
    f.close()


if __name__ == "__main__":
#    run_TES()
    run_TESe()
#    manuscript_table1()
