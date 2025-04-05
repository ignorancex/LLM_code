import os
import glob
from scipy.io import loadmat
import numpy as np



def read_ACM_dat(path, sample_fraction = 1, time_sample_fraction=1):

    data = loadmat(path)
    fields = data["data"]
    if np.size(fields,0)==3:
        p = fields[0, ...].T
        ux = fields[1, ...].T
        uy = fields[2, ...].T
        mask = np.zeros_like(p)
    else:
        mask = fields[0, ...].T
        p = fields[1, ...].T
        ux = fields[2, ...].T
        uy = fields[3, ...].T
    time = data["time"].flatten()
    time = time - time[0]

    Ngrid = [fields.shape[2] // sample_fraction, fields.shape[3] // sample_fraction]
    domain_size = data['domain_size'][0]
    dx = data["dx"][0]*sample_fraction
    frac = sample_fraction
    tfrac = time_sample_fraction
    return ux[::frac,::frac,::tfrac], uy[::frac,::frac,::tfrac], p[::frac,::frac,::tfrac], mask[::frac,::frac,::tfrac], time[::tfrac], Ngrid, dx, domain_size

ux_list, uy_list=None,None

def load_trajectories(data_path, component_list = None, params_id_regex = "ai_*", sample_fraction = 1, time_sample_fraction = 1 ):

    ux_list = []
    uy_list = []
    time_list = []
    mu_list = []
    p_list = []
    mask_list = []
    Nt_sum = 0
    Nfiles = 0
    for fpath in glob.glob(data_path + "/" + params_id_regex):
        Nfiles +=1
        fpath = fpath + "/ALL.mat"
        print("reading: ", fpath)
        ux, uy, p, mask, time, Ngrid, dX, L = read_ACM_dat(fpath, sample_fraction=sample_fraction, time_sample_fraction= time_sample_fraction)

        if "ux" in component_list: # for very large data this is necessary! otherwise it blows up the RAM
            ux_list.append(ux)
        if "uy" in component_list:
            uy_list.append(uy)
        if "p" in component_list:
            p_list.append(p)
        if "mask" in component_list:
            mask_list.append(mask)
        time_list.append(time)
        Nt = len(time)
        Nt_sum += Nt
        mu_vec = np.asarray([float(mu) for mu in fpath.split("/")[-2].split("_")[2:]])
        mu_list.append(mu_vec)

    assert (Nfiles >= 1), "No file found!"
    val_dict = {"ux": ux_list, "uy": uy_list, "p": p_list, "mask": mask_list,
                "mu": mu_list, "time": time_list, "dx" : dX, "domain_size": L}
    ret_list = []
    for component in component_list:
        ret_list.append(val_dict[component])

    print("reading done!")
    return ret_list