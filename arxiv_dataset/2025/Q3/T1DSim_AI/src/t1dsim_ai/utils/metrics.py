import numpy as np


def get_TIR(cgm, lim_inf=70, lim_sup=180):
    cgm = np.array(cgm)
    cgm_proc = cgm.copy()
    cgm_proc = cgm_proc[~np.isnan(cgm)]
    return np.sum((cgm_proc >= lim_inf) & (cgm_proc <= lim_sup)) / cgm_proc.size


def get_TBR70(cgm, lim_inf=70):
    cgm = np.array(cgm)
    cgm_proc = cgm.copy()
    cgm_proc = cgm_proc[~np.isnan(cgm)]
    return np.sum(cgm_proc < lim_inf) / cgm_proc.size


def get_TAR180(cgm, lim_sup=180):
    cgm = np.array(cgm)
    cgm_proc = cgm.copy()
    cgm_proc = cgm_proc[~np.isnan(cgm)]
    return np.sum(cgm_proc > lim_sup) / cgm_proc.size


def get_glucose_variability(cgm):
    cgm = np.array(cgm)
    return np.nanstd(cgm) / np.nanmean(cgm)
