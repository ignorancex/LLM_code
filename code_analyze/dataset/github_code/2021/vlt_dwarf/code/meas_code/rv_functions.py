import pyfits
import numpy as np
import math as m
import matplotlib.pyplot as plt

def lp_post(rv, rvmin, rvmax, mask, wl, model, obj, objerr):
    c = 2.99792e5
    lp = -np.inf
    #print d

    if rv < rvmax and rv > rvmin:
        z = rv/c
        lp_prior=0.0

        new_wl = wl*(1+z)
        model = np.interp(wl,new_wl,model)
        model = model[mask]
        obj = obj[mask]
        objerr = objerr[mask]

        a = -0.5 * np.sum(np.log(objerr**2))
        b = -0.5 * np.sum((obj-model)**2/(objerr**2))
        c = -1. * (obj.size)/2. * np.log(2*np.pi)

        lp_post = a + b + c
        #lp_post= - np.sum((obj-model)**2/(2.0*(objerr**2)))

        if np.isfinite(lp_post):
            lp=lp_post+lp_prior

    return lp

def chi2cal(theta, rvmin, rvmax, mask, wl, model, obj, objerr):
    c = 2.99792e5
    rv = theta
    z = rv/c

    new_wl = wl*(1+z)
    model = np.interp(wl,new_wl,model)
    model = model[mask]
    obj = obj[mask]
    objerr = objerr[mask]
    b = -0.5 * np.sum((obj-model)**2/(objerr**2))
    return b

def lp_post_err_scale(theta, rvmin, rvmax, mask, wl, model, obj, objerr):
    c = 2.99792e5
    lp = -np.inf
    rv, scale = theta
    #print d

    if rv < rvmax and rv > rvmin and scale > 0:
        z = rv/c
        lp_prior=0.0

        new_wl = wl*(1+z)
        model = np.interp(wl,new_wl,model)
        model = model[mask]
        obj = obj[mask]
        objerr = objerr[mask] * scale
        # break long equation into three parts
        a = -0.5 * np.sum(np.log(objerr**2))
        b = -0.5 * np.sum((obj-model)**2/(objerr**2))
        c = -1. * (obj.size)/2. * np.log(2*np.pi)
        lp_post = a + b + c
        if np.isfinite(lp_post):
            lp=lp_post+lp_prior

    return lp


def chi2cal_err_scale(theta, rvmin, rvmax, mask, wl, model, obj, objerr):
    c = 2.99792e5
    rv, scale = theta
    z = rv/c

    new_wl = wl*(1+z)
    model = np.interp(wl,new_wl,model)
    model = model[mask]
    obj = obj[mask]
    objerr = objerr[mask] * scale
    b = -0.5 * np.sum((obj-model)**2/(objerr**2))
    return b

def read_vlt_spec(inputfilename):
    temp = pyfits.open(inputfilename)
    hdr=temp[0].header
    xref=hdr['CRVAL2']
    delx=hdr['CDELT2']
    xrefpix=hdr['CRPIX2'] - 1.0
    num=hdr['NAXIS2']
    wl=np.arange(xref-xrefpix*delx,xref+(num-1e-10-xrefpix)*delx,delx)
    spec=temp[0].data
    dspec = temp[1].data
    return wl, spec, dspec


def normalize_spec(pixels, spec, dspec):
    idx = np.isnan(spec)
    spec[idx] = np.median(spec)
    dspec[idx] = np.median(dspec)

    z = np.polyfit(pixels, spec, 1, w = 1/dspec**2)
    p = np.poly1d(z)
    spec = spec/p(pixels)
    dspec = dspec/p(pixels)
    idx = spec < 0.85
    temp = np.array(list(spec))
    temp[idx] = np.median(spec)

    z = np.polyfit(pixels, temp, 0, w = 1/dspec**2)
    p = np.poly1d(z)
    spec = spec/p(pixels)
    dspec = dspec/p(pixels)
    return spec, dspec