import numpy as np
import os
import subprocess as subp
import time
import glob
from astropy.io import fits as pyfits
import scipy.optimize as opt
import scipy.stats as st
from itertools import groupby
import matplotlib.pylab as plt
from multiprocessing import Pool
import re

import aztools as az


def fit_xspec_model(fit_func, spec_ids, base_dir, suff='', **kwargs):
    '''Call fit_func from fit.tcl to model the spectra in spec_ids
        and read the fit parameters. If the fitting has already been done,
        just read it.
    
    Parameters:
        fit_func: tcl function in fit.tcl to all; 
            It should be of the form proc fit_2a {sfile {suff ""}} {...}
        spec_ids: list of spec ids so the files are: spec_$id.grp
        base_dir: directory containing the fit.tcl file
        suff: any extra suffix for the saved fit files. 
            The saved files will be: fits/{fit_func}{suff}__{ispec}
     
     Keywords:
        read_fit: read fit result? use False when errors are not needed, so
            no log files exist. Default: True
        spec_root: root name for the spectra. Default: spec_%d.grp
        ext_check: file extention to use when checking whether the fit has already run
            or not. Default: xcm 
        extra_args: extra arguments to fit_func; default nothing
        parallel: run in parallel; Default: True
            
    Returns: an array of the fit parameters of shape: (nspec, npar, 4(par, err, err+, err-))
    '''    
    read_fit = kwargs.get('read_fit', True)
    spec_root = kwargs.get('spec_root', 'spec_%d.grp')
    ext_check = kwargs.get('ext_check', 'xcm')
    extra_args = kwargs.get('extra_args', '')
    parallel = kwargs.get('parallel', True)
           
    
    procs = []
    for ispec in spec_ids:
        # if fit already done, skip
        if os.path.exists('fits/%s%s__%d.%s'%(fit_func, suff, ispec, ext_check)): continue
        
        tcl  = 'source %s/fit.tcl\n'%base_dir
        tcl += '%s %s %s__%d %s\nexit\n'%(fit_func, spec_root%ispec, suff, ispec, extra_args) 
        xcm = 'tmp_%d.xcm'%ispec
        with open(xcm, 'w') as fp: fp.write(tcl)
        cmd = 'xspec - %s > tmp_%d.log 2>&1'%(xcm, ispec)
        if parallel:
            time.sleep(0.1)
            p = subp.Popen(['/bin/bash', '-i', '-c', cmd])
            procs.append(p)
            if len(procs)==30:
                for p in procs: p.wait()
                procs = []
        else:
            p = subp.call(['/bin/bash', '-i', '-c', cmd])
    # wait for the tasks to end
    for p in procs: p.wait()
    _ = os.system('rm tmp_*.??? >/dev/null 2>&1')
    
    # read the fit #
    if not read_fit: return
    fit = []
    for ispec in spec_ids:
        # exception for missing mos-1 data
        if not os.path.exists('fits/%s%s__%d.log'%(fit_func, suff, ispec)):
            print('missing fits/%s%s__%d.log'%(fit_func, suff, ispec))
            fit.append(fit[-1]*np.nan)
            continue
        fit.append(np.loadtxt('fits/%s%s__%d.log'%(fit_func, suff, ispec), usecols=[0,1,2,3]))
    return np.array(fit)

def ftest(c2, d2, c1, d1):
    """Do F-test"""
    fstat = ((c1-c2)/(d1-d2)) / (c2/d2)
    fprob = st.f.cdf(fstat, d1-d2, d2)
    fpval = 1 - fprob
    return fstat, fprob, fpval


def write_resid(base_dir, spec_ids, suff, extra_cmds='', avg_iref=-1, avg_bin=True,
                     outdir='results', z=0.00618):
    """Plot the residuals from fits of the form fit_{suff}__{ispec}
    
    spec_ids: [1,2,3,...]
    suff: e.g. indiv_1l, 2a etc
    extra_cmds: any extra xspec commands between loading the data and plotting.
        e.g. removing cflux and renormalizing.
    avg_iref: reference ispec when averaging; -1, select the first even array
    outdir: output directory
    """
    os.system('mkdir -p %s'%outdir)
    outfile = '%s/fit_%s.plot'%(outdir,suff)
    # individual fits #
    for ispec in spec_ids:
        tcl  = 'source %s/fit.tcl\nsetpl ener\nsetpl redshift %g\n'%(base_dir, z)
        tcl += '@fits/fit_%s__%d.xcm\n%s\n'%(suff, ispec, extra_cmds)
        tcl += 'fit 1000\naz_plot_unfold u tmp_%d %s__%d 1 1\n'%(ispec, suff, ispec)
        with open('tmp.xcm', 'w') as fp: fp.write(tcl)
        cmd = 'xspec - tmp.xcm > tmp.log 2>&1'
        p = subp.call(['/bin/bash', '-i', '-c', cmd])
    os.system("ls -1 tmp_*.plot|sort -t'_' -n -k2| xargs cat > %s/fit_%s.plot"%
              (outdir, suff))
    _ = os.system('rm tmp.??? tmp_*plot')
    
    # average residuals #
    # read and group the spectral data #
    lines = open(outfile).readlines()
    grp = [list(v) for k,v in groupby(lines, lambda l: (len(l)==0 or l[0]=='d') )]
    grp = [np.array([x.split() for x in g if x!='\n'], np.double) for g in grp if len(g)>4]

    dat = grp[::4]
    mod = grp[1::4]
    mod_spec = []
    for m,d in zip(mod, dat):
        mod_spec.append(np.array(d))
        mod_spec[-1][:,3] = 0
        mod_spec[-1][:,2] = m[:,0]

    # choose  some reference grid #
    iref = avg_iref
    if iref == -1:
            ilen = [i for i,d in enumerate(dat) if len(d)%2==0]
            iref = ilen[0]
    egrid_ref = np.array([dat[iref][:,0]-dat[iref][:,1], dat[iref][:,0]+dat[iref][:,1]]).T
    if avg_bin:
        egrid_ref = np.array([egrid_ref[::2,0], egrid_ref[1::2,1]]).T

    # map of spectra and models to the same reference grid #
    en_new, spec_new = spec_common_grid(dat, egrid_ref)
    _, mod_new = spec_common_grid(mod_spec, egrid_ref)

    # calculate residuals #
    dat_tot = np.array([ np.mean(spec_new[:,0], 0), 
                        (np.sum(spec_new[:,1]**2, 0)**0.5)/len(spec_new)])
    mod_tot = np.mean(mod_new[:,0], 0)
    #return en_new, spec_new, mod_new, dat_tot, mod_tot
    del_tot = (dat_tot[0]-mod_tot) / dat_tot[1]
    rat_tot = [dat_tot[0]/mod_tot, dat_tot[1]/mod_tot]

    # update the results file #results/fit_indiv_1.plot
    text = '\n\ndescriptor en_%s__tot,+- del_%s__tot,+- rat_%s__tot,+-\n'%tuple([suff]*3)
    text += '\n'.join(['{:.5} {:.5} {:.5} 1.0 {:.5} {:.5}'.format(*z) for z in 
                       zip(en_new[0], en_new[1], del_tot, rat_tot[0], rat_tot[1])])
    
    # add binned individual spectra #
    txt1 = ' '.join(['del_%s__g%d,+- rat_%s__g%d,+-'%(suff, ispec, suff, ispec) 
                        for ispec in spec_ids])
    txt2 = '\n'.join([' '.join(['{:.5} 1.0 {:.5} {:.5}'.format(
            (spec_new[ispec,0,ie]-mod_new[ispec,0,ie])/spec_new[ispec,1,ie],
            spec_new[ispec,0,ie]/mod_new[ispec,0,ie], spec_new[ispec,1,ie]/mod_new[ispec,0,ie]
        )
        for ispec in range(len(spec_ids))]) for ie in range(len(en_new[0]))])
    text += '\n\ndescriptor ' + txt1 + '\n' + txt2
    with open(outfile, 'a') as fp: fp.write(text)



def spec_common_grid(spectra, egrid):
    """Map a list of spectra into a single energy grid
    spectra: a list of (x,xe,y,ye) spectra
    egrid: array(nen, 2) giving low/high bin boundaries
    
    Returns: en_new(2,nen), spectra_new(nspec,2,nen)
    """
    nspec = len(spectra)
    spectra_new = np.zeros((nspec, 3, len(egrid)))
    for ispec in range(nspec):
        # the energy grid to be aligned #
        ebins = np.array([spectra[ispec][:,0]-spectra[ispec][:,1],
                          spectra[ispec][:,0]+spectra[ispec][:,1]]).T
        # loop through elements of the reference grid #
        # for each, sum the bins from the other grids that
        # fall into the current bin of the reference
        for iref, eref in enumerate(egrid):
            for ibin, ebin in enumerate(ebins):
                if ebin[1] < eref[0] or ebin[0] > eref[1]:
                    continue
                erange = (np.min([eref[1], ebin[1]]) - np.max([eref[0], ebin[0]]))
                efrac  = erange / (ebin[1] - ebin[0])
                spectra_new[ispec, 0, iref] += efrac * spectra[ispec][ibin,2]
                spectra_new[ispec, 1, iref] += (efrac * spectra[ispec][ibin,3])**2
                spectra_new[ispec, 2, iref] += efrac
    # spectra_new[:,2] is the fractional exposure in each bin
    spectra_new[:,0] /= spectra_new[:,2]
    spectra_new[:,1] = (spectra_new[:,1]/spectra_new[:,2])**0.5
    spectra_new = spectra_new[:,:2,]
    en_new = np.array([np.sum(egrid, 1)/2, np.diff(egrid, 1)[:,0]/2])
    return en_new, spectra_new