import os
import sys
import numpy as np
import aztools as az
import matplotlib.pylab as plt
import subprocess as subp
import scipy.optimize as opt
import scipy.stats as st
from astropy.io import fits as pyfits
from astropy import time as TT

import fqlag

import importlib
importlib.reload(fqlag)
importlib.reload(az)



def read_lc(loc_info, nu_obsids, dt, nen, combine_ab=True, bgd=False):
    """Read multi-energy light curves and combine a and b
    Returns a list of shape: nen, nobs, (t,r,re)
    """
    base_dir, lcdir = loc_info
    lfile = 'lc_%s_%03g%s.npz'%(lcdir, dt, '_bgd' if bgd else '')

    if os.path.exists(lfile):
        print('reading data from %s ..'%lfile)
        LC = np.load(lfile, allow_pickle=True)['LC']
    else:
        # nobs*2, nen, 
        lc = [[az.LCurve.read_nu_lcurve('%s/data/nustar/%s_p/lc/%s/lc_%03g__%s__%d%s.lc'%(
                        base_dir, o, lcdir, dt, instr, ie+1, '_bk' if bgd else ''), min_exp=0.2)  
                      for ie in range(nen)] for o in nu_obsids for instr in ['a', 'b'] ]
        LC = []
        for ie in range(nen):
            Lc = []
            for ii in range(len(lc)//2):
                if combine_ab:
                    la, lb = az.LCurve.sync([lc[2*ii][ie], lc[2*ii+1][ie]])
                    Lc.append([la[0], la[1] + lb[1], (la[2]**2 + lb[2]**2)**0.5])
                else:
                    Lc.append([lc[2*ii  ][ie].time, lc[2*ii  ][ie].rate, lc[2*ii  ][ie].rerr])
                    Lc.append([lc[2*ii+1][ie].time, lc[2*ii+1][ie].rate, lc[2*ii+1][ie].rerr])
            LC.append(Lc)
        np.savez(lfile, LC=LC)
        print('saved data to %s!'%lfile)
    # LC: nen, nobs, (t,r,re)
    return LC


def remove_high_bgd(Lc, Lcb, nBins=15, ibin=-1):
    """Use background light curve to remove times of high flares
    We use a simple algorithm that use the histogram of background
    counts to remove outliers.
    
    Lc: nen, nlc, 3
    Lcb: nen, nlc, 3
    nBins: bins in histogram
    ib: which bin in Lcb to use. Default: last
    
    """
    # high background: use the last energy bin; then apply to all #
    igood = []
    for ilc,lc in enumerate(Lcb[ibin]):
        cut = 1000
        h  = np.histogram(lc[1][lc[1] < cut], nBins)
        i0 = np.where(h[0]==0)[0]
        i0 = i0[i0 > len(h[0])//2]
        while len(i0):
            cut = h[1][i0[0]+1]
            h  = np.histogram(lc[1][lc[1] < cut], nBins)
            i0 = np.where(h[0]==0)[0]
        igood.append(np.arange(len(lc[1]))[lc[1] < cut])
    Lc_ = [[np.array(list(l))[:,igood[il]] for il,l in enumerate(lc)] for lc in Lc]
    return Lc_


def split_LC_to_segments(LC, tlen, plot=False):
    """Split LC to segments
    LC is a list with shape: nen, nlc, 3 (t,r,re)
    tlen in same units as time axis
    
    return: newLC, sub_Idx
    where newLC has shape: nen, nseg, 3
    sub_Idx has shape: nen, nseg, stating the original index in LC of 
        earch segment (e.g. for plotting)
    
    """
    LC_, sub_Idx = [], []
    for ie, Lc in enumerate(LC):

        Lc_, sub_idx = [], []
        for ilc,lc in enumerate(Lc):

            tt = lc[0]
            tlen_f = (tt - tt[0]) / tlen
            tlen_f = np.abs(tlen_f - np.round(tlen_f))
            idx =  np.arange(len(tlen_f))[np.r_[True, tlen_f[1:] < tlen_f[:-1]] & 
                                          np.r_[tlen_f[:-1] < tlen_f[1:], True]]
            # re-adjust the segment lengths to accomodate leftover small segments
            if idx[-1] != len(tt):
                iextra = (len(tt)-1 - idx[-1])//(len(idx)-1)
                idx = [0] + [i+iextra for i in idx[1:-1]] + [len(tt)]

            for i,j in zip(idx[:-1], idx[1:]):
                if len(tt[i:j]) < 10: continue
                Lc_.append([lc[0][i:j], lc[1][i:j], lc[2][i:j]])
                sub_idx.append(ilc)
        sub_Idx.append(sub_idx)
        LC_.append(Lc_)
    
    if plot:
        ie = 0 # which energy?
        fig, ax = plt.subplots(len(LC[ie]), 1, figsize=(12, 12))
        tmax = np.max([t[0][-1]-t[0][0] for t in LC[ie]])/1e3
        for ilc,lc in enumerate(LC[ie]):
            ax[ilc].errorbar((lc[0] - lc[0][0])/1e3, lc[1], lc[2], fmt='o', ms=10, alpha=0.5)
            ax[ilc].set_xlim([0, tmax*1.2])

            idx = np.argwhere(np.array(sub_Idx[0]) == ilc)[:,0]
            for i in idx:
                lc_ = LC_[0][i]
                ax[ilc].errorbar((lc_[0] - lc[0][0])/1e3, lc_[1], lc_[2], fmt='s', ms=10, alpha=0.5)
    
    return LC_, sub_Idx


def get_fq_bins(Lc, dt, mode=1, add_dummy=True, nyquist=0.5, **kw):
    """Get Nfq frequency bins in logspace
    mode = 1; simple log using Nfq in **kw
    mode = 2; group_array with parameters e.g. by_n, min_per_bin in **kw
    
    """
    tArr = [l[0] for l in Lc]
    
    if mode == 1: # simple log
        Nfq = kw.get('Nfq', 10)
        fqmin = np.min([1./(t[-1]-t[0]) for t in tArr])
        fqL = np.logspace(np.log10(fqmin), np.log10(nyquist/dt), Nfq + (3 if add_dummy else 0))
    else:
        by_n = kw.get('by_n', [10, 1.5])
        min_per_bin = kw.get('min_per_bin', 10)
        fq = np.sort(np.concatenate([np.linspace(1./(t[-1]-t[0]), nyquist/dt, len(t)//2) for t in tArr]))
        idx = az.misc.group_array(fq, by_n=by_n, do_unique=True, min_per_bin=min_per_bin)
        fqL  = np.array([fq[i[0]] for i in idx] + [fq[idx[-1][-1]]])
    
    if add_dummy:
        fqL = np.r_[fqL[0]*0.5, 0.5*fqL[1], fqL[2:-1], fqL[-1]*2]

    nfq = len(fqL) - 1
    fqd = np.exp((np.log(fqL[1:])+np.log(fqL[:-1]))/2.0)
    print('nfq: ', nfq)
    print('fqL: ', ' '.join(['%g'%x for x in fqL]))
    
    return fqL, fqd


def calculate_psd(lc, fqL, npzfile, **kwargs):
    """Calculate psd and save it to npz or load it if saved
    lc: a list of shape (nlc, 3:t,r,re)
    fqL: an array or list of frequency bin limits

    kwargs:
    Dt: dt or None for the type of fqlag model to use
    do_bin: calculate psd at bins defined by fqL
    do_pl: fit pl model beyween fqL[0] and fqL[-1]
    do_bpl: fit bending pl model beyween fqL[0] and fqL[-1]
    NFQ: number of frequencies when using pl and bpl
    errors: calculate errors?
    """
    
    Dt     = kwargs.get('Dt', None)
    do_bin = kwargs.get('do_bin', True) 
    do_pl  = kwargs.get('do_pl', False)
    do_bpl = kwargs.get('do_pl', False)
    NFQ    = kwargs.get('NFQ', 30)
    errors = kwargs.get('errors', False)
    
    nfq = len(fqL) - 1
    fqd = np.exp((np.log(fqL[1:])+np.log(fqL[:-1]))/2.0)
    
    if npzfile[0] == '!':
        npzfile = npzfile[1:]
        os.system('rm %s '%npzfile)
    
    if os.path.exists(npzfile):
        result = np.load(npzfile, allow_pickle=True)['result'][()]
    else:
        result = {'fqL': fqL}
        # bins #
        if do_bin:
            mods   = [fqlag.lPsd(l[0], l[1], l[2], fqL, dt=Dt) for l in lc]
            modBin = fqlag.multiFqLagBin(mods)
            p0     = np.zeros(nfq) - 1.0
            fBin   = fqlag.misc.maximize(modBin, p0)
            if errors:
                fBin = fqlag.misc.errors(modBin, fBin[0])
                
            pBin   = np.array([fqd, fBin[0], fBin[1]])
            result.update({'p_Bin': pBin, 'mod_Bin':modBin, 'fit_Bin':fBin})

        # using Psdf for functional forms #
        Mod = []
        if do_pl:  Mod.append('pl')
        if do_bpl: Mod.append('bpl')
        for M in Mod:
            mods = [fqlag.Psdf(l[0], l[1], l[2], fqL[[0,-1]], M, dt=Dt, NFQ=NFQ) for l in lc]
            pmod = fqlag.multiFqLagBin(mods)
            p0  = np.array([-6., -2.]) if M=='pl' else np.array([-6., -2, -9.]) 
            fit = fqlag.misc.maximize(pmod, p0)
            if errors:
                fit = fqlag.misc.errors(pmod, fit[0])
            p   = fit[:2]
            ff  = np.logspace(np.log10(fqL[0]), np.log10(fqL[-1]), 50)
            pr  = np.random.randn(200)*p[1][:,None] + p[0][:,None]
            ppr = np.array([mods[0].psd_func(ff, x) for x in pr.T])
            m   = np.array([ff, np.log(ppr).mean(0), np.log(ppr).std(0)])
            result.update({'p_%s'%M:p, 'mod_%s'%M:pmod, 'm_%s'%M:m, 'fit_%s'%M:fit})
        np.savez(npzfile, result=result)
        
    return result


def calculate_lag(Lc, fqL, dt, ebins, npzfile, **kwargs):
    """Calculate psd and save it to npz or load it if saved
    Lc: a list of shape (nen, nlc, 3:t,r,re)
    dt: time bins
    ebins: a space-separated string of energy bin boundaries
    fqL: an array or list of frequency bin limits
    npzfile: npz file to save to or to read from if it exists

    kwargs:
    Dt: dt or None for the type of fqlag model to use
    iref: and index or list of energy bin indices for the reference band. Default: all
    iEn: a list of lists of bin indices to calculate the lags for relative to the reference.
    err_kw: dict to be passed to fqlag.misc.errors
    irand: a random number (int or str) for caching, if given, 
            we look for 'tmp_%s__%d.npz'%(irand, ie+1).
            otherwise create one. used in case calculations is interrupted.
    iLc: a list of light curve indices to use
    errors: calculate errors?
    return_models: return models
    """
    
    iref   = kwargs.get('iref', None)
    iEn    = kwargs.get('iEn', None)
    errors = kwargs.get('errors', False)
    mcmc   = kwargs.get('mcmc', None)
    Dt     = kwargs.get('Dt', None)
    err_kw = kwargs.get('err_kw', {})
    irand  = kwargs.get('irand', None)
    iLc    = kwargs.get('iLc', None)
    logmod = kwargs.get('logmod', True)
    return_models = kwargs.get('return_models', False)
    mcmc_sigma_f  = kwargs.get('mcmc_sigma_f', 0.3)
    
    # energy axis #
    enL = np.array(ebins.split(), np.double)
    en, ene = (enL[1:] + enL[:-1])/2, (enL[1:] - enL[:-1])/2
    nen = len(en)
    if nen != len(Lc):
        raise ValueError('ebins does not match the input data in Lc')
    if iEn is None:
        iEn = [[i] for i in range(nen)]

    # frequency axis #
    fqL = np.array(fqL)
    fqd = np.exp((np.log(fqL[1:])+np.log(fqL[:-1]))/2.0)
    nfq = len(fqd)

    # light curves to use #
    nlc = len(Lc[0])
    if iLc is None:
        iLc = list(range(nlc))
    

    # cache file reset? #
    if npzfile[0] == '!':
        npzfile = npzfile[1:]
        os.system('rm %s '%npzfile)
    
    # do we have a cache file? #
    if os.path.exists(npzfile):
        print('cache file %s found. Reading ...!'%npzfile)
        saved = np.load(npzfile, allow_pickle=True)
        Res = saved['Res']
        En, fqd, extra = saved['En'], saved['fqd'], saved['extra'][()]
        ###
        En = []
        for iiE,iE in enumerate(iEn):
            En_  = [(enL[np.max(iE)+1] + enL[np.min(iE)])/2, 
                    (enL[np.max(iE)+1] - enL[np.min(iE)])/2]
            En.append(En_)
        En  = np.array(En).T
        np.savez(npzfile, Res=Res, En=En, fqd=fqd, extra=extra)
        ###
        return Res, En, fqd, extra

    
    
    # reference #
    if iref is None:
        iref = list(range(nen))
    refR  = [np.sum([Lc[ie][ilc][1] for ie in iref], 0) for ilc in iLc]
    refRe = [np.sum([Lc[ie][ilc][2]**2 for ie in iref], 0)**0.5 for ilc in iLc]
    refT  = [Lc[0][ilc][0] for ilc in iLc]


    # random number to keep track of temporary files
    if irand is None:
        irand = '%08d'%np.random.randint(100000000)
    if not isinstance(irand, str): irand = '%d'%irand
    print('using irand: %s\n'%irand)
    os.write(1, ('\nusing irand: %s\n'%irand).encode())


    # loop through energy #
    Res, Fits, Mods, En, Chains = [], [], [], [], []
    for iiE,iE in enumerate(iEn):

        print('\n----- en: %d of %d -----\n'%(iiE+1, len(iEn)))
        tmpfile = 'tmp_%s__%d.npz'%(irand, iiE+1)

        # load cached file if we have one; otherwise, do calculations #
        if os.path.exists(tmpfile):
            saved = np.load(tmpfile, allow_pickle=True)
            res, fit, En_ = saved['res'], saved['fit'], saved['En']
        else:

            # band of interest
            r  = [np.sum([Lc[ie][ilc][1] for ie in iE], 0) for ilc in iLc]
            re = [np.sum([Lc[ie][ilc][2]**2 for ie in iE], 0)**0.5 for ilc in iLc]
            t  = [Lc[iE[0]][ilc][0] for ilc in iLc]

            # remove any contribution of iE to iref
            iOv = [i for i in iE if i in iref]
            if len(iOv):
                overlapR  = [np.sum([Lc[ie][ilc][1] for ie in iOv], 0) for ilc in iLc]
                overlapRe = [np.sum([Lc[ie][ilc][2]**2 for ie in iOv], 0)**0.5 for ilc in iLc]
                rRef  = [refR[ilc]  - overlapR[ilc] for ilc in range(len(iLc))]
                reRef = [(refRe[ilc]**2 - overlapRe[ilc]**2)**0.5 for ilc in range(len(iLc))]
            else:
                rRef  = list(refR)
                reRef = list(refRe)


            # psd of reference band #
            # first do it in log-space regardless of logmod because it is more stable
            mods     = [fqlag.Psd(t[ilc], rRef[ilc], reRef[ilc], fqL, dt=Dt, log=True) for ilc in range(len(iLc))]
            pMod_Ref = fqlag.multiFqLagBin(mods)
            p0       = np.zeros(nfq) + (-1.0 if logmod else 1.)
            #limits   = ([[-30,30]]*nfq) if logmod else ([[1e-6,1e10]]*nfq)
            limits   = ([[-30,30]]*nfq)
            pFit_Ref = fqlag.misc.maximize(pMod_Ref, p0, limits=limits)
            # perturb the solution to make sure it is the maximum likelihood
            for i in range(3):
                pnew = fqlag.misc.maximize(pMod_Ref, np.random.randn(nfq)*pFit_Ref[1]+pFit_Ref[0], limits=limits)
                if pnew[2].fun < pFit_Ref[2].fun:
                    pFit_Ref = pnew
            # if linear psd is requested, then we redo it
            if not logmod:
                mods     = [fqlag.Psd(t[ilc], rRef[ilc], reRef[ilc], fqL, dt=Dt, log=False) for ilc in range(len(iLc))]
                pMod_Ref = fqlag.multiFqLagBin(mods)
                p0     = np.exp(pFit_Ref[0])
                limits = ([[1e-6,1e10]]*nfq)
                pFit_Ref = fqlag.misc.maximize(pMod_Ref, p0, limits=limits)
            
            if errors:
                print('** calculating errors **')
                pFit_Ref = fqlag.misc.errors(pMod_Ref, pFit_Ref[0], skip_ipars=[0,nfq-1], limits=limits, **err_kw)

            # cross spectrum #
            cmods = [fqlag.Psi([t[ilc], t[ilc]], [rRef[ilc], r[ilc]], [reRef[ilc], re[ilc]], fqL, 
                                pFit_Ref[0], dt=Dt, log=logmod) for ilc in range(len(iLc))]
            p0     = np.zeros(2*nfq, np.double) if logmod else np.ones(2*nfq, np.double)
            limits = ([[-30,30]]*(2*nfq)) if logmod else (([[1e-6,1e10]]*nfq) + ([[-30,30]]*nfq))
            cMod  = fqlag.multiFqLagBin(cmods)
            cFit  = fqlag.misc.maximize(cMod, p0, limits=limits)
            # perturb the solution to make sure it is the maximum likelihood
            for i in range(2):
                cnew = fqlag.misc.maximize(cMod, np.random.randn(2*nfq)*cFit[1]+cFit[0], limits=limits)
                if cnew[2].fun < cFit[2].fun:
                    cFit = cnew
            if errors:
                print('** calculating errors **')
                cFit = fqlag.misc.errors(cMod, cFit[0], skip_ipars=[0, nfq-1, nfq, 2*nfq-1], limits=limits, **err_kw)
            amp, phi = np.split(np.array(cFit[:2]), 2, axis=-1)
            phi[0] = (phi[0] + np.pi) % (2*np.pi) - np.pi
            lag = phi / (2*np.pi*fqd)

            # shape: (9, nfq), 9: fq, p, pE, amp, ampE, phi, phiE, lag, lagE
            res  = np.concatenate([[fqd], np.array(pFit_Ref[:2]), amp, phi, lag])
            fit  = [pFit_Ref, cFit]
            mods = [pMod_Ref, cMod]
            En_  = [(enL[np.max(iE)+1] + enL[np.min(iE)])/2, 
                    (enL[np.max(iE)+1] - enL[np.min(iE)])/2]
            
            chain = None
            if not mcmc is None:
                print('** mcmc chain **')
                nwk, nrun = mcmc
                chain = fqlag.misc.run_mcmc(cMod, cFit[0], cFit[1], nwalkers=nwk, nrun=nrun, 
                            sigma_f=mcmc_sigma_f, limits=limits, 
                            iphi=list(range(len(cFit[0])//2, len(cFit[0]))))

            # save tmp file
            np.savez(tmpfile, res=res, fit=fit, mods=mods, En=En_, chain=chain)
            print('saved temporary data to %s'%tmpfile)

        Res.append(res)
        Fits.append(fit)
        Mods.append(mods)
        En.append(En_)
        Chains.append(chain)


    # Res: (nen, 9:[fq,p,pe,cx,cxe,phi,phie,lag,lage], nfq)
    Res = np.array(Res)
    En  = np.array(En).T
    Chains = np.array(Chains) if not mcmc is None else None
    extra = {'fits': Fits, 'fqL':fqL, 'en':[en, ene], 'chains':Chains}
    
    # models take a lot of memory, so we return them only when requested
    if return_models:
        extra['mods'] = Mods
    else:
        np.savez(npzfile, Res=Res, En=En, fqd=fqd, extra=extra)
    os.system('rm tmp_%s__*.npz'%(irand))
    print('cleaned temporary files, and saved results to %s\n'%npzfile)
    return Res, En, fqd, extra


def calculate_lag_cross(Lc, fqL, dt, ebins, npzfile, **kwargs):
    """Calculate psd and save it to npz or load it if saved; 
    similar to calculate_lag, but fits for the cross spectrum instead of the tranfer function
    
    
    Lc: a list of shape (nen, nlc, 3:t,r,re)
    dt: time bins
    ebins: a space-separated string of energy bin boundaries
    fqL: an array or list of frequency bin limits
    npzfile: npz file to save to or to read from if it exists

    kwargs:
    Dt: dt or None for the type of fqlag model to use
    iref: and index or list of energy bin indices for the reference band. Default: all
    iEn: a list of lists of bin indices to calculate the lags for relative to the reference.
    err_kw: dict to be passed to fqlag.misc.errors
    irand: a random number (int or str) for caching, if given, 
            we look for 'tmp_%s__%d.npz'%(irand, ie+1).
            otherwise create one. used in case calculations is interrupted.
    iLc: a list of light curve indices to use
    errors: calculate errors?
    return_models: return models
    """
    
    iref   = kwargs.get('iref', None)
    iEn    = kwargs.get('iEn', None)
    errors = kwargs.get('errors', False)
    mcmc   = kwargs.get('mcmc', None)
    Dt     = kwargs.get('Dt', None)
    err_kw = kwargs.get('err_kw', {})
    irand  = kwargs.get('irand', None)
    iLc    = kwargs.get('iLc', None)
    logmod = kwargs.get('logmod', True)
    return_models = kwargs.get('return_models', False)
    mcmc_sigma_f  = kwargs.get('mcmc_sigma_f', 0.3)
    pcxd   = kwargs.get('pcxd', False)
    
    # energy axis #
    enL = np.array(ebins.split(), np.double)
    en, ene = (enL[1:] + enL[:-1])/2, (enL[1:] - enL[:-1])/2
    nen = len(en)
    if nen != len(Lc):
        raise ValueError('ebins does not match the input data in Lc')
    if iEn is None:
        iEn = [[i] for i in range(nen)]

    # frequency axis #
    fqL = np.array(fqL)
    fqd = np.exp((np.log(fqL[1:])+np.log(fqL[:-1]))/2.0)
    nfq = len(fqd)

    # light curves to use #
    nlc = len(Lc[0])
    if iLc is None:
        iLc = list(range(nlc))
    

    # cache file reset? #
    if npzfile[0] == '!':
        npzfile = npzfile[1:]
        os.system('rm %s '%npzfile)
    
    # do we have a cache file? #
    if os.path.exists(npzfile):
        print('cache file %s found. Reading ...!'%npzfile)
        saved = np.load(npzfile, allow_pickle=True)
        Res = saved['Res']
        En, fqd, extra = saved['En'], saved['fqd'], saved['extra'][()]
        ###
        En = []
        for iiE,iE in enumerate(iEn):
            En_  = [(enL[np.max(iE)+1] + enL[np.min(iE)])/2, 
                    (enL[np.max(iE)+1] - enL[np.min(iE)])/2]
            En.append(En_)
        En  = np.array(En).T
        np.savez(npzfile, Res=Res, En=En, fqd=fqd, extra=extra)
        ###
        return Res, En, fqd, extra

    
    
    # reference #
    if iref is None:
        iref = list(range(nen))
    refR  = [np.sum([Lc[ie][ilc][1] for ie in iref], 0) for ilc in iLc]
    refRe = [np.sum([Lc[ie][ilc][2]**2 for ie in iref], 0)**0.5 for ilc in iLc]
    refT  = [Lc[0][ilc][0] for ilc in iLc]


    # random number to keep track of temporary files
    if irand is None:
        irand = '%08d'%np.random.randint(100000000)
    if not isinstance(irand, str): irand = '%d'%irand
    print('using irand: %s\n'%irand)
    os.write(1, ('\nusing irand: %s\n'%irand).encode())


    # loop through energy #
    Res, Fits, Mods, En, Chains = [], [], [], [], []
    for iiE,iE in enumerate(iEn):

        print('\n----- en: %d of %d -----\n'%(iiE+1, len(iEn)))
        tmpfile = 'tmp_%s__%d.npz'%(irand, iiE+1)

        # load cached file if we have one; otherwise, do calculations #
        if os.path.exists(tmpfile):
            saved = np.load(tmpfile, allow_pickle=True)
            res, fit, En_ = saved['res'], saved['fit'], saved['En']
        else:

            # band of interest
            r  = [np.sum([Lc[ie][ilc][1] for ie in iE], 0) for ilc in iLc]
            re = [np.sum([Lc[ie][ilc][2]**2 for ie in iE], 0)**0.5 for ilc in iLc]
            t  = [Lc[iE[0]][ilc][0] for ilc in iLc]

            # remove any contribution of iE to iref
            iOv = [i for i in iE if i in iref]
            if len(iOv):
                overlapR  = [np.sum([Lc[ie][ilc][1] for ie in iOv], 0) for ilc in iLc]
                overlapRe = [np.sum([Lc[ie][ilc][2]**2 for ie in iOv], 0)**0.5 for ilc in iLc]
                rRef  = [refR[ilc]  - overlapR[ilc] for ilc in range(len(iLc))]
                reRef = [(refRe[ilc]**2 - overlapRe[ilc]**2)**0.5 for ilc in range(len(iLc))]
            else:
                rRef  = list(refR)
                reRef = list(refRe)


            # psd of reference band #
            # first do it in log-space regardless of logmod because it is more stable
            mods     = [fqlag.Psd(t[ilc], rRef[ilc], reRef[ilc], fqL, dt=Dt, log=True) for ilc in range(len(iLc))]
            pMod_Ref = fqlag.multiFqLagBin(mods)
            p0       = np.zeros(nfq) + (-1.0 if logmod else 1.)
            #limits   = ([[-30,30]]*nfq) if logmod else ([[1e-6,1e10]]*nfq)
            limits   = ([[-30,30]]*nfq)
            pFit_Ref = fqlag.misc.maximize(pMod_Ref, p0, limits=limits)
            # perturb the solution to make sure it is the maximum likelihood
            for i in range(3):
                pnew = fqlag.misc.maximize(pMod_Ref, np.random.randn(nfq)*pFit_Ref[1]+pFit_Ref[0], limits=limits)
                if pnew[2].fun < pFit_Ref[2].fun:
                    pFit_Ref = pnew
            # if linear psd is requested, then we redo it
            if not logmod:
                mods     = [fqlag.Psd(t[ilc], rRef[ilc], reRef[ilc], fqL, dt=Dt, log=False) for ilc in range(len(iLc))]
                pMod_Ref = fqlag.multiFqLagBin(mods)
                p0     = np.exp(pFit_Ref[0])
                limits = ([[1e-6,1e10]]*nfq)
                pFit_Ref = fqlag.misc.maximize(pMod_Ref, p0, limits=limits)
            
            if errors:
                print('** calculating errors **')
                pFit_Ref = fqlag.misc.errors(pMod_Ref, pFit_Ref[0], skip_ipars=[0,nfq-1], limits=limits, **err_kw)
            
            
            #####################
            # psd of other band #
            # first do it in log-space regardless of logmod because it is more stable
            mods     = [fqlag.Psd(t[ilc], r[ilc], re[ilc], fqL, dt=Dt, log=True) for ilc in range(len(iLc))]
            pMod     = fqlag.multiFqLagBin(mods)
            p0       = np.zeros(nfq) + (-1.0 if logmod else 1.)
            #limits   = ([[-30,30]]*nfq) if logmod else ([[1e-6,1e10]]*nfq)
            limits   = ([[-30,30]]*nfq)
            pFit     = fqlag.misc.maximize(pMod, p0, limits=limits)
            # perturb the solution to make sure it is the maximum likelihood
            for i in range(3):
                pnew = fqlag.misc.maximize(pMod, np.random.randn(nfq)*pFit[1]+pFit[0], limits=limits)
                if pnew[2].fun < pFit[2].fun:
                    pFit = pnew
            # if linear psd is requested, then we redo it
            if not logmod:
                mods     = [fqlag.Psd(t[ilc], r[ilc], re[ilc], fqL, dt=Dt, log=False) for ilc in range(len(iLc))]
                pMod     = fqlag.multiFqLagBin(mods)
                p0     = np.exp(pFit[0])
                limits = ([[1e-6,1e10]]*nfq)
                pFit   = fqlag.misc.maximize(pMod, p0, limits=limits)
            
            if errors:
                print('** calculating errors **')
                pFit_Ref = fqlag.misc.errors(pMod, pFit[0], skip_ipars=[0,nfq-1], limits=limits, **err_kw)

                
                
            #####################
            # cross spectrum #
            cmods = [fqlag.Cxd([t[ilc], t[ilc]], [rRef[ilc], r[ilc]], [reRef[ilc], re[ilc]], fqL, 
                                pFit_Ref[0], pFit[0], dt=Dt, log=logmod) for ilc in range(len(iLc))]
            #p0     = np.zeros(2*nfq, np.double) if logmod else np.ones(2*nfq, np.double)
            p0     = np.concatenate([0.4*(pFit_Ref[0] + pFit[0]), np.zeros(nfq)])
            limits = ([[-30,30]]*(2*nfq)) if logmod else (([[1e-6,1e10]]*nfq) + ([[-30,30]]*nfq))
            cMod  = fqlag.multiFqLagBin(cmods)
            cFit  = fqlag.misc.maximize(cMod, p0, limits=limits)
            # perturb the solution to make sure it is the maximum likelihood
            for i in range(2):
                cnew = fqlag.misc.maximize(cMod, np.random.randn(2*nfq)*cFit[1]+cFit[0], limits=limits)
                if cnew[2].fun < cFit[2].fun:
                    cFit = cnew
            if errors:
                print('** calculating errors **')
                cFit = fqlag.misc.errors(cMod, cFit[0], skip_ipars=[0, nfq-1, nfq, 2*nfq-1], limits=limits, **err_kw)
                
            if pcxd:
                cmods = [fqlag.PCxd([t[ilc], t[ilc]], [rRef[ilc], r[ilc]], [reRef[ilc], re[ilc]], fqL, 
                                    dt=Dt, log=logmod) for ilc in range(len(iLc))]
                p0     = np.concatenate([pFit_Ref[0], pFit[0], cFit[0]])
                limits = ([[-30,30]]*(3*nfq)) if logmod else (([[1e-6,1e10]]*(3*nfq)) + ([[-30,30]]*nfq))
                cMod  = fqlag.multiFqLagBin(cmods)
                cFit  = fqlag.misc.maximize(cMod, p0, limits=limits)
                # perturb the solution to make sure it is the maximum likelihood
                for i in range(2):
                    cnew = fqlag.misc.maximize(cMod, np.random.randn(4*nfq)*cFit[1]+cFit[0], limits=limits)
                    if cnew[2].fun < cFit[2].fun:
                        cFit = cnew
                if errors:
                    print('** calculating errors **')
                    cFit = fqlag.misc.errors(cMod, cFit[0], skip_ipars=[0, nfq-1, nfq, 2*nfq-1], limits=limits, **err_kw)
            
            
                p1,p2,amp, phi = np.split(np.array(cFit[:2]), 4, axis=-1)
            else:
                amp, phi = np.split(np.array(cFit[:2]), 2, axis=-1)
            phi[0] = (phi[0] + np.pi) % (2*np.pi) - np.pi
            lag = phi / (2*np.pi*fqd)

            # shape: (9, nfq), 9: fq, p, pE, amp, ampE, phi, phiE, lag, lagE
            res  = np.concatenate([[fqd], np.array(pFit_Ref[:2]), amp, phi, lag])
            fit  = [pFit_Ref, cFit]
            mods = [pMod_Ref, cMod]
            En_  = [(enL[np.max(iE)+1] + enL[np.min(iE)])/2, 
                    (enL[np.max(iE)+1] - enL[np.min(iE)])/2]
            
            chain = None
            if not mcmc is None:
                print('** mcmc chain **')
                nwk, nrun = mcmc
                chain = fqlag.misc.run_mcmc(cMod, cFit[0], cFit[1], nwalkers=nwk, nrun=nrun, 
                            sigma_f=mcmc_sigma_f, limits=limits, 
                            iphi=list(range(len(cFit[0])//2, len(cFit[0]))))

            # save tmp file
            np.savez(tmpfile, res=res, fit=fit, mods=mods, En=En_, chain=chain)
            print('saved temporary data to %s'%tmpfile)

        Res.append(res)
        Fits.append(fit)
        Mods.append(mods)
        En.append(En_)
        Chains.append(chain)


    # Res: (nen, 9:[fq,p,pe,cx,cxe,phi,phie,lag,lage], nfq)
    Res = np.array(Res)
    En  = np.array(En).T
    Chains = np.array(Chains) if not mcmc is None else None
    extra = {'fits': Fits, 'fqL':fqL, 'en':[en, ene], 'chains':Chains}
    
    # models take a lot of memory, so we return them only when requested
    if return_models:
        extra['mods'] = Mods
    else:
        np.savez(npzfile, Res=Res, En=En, fqd=fqd, extra=extra)
    os.system('rm tmp_%s__*.npz'%(irand))
    print('cleaned temporary files, and saved results to %s\n'%npzfile)
    return Res, En, fqd, extra




def plot_lag(res, all_f=False):
    """Summary plot for the output of calculate_lag"""
    res, (en, ene), fqd = res[:3]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # fq #
    iFq = range(1,len(fqd)-1)
    if all_f: iFq = range(len(fqd))
    for ie in range(len(en)):
        ax[0].errorbar(fqd[iFq], res[ie,7][iFq], res[ie,8][iFq], fmt='o-')
    ax[0].plot(fqd[iFq], fqd[iFq]*0, '-')
    ax[0].set_xscale('log')
    
    # en #
    for ifq in iFq:
        ax[1].errorbar(en, res[:,7,ifq], res[:,8,ifq])
    ax[1].set_xscale('log')

    
def write_lag(Res, suff='', all_f=False, pha=False, null_tests=False, pha_fq=False):
    """Write the results of calculate_lag to ascii file
    with the option of saving pha files and running null tests for simple models
    """
    os.system('mkdir -p lag/plots')
    res, (en, ene), fqd = Res[:3]
    fqL = Res[3]['fqL']
    nen = len(en)
    iEn = range(nen)
    iFq = range(1,len(fqd)-1)
    if all_f: ii = range(len(fqd))
        
    if suff != '' and suff[0] != '_': suff = '_' + suff
        
    # vs fq #
    txt = '\ndescriptor fq%s %s\n'%(suff, ' '.join(['lfq%s__%d,+-'%(suff, i) for i in iEn]))
    txt += '\n'.join(['%g %s'%(fqd[ii], ' '.join(['%g %g'%(res[ie,7,ii], res[ie,8,ii]) 
                                                  for ie in iEn])) for ii in iFq])
 
    # vs en #
    txt += '\ndescriptor en%s,+- %s\n'%(suff, ' '.join(['len%s__%d,+-'%(suff, i) for i in iFq]))
    txt += '\n'.join(['%g %g %s'%(en[ie], ene[ie], ' '.join(['%g %g'%(res[ie,7,ii], res[ie,8,ii]) 
                                                  for ii in iFq])) for ie in iEn])
    # if we have errors from mcmc quantiles:
    if res.shape[1] == 11:
        txt += '\ndescriptor %s\n'%(' '.join(['len2%s__%d,+,-'%(suff, i) for i in iFq]))
        txt += '\n'.join(['%s'%( ' '.join(['%g %g %g'%(
            res[ie,7,ii], res[ie,9,ii]/(2*np.pi*res[ie,0,ii]), -res[ie,10,ii]/(2*np.pi*res[ie,0,ii])) 
                                                  for ii in iFq])) for ie in iEn])
    with open('lag/plots/lag%s.plot'%suff, 'w') as fp: fp.write(txt)
    
    # write pha #
    if pha_fq:
        os.system('mkdir -p lag/pha')
        os.system('rm lfq%s__* > /dev/null 2&1'%(suff))
        
        for ie in iEn:
            az.misc.write_pha_spec(fqL[:-1], fqL[1:], res[ie,7,:], res[ie,8,:], 
                                  'lfq%s__%d'%(suff, ie+1), verbosity=False)
        os.system('mv *.pha *.xsp *.rsp lag/pha')
    
    if pha:
        os.system('mkdir -p lag/pha')
        os.system('rm len%s__* > /dev/null 2&1'%(suff))
        
        
        for ii in iFq:
            az.misc.write_pha_spec(en-ene, en+ene, res[:,7,ii], res[:,8,ii], 
                                   'len%s__%d'%(suff, ii), verbosity=False)
            pha = 'len%s__%d.pha'%(suff, ii)
            with pyfits.open(pha) as fp:
                fp[1].header['XFLT0001'] = "fq: %g"%fqd[ii]
                fp.writeto('%s'%pha, overwrite=True)
        os.system('mv *.pha *.xsp *.rsp lag/pha')
    
    # null tests against a const and log-linear model #
    if null_tests:
        os.system('mkdir -p lag/model_tests')
        text = ''
        for ii in iFq:
            r = lag_en_tests(en, res[:,7,ii], res[:,8,ii], verbosity=False)
            text += '\n' + r[-1] + '\n'
        with open('lag/model_tests/test_%s.txt'%suff, 'w') as fp: fp.write(text)

            
def lag_en_tests(en, l, le, verbosity=True):
    """Test lag-enery against a null hypothesis of a const and log-linear model"""

    # const #
    def fit_1(x, *p): return x*0+p[0]
    f_1  = opt.curve_fit(fit_1, en, l, [l.mean()], sigma=le)
    c2_1 = np.sum(((l - fit_1(en, *f_1[0]))/le)**2)
    p_1  = 1 - st.chi2.cdf(c2_1, df=len(en)-1)
    nsigma = st.norm.ppf(p_1/2)
    text = '\n- fit 1: {} {:6.3} {:6.3} {:6.3}'.format(f_1[0], c2_1, p_1, nsigma)

    # log-linear
    def fit_2(x, *p): return p[0] + p[1] * np.log10(x)
    f_2  = opt.curve_fit(fit_2, en, l, [l.mean(), 0.1], sigma=le)
    c2_2 = np.sum(((l - fit_2(en, *f_2[0]))/le)**2)
    p_2  = 1 - st.chi2.cdf(c2_2, df=len(en)-2)
    nsigma = st.norm.ppf(p_2/2)
    text += '\n- fit 2: {} {:6.3} {:6.3} {:6.3}'.format(f_2[0], c2_2, p_2, nsigma)
    if verbosity:
        print(text)
    return [f_1, c2_1, p_1], [f_2, c2_2, p_2], text


def fit_pha_with_loglin(root, plot=True, recalc=False, do_f2=False):
    """Fit pha file of lag vs energy with log-linear model
    
    root: e.g. 22l3b_f7__1
    """
    
    # are we doing a second frequency simultaneously?
    xs2 = ''
    root2 = root
    if do_f2:
        xs2 = """
        da 2:2 len_{0}.pha
        set fq2 [scan [tcloutr xflt 2] "1 fq: %f"]
        new 8 $fq2 -1
        thaw 4
        fit 1000
        """.format(root.replace('__1', '__2'))
        root2 = root + '2'

    
    xs = """
    source ~/codes/xspec/az.tcl
    mkdir -p fits
    da len_{0}.pha
    setpl ener
    mdefine loglin a * log(e/eRef) * (f/1e-4)^(-b)
    set fq [scan [tcloutr xflt 1] "1 fq: %f"]
    set fq2 0
    mo loglin & 100 .1 0 0 & 7 .1& $fq -1& 1 -1& 1 -1; fit 1000
    {2}
    chain unload 1; chain len 2000; chain burn 10000; chain walk 10;para walk 20
    rm fits/*_{1}.* > /dev/null 2>&1 
    chain run fits/chain_{1}.fits
    az_calc_errors [az_free] fits/fit_{1} 1.0 

    # plot d res #
    cpd /gif
    set ptxt "# $fq $fq2\\n# en en_e d d_e res res_e del del_e; one per line"
    for {{set i 1}} {{$i <= [tcloutr datasets]}} {{incr i}} {{
        append ptxt "\\n[tcloutr plot d x $i]"
        append ptxt "\\n[tcloutr plot d xe $i]"
        append ptxt "\\n[tcloutr plot d y $i]"
        append ptxt "\\n[tcloutr plot d ye $i]"
        append ptxt "\\n[tcloutr plot res y $i]"
        append ptxt "\\n[tcloutr plot res ye $i]"
        append ptxt "\\n[tcloutr plot del y $i]"
        append ptxt "\\n[tcloutr plot del ye $i]"
    }}

    set fp [open fits/plot_{1}.dat w]
    puts $fp $ptxt
    close $fp
    exit
    """.format(root, root2, xs2)
    with open('tmp_%s.xcm'%root2, 'w') as fp: fp.write(xs)
    if not os.path.exists('fits/chain_%s.fits'%root2) or not os.path.exists('fits/plot_%s.dat'%root2) or recalc:
        print('chains for %s'%root2)
        info = subp.call(['/bin/bash', '-i', '-c', 'xspec - tmp_%s.xcm'%root2])
        os.system('rm tmp_%s.xcm > /dev/null 2&1'%root2)
    else:
        print('found fit files *%s*'%root2)
    

    # read the chain file #
    with pyfits.open('fits/chain_%s.fits'%root2) as fp:
        chain = np.array([fp[1].data.field(x.name) for x in fp[1].columns]) 
    plot_data = np.loadtxt('fits/plot_%s.dat'%root2)
    fq = np.double(open('fits/plot_%s.dat'%root2).readline().split()[1])
    fq2 = np.double(open('fits/plot_%s.dat'%root2).readline().split()[2])
    os.system('rm fits/plot_%s.dat'%root2)
    #plot_mod = chain[0][:,None] + chain[1][:,None] * np.log10(plot_data[0][None,:])
    if do_f2:
        plot_mod  = chain[0][:,None] * np.log10(plot_data[0][None,:]/chain[1][:,None]) * (fq/1e-4)**(-chain[2][:,None])
        plot_mod2 = chain[0][:,None] * np.log10(plot_data[0][None,:]/chain[1][:,None]) * (fq2/1e-4)**(-chain[2][:,None])
        plot_mod2_m, plot_mod2_s = plot_mod2.mean(0), plot_mod2.std(0)
    else:
        plot_mod = chain[0][:,None] * np.log10(plot_data[0][None,:]/chain[1][:,None]) * (fq/1e-4)**(-1)
    plot_mod_m, plot_mod_s = plot_mod.mean(0), plot_mod.std(0)
    txt = ('\ndescriptor en_mod_{0},+- dat_{0},+- res_{0},+- del_{0},+- mod_{0}_s1,+- '
           'mod_{0}_s2,+- mod_{0}_s3,+-\n').format(root2)
    txt += '\n'.join(['%g %g %g %g %g %g %g %g %g %g %g %g %g %g'%(
                        plot_data[0][i], plot_data[1][i], plot_data[2][i], plot_data[3][i],
                        plot_data[4][i], plot_data[5][i], plot_data[6][i], plot_data[7][i], 
                        plot_mod_m[i], plot_mod_s[i], plot_mod_m[i], plot_mod_s[i]*2, 
                        plot_mod_m[i], plot_mod_s[i]*3)
                     for i in range(len(plot_data[0]))])
    if do_f2:
        txt += ('\ndescriptor en_mod_{0},+- dat_{0},+- res_{0},+- del_{0},+- mod_{0}_s1,+- '
           'mod_{0}_s2,+- mod_{0}_s3,+-\n').format(root2+'_f2')
        txt += '\n'.join(['%g %g %g %g %g %g %g %g %g %g %g %g %g %g'%(
                        plot_data[0+8][i], plot_data[1+8][i], plot_data[2+8][i], plot_data[3+8][i],
                        plot_data[4+8][i], plot_data[5+8][i], plot_data[6+8][i], plot_data[7+8][i], 
                        plot_mod2_m[i], plot_mod2_s[i], plot_mod2_m[i], plot_mod2_s[i]*2, 
                        plot_mod2_m[i], plot_mod2_s[i]*3)
                     for i in range(len(plot_data[0]))])
    with open('fits/plot_%s.plot'%root2, 'w') as fp: fp.write(txt)
        
    # plot
    if not plot: return
    fig = plt.figure(figsize=(5,2.5))
    plt.errorbar(plot_data[0], plot_data[2], plot_data[3], fmt='o-')
    plt.xscale('log')
    for i in [1,2,3]:
        plt.fill_between(plot_data[0], plot_mod_m-i*plot_mod_s, 
                         plot_mod_m+i*plot_mod_s, alpha=0.15, color='C1')
    plt.title(root2)

    
def proc_lag_mcmc(root, hist_bins=30, pcxd=False):
    """Process mcmc chains from calculate_lag
    
    root: e.g. lag_22l3_k
    """
    npzfile = 'lag/%s.npz'%root
    saved = np.load(npzfile, allow_pickle=True)
    Res = saved['Res']
    En, fqd, extra = saved['En'], saved['fqd'], saved['extra'][()]
    # chains: (nen, nrun, 2*nfq+1)
    chains = extra['chains']
    chains = chains[:, -len(chains[0])//2:,:-1]
    if pcxd:
        _,_,amp, phi = np.split(chains, 4, axis=2)
    else:
        amp, phi = np.split(chains, 2, axis=2)
    phi = (phi+np.pi)%(2*np.pi)-np.pi
    
    phi_b = np.quantile(phi, [0.5, st.norm.cdf(-1), 1-st.norm.cdf(-1) ], axis=1)
    phi_b = np.array([phi_b[0], phi_b[0]-phi_b[1], phi_b[2]-phi_b[0], np.std(phi, axis=1)])
    amp_b = np.quantile(amp, [0.5, st.norm.cdf(-1), 1-st.norm.cdf(-1) ], axis=1)
    amp_b = np.array([amp_b[0], amp_b[0]-amp_b[1], amp_b[2]-amp_b[0], np.std(amp, axis=1)])
    # create a Res array from mcmc similar to Res
    Res_mc = Res * 1.0
    Res_mc[:,[1,2]] = np.nan
    Res_mc[:,[3,4]] = np.swapaxes(amp_b[[0,3]], 0, 1)
    Res_mc[:,[5,6]] = np.swapaxes(phi_b[[0,3]], 0, 1)
    Res_mc[:,[7,8]] = Res_mc[:,[5,6]] / (2*np.pi*Res_mc[:,[0]])
    
    # add +/- errors for phase; shape now is: (nen, 11, nfq)
    # 11: fq, p,pe, amp,ampe, phi,phie(std), lag,lage, dphi-,dphi+
    Res_mc = np.concatenate((Res_mc, np.swapaxes(phi_b[2:], 0, 1)), 1)
    
    
    # plot histograms #
    _, nen, nfq = phi_b.shape
    ncol, nrow = nen, nfq-2
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*1.4, nrow*2))
    ax = np.atleast_2d(ax)
    txt = ''
    for ir in range(nrow):
        for ic in range(ncol):
            h   = np.histogram(phi[ic,:,ir+1] , hist_bins, density=1)
            hx  = (h[1][1:] + h[1][:-1])/2
            twopf = 2*np.pi*Res_mc[ic,0,ir+1]
            hL  = np.histogram(phi[ic,:,ir+1]/twopf , hist_bins, density=1)
            hLx = (hL[1][1:] + hL[1][:-1])/2
            
            ax[ir, ic].fill_between(hx, h[0], step='mid', alpha=0.4)
            ax[ir, ic].errorbar(Res_mc[ic,5,ir+1], h[0].max()*0.6, [0], xerr=Res_mc[ic,6,ir+1], 
                                fmt='o', lw=2, capsize=6, ms=8, alpha=0.8)
            ax[ir, ic].errorbar(Res_mc[ic,5,ir+1], h[0].max()*0.7, [0], xerr=Res_mc[ic,[9,10],ir+1][:,None], 
                                fmt='o', lw=2, capsize=6, ms=8, alpha=0.8)
            ax[ir, ic].errorbar(Res[ic,5,ir+1], h[0].max()*0.8, [0], xerr=Res[ic,6,ir+1], 
                                fmt='s', lw=2, capsize=6, ms=8, alpha=0.8)
            ax[ir, ic].set_yticklabels([]); ax[ir, ic].set_yticks([])
            
            txt += '\n# ' + (' '.join(['%g'%x for x in h[1]]))
            txt += '\ndescriptor hx_{2}__f{0}_e{1} hy_{2}__f{0}_e{1}\n'.format(ir+1, ic+1, root)
            txt += '\n'.join(['%g %g'%z for z in zip(hx, h[0])])
            txt += ('\ndescriptor pnt_{2}__f{0}_e{1},+- pnt_y_{2}__f{0}_e{1} pntMC_{2}__f{0}_e{1},+- '
                    'pntMC_y_{2}__f{0}_e{1} pntMC2_{2}__f{0}_e{1},+,- pntMC2_y_{2}__f{0}_e{1}\n').format(ir+1, ic+1, root)
            txt += '%g %g %g %g %g %g %g %g %g %g'%(
                Res[ic,5,ir+1], Res[ic,6,ir+1], h[0].max()*0.8, 
                Res_mc[ic,5,ir+1], Res_mc[ic,6,ir+1], h[0].max()*0.7,
                Res_mc[ic,5,ir+1], Res_mc[ic,10,ir+1], -Res_mc[ic,9,ir+1], h[0].max()*0.6 )
            # do lag instead of phase #
            txt += '\n# ' + (' '.join(['%g'%x for x in hL[1]]))
            txt += '\ndescriptor hLx_{2}__f{0}_e{1} hLy_{2}__f{0}_e{1}\n'.format(ir+1, ic+1, root)
            txt += '\n'.join(['%g %g'%z for z in zip(hLx, hL[0])])
            txt += ('\ndescriptor pntL_{2}__f{0}_e{1},+- pntL_y_{2}__f{0}_e{1} pntLMC_{2}__f{0}_e{1},+- '
                    'pntLMC_y_{2}__f{0}_e{1}\n').format(ir+1, ic+1, root)
            txt += '%g %g %g %g %g %g'%(
                Res[ic,7,ir+1], Res[ic,8,ir+1], hL[0].max()*0.8, 
                Res_mc[ic,7,ir+1], Res_mc[ic,8,ir+1], hL[0].max()*0.7)
    plt.tight_layout(w_pad=-0.5)
    os.system('mkdir -p lag/plots')
    with open('lag/plots/%s__hist.plot'%root, 'w') as fp: fp.write(txt)
    return Res_mc, En, fqd, extra


def model_mcmc_lag(root):
    """Process mcmc chains from calculate_lag
    
    root: e.g. lag_22l3_k
    """
    npzfile = 'lag/%s.npz'%root
    saved = np.load(npzfile, allow_pickle=True)
    Res = saved['Res']
    En, fqd, extra = saved['En'], saved['fqd'], saved['extra'][()]
    # chains: (nen, nrun, 2*nfq+1)
    chains = extra['chains']
    chains = chains[:, -len(chains[0])//2:,:-1]
    amp, phi = np.split(chains, 2, axis=2)
    phi = (phi+np.pi)%(2*np.pi)-np.pi


def summarize_pha_123_fits(suff=''):
    """Summarize the fit to the combined frequencies fits; labeled 123
    in fits/ and kyn_fits/

    suff: '' or 'F2'. For len_22l3bMC_e1{suff}__123a
    
    call from within pha folder
    
    """
    xs = """
    @kyn_fits/load.xcm
    @{0}
    setpl ener

    cpd /gif
    set ptxt "# en en_e d d_e mod res res_e del del_e; one per line"
    foreach i {{1 2 3}} {{
     append ptxt "\\n[tcloutr plot d x $i]"
     append ptxt "\\n[tcloutr plot d xe $i]"
     append ptxt "\\n[tcloutr plot d y $i]"
     append ptxt "\\n[tcloutr plot d ye $i]"
     append ptxt "\\n[tcloutr plot d mod $i]"
     append ptxt "\\n[tcloutr plot res y $i]"
     append ptxt "\\n[tcloutr plot res ye $i]"
     append ptxt "\\n[tcloutr plot del y $i]"
     append ptxt "\\n[tcloutr plot del ye $i]"
    }}
    set fp [open tmp_{1}.dat w]
    puts $fp $ptxt
    close $fp
    exit
    """
    fits =  [
             f'kyn_fits/len_22l3bMC_e1{suff}__123',
             f'kyn_fits/len_22l3bMC_e1{suff}__123b',
             f'kyn_fits/len_22l3bMC_e1{suff}__123c',
             f'kyn_fits/len_22l3bMC_e1{suff}__123d',
             ]
    if suff in ['', 'F3']:
        if suff == '':
            fits.append(f'fits/len_22l3bMC_e1{suff}__123a')
        fits.append(f'kyn_fits/len_22l3bMC_e1{suff}__123e')
        fits.append(f'kyn_fits/len_22l3bMC_e1{suff}__123f')
    txt = ''
    for fit in fits:
        root = fit.split('/')[-1]
        with open(f'tmp_{root}.xcm', 'w') as fp: 
            fp.write(xs.format(fit, root))
        #info = subp.call(['/bin/bash', '-i', '-c', f'xspec - tmp_{root}.xcm'])
        info = subp.call(['xspec', '-', f'tmp_{root}.xcm'])
        plot_data = np.loadtxt(f'tmp_{root}.dat')
        plot_data = np.array(np.split(plot_data, 3))
        os.system(f'rm tmp_{root}.dat')
    
        for i in [1,2,3]:
            txt += ('\ndescriptor en{1}_fit_{0},+- dat{1}_fit_{0},+- mod{1}_fit_{0} res{1}_fit_{0},+- del{1}_fit_{0},+-\n').format(root, i)
            txt += '\n'.join([' '.join(['%g'%x for x in p])
                        for p in plot_data[i-1].T])
    
    with open('fits/plots_fit%s_all.plot'%suff, 'w') as fp: fp.write(txt)
