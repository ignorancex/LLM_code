import numpy as np
import os
import subprocess as subp
import time
import glob
from astropy.io import fits as pyfits
import scipy.stats as st
import matplotlib.pylab as plt
from scipy.ndimage import filters
import scipy.optimize as opt



import aztools as az

def read_pn_lc(obs, dt, enL, lcdir, data_dir, interp=False, suff='', **kwargs):
    """Read PN light curves
    
    obs: a list of observation folders
    dt: time bin
    enL: a string of space-separated bin boundaries
    lcdir: directory name indicating the bin type. e.g 8l for 8 bins in log space 
    data_dir: folder containing the obsids
    interp: if true, interpolate small gaps
    suff: extra suffix to lc name if needed. e.g. __s; default:''
    kwargs for read_pn_lcurve
    """
    do_raw = False
    #if do_raw: suff = ''
    
    enL       = np.array(enL.split(), np.double)
    en, ene   = (enL[1:] + enL[:-1])/2, (enL[1:] - enL[:-1])/2
    Lc        = [[az.LCurve.read_pn_lcurve('{}/{}/pn/lc/{}/lc_{:03g}__{}{}.fits'.format(
                        data_dir, o, lcdir, dt, ie+1, suff), **kwargs)
                        for o in obs] for ie in range(len(en))]
    
    if do_raw:
        rLc = [[az.LCurve.read_pn_lcurve('{}/{}/pn/lc/{}/lc_{:03g}__{}{}.fits'.format(
                    data_dir, o, lcdir, dt, ie+1, '__s'), **kwargs)
                    for o in obs] for ie in range(len(en))]
        LC = []
        for ie in range(len(Lc)):
            LC.append([])
            for io in range(len(Lc[0])):
                lc, rlc = Lc[ie][io], rLc[ie][io]
                tt  = np.intersect1d(lc.time, rlc.time)
                ii  = np.in1d(lc.time, tt)
                rii = np.in1d(rlc.time, tt)
                lc_new = az.LCurve(tt, rlc.rate[rii], rlc.rerr[rii], rlc.dt, lc.fexp[ii])
                LC[-1].append(lc_new)
        Lc = LC
        
    Lc = [[ll.make_even() for ll in l] for l in Lc]
    if interp:
        for l in Lc:
            for ll in l: ll.interp_small_gaps(np.int(1e3/dt), noise='norm', seed=345)
    return Lc, en, ene


def calc_cross_prodcuts(obsids, lcdir, dt, ebins, seglen, fqbin, **kwargs):
    """Calculate psd/rms/cov as a function of energy
    
    Parameters:
        obsids: a list of observation ids
        lcdir: name of folder conatining lc files, and it is the same
            where the psd/rms/cov files are created. e.g. 16ch
        dt: time sampling
        ebins: a string containing the bin boundaries
        seglen: segment length; if a list, it is interpreted ad [seglen, min_seg_length]

    Keywords:
        data_dir: folder conataining the obsids.
            default: /home/abzoghbi/data/ngc5506/spec_timing_paper/data/xmm
        iref: a list or bin numbers of the reference band. Default: -1 for all except current
        overlap: number of bins to overlap in the energy axis, .e.g 2, 3. Default: None
        verbose: Default: true
    """
    
    data_dir = kwargs.get('data_dir', '/home/abzoghbi/data/ngc5506/spec_timing_paper/data/xmm')
    iref = kwargs.get('iref', -1)
    overlap = kwargs.get('overlap', None)
    verbose = kwargs.get('verbose', True)
    if type(seglen) == list:
        min_len = seglen[1]
        seglen = seglen[0]
    else:
        min_len = 0
    
    # read light curves #
    Lc, en, ene = read_pn_lc(obsids, dt, ebins, lcdir, data_dir, min_exp=0.6, interp=True)
    nen = len(en)
    rarr = [i for lc in Lc for i in az.LCurve.create_segments(lc, seglen, 
                                                strict=False, min_seg_length=min_len)[:3]]
    # shape: nen, nseg, ...
    rate, rerr, time = rarr[::3], rarr[1::3], rarr[2::3]
    if verbose:
        print('segment lengths (ks): ', ' | '.join(['%4.1f'%(len(r)*dt/1e3) for r in rate[0]]))

    # the bins to use #
    ibins = list(range(nen))
    if not overlap is None and overlap != 1:
        ibins = [ibins[i:i+overlap] for i in range(nen-overlap+1)]
        enL = np.array(ebins.split(), np.double)
        en  = np.array([ (enL[ib[-1]]+enL[ib[0]])/2.0 for ib in ibins])
        ene = np.array([ (enL[ib[-1]]-enL[ib[0]])/2.0 for ib in ibins])
    if verbose:
        print('energy bin indices: ', ibins)

    # do calculations #
    crss = []
    for ib,ibin in enumerate(ibins):
        r, re, R, Re = az.LCurve.prepare_en_segments(rate, rerr, ibin, iref, 
                                ibin_exclude=(isinstance(iref, list) or iref==-1) )
        cx = az.LCurve.calculate_lag(r, R, dt, fqbin, rerr=re, Rerr=Re, taper=True, norm='rms')        
        crss.append(cx)
        if ib==0 and verbose:
            print('number of frequencies: ', cx[-1]['fqm'])
    rms = np.array([cx[-1]['rms'] for cx in crss])
    cov = np.array([cx[-1]['cov'] for cx in crss])
    coh = np.array([cx[-1]['coh'] for cx in crss])
    lag = np.array([cx[1:3] for cx in crss])
    res = {'rms': rms, 'cov': cov, 'lag':lag, 'coh':coh, 'freq': crss[0][0], 
           'en': [en, ene], 'crss':crss, 'lcdir':lcdir, 'overlap':overlap}
    return res
    
    
def write_cross_prodcuts(crss, **kwargs):
    """Write psd/rms/cov
    
    Parameters:
        crss: The output of calc_cross_products

    Keywords:
        out_dir: Where to write the files.
            default: /home/abzoghbi/data/ngc5506/spec_timing_paper/data/xmm_timing
        specfile: name of the enregy spectrum file. used to get the exposure and response file.
            assumed in the current dir; If none, no pha files are written
        nfq: number of frequencies to write (vs energy); Default: 4; if None, write
            product vs frequency and not energy
        suff: suffix for file names. default: ''; e.g. '_lo', '_hi'
        verbose: Default: true
    """
    
    lcdir = crss['lcdir']
    out_dir = kwargs.get('out_dir', 
            '/home/abzoghbi/data/ngc5506/spec_timing_paper/data/xmm_timing/cross/%s'%lcdir)
    specfile = kwargs.get('specfile', None)
    nfq = kwargs.get('nfq', 4)
    suff = kwargs.get('suff', '')
    verbose = kwargs.get('verbose', True)
    
    try:
        cwd = os.getcwd()
    except:
        cwd = out_dir
    os.system('mkdir -p %s'%out_dir)
    os.chdir(out_dir)
    
    # products #
    en = np.array(crss['en'])
    nen = en.shape[1]
    freq = np.array(crss['freq'])
    
    if nfq is None:
        # write things vs frequency #
        outfile = 'cross_freq%s.plot'%suff
        text = '\ndescriptor freq_%s%s\n'%(lcdir,suff)
        text += '\n'.join(['%g'%x for x in freq])
        for lab in ['rms', 'cov', 'lag']:
            par = crss[lab]
            text += '\ndescriptor %s\n'%(' '.join(['%s_%s%s_e%d,+-'%(lab, lcdir, suff, ie+1) 
                                                   for ie in range(nen)]))
            text += '\n'.join([' '.join(['%g %g'%(par[ie, 0, ii], par[ie, 1, ii]) 
                                for ie in range(nen)]) for ii in range(len(freq))])
            with open(outfile, 'w') as fp: fp.write(text)
        if verbose:
            print('saved products vs freq to %s'%outfile)
    else:
        # write things vs en
        outfile = 'cross_en%s.plot'%suff
        text = '\ndescriptor en_%s%s,+-\n'%(lcdir, suff)
        text += '\n'.join(['%g %g'%(x1, x2) for x1,x2 in en.T])
        for lab in ['rms', 'cov', 'lag']:
            par = crss[lab]
            if lab == 'lag':
                for ii in range(nfq):
                    lfit = az.misc.simple_fit(en[0], par[:,0,ii], par[:,1,ii], 'const', verbose=0)
                    text += lfit[-1]
                    text += '# --- #'
                    lfit = az.misc.simple_fit(np.log10(en[0]), par[:,0,ii], par[:,1,ii], 
                                              'linear', verbose=0)
                    text += lfit[-1]
            text += '\ndescriptor %s\n'%(' '.join(['%s_%s%s_f%d,+-'%(lab, lcdir, suff, ie+1) 
                                                   for ie in range(nfq)]))
            text += '\n'.join([' '.join(['%g %g'%(par[ie, 0, ii], par[ie, 1, ii]) 
                                for ii in range(nfq)]) for ie in range(nen)])
            with open(outfile, 'w') as fp: fp.write(text)
        if verbose:
            print('saved products vs freq to %s'%outfile)
        
        if not specfile is None:
            # write pha files #
            with pyfits.open(specfile) as fp:
                exposure = fp['spectrum'].header['exposure']
                respfile = fp['spectrum'].header['respfile']
            if verbose:
                print('frequency limits: ', ', '.join([
                        '%2.2e'%x for x in crss['crss'][0][3]['fqL'][:(nfq+1)]]))
            for ii in range(nfq):
                for lab in ['rms', 'cov']:
                    par = crss[lab]
                    x  = np.arange(nen+2)
                    y  = np.concatenate([[0], par[:,0,ii], [0]])
                    ye = np.concatenate([[0], par[:,1,ii], [0]])
                    ibad = ~np.isfinite(y)
                    y[ibad] = 0; ye[ibad] = np.max(ye[~ibad])
                    txt = '\n'.join(['%d %g %g'%z for z in zip(x,y,ye)])
                    with open('tmp.txt', 'w') as fp: fp.write(txt + '\n')
                    cmd = ('ascii2pha tmp.txt %s%s_f%d.pha chanpres=yes dtype=2 qerror=yes '
                           'rows=- tlmin=0 detchans=%d pois=no telescope=XMM instrume=PN '
                           'detnam=PN filter=NA exposure=%g respfile=%s')%(
                                lab, suff, ii+1, len(x), exposure, respfile)
                    os.system('rm %s%s_f%d.pha > /dev/null 2>&1'%(lab, suff, ii+1))
                    os.system(cmd)   
    os.chdir(cwd)
    



def lc_to_segments(Lc, seglen=10000, strict=False, **kwargs):
    """Split a Lc: a list of lists of LCurve to segments.
    No energy sum is done
    
    Lc: list of lists of LCurve: dims: nen, nobs; this is the output of read_pn_lc
    seglen, strict and kwargs: to be passed to az.misc.split_array
    
    Returns: rate_all, rerr_all, time_all, seg_indx. The first three are of dims: (nen, nseg_total),
        and seg_indx is a list of rate_all indices to which individual segments belong. 
    """

    nen, nobs = len(Lc), len(Lc[0])
    Lc = [[ll.make_even() for ll in l] for l in Lc]

    # arrays to be split, including rerr
    # arrs: (nobs, 2*nen(rate, rerr)); splt: (nobs, 2*nen, nseg)
    arrs = [[Lc[ie][io].rate for ie in range(nen)] + 
            [Lc[ie][io].rerr for ie in range(nen)] + 
            [Lc[ie][io].time for ie in range(nen)] for io in range(nobs)]
    splt = [az.misc.split_array(arr[0], seglen, strict, *arr, **kwargs)[2:] for arr in arrs]

    # segments: dim: (nseg_total, nen)
    rate_all = [s[:nen] for s in splt]
    rerr_all = [s[nen:(2*nen)] for s in splt]
    time_all = [s[(2*nen):] for s in splt]
    
    # indices for seprating the observations if needed #
    ic, seg_indx = 0, []
    for iar in splt:
        j = len(iar[0])
        seg_indx.append(list(range(ic, ic+j)))
        ic += j
        
    # flatten the lists so they have dims: (nseg_total, nen)
    rate_all = [[k for i in range(nobs) for k in rate_all[i][ie]] for ie in range(nen)]
    rerr_all = [[k for i in range(nobs) for k in rerr_all[i][ie]] for ie in range(nen)]
    time_all = [[k for i in range(nobs) for k in time_all[i][ie]] for ie in range(nen)]
    seg_indx = np.concatenate([[i]*len(s) for i,s in enumerate(seg_indx)])

    return rate_all, rerr_all, time_all, seg_indx


def chans_from_avg_spec(obsnum, e0, suff='', **kwargs):
    """Get average spectrum, group it, convert groups to bins, and return
    channel numbers
    
    Parameters:
        obsnum: a list of observations to average, 
                e.g [3,4,5,6,7,8,9], for all those used in timing.
        e0: the energy limits array to use a starting guide (e.g. np.linspace(2,10,17)).
        suff: suffix for the added spectrum; the result: spec_add{suff}
    
    Keywords:
        specdir: location of the spectral files
    """
    specdir = kwargs.get('specdir', '/home/abzoghbi/data/ngc5506/spec_timing_paper/data/xmm_spec')
    if len(obsnum) == 1: obsnum = [obsnum[0]]*2
    
    cwd = os.getcwd()
    os.chdir(specdir)
    # average spectrum #
    txt = '\n'.join(['spec_%d.grp'%i for i in obsnum])
    with open('tmp.add', 'w') as fp: fp.write(txt+'\n')
    os.system('rm spec_add%s* > /dev/null 2>&1'%suff)
    os.system('addspec tmp.add spec_add%s qaddrmf=yes qsubback=yes'%suff)
    
    # work out the grouping array from the guiding energy array #
    with pyfits.open('spec_add%s.rsp'%suff) as fp:
        chan = fp['ebounds'].data.field(0)
        ecent = (fp['ebounds'].data.field(1) + fp['ebounds'].data.field(2))/2
    ie = np.array([np.argmin(np.abs(ecent-v)) for v in e0])
    ibins = chan*0 - 1
    ibins[ie] = 1
    ibins[0] = 1
    
    # apply the grouping to the file; we create a grouped file to get the file struture #
    os.system('grppha spec_add%s.pha spec_add%s.grp "group min 1&exit" '%(suff, suff))
    with pyfits.open('spec_add%s.grp'%suff) as fp:
        hdu = fp['SPECTRUM']
        orig_cols = hdu.columns
        orig_cols['GROUPING'].array = ibins
        cols = pyfits.ColDefs(orig_cols)
        tbl = pyfits.BinTableHDU.from_columns(cols)
        hdu.header.update(tbl.header.copy())
        tbl.header = hdu.header.copy()
        grp = pyfits.HDUList([fp[0],tbl])
        os.system('rm spec_add%s.grp &> /dev/null'%suff)
        grp.writeto('spec_add%s.grp'%suff)

    # convert the grouping to bins: ->spec_add{suff}.grp.b and tmp.chan #
    hcmd = 'export HEADASNOQUERY=;export HEADASPROMPT=/dev/null;'
    os.system(hcmd + 'grp2bin.py spec_add%s.grp'%suff)
    chans = np.loadtxt('tmp.chan')
    chansL = np.concatenate([np.linspace(c[0], c[1]+1, np.int(1+(c[1]-c[0]+1)/c[2]))[:-1] 
                    for c in chans[1:-1]] + [np.array([chans[-1][0]])]) + 1
    os.chdir(cwd)
    print('files spec_add%s.??? created.'%suff)
    print('chans:', ' '.join(['%d'%i for i in chansL]))
    return 


def fit_psd(sfile, model='po', ibins=4, extra_cmd='', suff='tot__po'):
    """Fit psd with a model either directly or starting from an xcm file
    
    sfile: name of pha file
    model: po|lo or the name of a starting xcm file
    ibins: whittle#
    extra_cmd: extra xspec cmd; e.g. thaw, tie etc before running errors
    suff: of output fit file
    
    """
    os.system('mkdir -p fits')
    if model == 'po':
        mod = ('mo po+po+cpflux*po &0.1,,0 0 .2 .2&0 -1&0 -1&0.3&1e-5&1e-3&=p1^2.& '
               '2 .1 1.5 1.5 4 4& 1 -1')
    elif model == 'lo':
        mod = ('mo po+po+cpflux*lore &0.1,,0 0 .2 .2&0 -1&0 -1&0.3&1e-5&1e-3&=p1^2.& '
               '0 -1 & 1e-5,,1e-6 1e-6 & 1 -1')
    elif model == 'bpl':
        mod = ('mdefine bpl e^(-a1) * 1/(1 + (e/eb)^(a2-a1)); mo po+po+cpflux*bpl&'
               '0.1,,0 0 0.2 0.2& 0 -1 & 0 -1 & 0.3& 1e-5&1e-3&=p1^2.0&'
               '1 -1 & 1e-4 .1 1e-6 1e-6 1e-3 1e3& 2 .1 1.5 1.5 7 7 & 1 -1')
    elif model[-3:] == 'xcm':
        mod = '@%s'%model
    else:
        raise ValueError('Unknown model %s'%model)
    
    tcl = ['source ~/codes/xspec/az.tcl',
           mod, 'statis whittle%d'%ibins, 'da %s'%sfile, 
           'fit 1000', extra_cmd, 'fit 1000 .1',
           'az_calc_errors [az_free_params] fits/psd_%s 1.0'%suff, 'exit']
    tcl = '\n'.join(tcl)
    with open('tmp_%s.xcm'%suff, 'w') as fp: fp.write(tcl)
    cmd = 'xspec - tmp_%s.xcm > tmp_%s.log 2>&1'%(suff, suff)
    p = subp.call(['/bin/bash', '-i', '-c', cmd])
    if p==0: os.system('rm tmp_%s.???'%suff)

        
def calc_rms_cov(rate_all, rerr_all, fqbin, dt, iref=-1, taper=True):
    """Calculate rms and cov vs energy
    
    rate_all, rerr_all: list of light curves dims: (nen, nseg, ...)
    fqbin: frequency binning criteria
    dt
    iref: a list of a number of reference band. iref=-1 means use all except current
    """
    
    nen, nseg = len(rate_all), len(rate_all[0])
    
    if iref == -1:
        ref_bins = list(range(nen))
    elif type(iref) == int:
        ref_bins = [iref]
    else:
        ref_bins = iref
    
    ldata = []
    for ie in range(nen):
        r, re  = rate_all[ie], rerr_all[ie]
        if len(ref_bins) == 1 and ie==ref_bins[0]:
            R, Re = r, re
        else:
            R =  [np.sum([rate_all[je][i] for je in ref_bins if je!=ie], 0) 
                  for i in range(nseg)]
            Re = [np.sum([rerr_all[je][i]**2 for je in ref_bins if je!=ie], 0)**0.5 
                  for i in range(nseg)]
        l = az.LCurve.calculate_lag(r, R, dt, fqbin, rerr=re, Rerr=Re, taper=taper, 
                                    norm='rms', rms_snr_ratio=1.0)
        ldata.append(l)
    rms = np.array([l[-1]['rms'] for l in ldata])
    cov = np.array([l[-1]['cov'] for l in ldata])
    return rms, cov, ldata


def rms_cov_en_pipeline(lc_obsids, lcdir, ebins, dt, seglen, fqbin, **kwargs):
    """Calculate psd/rms/cov as a function of energy
    
Parameters:
    lc_obsids: a list of observation ids
    lcdir: name of folder conatining lc files, and it is the same
        where the psd/rms/cov files are created. e.g. 16ch
    ebins: a string containing the bin boundaries
    dt
Keywords:
    base_dir: folder conataining the data and timing products.
        default: /home/abzoghbi/data/ngc5506/spec_timing_paper
    data_dir: default: 'data/xmm'
    timing_dir: default: 'data/xmm_timing'
    NFQ: number of frequency bins to use when printing rms/cov; default: 8
    suff: suffix of the resulting files (and the average spectrum to be used too;
        assumed: spec_add_{lcdir}{suff}.rsp.b); the output will be: rms{suff}_f?.pha etc
        default: '' (other examples _lo).
    nowrite: if true, just return rms,cov,ldata without writing pha files. Default: False
    iref: a list or bin number of the reference band. Default: -1 for all except current
    wdir: work dir; default: psd (e.g. lag)
    """
    
    base_dir = kwargs.get('base_dir', '/home/abzoghbi/data/ngc5506/spec_timing_paper')
    data_dir = kwargs.get('data_dir', 'data/xmm')
    timing_dir = kwargs.get('timing_dir', 'data/xmm_timing')
    NFQ = kwargs.get('NFQ', 5)
    suff = kwargs.get('suff', '')
    nowrite = kwargs.get('nowrite', False)
    iref = kwargs.get('iref', -1)
    wdir = kwargs.get('wdir', 'psd')
    
    # go to the right place 
    cwd = base_dir #os.getcwd()
    os.chdir('%s/%s'%(base_dir, timing_dir))
    os.system('mkdir -p %s/%s'%(wdir, lcdir)); os.chdir('%s/%s'%(wdir, lcdir))
    
    
    # read light curves #
    Lc, en, ene = read_pn_lc(lc_obsids, dt, ebins, lcdir, '%s/%s'%(base_dir, data_dir), 
                         min_exp=0.1, interp=True, suff='')
    rate_all, rerr_all, time_all, seg_idx = lc_to_segments(Lc, seglen, strict=True)
    print('segment lengths (ks): ', ' | '.join(['%4.1f'%(len(r)*dt/1e3) for r in rate_all[0]]))
    
    # rms and covariance spectra #
    rms, cov, ldata = calc_rms_cov(rate_all, rerr_all, fqbin, dt, iref)
    rms_tot = np.array([l[-1]['rms_tot'] for l in ldata])
    cov_tot = np.array([l[-1]['cov_tot'] for l in ldata])
    rms = np.concatenate((np.expand_dims(rms_tot, 2), rms), -1)
    cov = np.concatenate((np.expand_dims(cov_tot, 2), cov), -1)
    if nowrite: return rms, cov, ldata
    
    # write the rms/cov as pha files #
    # copy the corresponding spectra and the response #
    os.system('cp %s/%s/../xmm_spec/spec_add_%s%s.???.b .'%(base_dir, timing_dir, lcdir, suff))
    exposure = pyfits.open('spec_add_%s%s.grp.b'%(lcdir, suff))['spectrum'].header['exposure']
    print('frequencies: ', ' | '.join(['%2.2e'%x for x in ldata[0][-1]['fqL'][:NFQ]]))
    # f0 is the total (all freq), f1, f2 .. are the individual frequencies
    for ii in range(NFQ+1):
        for pp in ['rms', 'cov']:
            x  = np.arange(len(en)+2)
            parr = rms if pp == 'rms' else cov
            y  = np.concatenate([[0], parr[:,0,ii], [0]])
            ye = np.concatenate([[0], parr[:,1,ii], [0]])
            txt = '\n'.join(['%d %g %g'%z for z in zip(x,y,ye)])
            with open('tmp.txt', 'w') as fp: fp.write(txt + '\n')
            cmd = ('ascii2pha tmp.txt %s%s_f%d.pha chanpres=yes dtype=2 qerror=yes rows=- tlmin=0 '
                  'detchans=%d pois=no telescope=XMM instrume=PN detnam=PN filter=NA '
                  'exposure=%g respfile=spec_add_%s%s.rsp.b')%(
                            pp, suff, ii, len(x), exposure, lcdir, suff)
            os.system('rm %s%s_f%d.pha > /dev/null 2>&1'%(pp, suff, ii))
            os.system(cmd)
    os.chdir(cwd)
    return rms, cov, ldata
    


def simulate_like(x, dt, nsim=1, seed=None):
    """simulate nsim light curve arrays like x
    powerlaw parameters are the average
    Use seed to create the same underlying light curve
    that is sampled through a different poisson noise.
    This allows zero lag light curves with perfect coherence
    to be simulated
    """
    # 2.2018e+00 2.2170e-01 2.4064e-01 -2.0275e-01 "PhoIndex "
    # 3.9876e-09 4.3754e-09 5.2287e-09 -3.5220e-09 "norm "
    psdpar = [1.36e-10, -2.548]
    nx  = len(x)
    mux = np.nanmean(x)
    inan = np.isnan(x)

    sim = az.SimLC(seed)
    # use the break from Papadakis+95
    sim.add_model('broken_powerlaw', [psdpar[0], -1, psdpar[1], 1e-7])
    ns  = max([2, nsim])
    y   = []
    for ii in range(ns//2+1):
        sim.simulate(nx*2, dt, mux, 'rms')
        y += np.split(sim.x, 2)
    y   = sim.add_noise(np.array(y), seed=None, dt=dt)[:nsim]
    y[:, inan] = np.nan
    return y


def combine_en_segments(rate_all, rerr_all, ibin, iref=None):
    """Combined segments from different energies to produce 1 (iref=None) or 2 bins.
    We assume the arrays from different energies have the same shape
    
    rate_all, rerr_all: the output of lc_to_segments with dim: (nen, nseg_total)
    ibin: bin of interest
    iref: reference band. -1 for all (excluding ibin of course)
    
    return: rate, rerr, Rate, Rerr
    """

    nen = len(rate_all)

    # make sure we are dealing with lists #
    if not isinstance(ibin, list): ibin = [ibin]

    # get the rate at the bins of interest #    
    rate = np.sum(np.array(rate_all)[ibin], 0)
    rerr = np.sum(np.square(rerr_all)[ibin], 0)**0.5
    
    # and the reference if needed #
    Rate, Rerr = None, None
    if not iref is None:
        if not isinstance(iref, list):
            iref = list(range(nen)) if iref==-1 else [iref]
        iref = [i for i in iref if not i in ibin]

        Rate = np.sum(np.array(rate_all)[iref], 0)
        Rerr = np.sum(np.square(rerr_all)[iref], 0)**0.5

    return rate, rerr, Rate, Rerr


def calc_lag_en(rate_all, rerr_all, dt, fqbin, indv=None, iref=-1, overlap=None, 
                fqen=None, **kwargs):
    """Calculate lag vs energy for the total and individual observations
    
    rate_all, rerr_all: the output of lc_to_segments. dim: (nen, nseg)
    dt: time bin
    fqbin: binning dict
    indv: a list of obs indices to group. e.g. [[0,1,2], [3,4,5]] etc.
    iref: reference band. -1 for all (excluding ibin of course)
    overlap: number of bins to overlap e.g. 1, 2, or 3. default None
    fqen: parameters for fq_en 2d plots
    **kwargs: any parameters to pass to az.LCurve.calculate_lag
    
    Return: lag (nen, 3, nfq), ilag (nen, nindv, 3, nfq)
    
    """

    # these are arrays of objects because the segments may have different lengths #
    rate_all = np.array(rate_all)
    rerr_all = np.array(rerr_all)
    
    nen, nseg = rate_all.shape

    ibins = list(range(nen))
    if not overlap is None:
        ibins = [ibins[i:i+overlap] for i in range(nen-overlap+1)]
    
    Lag, iLag, LagE, iLagE = [], [], [], []
    for ibin in ibins:
        rate, rerr, Rate, Rerr = combine_en_segments(rate_all, rerr_all, ibin, iref)
        lag  = az.LCurve.calculate_lag(rate, Rate, dt, fqbin, rerr=rerr, Rerr=Rerr, **kwargs)
        if not fqen is None:
            fraw = az.LCurve.calculate_lag(rate, Rate, dt, None, 
                                           rerr=rerr, Rerr=Rerr, **kwargs)[0]
            fraw = np.unique(fraw[(fraw>=fqen[0]) & (fraw<=fqen[1])])
            inum = np.array(fqen[2]*np.clip(1.01 * np.arange(1,200)*0.2, 1, np.inf), np.int)
            #fbins = [[fraw[i], fraw[i+np.int(fqen[2])]] for i in range(len(fraw)-fqen[2])]
            fbins, ii = [], 0
            for i in range(len(fraw)):
                if (i+inum[i])>= len(fraw): break
                fbins.append([fraw[i], fraw[i+inum[i]]])
            lag_ = [az.LCurve.calculate_lag(rate, Rate, dt, {'bins': fb}, 
                                    rerr=rerr, Rerr=Rerr, **kwargs) for fb in fbins]
            lag_fqen = [[l[0], l[1], l[2], l[3]['limit'], l[3]['Limit'], l[3]['limit_avg']] 
                            for l in lag_]
            lag[3]['fqen'] = np.array(lag_fqen)[:,:,0]
        Lag.append(lag[:3])
        LagE.append(lag[3])

        # indiviudal obs #
        ilag, ilagE = [], []
        if indv is None: continue
        for ii in indv:
            rate, rerr, Rate, Rerr = combine_en_segments(rate_all[:,ii], rerr_all[:,ii], ibin, iref)
            lag  = az.LCurve.calculate_lag(rate, Rate, dt, fqbin, rerr=rerr, Rerr=Rerr, **kwargs)
            if not fqen is None:
                fraw = az.LCurve.calculate_lag(rate, Rate, dt, None, 
                                               rerr=rerr, Rerr=Rerr, **kwargs)[0]
                fraw = np.unique(fraw[(fraw>=fqen[0]) & (fraw<=fqen[1])])
                #fbins = [[fraw[i], fraw[i+np.int(fqen[2])]] for i in range(len(fraw)-fqen[2])]
                inum = np.array(fqen[2]*np.clip(1.01 * np.arange(1,200)*0.2, 1, np.inf), np.int)
                fbins, ii = [], 0
                for i in range(len(fraw)):
                    if (i+inum[i])>= len(fraw): break
                    fbins.append([fraw[i], fraw[i+inum[i]]])
                lag_ = [az.LCurve.calculate_lag(rate, Rate, dt, {'bins': fb}, 
                                        rerr=rerr, Rerr=Rerr, **kwargs) for fb in fbins]
                lag_fqen = [[l[0], l[1], l[2], l[3]['limit'], l[3]['Limit'], l[3]['limit_avg']] 
                                for l in lag_]
                lag[3]['fqen'] = np.array(lag_fqen)[:,:,0]
            ilag.append(lag[:3])
            ilagE.append(lag[3])
        iLag.append(ilag)
        iLagE.append(ilagE)

    # nen, 3, nfq
    lag    = np.array(Lag)
    # nen, nindv, 3, nfq
    ilag   = np.array(iLag)
    return lag, ilag, LagE, iLagE

def lag_en_null_test(en, l, le, verbosity=True):
    """Test lag-enery against a null hypothesis of a const and log-linear model"""
    import scipy.optimize as opt
    import scipy.stats as st

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


def lag_en_pipeline(lcdata, fqbin=-1, indv=None, iref=-1, nsim=50, overlap=None, fqen=None):
    """read lc, calc lag, run simulations and plot it"""

    # some common input #
    if fqbin == -1:
        fqbin  = {'bins': [3e-5, 5e-4]}
    if indv == -1:
        indv   = [[0], [1], [2], [3], [4], [5], [6], [7], [8, 9],
                  [0,1,2,3,4,5,6,7,8,9]]
    kwargs = dict(taper=True)
    

    # read light curve files: Lc has dims: (nen, nobs) #
    print('prepare data ...')
    Lc, en, ene = lcdata 
    nen, nobs   = len(Lc), len(Lc[0])
    dt = Lc[0][0].dt

    if not overlap is None:
        ibins = list(range(nen))
        ibins = [ibins[i:i+overlap] for i in range(nen-overlap+1)]
        ene = np.array([( (en[i[-1]]+ene[i[-1]])-(en[i[0]]-ene[i[0]]) )/2 for i in ibins])
        en  = np.array([(en[i[0]]+en[i[-1]])/2 for i in ibins])
    
    
    min_length = np.int(2e3/dt)    
    rate_all, rerr_all, time_all, seg_idx = lc_to_segments(Lc, min_seg_length=min_length)

    
    # map indv which is for the obs number to indv_seg which is for the segment number #
    indv_seg = None
    if not indv is None:
        indv_seg = [[j for j,jj in enumerate(seg_idx) if jj in i] for i in indv]

    # calculate lag #
    print('calculate lag ...')
    lag, ilag, lagE, ilagE = calc_lag_en(rate_all, rerr_all, dt, fqbin, indv_seg, iref, 
                                        overlap, fqen, **kwargs)
    #return [None]*7
    
    # simulations #
    print('run simulations ...')
    rate_sim = []
    # use same seed for all energies; seperate for every segment, to simulate 0-lag
    seeds = np.random.randint(10000, size=len(seg_idx))
    for R in rate_all:
        rate_sim.append([simulate_like(r, dt, nsim, s) for r,s in zip(R, seeds)])
    # make rate_sim dims: (nsim, nseg, nen, ..)
    rate_sim = [[np.array([rate_sim[ie][iseg][isim] for iseg in range(len(seg_idx))])
             for ie in range(nen)] for isim in range(nsim)]
    rerr_sim = [[(r2/dt)**0.5 for r2 in r1] for r1 in rate_sim]
    
    LagS = [calc_lag_en(rs, rse, dt, fqbin, indv_seg, iref, overlap, **kwargs)[:2]
               for rs,rse in zip(rate_sim, rerr_sim)]
    # unzip so we have dims: (nsim, nen, 3, nfq) and (nsim, nen, nindv, 3, nfq)
    lagS, ilagS = zip(*LagS)
    lagS, ilagS = np.array(lagS), np.array(ilagS)
    

    # null tests for lag-vs-energy #
    print('calculating null tests ...')
    nfq = lag.shape[-1]
    fit_data  = [lag_en_null_test(en, lag[:,1,ifq], lag[:,2,ifq], 0) for ifq in range(nfq)]
    fit_sim  = [lag_en_null_test(en, lag[:,1,ifq], lagS[:,:,1,ifq].std(0), 0)
                        for ifq in range(nfq)]
    if not indv is None:
        ifit_data = [[lag_en_null_test(en, ilag[:,ii,1,ifq], ilag[:,ii,2,ifq], 0)
                        for ii in range(len(indv))] for ifq in range(nfq)]
        ifit_sim = [[lag_en_null_test(en, ilag[:,ii,1,ifq], ilagS[:,:,ii,1,ifq].std(0), 0)
                        for ii in range(len(indv))] for ifq in range(nfq)]

    text  = ''
    for ifq in range(nfq):
        text += fit_data[ifq][2].replace('\n','\n#') + fit_sim[ifq][2].replace('\n','\n#') + '\n'
        if not indv is None:
            text += ('\n'.join(['#- {} -# {}'.format(i+1, idv) + 
                   ifit_data[ifq][i][2].replace('\n','\n#') +
                   ifit_sim[ifq][i][2].replace('\n','\n#') for i,idv in enumerate(indv)]))
        text += '\n'
    
    extra = [text, lagE, ilagE]
    return en, ene, lag, lagS, ilag, ilagS, extra


def plot_ilag(en, lag, ilag, lagS=None, ilagS=None, figsize=None, extra=None):
    """plot the results of lag_en_pipeline, with simulations if given"""
    
    doindv = False
    try:
        _,nindv,_,nfq = ilag.shape
        doindv = True 
    except:
        nfq, nindv = lag.shape[-1], 1
    if figsize is None: figsize=(14,3*nfq)
    fig = plt.figure(figsize=figsize)
    for ifq in range(nfq):
        if not lagS is None:
            lm, ls = lagS[:,:,1,ifq].mean(0), lagS[:,:,1,ifq].std(0)
        for il in range(nindv):
            if nindv <= 10:
                ax = plt.subplot(nfq, nindv, il+ifq*nindv+1)
            else:
                ax = plt.subplot(nfq*2, nindv//2+1, il+ifq*nfq+1)
            ax.set_xscale('log')
            ax.set_ylim([2*np.min(lag[:,1,ifq])/1e3,2*np.max(lag[:,1,ifq])/1e3])
            ax.set_xticks([3,6])
            ax.xaxis.set_major_formatter(plt.ScalarFormatter())
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.errorbar(en, lag[:,1,ifq]/1e3, lag[:,2,ifq]/1e3, fmt='o-', alpha=0.5, color='C0')
            if not extra is None:
                llim = np.array([extra[1][ie]['Limit'][ifq] for ie in range(len(en))])
                ax.fill_between(en, -llim/1e3, llim/1e3, alpha=0.2, color='C0')
            if doindv:
                ax.errorbar(en, ilag[:,il,1,ifq]/1e3, ilag[:,il,2,ifq]/1e3, fmt='s-', color='C1')
                if not extra is None:
                    llim = np.array([extra[2][ie][il]['Limit'][ifq] for ie in range(len(en))])
                    ax.fill_between(en, -llim/1e3, llim/1e3, alpha=0.2, color='C1')

            # simulations #
            if not lagS is None:
                ax.fill_between(en, (lm-ls)/1e3, (lm+ls)/1e3, alpha=0.3)
                if doindv:
                    ilm, ils = ilagS[:,:,il,1,ifq].mean(0), ilagS[:,:,il,1,ifq].std(0)
                    ax.fill_between(en, (ilm-ils)/1e3, (ilm+ils)/1e3, alpha=0.3)
    plt.tight_layout(pad=0)

    
def write_ilag(en, ene, lag, ilag, lagS=None, ilagS=None, suff=''):
    """Write the lag results to a string to be written to a veusz file"""

    doindv = False
    try:
        nen,nindv,_,nfq = ilag.shape
        doindv = True
    except:
        nfq, nindv,nen = lag.shape[-1], 1, lag.shape[0]
        
    if suff != '' and suff[0] != '_': suff = '_%s'%suff

    text = ''
    for ifq in range(nfq):
        
        text += '\ndescriptor en_f{0}{1},+- lag_f{0}{1},+-\n'.format(ifq+1, suff)
        text += '\n'.join(['{} {} {} {}'.format(en[ie], ene[ie], lag[ie,1,ifq]/1e3,
                lag[ie,2,ifq]/1e3) for ie in range(nen)])
        if not lagS is None:
            lm, ls = lagS[:,:,1,ifq].mean(0), lagS[:,:,1,ifq].std(0)
            text += '\ndescriptor slag_f{0}{1},+- sLag_f{0}{1},+-\n'.format(ifq+1, suff)
            text += '\n'.join(['{0} {1} {2} {1}'.format(lm[ie]/1e3, ls[ie]/1e3, lag[ie,1,ifq]/1e3)
                          for ie in range(nen)])

        if not doindv: continue
        for il in range(nindv):
            text += '\ndescriptor lag_f{}_i{}{},+-\n'.format(ifq+1, il+1, suff)
            text += '\n'.join(['{} {}'.format(ilag[ie,il,1,ifq]/1e3, ilag[ie,il,2,ifq]/1e3)
                              for ie in range(nen)])
            # simulations #
            if not lagS is None:
                ilm, ils = ilagS[:,:,il,1,ifq].mean(0), ilagS[:,:,il,1,ifq].std(0)
                text += '\ndescriptor slag_f{0}_i{1}{2},+- sLag_f{0}_i{1}{2},+-\n'.format(
                            ifq+1, il+1, suff)
                text += '\n'.join(['{0} {1} {2} {1}'.format(ilm[ie]/1e3, ils[ie]/1e3,
                            ilag[ie,il,1,ifq]/1e3) for ie in range(nen)])
    return text
