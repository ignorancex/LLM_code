from rv_functions import *
import emcee
from scipy import stats
import os 
from rv_functions import *
from exptime_table import *
import numpy as np
from scipy import ndimage
from math import pi, sqrt, log10, log
import pyfits
import mpfit
import re
import matplotlib.pyplot as plt

showplot = 0  # show the plot or not
savefig = 1   # save the plot or not
savedata = 1  # save data or not
zquality = 0  # assign quality manually or not
snr_threshold = 0 # threshold above which to calculate rv/feh

#for metallicity code
saveplot = 1
display = 0

filedir = '../files/'


def mpfitexpr(func, x, y, err, start_params, control_params, check=True, full_output=True, **kw):
    """Fit the used defined expression to the data
    Input:
    - func: string with the function definition
    - x: x vector
    - y: y vector
    - err: vector with the errors of y
    - start_params: the starting parameters for the fit
    Output:
    - The tuple (params, yfit) with best-fit params and the values of func evaluated at x
    Keywords:
    - check: boolean parameter. If true(default) the function will be checked for sanity
    - full_output: boolean parameter. If True(default is False) then instead of best-fit parameters the mpfit object is returned
    Example:
    params,yfit=mpfitexpr('p[0]+p[2]*(x-p[1])',x,y,err,[0,10,1])

    If you need to use numpy and scipy functions in your function, then
            you must to use the full names of these functions, e.g.:
            numpy.sin, numpy.cos etc.

    This function is motivated by mpfitexpr() from wonderful MPFIT IDL package
            written by Craig Markwardt

    """

    def myfunc(p, fjac=None, x=None, y=None, err=None):
        return [0, eval('(y-(%s))/err' % func)]

    myre = "[^a-zA-Z]p\[(\d+)\]"
    r = re.compile(myre)
    maxp = -1
    for m in re.finditer(r, func):
        curp = int(m.group(1))
        maxp = curp if curp > maxp else maxp
    if check:
        if maxp == -1:
            raise Exception("wrong function format")
        if maxp + 1 != len(start_params):
            raise Exception("the length of the start_params != the length of the parameter verctor of the function")
    fa = {'x': x, 'y': y, 'err': err}
    res = mpfit.mpfit(myfunc, start_params, functkw=fa, quiet=1, parinfo=control_params, **kw)
    yfit = eval(func, globals(), {'x': x, 'p': res.params})
    if full_output:
        return (res, yfit)
    else:
        return (res.params, yfit)


def fit_all_cat_lines(fiberid, w, spec, dspec, w_mg, spec_mg, dspec_mg, figdir, velocity=0, gaussianonly=0):
    c = 2.99792458e5

    fitstart = (np.abs(w - 8482)).argmin()
    fitend = (np.abs(w - 8682)).argmin()
    dw = np.median(w[fitstart:fitend] - np.roll(w[fitstart:fitend], 1))
    npix = fitend - fitstart + 1

    contstart = (np.abs(w - 8563)).argmin()
    contend = (np.abs(w - 8577)).argmin()

    peakfindstart = (np.abs(w - 8538.09 * (1 + velocity / c))).argmin()
    peakfindend = (np.abs(w - 8546.09 * (1 + velocity / c))).argmin()

    sn = np.median(spec[fitstart:fitend] / dspec[fitstart:fitend])
    if sn < 10:
        lowsn = 1
    else:
        lowsn = 0

    contlevel = np.median(spec[contstart:contend])
    if contlevel == 0:
        contlevel = 1e-5
    spec = spec / contlevel
    dspec = dspec / contlevel

    smoothspec = ndimage.filters.uniform_filter(spec, size=5)
    linepos = smoothspec[peakfindstart:peakfindend].argmin()
    depth = min(smoothspec[peakfindstart:peakfindend]) - np.median(spec[fitstart:fitend])

    initial_guesses = np.zeros(10)
    param_control = [{'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.]} for i in range(10)]

    initial_guesses[0] = np.median(spec[contstart:contend])
    initial_guesses[1] = 0.5 * depth
    initial_guesses[2] = (w[fitstart:fitend])[linepos + peakfindstart - fitstart]
    initial_guesses[3] = 1.0
    initial_guesses[4] = 0.3 * depth
    initial_guesses[5] = 1.0
    initial_guesses[6] = 0.25 * depth
    initial_guesses[7] = 0.4 * depth
    initial_guesses[8] = 0.15 * depth
    initial_guesses[9] = 0.24 * depth

    param_control[1]['limited'][1] = 1
    param_control[1]['limits'][1] = -0.01
    param_control[1]['limited'][0] = 1
    param_control[1]['limits'][0] = -1
    param_control[4]['limited'][1] = 1
    param_control[4]['limits'][1] = 0
    param_control[4]['limited'][0] = 1
    param_control[4]['limits'][0] = -1
    param_control[6]['limited'][1] = 1
    param_control[6]['limits'][1] = 0
    param_control[6]['limited'][0] = 1
    param_control[6]['limits'][0] = -1
    param_control[7]['limited'][1] = 1
    param_control[7]['limits'][1] = 0
    param_control[7]['limited'][0] = 1
    param_control[7]['limits'][0] = -1
    param_control[8]['limited'][1] = 1
    param_control[8]['limits'][1] = 0
    param_control[8]['limited'][0] = 1
    param_control[8]['limits'][0] = -1
    param_control[9]['limited'][1] = 1
    param_control[9]['limits'][1] = 0
    param_control[9]['limited'][0] = 1
    param_control[9]['limits'][0] = -1

    # FORCE LINE WIDTHS TO BE AT LEAST 1 RESOLUTION ELEMENT (0.8AA) AND LESS THAN 300 KM/S
    param_control[3]['limited'][0] = 1
    param_control[3]['limits'][0] = 0.50
    param_control[3]['limited'][1] = 1
    param_control[3]['limits'][1] = 3.63
    param_control[5]['limited'][0] = 1
    param_control[5]['limits'][0] = 0.50
    param_control[5]['limited'][1] = 1
    param_control[5]['limits'][1] = 3.63

    if lowsn:
        initial_guesses[4] = 0.
        param_control[4]['fixed'] = 1
        param_control[5]['fixed'] = 1

    if gaussianonly:
        initial_guesses[4] = 0.
        initial_guesses[5] = 1.0
        param_control[4]['fixed'] = 1
        param_control[5]['fixed'] = 1

    gauss = 'p[1]*np.exp(-0.5*((x-p[2])/p[3])**2)+ p[6]*np.exp(-0.5*( (x-p[2]*0.994841)/p[3] )**2) + p[7]*np.exp(-0.5*( (x-p[2]*1.01405)/p[3] )**2)'
    lorentz = 'p[4]*p[5]/( (x-p[2])**2 + (p[5]/2.)**2 ) + p[8]*p[5]/( (x-p[2]*0.994841)**2 + (p[5]/2.)**2 ) + p[9]*p[5]/( (x-p[2]*1.01405)**2 + (p[5]/2.)**2 )'
    func = 'p[0] + ' + gauss + ' + ' + lorentz

    x = w[fitstart:fitend]
    y = spec[fitstart:fitend]
    err = dspec[fitstart:fitend]
    try:
        m, yfit = mpfitexpr(func, x, y, err, initial_guesses, param_control)
    except ValueError:
        print("Oops!  Something wrong.")

    modelgl = yfit
    covargl = m.covar
    errmsg = m.errmsg
    status = m.status
    perrorgl = m.perror
    chisqgl = sum((spec[fitstart:fitend] - modelgl) ** 2 / dspec[fitstart:fitend] ** 2)
    outparams = m.params
    lineparams = outparams
    bool = True
    try:
        for err in perrorgl:
            print(err)
            if err != None:
                bool = False
    except:
        bool = True
    print(bool)
    if bool:
        perrorgl = np.zeros(10) + 1
        covargl = np.zeros([10, 10]) + 1
    print(perrorgl)

    if showplot or saveplot:
        plt.figure()
        plt.plot(w[fitstart:fitend], spec[fitstart:fitend], c='y')
        plt.plot(w[fitstart:fitend], dspec[fitstart:fitend], '--', c='g')
        plt.plot(w[fitstart:fitend], modelgl, lw=2, c='k')

        if outparams[3] > outparams[5]:
            larger_width = outparams[3]
        else:
            larger_width = outparams[5]
        plt.title(fiberid)
        plt.ylim(-0.5, 1.5)
        if saveplot:
            plt.savefig(figdir + 'feh_' + str(fiberid) + '.png')
        if showplot:
            plt.show()
        plt.cla

    gaussian_integral = outparams[1] * outparams[3] * sqrt(2 * pi)
    dgaussian_integral = sqrt((outparams[1] * perrorgl[3] * sqrt(2 * pi)) ** 2 + \
                              (perrorgl[1] * outparams[3] * sqrt(2 * pi)) ** 2)

    lorentzian_integral = 2 * pi * outparams[4]
    dlorentzian_integral = 2 * pi * perrorgl[4]
    ew2_fit = gaussian_integral + lorentzian_integral
    dew2_fit = sqrt(dgaussian_integral ** 2 + dlorentzian_integral ** 2)

    v2 = (outparams[2] - 8542.09) / 8542.09 * c

    gaussian_integral = outparams[6] * outparams[3] * sqrt(2 * pi)
    dgaussian_integral = sqrt((outparams[6] * perrorgl[3] * sqrt(2 * pi)) ** 2 + \
                              (perrorgl[6] * outparams[3] * sqrt(2 * pi)) ** 2)

    lorentzian_integral = 2 * pi * outparams[8]
    dlorentzian_integral = 2 * pi * perrorgl[8]
    ew1_fit = gaussian_integral + lorentzian_integral
    dew1_fit = sqrt(dgaussian_integral ** 2 + dlorentzian_integral ** 2)

    gaussian_integral = outparams[7] * outparams[3] * sqrt(2 * pi)
    dgaussian_integral = sqrt((outparams[7] * perrorgl[3] * sqrt(2 * pi)) ** 2 + \
                              (perrorgl[7] * outparams[3] * sqrt(2 * pi)) ** 2)

    lorentzian_integral = 2 * pi * outparams[9]
    dlorentzian_integral = 2 * pi * perrorgl[9]
    ew3_fit = gaussian_integral + lorentzian_integral
    dew3_fit = sqrt(dgaussian_integral ** 2 + dlorentzian_integral ** 2)

    ews = ew1_fit + ew2_fit + ew3_fit
    dews = sqrt(dew1_fit ** 2 + dew2_fit ** 2 + dew3_fit ** 2)
    vcat = v2

    #Mg EW
    contstart = (np.abs(w_mg - 8798.8)).argmin()
    contend = (np.abs(w_mg - 8803.8)).argmin()
    contstart2 = (np.abs(w_mg - 8809.8)).argmin()
    contend2 = (np.abs(w_mg - 8814.8)).argmin()

    contlevel = np.median(np.concatenate([spec_mg[contstart:contend],spec_mg[contstart2:contend2]]))
    if contlevel == 0:
        contlevel = 1e-5
    spec = spec_mg/ contlevel
    dspec = dspec_mg / contlevel
    
    mask_mg = (w_mg > 8803.8) & (w_mg < 8809.8)
    w_mg = w_mg[mask_mg] 
    spec = spec[mask_mg]
    dspec = dspec[mask_mg]
 
    deltax = .2
    ew_mg = deltax*len(spec) - deltax*sum(spec)
    ew_mg2 = sum(1-spec)*deltax
   
    dew_mg = deltax * np.sqrt(sum(dspec**2))
    
    return ews, dews, ew1_fit, dew1_fit, ew2_fit, dew2_fit, ew3_fit, dew3_fit, vcat, ew_mg, dew_mg


def feh_cat_all(ews, dews, vmag, dvmag, distance, ddistance, ebv=0, gaussonly=0, velocity=0, dew_sys=0):
    dm = 5 * log10(distance / 10.)
    ddm = 25 * ddistance ** 2 / (distance * log(10)) ** 2
    A_V = 3.1 * ebv
    M_V = vmag - dm - A_V
    dM_V = sqrt(dvmag ** 2 + ddm ** 2)

    # CARRERA ET AL. (2013) CAT CALIBRATION TO ABSOLUTE V MAGNITUDE
    a = -3.45
    b = 0.16
    c = 0.41
    d = -0.53
    e = 0.019

    da = 0.04
    db = 0.01
    dc = 0.004
    dd = 0.11
    de = 0.002

    feh = a + b * M_V + c * (abs(ews)) + d * (abs(ews)) ** (-1.5) + e * (abs(ews)) * M_V
    dfeh = sqrt(1. * da ** 2 + M_V ** 2 * db ** 2 + (abs(ews)) ** 2 * dc ** 2 + (abs(ews)) ** (-3) * dd ** 2 + \
                ((abs(ews)) * M_V) ** 2 * de ** 2 + (b + e * (abs(ews))) ** 2 * dM_V ** 2 + \
                (c - 1.5 * d * (abs(ews)) ** (-2.5) + e * M_V) ** 2 * ((dews ** 2) + dew_sys ** 2))
    return feh, dfeh


def gmr_to_vmag(g, r, dg=0.03, dr=0.03):
    v = g - 0.487 * (g - r) - 0.025
    dv = sqrt((1 - 0.487) * dg ** 2 + 0.487 * dr ** 2)
    return v, dv

def rv_met(fl, num, combined=0):
    if combined == 0: 
        runname = fl+'_'+num
    else: 
        runname = fl+'_combined'
    fl_nm = '../'+fl
    figdir = fl_nm+'/RV_met_data/'+runname+'/'

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    if combined == 0:
        inputname = fl_nm+'/Reduced_data/science_rbnspectra_'+num+'_shift_fit_subsky_adjust_width.fits'
    else:
        inputname = fl_nm+'/'+fl+'.fits'
    filename = inputname.split('/')[-1]
    fl = inputname.split('.')[-2]

    outputfile = fl_nm+'/RV_met_data/rv_met_'+runname+'_testing.txt'
    
    import heliocorr
    rvc = heliocorr.compute_heliocorr_vlt(inputname)

    # display window for CaT lines
    CaT1min=8480
    CaT1max=8520

    CaT2min=8520
    CaT2max=8560

    CaT3min=8640
    CaT3max=8680


    # fitting window, a gap of 8630-8645 due to the discontinuity in DEIMOS templates
    CaTmin1=8480
    CaTmax1=8620

    CaTmin2=8645
    CaTmax2=8680

    nstars = 4
    c = 2.9979e5
    temp = pyfits.open('deimos-052715.fits')[0].data
    hdr = pyfits.open('deimos-052715.fits')[0].header
    coeff0 = hdr['COEFF0']
    coeff1 = hdr['COEFF1']
    rvwl = 10**(coeff0 + coeff1 * np.arange(7200))
    rvspec = temp[3:7]
    rvstar=np.array(['BD-185550','HD103095', 'BD+233912','HD109995'])

    rvobs = np.zeros(nstars)

    ######MCMC parameters######
    ndim=1
    nwalkers=20
    rvmin = -800
    rvmax = 800
    p0=np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    p0= p0 * rvmax * 2 - rvmax

    # MCMC needs some time to produce reasonable "d" from the likelihood, which is called the "burn-in" period.
    # Adjusting the "burn-in" period is quite empirical.
    nburn=100
    nsam=900

    ###########################
    inputfile = pyfits.open(inputname)
    inputspec = inputfile[0].data
    inputdspec = inputfile[1].data
    inputtbl = inputfile[2].data
    hdr = inputfile[0].header

    xref=hdr['CRVAL2']
    delx=hdr['CDELT2']
    xrefpix=hdr['CRPIX2'] - 1.0
    num=hdr['NAXIS2']
    wl=np.arange(xref-xrefpix*delx,xref+(num-1e-10-xrefpix)*delx,delx)

    wlmask = (wl > 8100) & (wl < 8950)
    wl = wl[wlmask]
    inputspec = inputspec[wlmask]
    inputdspec = inputdspec[wlmask]
    mjd = hdr['MJD-OBS']
    if combined:
        exp = exptime(fl_nm+'/Reduced_data')
    else:
        exp = hdr['EXPTIME']
    print("EXPOSURE\n")
    print(exp)
    rvdist = np.zeros([nstars, nwalkers * nsam])

    k = 0
    if savedata:
        f=open(outputfile,'a')
        print("Outputfil: ", outputfile)
        f.write('#index   ID    run   mjd   exptime  snr  rv  rverr  absdev  skew    kurt   rvc   template    redchi2     rv1   rverr1    redchi2_1   rv2    rverr2     redchi2_2   rv3   rverr3    redchi2_3   rv4    rverr4    redchi2_4   rvqual   ew1      ew1err   ew2    ew2err    ew3       ew3err     rvew     ewqual  ew_mg  ewerr_mg  ra   dec    mag      filename\n')
       
    for i in range(len(inputtbl)):
        print("FIBER ID = "+str(i))

        spec = inputspec[:,i]
        dspec = inputdspec[:,i] 
        if k == 0:
            speclen = len(wl)
            rvspec_s =  np.zeros([nstars, speclen])
            #mask = (wl < CaT1max) & (wl > CaT1min) | (wl < CaT2max) & (wl > CaT2min) | (wl < CaT3max) & (wl > CaT3min)
            mask_orig = (wl < CaTmax1) & (wl > CaTmin1) | (wl < CaTmax2) & (wl > CaTmin2)
            for j in range(nstars):
                rvspec_s[j] = np.interp(wl, rvwl, rvspec[j])
        k = k + 1


        cat = inputtbl['OBJECT'][i]
        ra = inputtbl['RA'][i]
        dec = inputtbl['DEC'][i]
        mag = inputtbl['MAGNITUDE'][i]
        snr = np.median(spec/dspec)

        if snr > snr_threshold:
            pixels = np.arange(len(spec))
            try: 
                spec, dspec = normalize_spec(pixels, spec, dspec)
            except: 
                med = np.nanmedian(spec)
                spec = spec/med
                dspec = dspec/med
            dspec = dspec
            chi2rv = np.zeros(nstars)

            for kk in range(nstars):
                mask = mask_orig
                specwl = wl
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_post, args=(rvmin, rvmax, mask, specwl, rvspec_s[kk], spec, dspec), a=0.01)
                pos, prob, state = sampler.run_mcmc(p0, nburn)
                sampler.reset()

                sampler.run_mcmc(pos, nsam)
                samples = sampler.chain[:, :, :].reshape((-1, ndim))

                rvdist[kk,:] = sampler.flatchain[:,0]
                temp = rvdist[kk,:]
                temp_mean = np.median(temp)
                if np.std(temp) > 20 : temp = temp[(temp < temp_mean + 70) & (temp > temp_mean -70)]
                temp = stats.sigmaclip(temp,low=5, high=5)[0]
                rv_mean = np.median(temp)
                rv_std = np.std(temp)
                chi2rv[kk] = chi2cal(rv_mean, rvmin, rvmax, mask, specwl, rvspec_s[kk], spec, dspec) * 2
                masklen=len(specwl[mask])
            chi2rv[chi2rv == 0] = -1e10
            rvidx = (chi2rv == np.max(chi2rv))
            jj = np.arange(0,nstars)[rvidx][0]


            poster = rvdist[rvidx].flatten()
            d = poster
            m1 = stats.median_abs_deviation(d)
            m2 = stats.skew(d)
            m3 = stats.kurtosis(d)
            d_mean = np.percentile(poster, 50)
            rv = np.percentile(poster, 50) + rvobs[jj] + rvc
            drv = np.abs(0.5 * (np.percentile(poster, 50. + 68.27 / 2) - np.percentile(poster, 50. - 68.27 / 2)))
            
            all_rv, all_drv, all_bestchi2 = [], [], []
            for ii in range(4):
                jj2 = ii
                poster2 = rvdist[ii].flatten()
                d_mean2 = np.percentile(poster2, 50)
                all_rv.append(np.percentile(poster2, 50) + rvobs[jj2] + rvc)
                all_drv.append(np.abs(0.5 * (np.percentile(poster2, 50. + 68.27 / 2) - np.percentile(poster2, 50. - 68.27 / 2))))
                all_bestchi2.append(-chi2rv[jj2]/masklen)
                     
            bestrvstar = rvstar[jj]
            bestchi2 = -chi2rv[jj]/masklen

            fig, axarr = plt.subplots(1, 5, figsize=(15,6))
            minn = np.percentile(d, 10)
            max = np.percentile(d, 90)
            axarr[0].hist(d, 100, range=(minn, max), color="k", histtype="step")
            axarr[0].set_title('RV Histogram', fontsize=16)
            #axarr[0].xaxis.set_major_locator(plt.MultipleLocator(5))
            axarr[0].set_xlabel('RV')

            axarr[1].plot(specwl, spec, 'y',lw=0.5)
            axarr[1].plot(specwl*(1+d_mean/c), rvspec_s[jj], 'b')
            axarr[1].set_xlim(CaT1min,CaT1max)
            axarr[1].set_ylim(-0.5,1.5)
            axarr[1].xaxis.set_major_locator(plt.MultipleLocator(20))
            #axarr[1].set_xlabel('Wavelength')

            axarr[2].plot(specwl, spec, 'y',lw=0.5)
            axarr[2].plot(specwl*(1+d_mean/c), rvspec_s[jj], 'b')
            axarr[2].set_xlim(CaT2min,CaT2max)
            axarr[2].set_ylim(-0.5,1.5)
            axarr[2].set_title(rvstar[jj]+' + '+str(cat))
            axarr[2].xaxis.set_major_locator(plt.MultipleLocator(20))
            axarr[2].set_xlabel('Wavelength')


            axarr[3].plot(specwl, spec, 'y',lw=0.5)
            axarr[3].plot(specwl*(1+d_mean/c), rvspec_s[jj], 'b')
            axarr[3].set_xlim(CaT3min,CaT3max)
            axarr[3].set_ylim(-0.5,1.5)
            axarr[3].xaxis.set_major_locator(plt.MultipleLocator(20))

            axarr[4].plot(specwl*(1-d_mean/c), spec , 'y', lw=0.5)
            axarr[4].axvline(8183.25, ls='--', color='r')
            axarr[4].axvline(8194.79, ls='--', color='r')
            axarr[4].set_xlim(8160, 8220)
            #axarr[4].axvline(8542.09, ls='--', color='r')
            #axarr[4].set_xlim(8530, 8560)
            axarr[4].set_ylim(-0.5, 1.5)
            axarr[4].xaxis.set_major_locator(plt.MultipleLocator(30))
            txt = "Velocity: "+str(round(rv,2))+" Error: "+str(round(drv,2))+"\nMagnitude: "+str(round(mag,2))+" SNR: "+str(round(snr,2))
            axarr[4].set_title(txt)
                

            if savefig:
                print(str(cat))
                plt.savefig(figdir+str(cat)+'_rv.png')
            if showplot:
                plt.show()
            plt.close()

            if zquality:
                temp = raw_input('quality (0 or 1) --> ')
                zq_vel = int(temp)
            else:
                zq_vel = -1

        else:
            rv = np.nan
            drv = np.nan
            bestrvstar = np.nan
            bestchi2 = np.nan
            zq_vel = -1



        dew_sys = 0.0


        wl_m, specarr_m, dspecarr_m = read_vlt_spec(inputname)

        fitstbl = pyfits.open(inputname)[2].data
    #rvtbl = np.genfromtxt('Phestream_sfield_combined_2.txt', names=True, dtype=None)
        j=0
        for obj in fitstbl['OBJECT']:
            obj = obj.rstrip()
            fitstbl['OBJECT'][j] = obj
            j = j+1
        mask_m = (wl_m > 8480) & (wl_m < 8695)
        wl_m_mg = wl_m
        wl_m = wl_m[mask_m]

        if snr > snr_threshold:
            idx = fitstbl['OBJECT'] == str(cat).rstrip()
            spec_m = specarr_m[:,idx]
            dspec_m = dspecarr_m[:,idx]

            spec_m_mg = spec_m
            dspec_m_mg = dspec_m
            
            spec_m = spec_m[mask_m]
            dspec_m = dspec_m[mask_m]

            spec_m_mg = spec_m_mg.flatten()
            dspec_m_mg = dspec_m_mg.flatten()
            spec_m = spec_m.flatten()
            dspec_m = dspec_m.flatten()
            num_m_mg = len(spec_m_mg)
            num_m = len(spec_m)
            pixels_m_mg = np.arange(num_m_mg)
            pixels_m = np.arange(num_m)
            try: 
                spec_m, dspec_m = normalize_spec(pixels_m, spec_m, dspec_m)
            except: 
                med = np.nanmedian(spec_m)
                spec_m = spec_m/med
                dspec_m = dspec_m/med
            try: 
                spec_m_mg, dspec_m_mg = normalize_spec(pixels_m_mg, spec_m_mg, dspec_m_mg)
            except: 
                med_mg = np.nanmedian(spec_m_mg)
                spec_m_mg = spec_m_mg/med_mg
                dspec_m_mg = dspec_m_mg/med_mg
            wl_m_mg=wl_m_mg-wl_m_mg*(rv-rvc)/c


            gaussonly2 = 0
            [ews_m, dews, ew1_fit, dew1_fit, ew2_fit, dew2_fit, ew3_fit, dew3_fit, vcat_m, ew_mg, dew_mg]=fit_all_cat_lines(cat, wl_m,spec_m,dspec_m, wl_m_mg, spec_m_mg, dspec_m_mg, figdir, rv, gaussonly2)

            if zquality:
                temp = raw_input('quality (0 or 1) --> ')
                zq_ew = int(temp)

            else:
                zq_ew = -1
            ews_merr = sqrt(dews**2+dew_sys**2)
        else:
            vcat_m = np.nan
            dew1_fit = np.nan
            ew1_fit = np.nan
            dew2_fit = np.nan
            ew2_fit = np.nan
            dew3_fit = np.nan
            ew3_fit = np.nan
            ew_mg = np.nan
            ewerr_mg = np.nan
            bestchi2 = np.nan
            template = np.nan
            drv = np.nan
            rv = np.nan
            zq_ew = np.nan
        if savedata:
            f.write('%4d  %15s %20s  %12.6f  %5d  %7.2f  %8.2f  %5.2f  %5.3f %5.3f %5.3f  %5.2f  %10s  %6.2f  %8.2f  %5.2f  %6.2f  %8.2f  %5.2f  %6.2f  %8.2f  %5.2f  %6.2f  %8.2f  %5.2f  %6.2f  %3d  %6.2f  %5.2f  %6.2f  %5.2f  %6.2f  %5.2f  %8.2f  %3f  %5.2f %5.2f %12.7f  %12.7f  %5.2f  %s \n' \
                    %(i, cat, runname, mjd, exp, snr, rv, drv, m1, m2, m3, rvc, bestrvstar, bestchi2,all_rv[0], all_drv[0], all_bestchi2[0], all_rv[1], all_drv[1], all_bestchi2[1], all_rv[2], all_drv[2], all_bestchi2[2], all_rv[3], all_drv[3], all_bestchi2[3], zq_vel, ew1_fit, dew1_fit, ew2_fit, dew2_fit, ew3_fit, dew3_fit, vcat_m, zq_ew, ew_mg, dew_mg, ra, dec, mag, filename))
    if savedata:
        f.close()