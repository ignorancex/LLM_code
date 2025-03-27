__author__ = 'sazabi'

# This routine does the sky subtraction
# It first pick the sky fibers and average them weighted by the variance to make the master sky spectrum
# then for each fiber, it calculate the scaling factor to the master sky using a list of sky lines
# and then subtract the scaled sky spectrum

#Input: GIRAFFE spectra and error spectra
#Output: GIRAFFE spectra + subsky
#Output has 5 extensions
# [1] sky subtracted spectra
# [2] error spectra (error from sky considered)
# [3] table for the fiber set up
# [4] master sky spectrum
# [5] master sky error spectrum

from functions import *
from scipy.ndimage.filters import gaussian_filter
import scipy
print scipy.version.version

def sub_sky(inputname, inputerror):
    data = pyfits.open(inputname)
    errordata = pyfits.open(inputerror)

    hdr = data[0].header
    spec = data[0].data
    dspec = errordata[0].data
    mask = dspec ==0
    dspec[mask] = 999999
    tab = data[1].data
    spec_aftersky = np.zeros_like(spec)
    dspec_aftersky = np.zeros_like(spec)

    x0 = hdr['CRVAL2'] * 10
    delta_lam = hdr['CDELT2'] * 10
    [num,fibernum] = spec.shape
    lamb = np.arange(x0,x0+delta_lam*num-0.00001,delta_lam)

    print "Start wavelength: "+str(lamb[0])+" A"
    print "Stop wavelength: "+str(lamb[-1])+" A"
    print "Delta lamba: "+str(delta_lam)+" A/pixel"
    print "Total pixel number: "+str(num)
    print "Total fiber number: "+str(fibernum)

    # make master sky spectra, weighted by variance
    idx = []
    for i in range(fibernum):
        if tab['magnitude'][i] == 0. or tab['magnitude'][i] > 99. : idx.append(i)
    idx = np.array(idx)
    skyraw = spec[:,idx]
    skyraw_err = dspec[:,idx]
    skyraw_wgt = 1/skyraw_err**2

    sky_mean = np.sum(skyraw * skyraw_wgt, axis=1) / np.sum(skyraw_wgt, axis=1)  #master sky spectrum
    sky_err = np.sqrt(1. / np.sum(skyraw_wgt, axis=1)) # not MAD here so no need?? * 1.4826 * sqrt(pi/2) #error of the master sky spectrum

    winsize = 4
    res = 1.5

    goodlines = np.loadtxt(filedir+'good_skylines.txt')

    sky_int = []
    sky_fwhm = []
    for i in goodlines:
        line = i
        fit_lamb = lamb[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
        fit_flux = sky_mean[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
        params = fit_spec(fit_lamb,fit_flux, line, res)
        sky_int.append(params[0]*params[2])
        sky_fwhm.append(params[2])
        #print params[2]
    # subtract from master sky

    for j in range(spec.shape[1]):
    #for j in [56]:
        flux = spec[:,j]
        spec_int = []
        spec_fwhm = []
        for i in goodlines:
            line = i
            fit_lamb = lamb[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
            fit_flux = flux[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
            #plt.plot(fit_lamb,fit_flux)
            params = fit_spec(fit_lamb,fit_flux, line, res)
            #plt.plot(fit_lamb,Flin(fit_lamb,params))
            #plt.show()
            spec_int.append(params[0]*params[2])
            spec_fwhm.append(params[2])
            #print params[2]

        spec_fwhm_median = np.median(spec_fwhm)
        sky_fwhm_median = np.median(sky_fwhm)
        orig_fwhm_ratio = np.median(np.array(spec_fwhm) / np.array(sky_fwhm))
        print 'original fwhm ratio = %5.3f' %(orig_fwhm_ratio)
        if orig_fwhm_ratio > 1.015:
            #print spec_fwhm_median, sky_fwhm_median
            #print np.sqrt(spec_fwhm_median**2-sky_fwhm_median**2)
            sky_temp = gaussian_filter(sky_mean, np.sqrt(spec_fwhm_median**2-sky_fwhm_median**2)/delta_lam)#, truncate = 3.0)
            #print all(sky_temp == sky_mean)
            rescale_sky_int = []
            rescale_sky_fwhm = []
            for i in goodlines:
                line = i
                fit_lamb = lamb[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
                fit_flux = sky_temp[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
                params = fit_spec(fit_lamb,fit_flux, line, res)
                rescale_sky_int.append(params[0]*params[2])
                rescale_sky_fwhm.append(params[2])
            fwhm_ratio = np.array(spec_fwhm) / np.array(rescale_sky_fwhm)
            ratio = np.array(spec_int) / np.array(rescale_sky_int)
            #spec_fwhm_median = np.median(spec_fwhm)
            #rescale_sky_fwhm_median = np.median(rescale_sky_fwhm)
            #print spec_fwhm_median, rescale_sky_fwhm_median
        elif orig_fwhm_ratio < 0.985:
            sky_temp = sky_mean
            #print spec_fwhm_median, sky_fwhm_median
            #print np.sqrt(sky_fwhm_median**2-spec_fwhm_median**2)
            flux = gaussian_filter(flux,  np.sqrt(sky_fwhm_median**2-spec_fwhm_median**2)/delta_lam)#, truncate = 3.0)
            spec_int = []
            spec_fwhm = []
            for i in goodlines:
                line = i
                fit_lamb = lamb[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
                fit_flux = flux[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
                params = fit_spec(fit_lamb,fit_flux, line, res)
                spec_int.append(params[0]*params[2])
                spec_fwhm.append(params[2])
            fwhm_ratio = np.array(spec_fwhm) / np.array(sky_fwhm)
            ratio = np.array(spec_int) / np.array(sky_int)
            #spec_fwhm_median = np.median(spec_fwhm)
            #sky_fwhm_median = np.median(sky_fwhm)
            #print spec_fwhm_median, sky_fwhm_median
        else:
            sky_temp = sky_mean
            fwhm_ratio = np.array(spec_fwhm) / np.array(sky_fwhm)
            ratio = np.array(spec_int) / np.array(sky_int)
            #spec_fwhm_median = np.median(spec_fwhm)
            #sky_fwhm_median = np.median(sky_fwhm)
            #print spec_fwhm_median, sky_fwhm_median


        #plt.plot(goodlines, ratio, 'o')
        #plt.plot(goodlines, fwhm_ratio, 'rd')
        #plt.show()
        #print ratio
        print 'Fiber ID = %3d, scale factor = %4.2f, fwhm ratio = %5.3f' %(j, np.median(ratio), np.median(fwhm_ratio))
        spec_aftersky[:,j] = flux - sky_temp * np.median(ratio)
        dspec_aftersky[:,j] = np.sqrt(dspec[:,j]**2 + sky_err**2 * np.median(ratio)**2)

    hdr['CRVAL2'] = lamb[0]
    hdr['CDELT2'] = delta_lam
    hdr['CRPIX2'] = 1
    hdr['CUNIT2'] = '        '
    arr = np.arange(0,num)
    hdu1 = pyfits.PrimaryHDU(spec_aftersky[arr], hdr)
    hdu2 = pyfits.ImageHDU(dspec_aftersky[arr], hdr)
    hdu3 = pyfits.BinTableHDU(data[1].data)
    hdu4 = pyfits.ImageHDU(sky_mean[arr], hdr)
    hdu5 = pyfits.ImageHDU(sky_err[arr], hdr)
    outContent = pyfits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5])
    outContent.writeto(outputname)

if len(sys.argv) > 1:
    inputname = str(sys.argv[1])
    inputerror = str(sys.argv[2])
else:
    print 'ERROR! Need an input file'
    pass

if inputname[-4:] != 'fits' or inputerror[-4:] != 'fits':
    print 'ERROR! Input must be fits file'
    pass
if os.path.isfile(inputname) and os.path.isfile(inputerror):
    outputname = inputname.split('.fits')[0] + '_subsky_adjust_width.fits'
    if os.path.isfile(outputname):
        print "File exist. Do you want to overwrite it?"
        yn = raw_input('Overwrite (y or n) > ')
        if yn == 'y' or yn == 'yes' or yn == 'Y' or yn == 'YES':
            os.remove(outputname)
            sub_sky(inputname, inputerror)
        elif yn == 'n' or yn == 'no'or yn == 'N' or yn == 'NO':
            pass
        else:
            print "Please type y or n"
    else:
        sub_sky(inputname, inputerror)
else:
    print "Input file not exist"