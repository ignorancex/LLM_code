__author__ = 'sazabi'

#This routine twicks the GIRAFFE spectra wavelength solution
#  by using a list of sky lines and apply a linear fit to get
#  a new solution and rebin the spectra

#Input: GIRAFFE spectra
#Output: GIRAFFE spectra + shift_fit

from robust_polyfit import *
from functions import *
import matplotlib.pyplot as plt

def spec_shift_fit(inputname):
    data = pyfits.open(inputname)
    hdr = data[0].header
    spec = data[0].data
    spec_aftershift = np.zeros_like(spec)

    x0 = hdr['CRVAL2'] * 10
    delta_lam = hdr['CDELT2'] * 10
    [num,fibernum] = spec.shape
    lamb = np.arange(x0,x0+delta_lam*num-0.00001,delta_lam)

    print "Start wavelength: "+str(lamb[0])+" A"
    print "Stop wavelength: "+str(lamb[-1])+" A"
    print "Delta lamba: "+str(delta_lam)+" A/pixel"
    print "Total pixel number: "+str(num)
    print "Total fiber number: "+str(fibernum)

    order = 1
    winsize = 1
    res = 1.5

    goodlines = np.loadtxt(filedir+'good_skylines.txt')
    s0 = np.zeros([spec.shape[1],order+1])
    for j in range(spec.shape[1]):
    #for j in range(2):
        flux = spec[:,j]
        #flux = spec_aftershift[:,j]
        x = []
        y = []
        for i in goodlines:
            line = i
            fit_lamb = lamb[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
            fit_flux = flux[find_nearest(lamb, line-winsize*res):find_nearest(lamb, line+winsize*res)]
            #plot(fit_lamb,fit_flux)
            params = fit_spec(fit_lamb,fit_flux, line, res)
            #plot(fit_lamb,Flin(fit_lamb,params))
            #show()
            #print line, params[1], line-params[1]
            x.append(line)
            y.append((params[1]-line)/line)

        x = np.array(x)
        y = np.array(y)
        [z, rms] = polyfit(x,y,order)
        p = np.poly1d(z)
        print 'Fiber ID: %3d,  RMS: %4.2f km/s' % (j, rms * 3e5)
        if abs(rms) * 3e5 > 1.5:
            plt.plot(x, y * 3e5, 'ok')
            plt.plot(x, p(x) * 3e5, '-r')
            plt.show()
            temp = raw_input('Delete a line? Put in 0 or line number (1, 2, 3...). >')
            if temp != '0':
                y = y[x != x[int(temp) - 1]]
                x = x[x != x[int(temp)-1]]
                [z, rms] = polyfit(x, y, order)
                p = np.poly1d(z)
                print 'Fiber ID: %3d,  RMS: %4.2f km/s' % (j, rms * 3e5)
                plt.plot(x, y * 3e5, 'ok')
                plt.plot(x, p(x) * 3e5, '-r')
                plt.show()
                temp = raw_input('Are you happy with this? (Y/N) >')
                if temp == 'N' or temp =='n' or temp == '0':
                    print 'Then I will use default value'
                    z = np.array([2.033108e-08, -1.843296e-04])
                    p = np.poly1d(z)

        new_x = lamb - p(lamb)*lamb
        spec_aftershift[:,j] = np.interp(lamb, new_x, flux)
        s0[j] = z

    plt.plot((s0[:,1]+s0[:,0]*8600) * 3e5,'ok', label = 'zero-order shift in km/s at 8600 A')
    plt.plot(s0[:,0] * 200 * 3e5 ,'^r', label = 'first-order shift in km/s/200A')
    plt.legend(loc=7)
    plt.title('0th- and 1st- order shift as a function of fiber ID')
    plt.xlabel('Fiber ID')
    plt.ylabel('$\Delta V(km/s)$')
    plt.show()

    hdu1 = pyfits.PrimaryHDU(spec_aftershift, hdr)
    hdu2 = pyfits.BinTableHDU(data[1].data)
    outContent = pyfits.HDUList([hdu1, hdu2])
    outContent.writeto(outputname)


if len(sys.argv) > 1:
    inputname = str(sys.argv[1])
else:
    print 'ERROR! must have an input'
    pass

if inputname[-4:] != 'fits':
    print 'ERROR! Input must be fits file'
    pass
if os.path.isfile(inputname):
    outputname = inputname.split('.fits')[0] + '_shift_fit.fits'
    if os.path.isfile(outputname):
        print "File exist. Do you want to overwrite it?"
        yn = raw_input('Overwrite (y or n) > ')
        if yn == 'y' or yn == 'yes' or yn == 'Y':
            os.remove(outputname)
            spec_shift_fit(inputname)
        elif yn == 'n' or yn == 'no' or yn == 'N':
            pass
        else:
            print "WRONG INPUT"
    else:
        spec_shift_fit(inputname)
else:
    print "Input file not exist"
