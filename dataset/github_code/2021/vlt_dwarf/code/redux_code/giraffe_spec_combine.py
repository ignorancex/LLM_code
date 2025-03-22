__author__ = 'sazabi'

# This routine combines the spectra from different exposures
# Since GIRAFFE has 2 setting for observations MEDUSA1 and MEDUSA2
# The fiber setup table might be different from exposure to exposure
# This routine combines all exposures

# Input: list of sky subtracted spectra to combine, .txt or .lst or .dat
# Output: combined spectra .fits
# Output has 3 extensions
# [1] combined spectra
# [2] combined error spectra
# [3] table for the target list

from functions import *
from astropy.table import Table
import heliocorr

def spec_combine(inputlist, outputname):
    c = 2.9979e5

    speclist = np.loadtxt(inputlist, dtype='str')

    data = pyfits.open(speclist[0])
    hdr = data[0].header
    xref=hdr['CRVAL2']
    delx=hdr['CDELT2']
    xrefpix=hdr['CRPIX2'] - 1.0
    num=hdr['NAXIS2']
    wl=np.arange(xref-xrefpix*delx,xref+(num-1e-10-xrefpix)*delx,delx)
    rvc0 = heliocorr.compute_heliocorr_vlt(speclist[0])
    #rvc0 = 0 # apply heliocentric correction (to 0) to all spectra before coadd
    #rvc0 = -14.3 # apply heliocentric correction (to -14.3) to all spectra before coadd

    biglist = []
    for i in range(len(speclist)):
        biglist.append(pyfits.open(speclist[i]))

    pixelnum = biglist[0][0].data.shape[0]
    expnum = len(speclist)
    merge_list = biglist[0][2].data['OBJECT']
    for i in range(expnum-1):
        first_list = list(merge_list)
        fs = set(first_list)
        second_list = list(biglist[i+1][2].data['OBJECT'])
        merge_list = first_list + [x for x in second_list if x not in fs]

    target = []
    for kk in range(len(merge_list)):
        #if merge_list[kk].isdigit():
        if merge_list[kk][0] != 's' and merge_list[kk][0] != 'S':
            print merge_list[kk]
            target.append(merge_list[kk])

    target = np.array(target)
    starnum = len(target)

    spec = np.zeros([pixelnum, starnum])
    dspec = np.zeros([pixelnum, starnum])

    t = Table(names=('INDEX', 'OBJECT', 'RA', 'DEC', 'MAGNITUDE', 'numexp'), dtype=('>i4', 'S19', '>f8', '>f8', '>f8', '>i4'))

    print starnum
    for kk in range(starnum):
    #for kk in [0]:
        print 'precessing star#: '+str(kk)
        spec_temp = np.zeros([pixelnum, expnum])
        dspec_temp = np.zeros([pixelnum, expnum]) + 9999999
        spec_wgt = np.zeros([pixelnum, expnum])

        count = 0
        for i in range(expnum):
            idx = np.arange(len(biglist[i][2].data['OBJECT']))[biglist[i][2].data['OBJECT'] == target[kk]]
            if len(idx) == 1 :
                idx = int(idx)
                spec_temp[:,i] = biglist[i][0].data[:,idx]
                dspec_temp[:,i] = biglist[i][1].data[:,idx]
                rvc = heliocorr.compute_heliocorr_vlt(speclist[i])
                z = (rvc - rvc0)/c
                if kk == 0:
                    print 'Heliocentric correction = %5.2f' %rvc
                #print 'exp #%2d, z = %6.4f Angstrom' %(i, z)
                spec_temp[:,i] = np.interp(wl, wl * (1 + z), spec_temp[:,i])
                dspec_temp[:,i] = np.interp(wl, wl * (1 + z), dspec_temp[:,i])
                unityfactor = np.median(spec_temp[:,i])
                spec_temp[:,i] = spec_temp[:,i]/unityfactor
                dspec_temp[:,i] = dspec_temp[:,i]/unityfactor #* 0.7
                spec_wgt[:,i] = 1/dspec_temp[:,i]**2
                count = count + 1
                if count == 1:
                    t_index = int(kk)
                    t_object = biglist[i][2].data['OBJECT'][idx]
                    t_ra = biglist[i][2].data['RA'][idx]
                    t_dec = biglist[i][2].data['DEC'][idx]
                    t_magnitude = biglist[i][2].data['MAGNITUDE'][idx]
            else:
                print 'star #' + str(kk)+' missing exp #'+str(i+1)
        spec[:,kk] = np.sum(spec_temp * spec_wgt, axis=1) / np.sum(spec_wgt, axis=1)
        dspec[:,kk] = np.sqrt(1. / np.sum(spec_wgt, axis=1))
        t_numexp = count
        t.add_row((t_index, t_object, t_ra, t_dec, t_magnitude, t_numexp))
        if False:
            #plt.plot(dspec[:,0],lw=2)
            plt.plot(dspec_temp[:,1],'r')
            #plt.plot(dspec_temp[:,2],'g')
            #plt.plot(dspec_temp[:,3],'b')
            #plt.plot(dspec_temp[:,4],'y--')
            #plt.plot(dspec_temp[:,5],'c')
            #plt.plot(dspec_temp[:,6],'m--')
            #plt.plot(dspec_temp[:,7],'b')
            #plt.plot(dspec_temp[:,8],'g')
            plt.plot(dspec_temp[:,9],'y')
            plt.plot(dspec_temp[:,10],'c')
            plt.ylim(0,50)
            plt.show()

    hdr = biglist[0][0].header
    hdu1 = pyfits.PrimaryHDU(spec, hdr)
    hdu2 = pyfits.ImageHDU(dspec, hdr)
    hdu3 = pyfits.BinTableHDU(np.array(t))
    outContent = pyfits.HDUList([hdu1, hdu2, hdu3])
    outContent.writeto(outputname)

if len(sys.argv) == 3:
    inputlist = str(sys.argv[1])
    outputname = str(sys.argv[2])
elif len(sys.argv) == 1:
    inputlist= 'Tucii.lst'
    outputname = '../TucII/TucII_combined.fits'
else:
    print "Need an input name and output name"

if outputname[-4:] != 'fits':
    print 'ERROR! Output must be fits file'
    pass
if inputlist[-3:] != 'lst' and inputlist[-3:] != 'txt' and inputlist[-3:] != 'dat':
    print 'ERROR! Input and output must be fits file'
    pass

if os.path.isfile(inputlist):
    if os.path.isfile(outputname):
        print "File exist. Do you want to overwrite it?"
        yn = raw_input('Overwrite (y or n) > ')
        if yn == 'y' or yn == 'yes' or yn == 'Y' or yn == 'YES':
            os.remove(outputname)
            spec_combine(inputlist, outputname)
        elif yn == 'n' or yn == 'no'or yn == 'N' or yn == 'NO':
            pass
        else:
            print "Please type y or n"
    else:
        spec_combine(inputlist, outputname)
else:
    print "Input file not exist"