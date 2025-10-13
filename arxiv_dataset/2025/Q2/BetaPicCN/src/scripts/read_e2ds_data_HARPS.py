from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import interpolate
from scipy import optimize
import fnmatch

# this takes about 18 minutes on desktop mac

def get_wave(data,header):
   '''gets the polynomial wavelength solution from the ESO keywords for a particular e2ds spectrum'''
   wave=data*0.
   no=data.shape[0]
   npix=data.shape[1]
   d=header['ESO DRS CAL TH DEG LL']
   xx0=np.arange(npix)
   xx=[]
   for i in range(d+1):
      xx.append(xx0**i)
   xx=np.asarray(xx)

   for o in range(no):
      for i in range(d+1):
         idx=i+o*(d+1)
         par=header['ESO DRS CAL TH COEFF LL%d' % idx]
         wave[o,:]=wave[o,:]+par*xx[i,:]
   return wave


def blaze_cor(spec,ref_spec,polyorder=2):
   '''blaze variation correction on a stack of spectra
   spec - spectrum to be corrected for blaze variation
   ref_spec - the reference spectrum to which the blaze variation is normalized
   polyorder - the polynomial order that is fit to the blaze variation '''

   # take the ratio of the input to the reference spectrum
   rat=spec/ref_spec

   # mask out edges of the spectrum to prevent problems with smoothing later on
   rat[0,0:200]=np.nan
   rat[0,-400:]=np.nan
   rat[:,0:50]=np.nan
   rat[:,-50:]=np.nan

   # get the number of spectral orders in the spectrum
   nord=rat.shape[0]

   # values used for fitting the polynomial in pixel space
   x0=np.arange(4096.)
   x1=x0.reshape(4096//256,256)

   # reshape the output spectrum for binning
   o_rat=rat.reshape(nord,4096//256,256)

   # take median of each bin...
   med=np.median(o_rat,axis=2)

   # take the median absoluate deviation (MAD) for each bin
   sig=np.nanmedian( np.abs(o_rat-med[:,:,np.newaxis]),axis=2)*1.5
   # flag bad binned data more than 1.75 sigma
   flag=np.abs(o_rat-med[:,:,np.newaxis])/sig[:,:,np.newaxis] > 1.75

   # flag bad pixel bins
   o_rat2=o_rat.copy()
   o_rat2[flag]=np.nan
   m_rat=np.nanmean(o_rat2,axis=2)

   # calculate error bars in each bin with sqrt(npix in a bin)
   w_rat=np.nanstd(o_rat2,axis=2)/np.sqrt(np.sum(1.-flag,axis=2))
   mdl=np.ones_like(rat)

   for o in range(nord): # loop over the orders to fit the blaze
      x=x1.copy()

      # flag out the bad pixels in the x-grid
      x[flag[o,:,:]]=np.nan

      # take the mean per bin
      x=np.nanmean(x,axis=1)

      # only pick bins with valid points
      idx=(np.isfinite(x)*np.isfinite(m_rat[o,:]))

      # if you have at least order+2 points in the binned space, fit a polynomial to it
      if np.sum(idx) > (polyorder+2):
         par=np.polyfit(x[idx],m_rat[o,idx],polyorder,w=1./w_rat[o,idx])
         mdl[o,:]=np.polyval(par,x0)

         # return the corrected spectrum with the residual blaze function removed
   return spec/mdl


def stack_spectra(fl,ref_wave,ref_spec):
   '''stack_spectra - makes nightly stacks of spectra
   reads in the data, does the blaze correction, gets interpolation for the same wlen grid
   fl - list of files of spectra
   ref_wave - reference wavlength
   ref_spec - reference spectrum ( a stack of one night of spectra used for blaze correction)
   '''

   fin = fl[0]
   print(f'Reading in spectum {fin}...')
   hdu=fits.open(fl[0])
   data=hdu[0].data
   header=hdu[0].header
   hdu.close()
   print('done')
   
   brv=header['ESO DRS BERV']
   mjd0=header['MJD-OBS']
   wave=get_wave(data,header)*(1.+brv/3e5)
   spec=ref_spec.copy()
   tspec=np.zeros_like(ref_spec)

   tspec=blaze_cor(data,ref_spec)
   sp=np.zeros_like(ref_spec)

   # number of orders in the spectrum 
   no=ref_wave.shape[0]

   # put on a common wavelength grid
   for o in range(no):
      speci=interpolate.interp1d(wave[o,:],tspec[o,:],bounds_error=False,fill_value=0.)
      sp[o,:]=speci(ref_wave[o,:])

   # prepare output variables
   obsdate=[]
   baryvel=[]
   spectra=[]
   nspec=[]
   norm=[]
   j=0.
   k=0.
   date=0.
   tnorm=0.
   norm_compl=0.
   bad=0.
   SN=[]

   # the blue orders are from 0 to 11 (around the CaII line originally...)
   # put in the header for the first frame
   
   for t in np.arange(0,11):
      SN.append(header['HIERARCH ESO DRS SPE EXT SN%i' % t])
   if 1./np.mean(np.asarray(SN)) < 0.04: # select good signal to noise 
      tnorm=np.mean(np.asarray(SN))**2
      norm_compl=np.mean(np.asarray(SN))**2
      spec=sp.copy()*np.mean(np.asarray(SN))**2
      spec_compl=sp.copy()*np.mean(np.asarray(SN))**2
      date=mjd0
      brvel=brv
   else: # otherwise fill in as a blank
      spec=np.zeros_like(sp)
      spec_compl=np.zeros_like(sp)
      norm_compl=0.
      tnorm=0.
      bad=1.
      brvel=0.
   j=j+1
   night=0

   print(f'Number of spectra to process: {np.size(fl[1:])}')
   for counter,fn in enumerate(fl[1:]): # and now loop over the remaining frames
      hdu=fits.open(fn)
      if(counter%50) == 0:
         print(f'{counter}...',end='',flush=True)
      data=hdu[0].data
      header=hdu[0].header
      hdu.close()
      brv=header['ESO DRS BERV']
      mjd=header['MJD-OBS']
      wave=get_wave(data,header)*(1.+brv/3e5)
      tspec=blaze_cor(data,ref_spec)
      sp=ref_spec.copy()*0.
      for o in range(no):
         speci=interpolate.interp1d(wave[o,:],tspec[o,:],bounds_error=False,fill_value=0.)
         sp[o,:]=speci(ref_wave[o,:])

      # is it a new night? if yes, append the previous stack to the variables and reset
      if (mjd-mjd0) >0.4: # becasue dates are sorted in increasing time, here increase the night number
         night=night+1
         if (j-bad) >0:
            norm.append(tnorm)
            nspec.append(j-bad)
            obsdate.append(date/(j-bad))
            baryvel.append(brvel/(j-bad))
#            print('S2d: %7.1f' % (date/(j-bad)))
            spectra.append(spec/tnorm)
         mjd0=mjd
         j=0.
         bad=0.
         tnorm=0.
         brvel=0.

      SN=[]
      for t in np.arange(0,11):
         SN.append(header['HIERARCH ESO DRS SPE EXT SN%i' % t])

      if 1./np.mean(np.asarray(SN)) >= 0.04:
 #        print('bad...  %i %i' % (len(norm),j))
         bad=bad+1
         if j==0:
            spec=sp.copy()*0.
            date=mjd*0.
            brvel=brv*0.
      else:
         if j==0:
            spec=sp.copy()*np.mean(np.asarray(SN))**2
            date=mjd
            brvel=brv
         else:
            brvel=brvel+brv
            date=date+mjd
            spec=spec+sp*np.mean(np.asarray(SN))**2
         tnorm=tnorm+np.mean(np.asarray(SN))**2
         spec_compl=spec_compl+sp.copy()*np.mean(np.asarray(SN))**2
         norm_compl=norm_compl+np.mean(np.asarray(SN))**2
      j=j+1.
      k=k+1.

   # for the one remaining last night, push this night into output variables
   if (j-bad) >0:
      obsdate.append(date/(j-bad))
      baryvel.append(brvel/(j-bad)) 
      spectra.append(spec/tnorm)
      nspec.append(j-bad)
      norm.append(tnorm)

   return np.asarray(obsdate),np.asarray(spectra),np.asarray(baryvel),np.asarray(nspec),np.asarray(spec_compl),norm_compl,norm




# read in all spectra

fl=glob.glob('e2ds/20*/*.fits')
fl=np.asarray(fl)
nspec=np.size(fl)
mjd=[]
print(f'Collecting the JDs from {nspec} spectra...')
for i, fn in enumerate(fl):
   hdu=fits.open(fn)
   date=hdu[0].header['MJD-OBS']
   # occasionally the wrong star was observed (HD*640*) and these need to be rejected
   if fnmatch.fnmatch(hdu[0].header['OBJECT'],'*640*'):
      print ('Bad frame: '+fn)
      # flag bad spectra with large MJD for future rejection
      date=90000.
   mjd.append(date)
   hdu.close()
   if (i%50)==0:
      print(f'{i}...',end='',flush=True)
print('done.')

# sort all valid spectra by MJD

mjd=np.asarray(mjd)
# reject bad frames with high MJD
idx=(mjd < 62000.)
mjd=mjd[idx]
fl=fl[idx]
s1=np.argsort(mjd)
fl=fl[s1]
mjd=mjd[s1]

# pick one night with multiple spectra to act as a blaze and wavelength reference
# this is one of the largest number of spectra taken in one night

ref_night = 58244.0
idx=np.round(mjd) == ref_night

print(f'Using MJD {ref_night} as reference spectra because of multiple spectra')

fl2=fl[idx].copy()

# number of fl2 is number of spectra TODO
print(f'Number of spectra in ref night {np.size(fl2)}')
hdu=fits.open(fl2[0])
data=hdu[0].data
header=hdu[0].header
brv=header['ESO DRS BERV']
ref_wave=get_wave(data,header)
ref_spec=data*0.
hdu.close()

# reading and stacking spectra from one good night ref_night and stacking them together
for fn in fl2:
   hdu=fits.open(fn)
   data=hdu[0].data
   header=hdu[0].header
   wave=get_wave(data,header)*(1.+brv/3e5)
   ref_spec=ref_spec+data
   hdu.close()

# divide by the number of spectra to get the mean
ref_spec=ref_spec/float(fl2.shape[0])

# something big

obsdate,spec,baryvel,nspec,spec_compl,norm_compl,norm=stack_spectra(fl,ref_wave,ref_spec)

# finished!
hdu=fits.PrimaryHDU(ref_wave)
hdu.writeto('2d_wavelength.fits',overwrite=True)
hdu=fits.PrimaryHDU(norm)
hdu.writeto('2d_weights.fits',overwrite=True)
hdu=fits.PrimaryHDU(nspec)
hdu.writeto('2d_nspec.fits',overwrite=True)
hdu=fits.PrimaryHDU(spec)
hdu.writeto('2d_spec.fits',overwrite=True)
hdu=fits.PrimaryHDU(spec_compl/norm_compl)
hdu.writeto('2d_spec_compl.fits',overwrite=True)
hdu=fits.PrimaryHDU(obsdate)
hdu.writeto('2d_obsdate.fits',overwrite=True)
hdu=fits.PrimaryHDU(baryvel)
hdu.writeto('2d_baryvel.fits',overwrite=True)

hdu=fits.PrimaryHDU(ref_spec)
hdu.writeto('2d_refspec.fits',overwrite=True)
print('done')
