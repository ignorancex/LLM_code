"""
Created on Tue Jul 26 15:54:53 2022

@author: demooij
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl
import paths

#from pylab import rcParams
from scipy import interpolate
from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip
from astropy.stats.sigma_clipping import sigma_clipped_stats
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial as P
from scipy.optimize import curve_fit
import scipy.stats as stats
import time
import gc



## this is to convolve an arbitrary spectrum in (wave, spec) with velocity broadening effects. 
## dv is in km/s (which here we're setting to c/R)
def wavegrid(lmin,lmax,R):
    '''lmin is the minimum wavelength, lmax is the maximum wavelength you want, R is the spectral resolution
    returns a wavelength vector that is constant in velocity space.
    then interpolate to this wavelength grid'''
    dx=np.log(1.+1./R) ## the log10 gives you constant velocity spacing
    x=np.arange(np.log(lmin),np.log(lmax),dx)
    return np.exp(x)


## Read in the model from a file and convert the crossesction to opacity for a given columndensity.
def get_mdl(fn='CN/CN_0300K.npy',N=1e15,FWHM_kernel=3e5/1.1e5):
    
    ## set sampling for convolution to 0.25 km/s
    res_interp=12e5
    
    c = np.load(paths.data/fn)
    w = np.load(paths.data/'CN/CN_wavelengths.npy') #np.arange(3850, 3885, 0.0001)
    tau = N * c
    prof = np.exp(-tau)
    
      
    ## Sample the model at a constant resolution for convolution
    w_mdl=wavegrid(np.min(w)-1., np.max(w)+1.,res_interp)
    prof_i=interp1d(w,prof,kind='linear',bounds_error=False,fill_value=1.)
    mdl=prof_i(w_mdl)

    ## generate a Gaussian kernel with the FWHM at the instrumental resolution & convolve model
    
    x_kernel=np.arange(-90.,90.001,(3e5/res_interp))
    kernel=np.exp( - ((x_kernel)**2)/2./(FWHM_kernel/2.35)**2)
    kernel=kernel/np.sum(kernel)
    mdl_conv=np.convolve(mdl,kernel,mode='same')
    idx=(w_mdl >= np.min(w))*(w_mdl <= np.max(w))

    ## set up interpolation object for the model to use in the CCF code
    imdl_conv=interp1d(w_mdl[idx],mdl_conv[idx],kind='linear',bounds_error=False,fill_value=1.)

    return w,prof,imdl_conv



##Read in data
def read_data(order=[0,1,2,3,4,5,6,7]):
    vsys=20.    #Systemic velocity of beta Pic estimated from data
    nspec = fits.getdata(paths.data/'2d_nspec.fits')
    spec = fits.getdata(paths.data/'2d_spec.fits')
    wlen = fits.getdata(paths.data/'2d_wavelength.fits')/(1.+vsys/3e5)
    bary = fits.getdata(paths.data/'2d_baryvel.fits')
    obsdate = fits.getdata(paths.data/'2d_obsdate.fits')
    return nspec,spec[:,order,:],wlen[order,:],bary,obsdate


## Estimate residual blaze
def remove_resid_blaze(pixels,spec,star_spec,ord=3,poly_order=6):
    # order start and order end (the pixel point where there is flux in the order)
    # for the first eight orders
    o_start=np.array([ 225,  15,  15,  15,  15,  15,  15,  15])
    o_end  =np.array([3670,4050,4060,4060,4060,4060,4060,4060])
    
    # take the ratio of the spectrum to the mean spectrum...
    spec_cor=spec[:,ord,:]/star_spec[np.newaxis,ord,:]
    
    binsize=150
    idx=(pixels>o_start[ord])*(pixels<o_end[ord])
    nbins=pixels.shape[0] // binsize



    
    # fit a polynomial to the this ration (should be flat if there was no small residuals)
    # and then divide it out to make this order relatively flat
    for i in range(spec.shape[0]):
        x0=pixels.copy()
        bin_spec=spec_cor[i,:nbins*binsize].reshape(nbins,binsize).copy()
        bin_x=x0[:nbins*binsize].reshape(nbins,binsize).astype(float)

        #Quick clipping of outliers in each bin using the MAD
        med_spec=np.median(bin_spec,axis=1)
        mad_spec=np.median( np.abs(bin_spec-med_spec[:,np.newaxis]), axis=1)
        idx= ( np.abs(bin_spec-med_spec[:,np.newaxis])/mad_spec[:,np.newaxis] > 4.5)*(bin_x>o_start[ord])*(bin_x<o_end[ord])*(bin_spec>0)*np.isfinite(bin_spec)
        if np.sum(idx)>0:
            bin_spec[idx]=np.nan
            bin_x[idx]=np.nan
        
        bin_spec=np.nanmean(bin_spec,axis=1)
        bin_x=np.nanmean(bin_x,axis=1)
        idx=np.isfinite(bin_spec)
       
        l=np.polyfit(bin_x[idx],bin_spec[idx],poly_order)
        spec_cor[i,:]=spec[i,ord,:]/np.polyval(l,pixels)
    return spec_cor



def remove_stellar_envelope(spec,ord):
    spec0 = spec[:,ord,:].copy()

    ##Estimate the upper envelope. In this case use the [5,15] brightest pixels. Note: This is not robust against outliers...
    spec_star_only0=np.zeros(spec0.shape[1])
    for i in range(spec0.shape[1]):
        tmp_array=spec0[:,i].copy()
        tmp_array.sort()
        spec_star_only0[i]=np.median(tmp_array)#[-25:-15])

        
    #Now fit the overall envelope but leaving narrow features hopefully intact
    binsize=21
    nbins=spec_star_only0.shape[0] // binsize
    x0=np.arange(spec_star_only0.shape[0]).astype(float)
    bin_spec=spec_star_only0[:nbins*binsize].reshape(nbins,binsize).copy()
    bin_x=x0[:nbins*binsize].reshape(nbins,binsize).copy()

    #Quick clipping of outliers in each bin using the MAD
    med_spec=np.median(bin_spec,axis=1)
    mad_spec=np.median( np.abs(bin_spec-med_spec[:,np.newaxis]), axis=1)
    idx= ( np.abs(bin_spec-med_spec[:,np.newaxis])/mad_spec[:,np.newaxis] > 4.5)
    if np.sum(idx)>0:
       bin_spec[idx]=np.nan
       bin_x[idx]=np.nan
       
    bin_spec=np.nanmean(bin_spec,axis=1)
    bin_x=np.nanmean(bin_x,axis=1)
    idx=(np.isfinite(bin_spec))
    if np.sum(idx)>5:
        bin_spec=bin_spec[idx]
        bin_x=bin_x[idx]
        ibin_spec=interp1d(bin_x,bin_spec,kind='cubic',fill_value=(bin_spec[0],bin_spec[-1]),bounds_error=False)
        spec_star_only=ibin_spec(x0)
    else:
        spec_star_only=np.ones_like(spec_star_only0)
    spec_final_norm = spec0/spec_star_only[np.newaxis,:]

    
    return spec_final_norm



####################################################################################
###### Only used for finding comets ################################################
####################################################################################




## Find the comets in the Ca II H line
def find_comets(wlen,spec_m,ord_FEB=7,use_all=True):
    #Select the order with Ca II H line
    wlen_FEB = wlen[ord_FEB]
    spec_FEB = np.transpose(spec_m[:,ord_FEB,:])
    
    #remove outliers to get a good sigma-clipped spectrum for stationary features
    filtered_data = sigma_clip(spec_FEB, sigma_lower=1.0, sigma_upper=5, maxiters=5, axis=1, copy=True, masked=True)
    spec_Ca_only = np.ma.mean(filtered_data, axis=1)
    spec_FEB_norm = spec_FEB/spec_Ca_only[:,np.newaxis]


    #write Ca spectrum prior to removal of static features
    hdu=fits.PrimaryHDU(spec_FEB)
    hdu.writeto(paths.data/"FEB_spec_orig.fits",overwrite=True)
    
    #write spectrum used for FEB selection
    hdu=fits.PrimaryHDU(spec_FEB_norm.filled(np.nan))
    hdu.writeto(paths.data/"FEB_spec.fits",overwrite=True)
    
    # rest wavelength for CaII H line in the stellar rest frame
    lam0=3968.469
    # converting the FEB features into a relative velocity for the comet
    v_comet=3e5*(wlen_FEB-lam0)/lam0
    #Wavelengths assiociated with spectrum for FEB selection
    hdu=fits.PrimaryHDU(v_comet)
    hdu.writeto(paths.data/"FEB_vgrid.fits",overwrite=True)
    
    # look between -200 to -8km/s as well as 8km/s to 200km/s, find the minima (=maximum absorption)
    # range chosen to avoid most of the core. 
    # use_all will look for blueshifted comets, otherwise only redshifted comets are used.
    if use_all:
        m_p = (np.abs(v_comet)>8) * (np.abs(v_comet)<200.)
    else:
        m_p = (v_comet>10.)*(v_comet<200.)
    wlen_peak = wlen_FEB[m_p]
    spec_peak = 1. - (spec_FEB_norm[m_p])
    
    wheremax = np.argmax(spec_peak, axis=0)
    ampl_FEB_clip = np.max(spec_peak, axis=0)
    wlen_FEB_clip = wlen_peak[wheremax]

    # we now have the heights of the peaks in what_max, and the positions of the peaks in wlen_max
    # we make two separate sets to make: FEBs and no FEBs
    # 0.6 means 60% strength for pickup as an FEB
    # 0.15 means less than 15% strength for no FEB
    FEB_sel =   (ampl_FEB_clip > 0.6)
    star_sel =  (ampl_FEB_clip < 0.15)
    print("Number of strong comets: {0:}".format(np.sum(FEB_sel)))
    print("Number of weak comets: {0:}".format(np.sum(star_sel)))

    hdu=fits.PrimaryHDU(FEB_sel.filled(False).astype(int))
    hdu.writeto(paths.data/"FEB_idx.fits",overwrite=True)
    hdu=fits.PrimaryHDU(star_sel.filled(False).astype(int))
    hdu.writeto(paths.data/"NO_FEB_idx.fits",overwrite=True)


    v_FEB_clip=3e5*(wlen_FEB_clip-lam0)/lam0

    hdu=fits.PrimaryHDU(ampl_FEB_clip.filled(np.nan))
    hdu.writeto(paths.data/"FEB_depths.fits",overwrite=True)
    hdu=fits.PrimaryHDU(v_FEB_clip)
    hdu.writeto(paths.data/"FEB_velocities.fits",overwrite=True)

    
    return FEB_sel,star_sel,wlen_FEB_clip




#Align the spectra in the comet restframe for stacking. Note that for Ca, the line centres are set to NaN.
def align_spectra(wave,spec0,v_FEB):
    spec=spec0.copy()
    FEB_spec=spec.copy()
    for i in range(spec.shape[0]):
        z=1.+v_FEB[i]/3e5
        for o in range(spec.shape[1]):
            speci=interp1d(wave[o,:],spec[i,o,:],kind='linear',bounds_error=False,fill_value=np.nan)
            FEB_spec[i,o,:]=speci(wave[o,:]*z)
            idx=(np.abs( 3e5*(wave[o,:]*z-3933.663)/3933.663) < 10)+(np.abs( 3e5*(wave[o,:]*z-3968.469)/3968.469) < 10)
            if np.sum(idx)>0:
                FEB_spec[i,o,idx]=np.nan
    return FEB_spec
        

####################################################################################
####################################################################################
####################################################################################


##Set up a simple CCF code. N.B. This is the unweighted CCF
def c_correlate(l_data,data,imodel,dv=np.linspace(-50,50,101)):
    npix=dv.shape[0]
    ccf=np.zeros((data.shape[0],npix))
    ccf0=np.zeros((data.shape[0],npix))
    dat=data-np.mean(data,axis=1)[:,np.newaxis]

    
    z0=1.-dv/3e5
    for i,z in enumerate(z0):
        mdl=imodel(l_data*z)
        mdl=mdl-np.mean(mdl)
        ccf[:,i]=np.sum( data*mdl[np.newaxis,:],axis=1)
        ccf0[:,i]=ccf[:,i].copy()


    return ccf,ccf0




## Inject the model into the data 
def inject_mdl(l_data,data,imodel,dv=np.zeros(51)):
    sim_dat=data.copy()
    z0=1.-dv/3e5
    for i in range(data.shape[0]):
        mdl=imodel(l_data*z0[i])
        sim_dat[i,:]=data[i,:]*mdl
    return sim_dat



def generate_CCFs_orders(wlens,data_for_ccf,spec,pixels,broadening=5.,binsize=80,T_gas=10,N=1e12,v_comet=0.,idx_star=0,idx_pre=0,idx_post=0):
    
    # order start and order end (the pixel point where there is flux in the order)
    # for the first eight orders. 200 is added/subtracted to avoid edge of blaze correction
    o_start=np.array([ 225,  15,  15,  15,  15,  15,  15,  15])+200
    o_end  =np.array([3670,4050,4060,4060,4060,4060,4060,4060])-200

    
    ## define arrays to store ccfs 
    v_ccf=np.arange(-250.,250.,1.)
    m_ccf=np.zeros((data_for_ccf.shape[0],wlens.shape[0],v_ccf.shape[0]))
    m_sim_ccf=np.zeros((data_for_ccf.shape[0],wlens.shape[0],v_ccf.shape[0]))

    indexing=np.arange(wlens.shape[1])
    # sim_dat is an array to inject the model into
    sim_dat=np.zeros((data_for_ccf.shape))
    # star_spec is the specra with NO comets used to divide out the stellar spectrum
    star_spec_pre=np.mean(spec[idx_star*idx_pre,:,:],0)
    star_spec_post=np.mean(spec[idx_star*idx_post,:,:],0)

    for ord in range(wlens.shape[0]): # loop over the orders
        
        ## get the opacities and convert to line-profile
        w,prof,imdl_conv=get_mdl('CN/CN_{0:04d}K.npy'.format(T_gas),N=N,FWHM_kernel=broadening)
        w_sim,prof_sim,imdl_conv_sim=get_mdl('CN/CN_{0:04d}K.npy'.format(T_gas),N=N,FWHM_kernel=broadening)

        ## Calculate CCF for data
        ## Filter out the strongest circumstellar Fe I lines, that could contaminate the signal.
        idx=(indexing >= o_start[ord])*(indexing<o_end[ord])*(np.abs(wlens[ord,:]-3860.18)>0.35)*(np.abs(wlens[ord,:]-3856.63)>0.15)*(np.abs(wlens[ord,:]-3865.78)>0.1)*(np.abs(wlens[ord,:]-3878.83)>0.1)*(np.abs(wlens[ord,:]-3886.54)>0.1)
        

        ccf,ccf0=c_correlate(wlens[ord,idx],data_for_ccf[:,ord,idx],imdl_conv,dv=v_ccf)
        m_ccf[:,ord,:]=ccf.copy()

        
        ## Inject model into data 
        sim_dat[:,ord,:]=inject_mdl(wlens[ord,:],spec[:,ord,:],imdl_conv_sim,dv=v_comet)

        star_spec_pre=np.mean(sim_dat[idx_star*idx_pre,:,:],0)
        star_spec_post=np.mean(sim_dat[idx_star*idx_post,:,:],0)

        sim_dat[idx_pre ,ord,:]=remove_resid_blaze(pixels,sim_dat[idx_pre ,:,:],star_spec_pre ,ord,poly_order=8)
        sim_dat[idx_post,ord,:]=remove_resid_blaze(pixels,sim_dat[idx_post,:,:],star_spec_post,ord,poly_order=8)
        sim_dat[idx_pre ,ord,:]=remove_stellar_envelope(sim_dat[idx_pre ,:,:],ord)
        sim_dat[idx_post,ord,:]=remove_stellar_envelope(sim_dat[idx_post,:,:],ord)
    
        # cross correlate simulated spectra
        sim_ccf,sim_ccf0=c_correlate(wlens[ord,idx],sim_dat[:,ord,idx],imdl_conv,dv=v_ccf)
        m_sim_ccf[:,ord,:]=sim_ccf.copy()

    return v_ccf,m_ccf,m_sim_ccf






def gauss(x,A,mu,sig,offset):
    return(offset+A*np.exp( - (x-mu)**2/2./sig**2))

def gen_gauss(v_c):
    def n_gauss(x,A,sig,offset,slope,quadratic):
        return(offset+slope*(x-v_c)+quadratic*(x-v_c)**2+A*np.exp( - (x-v_c)**2/2./sig**2))
    return n_gauss


def estimate_significance(v_ccf,ccf,v_c=0.,dv=15.,sim=False):
    # 40 km/s from the line central position is the outside limit in order to avoid aliases in the CCF
    v_outer = 30.

    idx=(np.abs(v_ccf-v_c)<v_outer)

    fit=np.zeros((ccf.shape[0]+1,5))
    err=np.zeros((ccf.shape[0]+1,5))
    sig=np.std(ccf,axis=0)
      
    ccf_cor=ccf.copy()

    for i in range(ccf.shape[0]): # fit linear function in background region
        try:
            
            #set bounds for the fit to avoid over wide gaussians etc. :
            #  amplitude:  -max range to +max range
            #  sigma such that FWHM is less than half the window width
            #  Offset:  -1 ... 1
            #  slope:   -1 ... 1 [1/(km/s)]
            ccf_range=np.max(ccf)-np.min(ccf)
            bounds=[[-ccf_range*1.1,1.,-100,-100,-100],[ccf_range*1.1,1.5*v_outer/2.35,100,100,100]]

            #generate the function to force the Gaussian to be at v_c, while the other parameters are free
            func=gen_gauss(v_c)

            #Initial fit to the function to try and determine uncertainties
            popt,pcov=curve_fit(func,v_ccf[idx],ccf[i,idx],p0=[0.,dv/2.35,0.,0.,0.],sigma=sig[idx],bounds=bounds)
            
            ccf_cor[i,:]=ccf[i,:]-(popt[2]+popt[3]*(v_ccf-v_c)+popt[4]*(v_ccf-v_c)**2) 
            perr=np.sqrt(np.diag(pcov))                     
            fit[i,:]=np.array(popt)
            err[i,:]=np.array(perr)

        except:
            print("no convergence")
            plt.figure()
            plt.plot(v_ccf[idx],ccf[i,idx])
            plt.show()

    m_ccf=np.mean(ccf_cor,axis=0)
    s_ccf=np.std(ccf_cor,axis=0)/np.sqrt(ccf.shape[0])
    plt.close('all')
    # refit again using sig as the error on the weighted mean
    
    #set bounds for the fit to avoid over wide gaussians etc. :
    #  amplitude:  -max range to +max range
    #  sigma such that FWHM is less than half the window width
    #  Offset:  -1 ... 1
    #  slope:   -1 ... 1 [1/(km/s)]
    try:
        ccf_range=np.max(m_ccf)-np.min(m_ccf)
        bounds=[[-ccf_range*1.1,1.,-100,-100,-100.],[ccf_range*1.1,1.5*v_outer/2.35,100,100,100.]]
        popt,pcov=curve_fit(gen_gauss(v_c),v_ccf[idx],m_ccf[idx],p0=[0.,dv/2.35,0.,0.,0.],sigma=s_ccf[idx],bounds=bounds)
        perr=np.sqrt(np.diag(pcov))              
        fit[-1,:]=np.array(popt)
        err[-1,:]=np.array(perr)

        print(popt[0]/perr[0],popt[1]/perr[1])
    except:
        print("comb, no convergence")
    return fit,err



def fold_comet(v_ccf,ccf,sim_ccf,binsize,idx_comets,v_comets):

    ccf_comets=ccf
    sim_ccf_comets=sim_ccf

    # velocity range over which we try to line up comets
    v_phasefold=np.arange(-90,90.1)
    
    # output arrays
    phasefold=np.zeros((ccf.shape[0],ccf.shape[1],v_phasefold.shape[0]))
    sim_phasefold=np.zeros((ccf.shape[0],ccf.shape[1],v_phasefold.shape[0]))

    for ord in range(3,4): # only use the order with the dominant CN lines (can be modified if weaker lines need to be considered)
        for i in range(phasefold.shape[0]): # for each night...
            ccf_i=interp1d(v_ccf,ccf[i,ord,:],kind='linear',bounds_error=False,fill_value=np.nan)
            sim_ccf_i=interp1d(v_ccf,sim_ccf[i,ord,:],kind='linear',bounds_error=False,fill_value=np.nan)
            phasefold[i,ord,:]=ccf_i(v_phasefold+v_comets[i])
            sim_phasefold[i,ord,:]=sim_ccf_i(v_phasefold+v_comets[i])

        if ord == 3: # the CN spectral order
            fit_sim,err_sim=estimate_significance(v_phasefold,sim_phasefold[idx_comets,ord,:],v_c=0.,sim=True)
            fit,err=estimate_significance(v_phasefold,phasefold[idx_comets,ord,:],v_c=0.)

    return v_phasefold,phasefold,sim_phasefold,fit,err,fit_sim,err_sim




## run ccf for a range of temperatures
def run_ccf_ord_multi_temp(wlens,data_for_ccf,spec,pixels,f,binsize=81,v_comet=0.,idx_comet=0,idx_star=0,idx_pre=0,idx_post=0,restframe='stellar'):
    ## Define a range of column densities to use (mainly injection)
    N0=np.hstack([np.arange(1,10)*1e11,np.arange(1,10)*1e12,np.arange(1,10)*1e13,np.arange(1,10)*1e14,np.arange(1,10)*1e15])
    N0=10.**(np.linspace(11.+np.log10(2),15.+np.log10(2),50))
    print(N0)
    T_gas0=np.asarray([10, 15, 20, 30,40, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000])

    # tried different line broadenings for the CN model (FWHM in km/s) [from instrumental resolution up to 20km/s
    broadening0=np.array([3e5/1.1e5,5.,10.,15.,20.])
    
    # short run version for quick examination
    #N0=np.hstack([np.arange(1,4)*1e13])
    #N0=np.asarray([1e12,1e13,1e14,1e15])
    #T_gas0=np.asarray([30,300,3000])
    #broadening0=np.array([3e5/1.1e5,5.])

    # how many total iterations to print progress
    ntot = np.size(N0)*np.size(T_gas0)*np.size(broadening0)
    count = 1
    for k,broadening in enumerate(broadening0): # line broadening...
       for i,T_gas in enumerate(T_gas0): # CN gas temperature...
           for j,N in enumerate(N0): # column density...
               print('Running model {0} of {1} models for N0={2:.2e}, T_gas={3:4d}K and v_broad={4:.1f}km/s...'.format(count,ntot,N,T_gas,broadening))
               v_ccf,ccf,sim_ccf=generate_CCFs_orders(wlens,data_for_ccf,spec,pixels,broadening=broadening,binsize=binsize,T_gas=T_gas,N=N,v_comet=v_comet,idx_star=idx_star,idx_pre=idx_pre,idx_post=idx_post)
       
               

               
               # align everything up in the comet rest frame and estimate the significance of any (non)detection.
               v_phase,phase,sim_phase,fit,err,fit_sim,err_sim=fold_comet(v_ccf,ccf,sim_ccf,binsize,idx_comet,v_comet)
               
            
               f.write("{0:.0f}  {1:.1e} {2:.1f}  {3:.2f}  {4:.2f}\n".format(T_gas, N, broadening, fit[-1,0]/err[-1,0],fit_sim[-1,0]/err_sim[-1,0]) )
       
               if count==1:
                   fit_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],fit.shape[0],fit.shape[1]))
                   fit_err_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],fit.shape[0],err.shape[1]))
                   sim_fit_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],fit.shape[0],fit.shape[1]))
                   sim_fit_err_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],fit.shape[0],err.shape[1]))
                   ccf_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],ccf.shape[0],ccf.shape[2]))
                   sim_ccf_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],ccf.shape[0],ccf.shape[2]))
                   phase_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],phase.shape[0],phase.shape[2]))
                   sim_phase_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],phase.shape[0],phase.shape[2]))
               time.sleep(0.5)
            

               
               fit_cube[i,j,k,:,:]=fit.copy()
               fit_err_cube[i,j,k,:,:]=err.copy()
               sim_fit_cube[i,j,k,:,:]=fit_sim.copy()
               sim_fit_err_cube[i,j,k,:,:]=err_sim.copy()
               ccf_cube[i,j,k,:,:]=ccf[:,3,:].copy()
               sim_ccf_cube[i,j,k,:,:]=sim_ccf[:,3,:].copy()
               phase_cube[i,j,k,:,:]=phase[:,3,:].copy()
               sim_phase_cube[i,j,k,:,:]=sim_phase[:,3,:].copy()
               count = count + 1

    hdu=fits.PrimaryHDU(fit_cube)
    hdu.writeto(paths.data/"fit_cube_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(fit_err_cube)
    hdu.writeto(paths.data/"fit_err_cube_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(sim_fit_cube)
    hdu.writeto(paths.data/"sim_fit_cube_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(sim_fit_err_cube)
    hdu.writeto(paths.data/"sim_fit_err_cube_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(ccf_cube)
    hdu.writeto(paths.data/"ccf_cube_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(sim_ccf_cube)
    hdu.writeto(paths.data/"sim_ccf_cube_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(v_phase)
    hdu.writeto(paths.data/"v_phasefold.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(v_ccf)
    hdu.writeto(paths.data/"v_ccf_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(phase_cube)
    hdu.writeto(paths.data/"phase_cube_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(sim_phase_cube)
    hdu.writeto(paths.data/"sim_phase_cube_{0:}.fits".format(restframe),overwrite=True)
    hdu=fits.PrimaryHDU(T_gas0)
    hdu.writeto(paths.data/"T_gas.fits",overwrite=True)
    hdu=fits.PrimaryHDU(N0)
    hdu.writeto(paths.data/"Nf.fits",overwrite=True)
    hdu=fits.PrimaryHDU(broadening0)
    hdu.writeto(paths.data/"broadening.fits",overwrite=True)




if __name__ == "__main__":
   '''
   MAIN CODE FOR RUNNING FITS
   MAIN CODE FOR RUNNING FITS
   MAIN CODE FOR RUNNING FITS
   MAIN CODE FOR RUNNING FITS
   MAIN CODE FOR RUNNING FITS
   MAIN CODE FOR RUNNING FITS
   '''
   

#   mpl.use('agg')##macosx')
   #mpl.use('macosx')
   mpl.rcParams['figure.figsize'] = 12, 8
   mpl.rc('image', interpolation='nearest', origin='lower', cmap = 'gray')


   
   # read it all in
   nspec,spec,wlen,bary,obsdate=read_data()

   
   CN_ord=3
   
   ##HARPS Fiber change in June 2015 (JD~2457180) attempt to improve correction by treating pre- and post-fiberchange spectra separately. 
   idx_pre=obsdate<57200.
   idx_post=obsdate>=57200.

   
   # two binsizes - one for small scale and one for larger scales... determined empirically
   binsize=81 # in pixels
   
   nskip=25 # how many pixels to skip at the start of an echelle order to avoid edge effects
   
   
   ## Find the strongest comet in the Ca II H line for each of the spectra, before doing so,
   ## remove the overall line-shape by taking the envelope for each point using the top 10 pixels,
   ## but prior to that, do an extra pass at the residual blaze correction but here ignoring the
   ## fiber change (note: this is only used to find exocomets, so should not be an issue)
   ord_FEB=7
   cor=spec.copy()
   pixels=np.arange(wlen.shape[1])
   star_spec=np.median(spec,axis=0)

   for ord in range(8):
       cor[:,ord,:]=remove_resid_blaze(pixels,spec,star_spec,ord,poly_order=8)

   # we split the spectra into 'comets' and 'no comets'
   FEB_idx,star_idx,FEB_wave=find_comets(wlen,cor,ord_FEB=ord_FEB,use_all=True)
   
   # rest wavelength for CaII H line in the stellar rest frame
   lam0=3968.469
   
   # converting the FEB features into a relative velocity for the comet
   v_comet=3e5*(FEB_wave-lam0)/lam0
    
   
   # only continue with orders 0 to 3 
   # (cut down on time and memory)
   wlens=wlen[0:8,:]
   spec_cor=spec[:,0:8,:].copy()
   spec_cor2=spec[:,0:8,:].copy()    
   
   # residual blaze corrections, separate pre- and post-fiber change
   pixels=np.arange(wlens.shape[1])
   star_spec=np.median(spec[star_idx,:,:],axis=0)
   star_spec_pre=np.median(spec[star_idx*idx_pre,:,:],axis=0)
   star_spec_post=np.median(spec[star_idx*idx_post,:,:],axis=0)
   print('N_star:      ',np.sum(star_idx))
   print('N_star,pre:  ',np.sum(star_idx*idx_pre))
   print('N_star,post: ',np.sum(star_idx*idx_post))

   # pre and post fiber change, because the blaze function could have changed, subsequently remove
   for ord in range(8):
       spec_cor[idx_pre,ord,:]=remove_resid_blaze(pixels,spec[idx_pre,:,:],star_spec_pre,ord,poly_order=8)
       spec_cor[idx_post,ord,:]=remove_resid_blaze(pixels,spec[idx_post,:,:],star_spec_post,ord,poly_order=8)
       #spec_cor[:,ord,:]=remove_resid_blaze(pixels,spec,star_spec,ord,poly_order=4)
       spec_cor2[idx_pre,ord,:]=remove_stellar_envelope(spec_cor[idx_pre,:,:],ord)
       spec_cor2[idx_post,ord,:]=remove_stellar_envelope(spec_cor[idx_post,:,:],ord)
     
   data_for_ccf=spec_cor2[:,0:8,:]

   '''
   ##Used to verify interpolation
   v_comett=v_comet.copy()
   v_comet[:]=0.
   v_comet[FEB_idx]=v_comett[FEB_idx].copy()
   '''
   
   #Store spectra in stellar and cometary restframe to be stacked and plotted
   hdu=fits.PrimaryHDU(wlens)
   hdu.writeto(paths.data/"wave_corrected_spectra.fits",overwrite=True)
   hdu=fits.PrimaryHDU(data_for_ccf)
   hdu.writeto(paths.data/"corrected_spectra_stellar.fits",overwrite=True)
   FEB_spec=align_spectra(wlens,data_for_ccf,v_comet)
   hdu=fits.PrimaryHDU(FEB_spec)
   hdu.writeto(paths.data/"corrected_spectra_comet.fits",overwrite=True)



   '''
   # CHECK POINT to see that this looks normal. REMOVE in final version 
   hdu=fits.PrimaryHDU(FEB_spec.swapaxes(1,0))
   hdu.writeto("tmp3.fits",overwrite=True)    
   hdu=fits.PrimaryHDU(data_for_ccf[:,0:4,:].swapaxes(1,0))
   hdu.writeto('tmp.fits',overwrite=True)
   hdu=fits.PrimaryHDU(spec_cor2.swapaxes(1,0))
   hdu.writeto('tmp2.fits',overwrite=True)
   '''

   
   # Do the analysis for the strongest comets selected using find_comets()
   f = open(paths.data/"results_HARPS_comet.txt", "w")
   run_ccf_ord_multi_temp(wlens[0:4,:],data_for_ccf[:,0:4,:],spec[:,0:4,:],pixels,f,binsize=binsize,v_comet=v_comet,idx_comet=FEB_idx,idx_star=star_idx,idx_pre=idx_pre,idx_post=idx_post,restframe='comet')
   f.close()
   
   

   star_idx[:]=True
   FEB_idx[:]=True
   v_comet[:]=0.
   f = open(paths.data/"results_HARPS_star.txt", "w")
   run_ccf_ord_multi_temp(wlens[0:4,:],data_for_ccf[:,0:4,:],spec[:,0:4,:],pixels,f,binsize=binsize,v_comet=v_comet,idx_comet=FEB_idx,idx_star=star_idx,idx_pre=idx_pre,idx_post=idx_post)
   f.close()
   
   
