import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.ticker as tck
from matplotlib import colors

import numpy as np

from scipy.interpolate import interp1d

import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.constants as const

from astropy.cosmology import default_cosmology
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from astropy.io import fits
from astropy import constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.imaging_data import ImageData
from astropy.constants import G, c, M_sun
import astropy.io.fits as pyfits
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Util import kernel_util
import emcee
import corner
from multiprocessing import Pool
import dynesty
import pickle

mpi = False  # MPI possible, but not supported through that notebook.

from lenstronomy.Workflow.fitting_sequence import FittingSequence
import pyswarms as ps
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
import warnings

warnings.filterwarnings('ignore')

import astropy.io.fits as pyfits
from astropy.io import fits

nparal = 32

likemask = np.ones([70,70],dtype=bool)

for i in range(70):
    for j in range(70):
        if (i-32)**2. + (j-34)**2. < 30:
            likemask[i,j] = False
        if (i-34)**2. + (j-34)**2. > 1300:
            likemask[i,j] = False
            
imagedata = np.load('sersic_source_subtracted.npy')*likemask#np.load('mock_image_JVAS2.npy')
source_res = 4
observe_kwargs = {'read_noise': 4.0,
                  'pixel_scale': 0.025, 
                  'ccd_gain': 2.5, 
                  'exposure_time': 6976., 
                  'sky_brightness': 22.3, 
                  'magnitude_zero_point': 25.96, 
                  'num_exposures': 1, 
                  'seeing': 0.07, 
                  'psf_type': 'PIXEL'}
hiresforsource = {'read_noise': 4.0, 
                  'pixel_scale': 0.025/source_res, 
                  'ccd_gain': 2.5, 
                  'exposure_time': 6976., 
                  'sky_brightness': 22.3, 
                  'magnitude_zero_point': 25.96, 
                  'num_exposures': 1, 
                  'seeing': 0.07, 
                  'psf_type': 'PIXEL'}
background_rms = np.load('bckg.npy')
poisson = np.load('pois.npy')
errtot = np.power(np.power(background_rms,2.) + np.power(poisson,2.),0.5)
exp_time = observe_kwargs['exposure_time']
z_interloper = 0.881
z_lens =  0.881
z_source =  2.059
numpix = 70
deltaPix = 0.025
ext = numpix*deltaPix/2.
extent = [-ext,ext,-ext,ext]
fwhm = 0.14
kernel_size = 91
kernel_cut = np.zeros([15,15])
kernel_p = np.zeros([15,15])
kernel_p[7,7] = 1.
for i in range(15):
    for j in range(15):
        sigm = (1./2.355)*(fwhm/deltaPix)
        kernel_cut[i,j] = np.exp(-(0.5/sigm**2.)*((i-7)**2. + (j-7)**2.))
kwargs_psf = {'psf_type': 'PIXEL', 'fwhm': fwhm, 'pixel_size': deltaPix, 'kernel_point_source': kernel_cut}
kwargs_point = {'psf_type': 'PIXEL', 'fwhm': fwhm, 'pixel_size': deltaPix, 'kernel_point_source': kernel_p}
psf_class = PSF(**kwargs_psf)
psf = PSF(**kwargs_p)
kwargs_numerics = {'point_source_supersampling_factor': 1}
#https://arxiv.org/pdf/1402.7073.pdf

def bfunc(z):
    return -0.101 + 0.026*z

def afunc(z):
    return 0.520 + (0.905 - 0.520)*np.exp(-0.617*(z**1.21))

def c200vsM200(M200,z):
    # This is in log10 units. Mass is h^-1 Msun
    return bfunc(z)*(M200-12) + afunc(z)

def nfw_from_M200(M200,z):
    # takes in logM200 in Msun
    # returns nfw params in arcsec
    lens_cosmo = LensCosmo(z_lens=z, z_source=z_source, cosmo=cosmo)
    Mh = M200 + np.log10(cosmo.h)
    concen = c200vsM200(Mh,z)
    
    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=10**Mh, c=10**concen)
    return Rs_angle,alpha_Rs
    
def nfw_fromM200andC200(M200,c200,z):
    # takes in logM200 in Msun and c200
    # returns nfw params in arcsec
    lens_cosmo = LensCosmo(z_lens=z, z_source=z_source, cosmo=cosmo)
    Mh = M200 + np.log10(cosmo.h)
    
    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=10**Mh, c=c200)
    
    return Rs_angle,alpha_Rs
    
def model_shapeflat(data0,main_lens,shear,subhalo,beta_model,source_x,source_y,n_max_model,Mtype,Itype):
    lens_model_list = ['PEMD','SHEAR',Itype]

    kwargs_model_shape = {'lens_model_list': ['PEMD','SHEAR',Itype],  # list of lens models to be used
                          'lens_light_model_list': [],  # list of unlensed light models to be used
                          'source_light_model_list': ['SHAPELETS_POLAR'],  # list of extended source models to be used
                          'z_lens': z_lens, 'z_source': z_source,}
    kwargs_model_source = {'lens_model_list': ['PEMD','SHEAR',Itype],  # list of lens models to be used
                          'lens_light_model_list': [],  # list of unlensed light models to be used
                          'source_light_model_list': ['SHAPELETS_POLAR'],  # list of extended source models to be used
                          'z_lens': z_lens, 'z_source': z_source,}

    simhst_shape = SimAPI(numpix=numpix, kwargs_single_band=observe_kwargs, kwargs_model=kwargs_model_shape)
    simhst_source = SimAPI(numpix=source_res *numpix, kwargs_single_band=hiresforsource, kwargs_model=kwargs_model_source)

    kwargs_data_real = sim_util.data_configure_simple(numpix, deltaPix, exp_time, background_rms)
    data_real = ImageData(**kwargs_data_real)

    data_real.update_data(data0)
    psf = psf_point
        
    theta_E,gamma,clx,cly,el1,el2 = main_lens
    gamma1, gamma2 = shear
    
    ra = 1e-4

    if Mtype == 'SMOOTH':
        centerxproj, centeryproj,z_int = 0.,0.,z_lens
        if Itype == 'SIS' or Itype == 'PJAFFE':
            theta_e = 0.
            rs = np.sqrt(theta_e*2e-3*theta_E)
            sig_0 = (theta_e*2e-3*(rs-ra))/(2.*ra*rs)
        else:
            rs, alrs = 1.,0.
    if Mtype == 'SUB':
        z_int = z_lens
        if Itype == 'SIS':
            theta_e, centerxproj, centeryproj = subhalo
        elif Itype == 'PJAFFE':
            theta_e, centerxproj, centeryproj = subhalo
            rs = np.sqrt(theta_e*2e-3*theta_E)
            sig_0 = (theta_e*2e-3*(rs-ra))/(2.*ra*rs)
        else:
            M, centerxproj, centeryproj = subhalo
            logM = np.log10(M*(10.**10.))
            rs, alrs = nfw_from_M200(logM,z_int)
            
    if Mtype == 'SUBs':
        ## I type has to be 'PEMD'
        z_int = z_lens
        theta_e, centerxproj, centeryproj,slope = subhalo
            
    if Mtype == 'SUBc':
        z_int = z_lens
        M, centerxproj, centeryproj,c200 = subhalo
        logM = np.log10(M*(10.**10.))
        rs, alrs = nfw_fromM200andC200(logM,c200,z_int)
    
    if Mtype =='INT':
        if Itype == 'SIS':
            theta_e, centerxproj, centeryproj, z_int = subhalo
        elif Itype == 'PJAFFE':
            theta_e, centerxproj, centeryproj, z_int = subhalo
            rs = np.sqrt(theta_e*2e-3*theta_E)
            sig_0 = (theta_e*2e-3*(rs-ra))/(2.*ra*rs)
        else:
            M, centerxproj, centeryproj,z_int = subhalo
            logM = np.log10(M*(10.**10.))
            rs, alrs = nfw_from_M200(logM,z_int)
            
    if Mtype =='INTc':
        M, centerxproj, centeryproj,z_int,c200 = subhalo
        logM = np.log10(M*(10.**10.))
        rs, alrs = nfw_fromM200andC200(logM,c200,z_int)
        
    if Mtype == 'INTs':
        ## I type has to be 'PEMD'
        theta_e, centerxproj, centeryproj,z_int, slope = subhalo
    
    ######################Calculating Projectionn

    if z_int <= z_lens:
        centerx,centery = centerxproj,centeryproj
    else:
        kwargs_lens_model_macro = [{'theta_E': theta_E,'gamma':gamma, 'center_x': clx, 'center_y': cly, 'e1': el1, 'e2': el2},
                             {'gamma1': gamma1, 'gamma2': gamma2}]

        lens_model_list_macro = ['PEMD','SHEAR']
        redshifts_macro = [z_lens,z_lens]

        lens_model_class_macro = LensModel(lens_model_list=lens_model_list_macro, z_source=z_source, 
                                 lens_redshift_list=redshifts_macro, multi_plane=True)
        sourceplanex,sourceplaney = lens_model_class_macro.ray_shooting(centerxproj, centeryproj, kwargs_lens_model_macro, k=None)
            
        alpha_l = np.array([centerxproj,centeryproj])
        alpha_s = np.array([sourceplanex,sourceplaney])
        chis, chil, chi = cosmo.comoving_distance([z_source,z_lens,z_int])
            
        veca = (chis*alpha_s - chil*alpha_l)/(chis-chil)
        vecb = (alpha_l-alpha_s)*((chis*chil)/(chis-chil))
            
        centercood = veca + (1./chi)*vecb
        centerx,centery = float(centercood[0]),float(centercood[1])
         
    ###############################################
    
    if Itype == 'SIS':
        kw_int = {'theta_E':theta_e*2e-3, 'center_x':centerx, 'center_y':centery}
    elif Itype == 'PJAFFE':
        kw_int = {'Ra': ra, 'Rs': rs, 'center_x': centerx, 'center_y': centery, 'sigma0': sig_0}
    elif Itype == 'PEMD':
        kw_int = {'theta_E': theta_e*2e-3,'gamma':slope, 'center_x': centerx, 'center_y': centery, 'e1': 0., 'e2': 0.}
    else:
        kw_int = {'Rs':rs, 'alpha_Rs':alrs, 'center_x':centerx, 'center_y':centery}    
        
    kwargs_lens_model = [{'theta_E': theta_E,'gamma':gamma, 'center_x': clx, 'center_y': cly, 'e1': el1, 'e2': el2},
                         {'gamma1': gamma1, 'gamma2': gamma2},
                         kw_int]
    redshifts_func = [z_lens,z_lens,z_int]
    
    lens_model_class_func = LensModel(lens_model_list=lens_model_list, z_source=z_source, 
                         lens_redshift_list=redshifts_func, multi_plane=True)


    sourceLightModel_reconstruct = LightModel(['SHAPELETS_POLAR'])
    lensLightModel_reconstruct = LightModel([])
    
    
    
    kwargs_shapelet = [{'n_max': n_max_model, 'beta': beta_model, 'center_x': source_x, 'center_y': source_y}]
    
    kwargs_lens_light2 = []
    
    imageModel = ImageLinearFit(data_class=data_real, psf_class=psf, kwargs_numerics=kwargs_numerics, 
                                lens_model_class=lens_model_class_func, source_model_class=sourceLightModel_reconstruct,
                                lens_light_model_class = lensLightModel_reconstruct,
                               likelihood_mask=likemask)
    
    wls_model, model_error, cov_param, coeffsq = imageModel.image_linear_solve(kwargs_lens_model, kwargs_shapelet, 
                                                                   kwargs_lens_light=kwargs_lens_light2, kwargs_ps=None, inv_bool=False)
    
    kwargs_shapelet_result = {'amp':coeffsq,'n_max': n_max_model, 'beta': beta_model, 'center_x': source_x, 'center_y': source_y}


    sourcelight = [kwargs_shapelet_result]
    lenslight = []
    
    imSim_HST_func = ImageModel(simhst_shape.data_class, psf, lens_model_class_func, simhst_shape.source_model_class,
                          simhst_shape.lens_light_model_class, simhst_shape.point_source_model_class, kwargs_numerics=kwargs_numerics)
    im2model = imSim_HST_func.image(kwargs_lens_model, sourcelight, lenslight)
        
    return np.ndarray.flatten(im2model),coeffsq

for i in range(5,20):
    n_max = i

    itype = 'PEMD' # NFW or SIS or PJAFFE pr PEMD

    mtype = 'INTs' # SMOOTH SUB or INT

    ndeg = np.count_nonzero(likemask)

    if mtype == 'SMOOTH':
        npar = 11
    if mtype == 'SUB':
        npar = 14
    if mtype == 'INT':
        npar = 15
    
    if mtype == 'SUBc':
        npar = 15
    if mtype == 'SUBs':
        npar = 15    
    if mtype == 'INTc':
        npar = 16
    if mtype == 'INTs':
        npar = 16
        
    smoothmin = [0.43, 1.8,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3]
    shapemin = [0.01,-0.3,-0.3]
    if itype == 'SIS' or itype == 'PJAFFE':
        submin = [0.0,-0.15,0.4]
    elif itype == 'PEMD':
        submin = [0.0,-0.15,0.4]
    else:
        submin = [0.0001,-0.15,0.4]
    cmin = [2.]
    slopemin = [1.3]
    
    zmin = [0.3]
            
    smoothmax = [0.50, 2.7,+0.3,+0.3,+0.3,+0.3,+0.3,+0.3]
    shapemax = [0.45/np.sqrt(n_max+1),0.3,0.3] #Shapelets – I. A method for image analysis eq 24
    if itype == 'SIS' or itype == 'PJAFFE':
        submax = [35.,0.15,0.6]
    elif itype == 'PEMD':
        submax = [35.,0.15,0.6]
    else:
        submax = [100.,0.15,0.6]
    
    cmax = [120.]
    slopemax = [4.]
    
    zmax = [1.9]
        
    if mtype == 'SMOOTH':
        minparam = smoothmin + shapemin
        maxparam = smoothmax + shapemax
     
    if mtype == 'SUB':
        minparam = smoothmin + submin + shapemin
        maxparam = smoothmax + submax + shapemax
        
    if mtype == 'SUBc':
        minparam = smoothmin + submin + cmin + shapemin
        maxparam = smoothmax + submax + cmax + shapemax
        
    if mtype == 'SUBs':
        minparam = smoothmin + submin + slopemin + shapemin
        maxparam = smoothmax + submax + slopemax + shapemax    
        
    if mtype == 'INT':
        minparam = smoothmin + submin + zmin + shapemin
        maxparam = smoothmax + submax + zmax + shapemax
      
    if mtype == 'INTc':
        minparam = smoothmin + submin + zmin + cmin + shapemin
        maxparam = smoothmax + submax + zmax + cmax + shapemax
        
    if mtype == 'INTs':
        minparam = smoothmin + submin + zmin + slopemin + shapemin
        maxparam = smoothmax + submax + zmax + slopemax + shapemax

    def lnlike(params):
        pn = 0.
        if mtype == 'SMOOTH':
            main_lens = params[0],params[1],params[2],params[3],params[4],params[5]
            shear = params[6],params[7]
            subhalo = params[0],params[1],params[2],params[3]       
            betaval,sx,sy = params[8],params[9],params[10]
                
        if mtype == 'SUB':
            main_lens = params[0],params[1],params[2],params[3],params[4],params[5]
            shear = params[6],params[7]
            subhalo = params[8],params[9],params[10]      
            betaval,sx,sy = params[11],params[12],params[13]
            
        if mtype == 'SUBc' or mtype == 'SUBs':
            main_lens = params[0],params[1],params[2],params[3],params[4],params[5]
            shear = params[6],params[7]
            subhalo = params[8],params[9],params[10],params[11]      
            betaval,sx,sy = params[12],params[13],params[14]
            

        if mtype == 'INT':
            main_lens = params[0],params[1],params[2],params[3],params[4],params[5]
            shear = params[6],params[7]
            subhalo = params[8],params[9],params[10],params[11]       
            betaval,sx,sy = params[12],params[13],params[14]
            
        if mtype == 'INTc' or mtype == 'INTs':
            main_lens = params[0],params[1],params[2],params[3],params[4],params[5]
            shear = params[6],params[7]
            subhalo = params[8],params[9],params[10],params[11],params[12]    
            betaval,sx,sy = params[13],params[14],params[15]

        model, coeffs = model_shapeflat(imagedata,main_lens,shear,subhalo,betaval,sx,sy,n_max,mtype,itype)


        res = np.ndarray.flatten(imagedata) - model
        sqrd = np.ndarray.flatten(likemask)*(res/background_rms)**2.
        
        return -0.5*(np.sum(sqrd)+pn)
                   

    # Define our uniform prior.
    def ptform(u):
        """Transforms samples `u` drawn from the unit cube to samples to those
        from our uniform prior within [-10., 10.) for each variable."""
        return minparam*(1.-u) + maxparam*u
        
        
    pool = Pool(nparal) 
    # "Static" nested sampling.
    #sampler = dynesty.NestedSampler(loglike, ptform, ndim)
    #sampler.run_nested()
    #sresults = sampler.results

    # "Dynamic" nested sampling.
    dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, npar,pool=pool, queue_size=nparal)
    dsampler.run_nested()
    dresults = dsampler.results


    def save_obj(obj, name ):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            
    save_obj(dresults,'mask3013_NARROWSLOPE_on_jvas_70pixSUBTR_n_max'+str(n_max)+mtype+itype)