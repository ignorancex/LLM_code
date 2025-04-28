import scipy
import numpy as np
from astropy.nddata import Cutout2D
import lenstronomy.Util.util as util


def noise_creation(pmap,mag_zeropoint,back_mag,exp_time,exp_num,ron,res_in_arcsec,fix_seed=True,seed=1235):
    """
    creation of noise map and a realistation of it
    Param
    -----
    pmap : image in count/s in 1D array !
    mag_zeropoint : see glamer for more info
    back_mag : see glamer for more info
    exp_time : s
    exp_num : integer
    ron : read out noise
    res_in_arcsec : pixel size (in arcsec)
    fix_seed : boolean
    seed : integer (only used if fix_seed is True)
    Output
    ------
    noise_realasation,noise_map (in count/s) in 1D array
    """
    if fix_seed is True :
        np.random.seed(seed)
    Q=10**(0.4*(mag_zeropoint+48.6))
    back_mean = 10**(-0.4*(48.6+back_mag))*res_in_arcsec**2*Q*exp_time

    rms2=np.sqrt(exp_num*ron**2)
    P=scipy.stats.poisson.rvs(pmap*exp_time+back_mean)
    noise= np.random.randn(pmap.__len__())*rms2
    noise_realisation=(P-(pmap*exp_time+back_mean)+noise)/exp_time

    noise_bis= rms2*1.
    pois=np.sqrt(pmap*exp_time + back_mean)
    noise_map_general=np.sqrt((pois)**2+(noise_bis)**2)/exp_time
    return noise_realisation,noise_map_general
    
def make_mask(pmap,mag_zeropoint,back_mag,exp_time,exp_num,ron,res_in_arcsec,threshold=3.,number_realisation=2,
              fix_seed=True,seed=1235):
    """
    Create a mask based on noise with 1. when pmap+noise_add>threshold*background_sigma (median between the different
    realisations).
    
    Param
    -----
    pmap : image in count/s in 1D array !
    mag_zeropoint : see glamer for more info
    back_mag : see glamer for more info
    exp_time : s
    exp_num : integer
    ron : read out noise
    res_in_arcsec : pixel_size (in arcsec)
    threshold : threshold such that pixels above threshold*background_sigma are 1. in the mask
    number_realisation : number of realisation to avoid random features in the mask
    fix_seed : boolean
    seed : integer (only used if fix_seed is True) and for other realisations, seed+=1
    Output
    ------
    mask in 1D array
    """
    mask=np.zeros(shape=(number_realisation,len(pmap)))
    for i in range(number_realisation):
        noise_add,noise_gen = noise_creation(pmap,mag_zeropoint,back_mag,exp_time,exp_num,ron,res_in_arcsec,
                                             fix_seed=fix_seed,seed=seed+i)
        cut_background = Cutout2D(util.array2image(pmap+noise_add),(10,10),(20,20))
        background_rms=cut_background.data.std()
        mask[i,np.abs(pmap+noise_add)>threshold*background_rms]=1
    mask_final = np.median(mask,axis=0)
    mask_final[mask_final<=0.5]=0
    return mask_final
