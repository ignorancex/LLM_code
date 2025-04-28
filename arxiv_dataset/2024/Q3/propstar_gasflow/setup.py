# configuration file for the project, not for the nested sampling
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import astropy.units as u
from pathlib import Path
import sys
sys.path.append('../')

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
rc('font',**{'family':'serif'})#,'sans-serif':['Helvetica']})
rc('text', usetex=True)

mainfolder = str(Path.home()) + '/propstar_gasflow/'

datafolder = mainfolder + 'data/'
firstlookfolder = mainfolder + 'first_look/'
figfolder = mainfolder + 'figures/'
bayesfolder = mainfolder + 'bayes_frame/'

hc3n_10_9_cube_original = datafolder + 'NGC1333_HC3N_L24-merged'
hc3n_10_9_cube = datafolder + 'NGC1333_HC3N_L24-merged_small_K'
n2hp_1_0_cube_original = datafolder + 'NGC1333-N2Hp_match_kms'
n2hp_1_0_cube = datafolder + 'NGC1333-N2Hp_match_kms_F1F_01_12' # we need to integrate the component bw 5 and 10 km/s
# hc3n_10_9_cube_s = hc3n_10_9_cube +'_small'

# files for gaussian fit
fitdir = firstlookfolder + 'gaussfit/'
rmsfile = hc3n_10_9_cube + '_rms.fits'
snrfile = hc3n_10_9_cube + '_-3.0_18.0_snr.fits'
maskfile = fitdir + 'HC3N_10_9_mask'
fitfilebase = fitdir + 'HC3N_10_9_{}G_fitparams'

n2hpfitdir = firstlookfolder + 'fit_N2Hp/'


#files for nested sampling results
bayesfitfilebase = bayesfolder + 'nested-sampling/NGC1333-SE-mle-x{}.fits'
bayesnpeaksfile = bayesfolder + 'nested-sampling/npeaks_cut5.fits'

distance = 294 * u.pc # pc, Zucker et al 2018?
hc3n_10_9_rms = 1.1508e-2 # Jy/beam
velrange = np.array([-3, 18])
fwhm_to_sigma = 1. / (8 * np.log(2))**0.5

def beam_size(header):
    beam_fwhm_maj = header['BMAJ'] * u.deg
    beam_fwhm_min = header['BMIN'] * u.deg
    pixsize = np.abs(header['CDELT2']) * u.deg
    beam_sigma_maj = beam_fwhm_maj * fwhm_to_sigma
    beam_sigma_min = beam_fwhm_min * fwhm_to_sigma
    omega_B = 2 * np.pi * beam_sigma_maj * beam_sigma_min
    omega_B_pix2 = (omega_B / pixsize**2).value
    return omega_B, omega_B_pix2

def get_vc_r(velfield, header, centerra, centerdec, distance, region_file=None):
    
    from regions import Regions
    from astropy.wcs import WCS
    from astropy.io import fits
    import velocity_tools.velocity_tools.coordinate_offsets as c_offset
    # load region file and WCS structures
    if region_file is not None: regions = Regions.read(region_file)
    wcs_Vc = WCS(header).celestial
    hd_Vc = wcs_Vc.to_header()
    hd_Vc['naxis'] = 2
    hd_Vc['naxis1'] = header['naxis1']
    hd_Vc['naxis2'] = header['naxis2']
    results = c_offset.generate_offsets(hd_Vc, centerra, centerdec, pa_angle=0*u.deg, inclination=0*u.deg)
    rad_au = (results.r * distance*u.pc).to(u.au, equivalencies=u.dimensionless_angles())
    if region_file is not None:
        
        mask_Vc = (regions[0].to_pixel(wcs_Vc)).to_mask()
        Vc_cutout = mask_Vc.cutout(velfield)
        rad_cutout = mask_Vc.cutout(rad_au)
        gd = (mask_Vc.data == 1)
    else:
        Vc_cutout = velfield
        gd = ~np.isnan(Vc_cutout)
        rad_cutout = rad_au
    v_los = Vc_cutout[gd]*u.km/u.s
    r_proj = rad_cutout[gd]
    return r_proj, v_los
