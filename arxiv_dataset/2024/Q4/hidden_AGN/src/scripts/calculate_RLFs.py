import paths
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy.table import Table, Column, join, vstack
from helper_functions import *
from astropy.io import fits
## WMAP9 is Hinshaw et al. 2013, H_0=69.3, Omega=0.287

##############################################################
## Plotting housekeeping

matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['axes.labelsize'] = 'large'
## set up some colours
n = 255
mycols = plt.cm.viridis(np.linspace(0, 1,n))
mycols_m = plt.cm.inferno(np.linspace(0, 1,n))

##############################################################

## set a default spectral index
si = -0.7
zmin = 0.003  ## matches Mauch & Sadler 2007
zmax = 0.3   ## matches Mauch & Sadler 2007
dz = 0.0001  ## matches Cochrane and Kondapally
lmin = 20.5
lmax = 27
dl = 0.3

## read in the data 
t = Table.read( paths.data / 'vmaxes/redshift_bins.csv', format='csv' )
zbin_starts = t['zbin_starts']
zbin_ends = t['zbin_ends']

fields = ['lockman','en1']

os.makedirs( paths.data / 'rlfs', exist_ok=True )

for i in np.arange(0,5):
    zmin = zbin_starts[i]
    zmax = zbin_ends[i]
    keep_cols = ['Total_flux_dr','Z_BEST','vmax','agn_vmax','sf_vmax','AGN_flux','SF_flux', 'Overall_class','Mass_cons','SFR_cons']
    vmaxes = Table()
    for field in fields:
        infile = 'vmaxes/{:s}_vmaxes_zmin{:s}_zmax{:s}.fits'.format(field,str(zmin),str(zmax))
        tmp = Table.read( paths.data / infile, format='fits' )
        vmaxes = vstack([vmaxes,tmp[keep_cols]],metadata_conflicts='silent')

    lum_bins, lum_func, agn_lum_func, sf_lum_func, gal_agn_lum_func, gal_sf_lum_func = get_RLFs( vmaxes, zmin, zmax, lmin=lmin, lmax=lmax, dl=dl, si=si )
    e_agn_lum_func, e_sf_lum_func, e_gal_agn_lum_func, e_gal_sf_lum_func = random_resample( agn_lum_func, sf_lum_func, gal_agn_lum_func, gal_sf_lum_func, vmaxes, zmin, zmax, lmin=lmin, lmax=lmax, dl=dl, si=si, nsamp=1000 )

    t = Table()
    t.add_column( lum_func, name='lum_func' )
    t.add_column( agn_lum_func, name='agn_lum_func' )
    t.add_column( sf_lum_func, name='sf_lum_func' )
    t.add_column( gal_agn_lum_func, name='gal_agn_lum_func' )
    t.add_column( gal_sf_lum_func, name='gal_sf_lum_func' )
    t.add_column( e_agn_lum_func, name='e_agn_lum_func' )
    t.add_column( e_sf_lum_func, name='e_sf_lum_func' )
    t.add_column( e_gal_agn_lum_func, name='e_gal_agn_lum_func' )
    t.add_column( e_gal_sf_lum_func, name='e_gal_sf_lum_func' )

    outfile = paths.data / 'rlfs/rlfs_zmin{:s}_zmax{:s}_lmin{:s}_lmax{:s}_dl{:s}.fits'.format(str(zmin),str(zmax),str(lmin),str(lmax),str(dl))

    t.write( outfile )


