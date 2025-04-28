import paths
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy.table import Table, Column, join, vstack
from helper_functions import *
from astropy.io import fits
import os
## cosmology to match Kondapally and Cochrane
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import time

#########################################
## default parameters to adjust
si = -0.7
#zmin = 0.003  ## matches Mauch & Sadler 2007
#zmax = 0.3   ## matches Mauch & Sadler 2007
dz = 0.0001  ## matches Cochrane and Kondapally
## for T_b
T_e = 1e4
ref_freqs = np.array([0.003,0.01,0.03,0.1,0.3,1,3])*1e9
freqs_GHz = np.arange( 1e-3, 1e2, 1e-3 )

sigma_cut = 5.


fields = ['lockman', 'en1']

mycat = Table()

for field in fields:
    print('Starting with field: {:s}'.format(field))
    infile = paths.data / '{:s}_03_matched_inMOC_inHR.fits'.format(field)
    field = os.path.basename(infile).split('_')[0]
    lotss = Table.read( infile, format='fits' )
    ## add brightness temperature information
    lotss = get_tb_information( lotss, im_weight=0.5, maj_lim=0.4, min_lim=0.3, T_e=T_e, alpha=si, ref_freqs=ref_freqs, freqs_GHz=freqs_GHz, use_z=False )
    lotss = do_SFR_AGN_separation( lotss )
    ## get number of detectable sources
    ndet = len( np.where(lotss['Detectability_SNR'] > 5.)[0] )
    with open( paths.output / '{:s}_detectable.txt'.format(field), 'w' ) as f:
        f.write( "{:,}".format(ndet) )
    mycat = vstack([mycat,lotss],metadata_conflicts='silent')


## first get radio detected
detected = np.where( mycat['Detectability_SNR'] > 5. )[0]
mycat = mycat[detected]

## radio excess:
n_radio_excess = len( np.where( mycat['Radio_excess'] >= 0.7 )[0] )

## not radio excess
n_not_excess = len( np.where( mycat['Radio_excess'] < 0.7 )[0] )

## number not excess & resolved
n_unres = len( np.where( np.logical_and( mycat['Radio_excess'] < 0.7, mycat['Resolved'] == 'U' ) )[0] )

## tb-identified
tb_idx = np.where( np.logical_and( mycat['tb_from'] > 0.0, mycat['Resolved'] == 'U' ) )[0]
not_excess_idx = np.where( mycat['Radio_excess'] < 0.7 )[0]
combined_idx = np.intersect1d(tb_idx,not_excess_idx)

n_tbid = len( combined_idx )

leftover = n_unres - n_tbid

with open( paths.output / 'flowchart_numbers.txt', 'w' ) as f:
    f.write( 'In summary, we start with {:,} total sources. {:,} have a radio excess, while {:,} do not. Of those without a radio excess, there are {:,} unresolved sources for which their $T_b$ is checked. Of those, {:,} are $T_b$-identified AGN, while {:,} are not. In the final sample, there are {:,} sources contributing to the AGN category and {:,} sources contributing to the SF category. This is more than the total number of sources due to the $T_b$-identified AGN which had a portion of their flux density shifted from the SF to the AGN category.'.format(len(detected), n_radio_excess, n_not_excess, n_unres, len(combined_idx), leftover, n_radio_excess+len(combined_idx), n_not_excess) )


