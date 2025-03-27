###############################################################################
# spc2dat.py
###############################################################################
#
# Process output of XMM data pipeline into numpy files ready for analysis
#
###############################################################################


import numpy as np
from astropy.io import fits
import h5py

# Parse keyword arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--xmmdata',action='store',dest='xmmdata',
                    default='False',type=str)
parser.add_argument('--obsID',action='store',dest='obsID',
                    default='False',type=str)
parser.add_argument('--prefix',action='store',dest='prefix',
                    default='False',type=str)
results = parser.parse_args()
xmmdata = results.xmmdata
obsID = results.obsID
prefix = results.prefix


# Process files
base = xmmdata + '/' + obsID + '/odf/' + prefix
obj = fits.open(base + '-obj.pi')
arf = fits.open(base + '.arf')
rmf = fits.open(base + '.rmf')
qpb = fits.open(base + '-back.pi')

# Extract the raw X-ray counts in each CCD channel
# Number of channels is different if mos or pn camera
# Each channel is associated with an energy, as extracted from the detector
# response files below
counts = obj['SPECTRUM'].data['COUNTS']

# Extract the exposure time for the entire observation
# Not vignetting corrected
exp = obj['SPECTRUM'].header['EXPOSURE'] # [s]

# Extract the size of the ROI from backscale
# units are (0.05'')^2, so convert to sr
roi_size = obj['SPECTRUM'].header['BACKSCAL']*(0.05*1./60./60.*np.pi/180.)**2.


# Calculate the detector response, using the arf and rmf
# This maps from a series of input channels (energies) to output ones,
# accounting for the effective area. This is a square matrix for the mos
# camera, but not for the pn

# Get the input and output energy arrays - for pn not uniformly spaced
cin_min = rmf[1].data['ENERG_LO'] # [keV]
cin_max = rmf[1].data['ENERG_HI'] # [keV]

cout_min = rmf[2].data['E_MIN'] # [keV]
cout_max = rmf[2].data['E_MAX'] # [keV]
cout_de = cout_max - cout_min # [keV]

# Extract the effective area as a function of input energy
# NB: this is vignetting corrected
effA = arf[1].data['SPECRESP'] # [cm^2]

# Extract the rmf matrix, which gives the probability of an input X-ray with
# true energy in an input channel is reconstructed into a given output channel
# At the same time we also multiply in the effective area through, combining
# to give the full detector response
# NB: this matrix is stored in a sparse format, so we have to reconstruct it

# Check shape of F_CHAN, it and N_CHAN have different shape for pn than mos
# For mos, N_GRP is always 0 or 1, for pn it can be larger
pndat = 0
if len(np.shape(rmf[1].data['F_CHAN'])) == 2:
    pndat = 1

det_res = np.zeros((len(cout_min),len(cin_min)))

for i in range(len(cin_min)):
    
    # Reconstruct sparse matrix
    det_col = np.zeros(len(cout_min))
    jtot = 0
    for j in range(rmf[1].data['N_GRP'][i]): # Number of groups
        # Account for different file strucutre in pn and mos
        if pndat:
            fc = rmf[1].data['F_CHAN'][i][j] # First channel
            sl = rmf[1].data['N_CHAN'][i][j] # Channels in this group
        else:
            fc = rmf[1].data['F_CHAN'][i] # First channel
            sl = rmf[1].data['N_CHAN'][i] # Channels in this group
        det_col[fc:fc+sl] = rmf[1].data['MATRIX'][i][jtot:jtot+sl]
        jtot += sl

    # To help compress matrix, set all values < 1.e-5 to 0
    # Recall this is a pdf, so these entries have negligible impact
    tocut = np.where(det_col < 1.e-5)
    det_col[tocut] = 0.

    # Save and account for effective area
    det_res[:,i] = det_col * effA[i] # [cm^2]


# Calculate the differential flux in each bin
# NB: this has units of [cts/s/keV/sr], which is often how X-ray results
# are plotted. The cm^2 is wrapped up in the detector response, and stored
# seprately
flux = counts/cout_de/exp/roi_size


# Extract the Quiescent Particle Background (QPB)
# This is stored in units of counts for mos, or rate = counts/exposure for PN
# Also extract the error, which is not sqrt(counts) because QPB is measured in
# a smaller region. This is also why counts is not an integer
if pndat:
    bkg_eff_cts = qpb['SPECTRUM'].data['RATE']*exp
    bkg_eff_cts_err = qpb['SPECTRUM'].data['STAT_ERR']*exp
else:
    bkg_eff_cts = qpb['SPECTRUM'].data['COUNTS']
    bkg_eff_cts_err = qpb['SPECTRUM'].data['STAT_ERR']


# Write the output as an h5 file, compressing the detector response
out_file = xmmdata + '/' + obsID + '/' + prefix + '_processed.h5'
h5f = h5py.File(out_file, 'w')
h5f.create_dataset('counts',data=counts)
h5f.create_dataset('flux',data=flux)
h5f.create_dataset('det_res',data=det_res,compression='gzip',compression_opts=9)
h5f.create_dataset('exp',data=exp)
h5f.create_dataset('roi_size',data=roi_size)
h5f.create_dataset('cin_min',data=cin_min)
h5f.create_dataset('cin_max',data=cin_max)
h5f.create_dataset('cout_min',data=cout_min)
h5f.create_dataset('cout_max',data=cout_max)
h5f.create_dataset('bkg_eff_cts',data=bkg_eff_cts)
h5f.create_dataset('bkg_eff_cts_err',data=bkg_eff_cts_err)
h5f.close()
