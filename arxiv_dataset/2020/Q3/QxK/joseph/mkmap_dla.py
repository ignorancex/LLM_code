import astropy as ap
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle

Nside = 2048

mask  = hp.read_map("equatorialRealPlanckMask2048.fits")

dla = open('DLA_DR12_v1.dat')
dla_ra = []
dla_dec = []

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

for line in dla:
    ra = [line.split()][0][2]
    dec = [line.split()][0][3]
    if isfloat(ra):
        dla_ra.append(float(ra))
        dla_dec.append(float(dec))

dla_ra = np.array(dla_ra)
dla_dec = np.array(dla_dec)
dla_phi = dla_ra/180.*np.pi
dla_theta = (90.- dla_dec)*np.pi/180.
dla_ipix = hp.ang2pix(Nside, dla_theta, dla_phi)
dla_mask = mask[dla_ipix]==1
dla_ipix = dla_ipix[dla_mask]

map = np.bincount(dla_ipix)
map = np.append(map,np.zeros(12*Nside**2-len(map)))

plt.show(hp.mollview(map))
plt.close()
#astropy.io.fits.write_map('new_mask.fits',map, nest=False)

output = open('dla_map.pkl','wb')
pickle.dump(map,output)
output.close()
