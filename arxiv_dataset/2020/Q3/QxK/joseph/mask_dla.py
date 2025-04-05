import astropy as ap
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle

Nside = 2048
newNside = 32

n = Nside**2/newNside**2

mask = np.around(pickle.load(open("/Users/Joseph/Brookhaven/Research/data/mask_celestial.pkl",'rb')))

dla = open('/Users/Joseph/Brookhaven/Research/data/DLA_DR12_v1.dat') 
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

map = np.zeros(12*Nside**2)
map[dla_ipix] = 1.

map = np.array(map)
map = hp.ud_grade(map,nside_out=newNside)
map = np.ceil(map)
map = hp.ud_grade(map,nside_out=Nside)

plt.show(hp.mollview(map))
plt.close()
#astropy.io.fits.write_map('new_mask.fits',map, nest=False)

output = open('/Users/Joseph/Brookhaven/Research/pickled/mask_dla.pkl','wb')
pickle.dump(map,output)
output.close()
