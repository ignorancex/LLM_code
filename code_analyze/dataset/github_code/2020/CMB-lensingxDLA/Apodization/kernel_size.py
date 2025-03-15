# Data preparation, SGC+NGC

import pandas as pd
from astropy.io import fits
import numpy as np

DR12 = fits.open('../DR12Q.fits')

q_dr12 = DR12[1].data[(DR12[1].data['PSFMAG'][:,1]<=22)& (DR12[1].data['PSFMAG'][:,1]>=5.)
                            &(DR12[1].data['MI']<-23.78)&(DR12[1].data['MI']>=-28.74)
                            &(DR12[1].data['Z_VI']<=3.4)&(DR12[1].data['Z_VI']>=2.2)
                            &(DR12[1].data['FIRST_MATCHED']==0)
                            &(DR12[1].data['BOSS_TARGET1']!=0)]
q_dr12 = pd.DataFrame(np.array([q_dr12['RA'],q_dr12['DEC'],q_dr12['Z_VI'],q_dr12['PSFMAG'][:,1],q_dr12['MI']]).T,columns=('ra','dec','z','MAG','MI'))



'''
exclude SGC and plot maps
'''

from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp

n = SkyCoord(ra=q_dr12['ra'],dec=q_dr12['dec'],unit='deg',frame='icrs')
ng = n.galactic
l,b = ng.l,ng.b
q_dr12 = q_dr12[b>0]

g = SkyCoord(ra=q_dr12['ra'],dec=q_dr12['dec'],unit='deg',frame='icrs')
gg = g.galactic
l = gg.l.degree
b = gg.b.degree

q_indice = hp.pixelfunc.ang2pix(32,l,b,lonlat=True)     
q_map = np.zeros(hp.nside2npix(32), dtype=np.float)      
for i in range(len(q_indice)):                             
    q_map[q_indice[i]] += 1

mask_q = np.zeros(len(q_map))   # construct QSO mask with Nside=32               
for i in range(len(q_map)):
    if q_map[i] == 0:           # identify empty pixels
        mask_q[i] = 0
    else:
        mask_q[i] = 1      

mask_q = hp.pixelfunc.ud_grade(mask_q,nside_out=2048)  # upgrade the mask to Nside=self.nside

mask_k = hp.read_map('../mask.fits')

# simulate 

from pipeline_mock import *

# uniform distribution

z_min = 2.2
z_max = 3.4
s = 2/5


def mock_spectrum(Ckk,Ckq,Cqq,bias,fwhm_list):

    alm_k,seed = hp.sphtfunc.synalm(Ckk)
    alm_q1,seed = hp.sphtfunc.synalm(Ckq**2/Ckk,seed=seed)
    alm_q2,seed = hp.sphtfunc.synalm(Cqq-Ckq**2/Ckk)

    alm_q = alm_q1 + alm_q2 

    map_k = mask_k*hp.alm2map(alm_k,nside=2048)
    map_q = mask_q*hp.alm2map(alm_q,nside=2048)
    
    mask_kq = mask_k*mask_q

    re_spec = []
    for fwhm in fwhm_list:
        
        if fwhm!=None:
            mask_kq = hp.sphtfunc.smoothing(mask_kq,fwhm=fwhm)  
            
        fkq = np.mean(mask_kq*mask_k*mask_q)
    
        re = hp.sphtfunc.anafast(map_k*mask_kq,map_q*mask_kq,lmax=1200)
        re = re/fkq
        re_spec.append(re)
        
    return re_spec
    
kernel_size = [None,np.deg2rad(10/60),np.deg2rad(30/60),np.deg2rad(1)]    
round_N = 200
bias = 2.5

# uniform sample
uniform_cl = [[],[],[],[]]
uniform_l = []
uniform_model = []

n_tot = 2*10**8
sample = np.random.uniform(z_min,z_max,n_tot)
model = mock(sample,z_min,z_max,s,z_reso=100,lmax=1200)
Npix = hp.nside2npix(2048)
n_avg = n_tot/Npix
shot_noise = 1/n_avg

ell_kq, Ckq = model.Ckq(b=bias,x=range(1200),line=True)
ell_kk, Ckk = model.Ckk(x=range(800),line=True)
Ckk[0] = 1E-20
ell_qq, Cqq = model.Cqq(b=bias,x=range(1200),line=True)
#Cqq = Cqq + shot_noise

for i in range(200):
    re_spec = mock_spectrum(Ckk,Ckq,Cqq,bias=bias,fwhm_list=kernel_size)
    #l_mk,cl_mk = bin_corr(Ckq)
    uniform_model.append(Ckq)
    #uniform_l.append(l_mk)
    for j,spec in enumerate(re_spec):
        #l_re,cl_re = bin_corr(spec)
        uniform_cl[j].append(spec)
    print('uniform:%d finished'%i)

np.save('cl_full.npy',np.array(uniform_cl))
#np.save('l.npy',np.array(uniform_l))
np.save('model_full.npy',np.array(uniform_model))



