import tarfile
import os
import pandas as pd
from astropy.io import fits
import numpy as np
from pipeline_matrix import *
import shutil

# Data preparation
DLA = pd.read_csv('../matched_updated_dla.csv')  # updated dla catalogue 
DR12 = fits.open('../DR12Q.fits')
q_mag = []
for i in DLA['PSFMAG']:
    q_mag.append(float(i.split(',')[1]))
q_mag = np.array(q_mag)    
q = DLA[(q_mag<=21.5)&(q_mag>=5.0)
                   &(DLA['MI']<-23.78)&(DLA['MI']>-28.74)
                   &(DLA['FIRST_MATCHED']==0.)
                   &(DLA['Z_VI']<=3.4)&(DLA['Z_VI']>=2.2)
                   &(DLA['conf']>0.3)
                   &(DLA['zabs']<DLA['Z_VI'])]
q_mag = []
for i in q['PSFMAG']:
    q_mag.append(float(i.split(',')[1]))
q_mag = np.array(q_mag) 

q_dla = pd.DataFrame(np.array([q['RA_2'],q['DEC_2'],q['Z_VI'],q['zabs'],q_mag,q['MI']]).T,columns=('ra','dec','z','z_dla','MAG','MI'))


q_dr12 = DR12[1].data[(DR12[1].data['PSFMAG'][:,1]<=21.5)& (DR12[1].data['PSFMAG'][:,1]>=5.)
                            &(DR12[1].data['MI']<-23.78)&(DR12[1].data['MI']>=-28.74)
                            &(DR12[1].data['Z_VI']<=3.4)&(DR12[1].data['Z_VI']>=2.2)
                            &(DR12[1].data['FIRST_MATCHED']==0)
                            &(DR12[1].data['BOSS_TARGET1']!=0)]
q_dr12 = pd.DataFrame(np.array([q_dr12['RA'],q_dr12['DEC'],q_dr12['Z_VI'],q_dr12['PSFMAG'][:,1],q_dr12['MI']]).T,columns=('ra','dec','z','MAG','MI'))



from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp

n = SkyCoord(ra=q_dla['ra'],dec=q_dla['dec'],unit='deg',frame='icrs')
ng = n.galactic
l,b = ng.l,ng.b
q_dla = q_dla[b>0]

q = np.unique(np.array([q_dla['ra'],q_dla['dec'],q_dla['z'],q_dla['MI'],q_dla['MAG']]).T,axis=0)
q = pd.DataFrame(q,columns=('ra','dec','z','MI','MAG'))


n = SkyCoord(ra=q_dr12['ra'],dec=q_dr12['dec'],unit='deg',frame='icrs')
ng = n.galactic
l,b = ng.l,ng.b
q_dr12 = q_dr12[b>0]



# read CMB lensing realizations
files = os.listdir('../CMB_Lensing_SIM')
for file in files:
    if file.split('.')[-1] == 'tar':
        print(file)
        tar = tarfile.open('../CMB_Lensing_SIM/'+file)
        tar.extractall('../CMB_Lensing_SIM')
        index = int(file.split('.')[0].split('_')[-1])
        index = [index-29,index]
        
        q_cl = []
        dla_cl = []

        for j in range(index[0],index[-1]+1):

            cor = corr('../CMB_Lensing_SIM/MV/sim_klm_%03d.fits'%j,'../mask.fits',q_dr12,nside=2048)
            cor.cal_q_mask()       
            cor.cal_qso_overdensity_map()
            cor.cal_kq_mask()
            c_kq = cor.corr(lmax=1200)
            bin_l,bin_cl = cor.bin_corr(c_kq,l_min=30,l_max=1200,band=10)

            q_cl.append(np.array(bin_cl))


            cor_dla = corr('../CMB_Lensing_SIM/MV/sim_klm_%03d.fits'%j,'../mask.fits',q,nside=2048,q_mask=cor.q_mask)
            cor_dla.cal_kq_mask()
            cor_dla.cal_qso_overdensity_map()
            c_kq_dla = cor_dla.corr(lmax=1200)
            bin_l_dla,bin_cl_dla = cor_dla.bin_corr(c_kq_dla,l_min=30,l_max=1200,band=10)

            dla_cl.append(np.array(bin_cl_dla))

        np.save('./10arcmin_both/sim_%03d_%03d_q.npy'%(index[0],index[-1]),np.array(q_cl))
        np.save('./10arcmin_both/sim_%03d_%03d_dla.npy'%(index[0],index[-1]),np.array(dla_cl))
        
        shutil.rmtree('../CMB_Lensing_SIM/MV')
        print('%03d-%03d:finished'%(index[0],index[-1]))
        
        
        
