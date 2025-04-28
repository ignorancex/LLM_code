# Helper functions to get stats for one mock
import argparse
from os.path import join
import numpy as np
from lim_lytomo import stats
from lim_lytomo import comap, mock_lya
from os.path import join

## Specifics of the simulation
boxsize= [250]*3
seed = 13
snap='z2.5'
#snap = None
basepath='/rhome/mqezl001/bigdata/ASTRID/subfind/'

spec_file = '/rhome/mqezl001/bigdata/ASTRID/maps/spectra_ASTRID_noiseless_z2.5_1000_voxels.hdf5'
noiseless_file = '/rhome/mqezl001/bigdata/ASTRID/maps/map_ASTRID_true_0.25_z2.5.hdf5'


## Specifics of the mocks
k_par_min= None
Nmu = 30
#HCD_mask={'type':'NHI', 'thresh':10**19.7, 'vel_width':200 }
HCD_mask = {'type':None}

co_model = 'COMAP+21'
Li16_params={'alpha':1.17, 'beta':-0.21, 'sigma_co':0.37, 'delta_mf':1, 
            'behroozi_avsfr':'/rhome/mqezl001/bigdata/LIM/behroozi+13/sfr_release.dat', 
            'sfr_type':'behroozi+13'}

COMAP21_params={'A':-3.71, 'B':0.41, 'C':10**10.8, 'M':10**12.5, 'sigma_co':0.371}
#COMAP21_params={'A':-2.85, 'B':-0.42, 'C':10**10.63, 'M':10**12.3, 'sigma_co':0.42}



def get_stats(dperp, savefile, source_pk_file=None):
    lya_mock = mock_lya.MockLya(noiseless_file=noiseless_file, spec_file=spec_file, 
                    source_pk_file= source_pk_file, boxsize=boxsize, dperp=dperp, 
                    HCD_mask= HCD_mask, silent_mode=False, transpose=(1,0,2))
    if snap is not None:
        lim_mock = comap.MockComap(snap=snap, axis=3, basepath=basepath, fine_Nmesh=lya_mock.Nmesh,
                                boxsize=boxsize, halo_type='Subhalo', silent_mode=False, seed=seed, 
                                Li16_params=Li16_params, COMAP21_params=COMAP21_params, co_model='COMAP+21')
        st = stats.Stats(mock_lim=lim_mock, mock_lya=lya_mock, vol_ratio=1, k_par_min=k_par_min, Nmu=Nmu)
        st.get_lim_lya_sn()
    else:
        lim_mock = None
        st = stats.Stats(mock_lim=lim_mock, mock_lya=lya_mock, vol_ratio=1, k_par_min=k_par_min, Nmu=Nmu)
        st.get_lya_sn()
    
    st.save_stat(savefile)
    print(savefile)

if __name__ =='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--surveytype', type=str, required=True, help='')
   args = parser.parse_args()
   if args.surveytype == 'LATIS':
      dperp = 2.5
      savefile = './Gaussianized_peak/astrid_LATIS_stats_gaussianized_peak.hdf5'
   elif args.surveytype == 'PFS':
      dperp = 3.7
      savefile = './Gaussianized_peak/astrid_PFS_stats_gaussianized_peak.hdf5'
   elif args.surveytype == 'eBOSS':
      dperp = 13
      savefile = './Gaussianized_peak/astrid_eBOSS_stats_gaussianized_peak.hdf5'
   elif args.surveytype == 'DESI':
      dperp = 10
      savefile = './Gaussianized_peak/astrid_DESI_stats_gaussianized_peak.hdf5'
   #source_pk_file = 'astrid_'+args.surveytype+'_Cq.hdf5'
   source_pk_file = None
   get_stats(dperp=dperp, savefile=savefile, source_pk_file=source_pk_file)



