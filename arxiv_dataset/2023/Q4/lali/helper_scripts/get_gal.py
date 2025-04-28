# Helper functions to get stats for one mock
import argparse
from os.path import join
import numpy as np
from lim_lytomo import stats
from lim_lytomo import comap, mock_galaxy
from os.path import join

## Specifics of the simulation
boxsize= [250,250,250]
seed = 13
snap='z2.5'
basepath='/rhome/mqezl001/bigdata/ASTRID/subfind/'

## Specifics of the mocks
k_par_min= None
co_model = 'COMAP+21'
Li16_params={'alpha':1.17, 'beta':-0.21, 'sigma_co':0.37, 'delta_mf':1, 
            'behroozi_avsfr':'/rhome/mqezl001/bigdata/LIM/behroozi+13/sfr_release.dat', 
            'sfr_type':'behroozi+13'}
COMAP21_params={'A':-3.71, 'B':0.41, 'C':10**10.8, 'M':10**12.5, 'sigma_co':0.371}

def get_stats(Rz, savefile, gal_map_path, mass_cut):
    print(gal_map_path)
    gal_mock = mock_galaxy.MockGalaxy(snap=snap, axis=3, basepath=basepath,
                    boxsize=boxsize, halo_type='Subhalo', map_path= gal_map_path,
                    silent_mode=False, seed=seed, Rz=Rz, mass_cut=mass_cut)
            
    lim_mock = comap.MockComap(snap=snap, axis=3, basepath=basepath, fine_Nmesh=gal_mock.fine_Nmesh,
                                boxsize=boxsize, co_model=co_model, Li16_params=Li16_params, COMAP21_params=COMAP21_params,
                                halo_type='Subhalo', silent_mode=False, seed=seed)
                                
    st = stats.Stats(mock_lim=lim_mock, mock_galaxy=gal_mock, vol_ratio=1, k_par_min=k_par_min)
    
    st.get_lim_gal_sn()
    st.save_stat(savefile)

if __name__ =='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--Rz', type=float, required=True, help='')
   parser.add_argument('--n', type=int, required=False, help='iteration')
   args = parser.parse_args()
   if args.Rz == 7e-4:
   if args.Rz == 7e-4:
      savefile = 'astrid_Rz7e-4_stats_masscut11.9_gaussianized_peak_kcut0.03_n'+str(args.n)+'.hdf5'
      gal_map_path = './gal_map_0.25_z2.4_masscut11.9_n'+str(args.n)+'.hdf5'
      mass_cut = 11.9
   if args.Rz == 0.015:
      #savefile = 'astrid_Rz5e-4_stats_masscut11.51.hdf5'
      #gal_map_path = './gal_map_0.25_z2.4_masscut11.51.hdf5'
      #mass_cut = 11.51
      savefile = 'astrid_Rz015_stats_masscut12.1149_test.hdf5'
      gal_map_path = './gal_map_0.25_z2.4_masscut12.1149.hdf5'
      mass_cut = 12.1149

   if args.Rz == 0.02:
      savefile = './B-0.42/astrid_Rz02_stats_masscut11.51_gaussianized_peak.hdf5'
      gal_map_path = 'gal_map_0.25_z2.4_masscut11.51.hdf5'
      mass_cut = 11.51
   elif args.Rz == 0.03:
      savefile = './B-0.42/astrid_Rz03_stats_masscut11.21_gaussianized_peak.hdf5'
      gal_map_path = 'gal_map_0.25_z2.4_masscut11.21.hdf5'
      mass_cut = 11.21
   elif args.Rz == 0.04:
      savefile = './B-0.42/astrid_Rz04_stats_masscut11.31_gaussianized_peak.hdf5'
      gal_map_path = 'gal_map_0.25_z2.4_masscut11.31.hdf5'
      mass_cut = 11.31
   elif args.Rz == 0.06:
      savefile = './B-0.42/astrid_Rz06_stats_masscut11.01_gaussianized_peak.hdf5'
      gal_map_path = './gal_map_0.25_z2.4_masscut11.01.hdf5'
      mass_cut = 11.01

   elif args.Rz == 0.09:
      savefile = 'astrid_Rz09_stats_masscut10_test.hdf5'
      gal_map_path = './gal_map_0.25_z2.4_masscut10.hdf5'
      mass_cut = 10
   elif args.Rz == 0.2:
      savefile = 'astrid_Rz0.2_stats_masscut10.5_test.hdf5'
      Gal_map_path = './gal_map_0.25_z2.4_masscut10.5.hdf5'
      mass_cut = 10.5


   get_stats(Rz=args.Rz, savefile=savefile, gal_map_path=gal_map_path, mass_cut=mass_cut)



