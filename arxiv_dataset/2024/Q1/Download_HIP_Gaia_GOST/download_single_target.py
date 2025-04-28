import os
import numpy as np
import pandas as pd
from frame_rotation_correction import Collaborator
from astroquery.gaia import Gaia
from astropy.io import ascii
import urllib
import bs4
from get_hipIAD1997 import get_hipIAD1997
from Obtain_GOST import find_target
import requests
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from datetime import timedelta

class Downloader:
    def __init__(self):
        print('Initiating downloader...')
        self.collaborator = Collaborator()
        self.HIP = pd.read_csv('input_data/HIP.csv',sep = '|',comment = '#',dtype=str)
        self.HIP2 = pd.read_csv('input_data/HIP2.csv',sep = '|',comment = '#',dtype=str)
        self.TYC = pd.read_csv('input_data/TYC.csv',sep = '|',comment = '#',dtype=str)
        print('Done!')
    def download(self,hip_epoch = False,GOST = False,**kargs):
        query = '''
SELECT TOP 1
dr3.source_id AS dr3_source_id,
dr3.phot_g_mean_mag AS Gmag_dr3,
dr3.bp_rp AS bp_rp_dr3,
dr3.teff_gspphot AS teff_dr3,
dr3.ra AS dr3_ra,
dr3.ra_error AS dr3_ra_error,
dr3.dec AS dr3_dec,
dr3.dec_error AS dr3_dec_error,
dr3.parallax AS dr3_parallax,
dr3.parallax_error AS dr3_parallax_error,
dr3.pmra AS dr3_pmra,
dr3.pmra_error AS dr3_pmra_error,
dr3.pmdec AS dr3_pmdec,
dr3.pmdec_error AS dr3_pmdec_error,
dr3.ruwe AS dr3_ruwe,
dr3.radial_velocity AS dr3_radial_velocity,
dr3.radial_velocity_error AS dr3_radial_velocity_error,
dr3.ra_dec_corr AS dr3_ra_dec_corr,
dr3.ra_parallax_corr AS dr3_ra_parallax_corr,
dr3.ra_pmra_corr AS dr3_ra_pmra_corr,
dr3.ra_pmdec_corr AS dr3_ra_pmdec_corr,
dr3.dec_parallax_corr AS dr3_dec_parallax_corr,
dr3.dec_pmra_corr AS dr3_dec_pmra_corr,
dr3.dec_pmdec_corr AS dr3_dec_pmdec_corr,
dr3.parallax_pmra_corr AS dr3_parallax_pmra_corr,
dr3.parallax_pmdec_corr AS dr3_parallax_pmdec_corr,
dr3.pmra_pmdec_corr AS dr3_pmra_pmdec_corr,
dr2.source_id AS dr2_source_id,
dr2.ra AS dr2_ra,
dr2.ra_error AS dr2_ra_error,
dr2.dec AS dr2_dec,
dr2.dec_error AS dr2_dec_error,
dr2.parallax AS dr2_parallax,
dr2.parallax_error AS dr2_parallax_error,
dr2.pmra AS dr2_pmra,
dr2.pmra_error AS dr2_pmra_error,
dr2.pmdec AS dr2_pmdec,
dr2.pmdec_error AS dr2_pmdec_error,
dr2.radial_velocity AS dr2_radial_velocity,
dr2.radial_velocity_error AS dr2_radial_velocity_error,
dr2.ra_dec_corr AS dr2_ra_dec_corr,
dr2.ra_parallax_corr AS dr2_ra_parallax_corr,
dr2.ra_pmra_corr AS dr2_ra_pmra_corr,
dr2.ra_pmdec_corr AS dr2_ra_pmdec_corr,
dr2.dec_parallax_corr AS dr2_dec_parallax_corr,
dr2.dec_pmra_corr AS dr2_dec_pmra_corr,
dr2.dec_pmdec_corr AS dr2_dec_pmdec_corr,
dr2.parallax_pmra_corr AS dr2_parallax_pmra_corr,
dr2.parallax_pmdec_corr AS dr2_parallax_pmdec_corr,
dr2.pmra_pmdec_corr AS dr2_pmra_pmdec_corr,
dr1.source_id AS dr1_source_id,
dr1.ra AS dr1_ra,
dr1.ra_error AS dr1_ra_error,
dr1.dec AS dr1_dec,
dr1.dec_error AS dr1_dec_error,
dr1.parallax AS dr1_parallax,
dr1.parallax_error AS dr1_parallax_error,
dr1.pmra AS dr1_pmra,
dr1.pmra_error AS dr1_pmra_error,
dr1.pmdec AS dr1_pmdec,
dr1.pmdec_error AS dr1_pmdec_error,
dr1.ra_dec_corr AS dr1_ra_dec_corr,
dr1.ra_parallax_corr AS dr1_ra_parallax_corr,
dr1.ra_pmra_corr AS dr1_ra_pmra_corr,
dr1.ra_pmdec_corr AS dr1_ra_pmdec_corr,
dr1.dec_parallax_corr AS dr1_dec_parallax_corr,
dr1.dec_pmra_corr AS dr1_dec_pmra_corr,
dr1.dec_pmdec_corr AS dr1_dec_pmdec_corr,
dr1.parallax_pmra_corr AS dr1_parallax_pmra_corr,
dr1.parallax_pmdec_corr AS dr1_parallax_pmdec_corr,
dr1.pmra_pmdec_corr AS dr1_pmra_pmdec_corr,
dr3_hip.original_ext_source_id AS hip_id,
dr3_tyc.original_ext_source_id AS tyc_id

FROM gaiadr3.gaia_source AS dr3 
LEFT JOIN gaiadr3.dr2_neighbourhood AS dr3_dr2
ON dr3.source_id = dr3_dr2.dr3_source_id
LEFT JOIN gaiadr2.gaia_source AS dr2
ON dr3_dr2.dr2_source_id = dr2.source_id
LEFT JOIN gaiadr2.dr1_neighbourhood AS dr2_dr1
ON dr2_dr1.dr2_source_id = dr2.source_id
LEFT JOIN gaiadr1.gaia_source AS dr1
ON dr2_dr1.dr1_source_id = dr1.source_id
LEFT JOIN gaiadr3.hipparcos2_neighbourhood AS dr3_hip
ON dr3_hip.source_id = dr3.source_id
LEFT JOIN gaiadr3.tycho2tdsc_merge_neighbourhood AS dr3_tyc
ON dr3_tyc.source_id = dr3.source_id
WHERE 

'''
        if kargs['catalogue']=='Gaia DR3':
            query += 'dr3.source_id = '+str(kargs['id_star'])
        if kargs['catalogue']=='Gaia DR2':
            query += 'dr2.source_id = '+str(kargs['id_star'])
        if kargs['catalogue']=='Gaia DR1':
            query += 'dr1.source_id = '+str(kargs['id_star'])
        if kargs['catalogue']=='HIP':
            query += 'dr3_hip.original_ext_source_id = '+str(kargs['id_star'])
        if kargs['catalogue']=='TYC':
            query += "dr3_tyc.original_ext_source_id = \'"+str(kargs['id_star'])+"\'"
        job = Gaia.launch_job(query=query)
        Gaia_out = job.get_results()
        ascii.write(Gaia_out,'temp/temp_'+kargs['catalogue']+'_'+str(kargs['id_star'])+'.csv',format='csv',overwrite=True)
        query_res = pd.read_csv('temp/temp_'+kargs['catalogue']+'_'+str(kargs['id_star'])+'.csv')
    
        if GOST:
            print('Downloading GOST...')
            GOST_res = find_target(float(query_res.loc[0,'dr3_ra']),float(query_res.loc[0,'dr3_dec']),'Gaia DR3 '+str(query_res.loc[0,'dr3_source_id']))
            GOST_res.to_csv('results/GOST/'+kargs['catalogue']+'_'+str(kargs['id_star'])+'=Gaia DR3 '+str(query_res.loc[0,'dr3_source_id'])+'_GOST.csv',index = False)
            print('Done!')
        #------------------------------------------------ Collaborate GDR2->GDR3
        if query_res.loc[0,'dr2_source_id']==query_res.loc[0,'dr2_source_id']: #have GDR2 data
            input_astrometry = {'ra':query_res.loc[0,'dr2_ra'],'dec':query_res.loc[0,'dr2_dec'],'mag':query_res.loc[0,'Gmag_dr3'],'br':query_res.loc[0,'bp_rp_dr3']}
            if query_res.loc[0,'dr2_pmra']==query_res.loc[0,'dr2_pmra'] and query_res.loc[0,'dr2_pmdec']==query_res.loc[0,'dr2_pmdec']:
                input_astrometry['pmra'] = query_res.loc[0,'dr2_pmra']
                input_astrometry['pmdec'] = query_res.loc[0,'dr2_pmdec']
            if query_res.loc[0,'dr2_parallax']==query_res.loc[0,'dr2_parallax']:
                input_astrometry['plx'] = query_res.loc[0,'dr2_parallax']
            if float(query_res.loc[0,'Gmag_dr3'])<10.5:
                collaborate_res = self.collaborator.collaborate(input_astrometry,catalogue = 'GDR2toGDR3',mode='SS')
            else:
                collaborate_res = self.collaborator.collaborate(input_astrometry,catalogue = 'GDR2toGDR3',mode='NST')
            print(query_res,collaborate_res)
            query_res['dr2_col_ra'] = [collaborate_res['ra']]
            query_res['dr2_col_dec'] = [collaborate_res['dec']]
            query_res['dr2_col_pmra'] = [collaborate_res['pmra']]
            query_res['dr2_col_pmdec'] = [collaborate_res['pmdec']]
            query_res['dr2_col_parallax'] = [collaborate_res['plx']]
        #------------------------------------------------ Collaborate GDR1->GDR3
        if query_res.loc[0,'dr1_source_id']==query_res.loc[0,'dr1_source_id']: #have GDR1 data
            input_astrometry = {'ra':query_res.loc[0,'dr1_ra'],'dec':query_res.loc[0,'dr1_dec']}
            if query_res.loc[0,'dr1_pmra']==query_res.loc[0,'dr1_pmra'] and query_res.loc[0,'dr1_pmdec']==query_res.loc[0,'dr1_pmdec']:
                input_astrometry['pmra'] = query_res.loc[0,'dr1_pmra']
                input_astrometry['pmdec'] = query_res.loc[0,'dr1_pmdec']
            if query_res.loc[0,'dr1_parallax']==query_res.loc[0,'dr1_parallax']:
                input_astrometry['plx'] = query_res.loc[0,'dr1_parallax']
            if float(query_res.loc[0,'Gmag_dr3'])>10.5 or len(input_astrometry)==2:
                collaborate_res = self.collaborator.collaborate(input_astrometry,catalogue = 'GDR1toGDR3',mode='NSTP2')
            else:
                collaborate_res = self.collaborator.collaborate(input_astrometry,catalogue = 'GDR1toGDR3',mode='SSP5')
            query_res['dr1_col_ra'] = [collaborate_res['ra']]
            query_res['dr1_col_dec'] = [collaborate_res['dec']]
            query_res['dr1_col_pmra'] = [collaborate_res['pmra']]
            query_res['dr1_col_pmdec'] = [collaborate_res['pmdec']]
            query_res['dr1_col_parallax'] = [collaborate_res['plx']]
        #------------------------------------------------ JOIN TYC
        if query_res.loc[0,'tyc_id']==query_res.loc[0,'tyc_id']: #have TYC id
            try:
                TYC_name_part = str(query_res.loc[0,'tyc_id']).split('-')
                TYC_full_name = (4-len(TYC_name_part[0]))*' '+TYC_name_part[0]+(6-len(TYC_name_part[1]))*' '+TYC_name_part[1]+(2-len(TYC_name_part[2]))*' '+TYC_name_part[2]
                this_TYC = self.TYC.loc[self.TYC['TYC']==TYC_full_name]

                query_res['tyc_ra'] = [this_TYC['RAICRS'].values[0].strip(' ')]
                query_res['tyc_ra_error'] = [this_TYC['e_RAICRS'].values[0].strip(' ')]
                query_res['tyc_dec'] = [this_TYC['DEICRS'].values[0].strip(' ')]
                query_res['tyc_dec_error'] = [this_TYC['e_DEICRS'].values[0].strip(' ')]

                query_res['tyc_pmra'] = [this_TYC['pmRA'].values[0].strip(' ')]
                query_res['tyc_pmra_error'] = [this_TYC['e_pmRA'].values[0].strip(' ')]
                query_res['tyc_pmdec'] = [this_TYC['pmDE'].values[0].strip(' ')]
                query_res['tyc_pmdec_error'] = [this_TYC['e_pmDE'].values[0].strip(' ')]

                query_res['tyc_parallax'] = [this_TYC['Plx'].values[0].strip(' ')]
                query_res['tyc_parallax_error'] = [this_TYC['e_Plx'].values[0].strip(' ')]
                
                query_res['tyc_ra_dec_corr'] = [this_TYC['DE:RA'].values[0].strip(' ')]	
                query_res['tyc_ra_parallax_corr'] = [this_TYC['Plx:RA'].values[0].strip(' ')]	
                query_res['tyc_ra_pmra_corr'] = [this_TYC['pmRA:RA'].values[0].strip(' ')]	
                query_res['tyc_ra_pmdec_corr'] = [this_TYC['pmDE:RA'].values[0].strip(' ')]	
                query_res['tyc_dec_parallax_corr'] = [this_TYC['Plx:DE'].values[0].strip(' ')]	
                query_res['tyc_dec_pmra_corr'] = [this_TYC['pmRA:DE'].values[0].strip(' ')]	
                query_res['tyc_dec_pmdec_corr'] = [this_TYC['pmDE:DE'].values[0].strip(' ')]	
                query_res['tyc_parallax_pmra_corr'] = [this_TYC['pmRA:Plx'].values[0].strip(' ')]	
                query_res['tyc_parallax_pmdec_corr'] = [this_TYC['pmDE:Plx'].values[0].strip(' ')]	
                query_res['tyc_pmra_pmdec_corr'] = [this_TYC['pmDE:pmRA'].values[0].strip(' ')]
            except:
                print('TYC failed!')
        #---------------------------------------------------------------- JOIN HIP and HIP->GDR3
        if query_res.loc[0,'hip_id']==query_res.loc[0,'hip_id']: #have TYC id
            if hip_epoch:
                print('Obtaining HIPPARCOS epoch data...')
                get_hipIAD1997(query_res.loc[0,'hip_id'],relative_path = 'results/HIP_epoch/'+kargs['catalogue']+'_'+str(kargs['id_star'])+'=HIP'+str(query_res.loc[0,'hip_id'])+'_epoch.csv')
                print('Done!')
            HIP_name = (6-len(str(query_res.loc[0,'hip_id'])))*' '+str(query_res.loc[0,'hip_id'])
            this_HIP = self.HIP.loc[self.HIP['HIP']==HIP_name]

            query_res['hip_ra'] = [this_HIP['RAICRS'].values[0].strip(' ')]
            query_res['hip_ra_error'] = [this_HIP['e_RAICRS'].values[0].strip(' ')]
            query_res['hip_dec'] = [this_HIP['DEICRS'].values[0].strip(' ')]
            query_res['hip_dec_error'] = [this_HIP['e_DEICRS'].values[0].strip(' ')]

            query_res['hip_pmra'] = [this_HIP['pmRA'].values[0].strip(' ')]
            query_res['hip_pmra_error'] = [this_HIP['e_pmRA'].values[0].strip(' ')]
            query_res['hip_pmdec'] = [this_HIP['pmDE'].values[0].strip(' ')]
            query_res['hip_pmdec_error'] = [this_HIP['e_pmDE'].values[0].strip(' ')]

            query_res['hip_parallax'] = [this_HIP['Plx'].values[0].strip(' ')]
            query_res['hip_parallax_error'] = [this_HIP['e_Plx'].values[0].strip(' ')]
            
            query_res['hip_ra_dec_corr'] = [this_HIP['DE:RA'].values[0].strip(' ')]	
            query_res['hip_ra_parallax_corr'] = [this_HIP['Plx:RA'].values[0].strip(' ')]	
            query_res['hip_ra_pmra_corr'] = [this_HIP['pmRA:RA'].values[0].strip(' ')]	
            query_res['hip_ra_pmdec_corr'] = [this_HIP['pmDE:RA'].values[0].strip(' ')]	
            query_res['hip_dec_parallax_corr'] = [this_HIP['Plx:DE'].values[0].strip(' ')]	
            query_res['hip_dec_pmra_corr'] = [this_HIP['pmRA:DE'].values[0].strip(' ')]	
            query_res['hip_dec_pmdec_corr'] = [this_HIP['pmDE:DE'].values[0].strip(' ')]	
            query_res['hip_parallax_pmra_corr'] = [this_HIP['pmRA:Plx'].values[0].strip(' ')]	
            query_res['hip_parallax_pmdec_corr'] = [this_HIP['pmDE:Plx'].values[0].strip(' ')]	
            query_res['hip_pmra_pmdec_corr'] = [this_HIP['pmDE:pmRA'].values[0].strip(' ')]

            input_astrometry = {'ra':float(query_res.loc[0,'hip_ra']),'dec':float(query_res.loc[0,'hip_dec'])}
            if len(query_res.loc[0,'hip_pmra'])!=0 and len(query_res.loc[0,'hip_pmra'])!=0:
                input_astrometry['pmra'] = float(query_res.loc[0,'hip_pmra'])
                input_astrometry['pmdec'] = float(query_res.loc[0,'hip_pmdec'])
            if len(query_res.loc[0,'hip_parallax'])!=0:
                input_astrometry['plx'] = float(query_res.loc[0,'hip_parallax'])
            collaborate_res = self.collaborator.collaborate(input_astrometry,catalogue = 'HIPtoGDR3')
            query_res['hip_col_ra'] = [collaborate_res['ra']]
            query_res['hip_col_dec'] = [collaborate_res['dec']]
            query_res['hip_col_pmra'] = [collaborate_res['pmra']]
            query_res['hip_col_pmdec'] = [collaborate_res['pmdec']]
            query_res['hip_col_parallax'] = [collaborate_res['plx']]
        #---------------------------------------------------------------- JOIN HIP2 and HIP2->GDR3
        if query_res.loc[0,'hip_id']==query_res.loc[0,'hip_id']: #have TYC id
            HIP_name = (6-len(str(query_res.loc[0,'hip_id'])))*' '+str(query_res.loc[0,'hip_id'])
            this_HIP = self.HIP2.loc[self.HIP2['HIP']==HIP_name]

            query_res['hip2_ra'] = [this_HIP['RArad'].values[0].strip(' ')]
            query_res['hip2_ra_error'] = [np.sqrt(float(this_HIP['e_RArad'].values[0].strip(' '))**2+2.16**2)]
            query_res['hip2_dec'] = [this_HIP['DErad'].values[0].strip(' ')]
            query_res['hip2_dec_error'] = [np.sqrt(float(this_HIP['e_DErad'].values[0].strip(' '))**2+2.16**2)]

            query_res['hip2_pmra'] = [this_HIP['pmRA'].values[0].strip(' ')]
            query_res['hip2_pmra_error'] = [this_HIP['e_pmRA'].values[0].strip(' ')]
            query_res['hip2_pmdec'] = [this_HIP['pmDE'].values[0].strip(' ')]
            query_res['hip2_pmdec_error'] = [this_HIP['e_pmDE'].values[0].strip(' ')]

            query_res['hip2_parallax'] = [this_HIP['Plx'].values[0].strip(' ')]
            query_res['hip2_parallax_error'] = [this_HIP['e_Plx'].values[0].strip(' ')]


            input_astrometry = {'ra':float(query_res.loc[0,'hip2_ra']),'dec':float(query_res.loc[0,'hip2_dec'])}
            if len(query_res.loc[0,'hip2_pmra'])!=0 and len(query_res.loc[0,'hip2_pmra'])!=0:
                input_astrometry['pmra'] = float(query_res.loc[0,'hip2_pmra'])
                query_res['hip2_pmra_error'] = [np.sqrt(float(this_HIP['e_pmRA'].values[0].strip(' '))**2+2.16**2)]
                input_astrometry['pmdec'] = float(query_res.loc[0,'hip2_pmdec'])
                query_res['hip2_pmdec_error'] = [np.sqrt(float(this_HIP['e_pmDE'].values[0].strip(' '))**2+2.16**2)]
            if len(query_res.loc[0,'hip2_parallax'])!=0:
                input_astrometry['plx'] = float(query_res.loc[0,'hip2_parallax'])
            collaborate_res = self.collaborator.collaborate(input_astrometry,catalogue = 'HIPtoGDR3')
            query_res['hip2_col_ra'] = [collaborate_res['ra']]
            query_res['hip2_col_dec'] = [collaborate_res['dec']]
            query_res['hip2_col_pmra'] = [collaborate_res['pmra']]
            query_res['hip2_col_pmdec'] = [collaborate_res['pmdec']]
            query_res['hip2_col_parallax'] = [collaborate_res['plx']]
        query_res.to_csv('results/GDR123_HIP_TYC_collaborated/'+kargs['catalogue']+'_'+str(kargs['id_star'])+'.csv',index = False)
        print('Results are available in '+'results/'+kargs['catalogue']+'_'+str(kargs['id_star'])+'.csv')



        #print()
    
Dl = Downloader()
import glob
target_list = pd.read_csv('/Users/ruiyicheng/Documents/code/projects/astrometry_periodogram/Barnards_star_vicinity_1000.csv')
path_list = glob.glob('/Users/ruiyicheng/Documents/code/projects/astrometry_periodogram/results/GDR123_HIP_TYC_collaborated/*.csv')
print(path_list)
for sid in target_list['source_id']:
    if '/Users/ruiyicheng/Documents/code/projects/astrometry_periodogram/results/GDR123_HIP_TYC_collaborated/Gaia DR3_'+str(sid)+'.csv' in path_list:
        print('dwonloaded')
        continue
    print(sid)
    Dl.download(hip_epoch = True, GOST = True,catalogue='Gaia DR3',id_star =str(sid))

# Dl.download(catalogue='HIP',id_star ='87937')
# Dl.download(catalogue='Gaia DR2',id_star ='4472832130942575872')
# Dl.download(catalogue='Gaia DR3',id_star ='4472832130942575872')
path_list = glob.glob('/Users/ruiyicheng/Documents/code/projects/astrometry_periodogram/results/GDR123_HIP_TYC_collaborated/*.csv')
df = pd.read_csv(path_list[0])
ct =0
for p in path_list:
    if ct==0:
        ct = 1
        continue
    df_m = pd.read_csv(p)
    df = pd.concat([df,df_m])

df.to_csv('merge_results_barnards_star_1000.csv',index = False)


