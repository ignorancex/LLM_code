import numpy as np
import pandas as pd
from collections import Counter
from astropy.io.fits import getdata
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astroquery.vizier import Vizier
from astropy.table import Table
from astroquery.xmatch import XMatch
from astroquery.simbad import Simbad
import time
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import sys  
import os
from os import path
import multiprocessing as mp
from pathlib import Path

gaia_search_radius = 10 # arcsec
nway_dir = '/Users/huiyang/Softwarses/nway-master/'

def CSCviewsearch(field_name, ra, dec, radius,query_dir,csc_version='2.0'):
    
    ra_low  = ra - radius/(60.*np.cos(dec*np.pi/180.))
    ra_upp  = ra + radius/(60.*np.cos(dec*np.pi/180.))
    dec_low = dec - radius/60
    dec_upp = dec + radius/60
    rad_cone = radius
    

    
    f = open(f'{query_dir}/template/csc_query_cnt_template.adql', "r")
    adql = f.readline()

    ra_temp = '266.599396'
    dec_temp = '-28.87594'
    ra_low_temp = '266.5898794490786'
    ra_upp_temp = '266.60891255092145'
    dec_low_temp = '-28.884273333333333'
    dec_upp_temp = '-28.867606666666667'
    rad_cone_temp = '0.543215'
    #'''
    for [str1, str2] in [[rad_cone, rad_cone_temp], [ra, ra_temp], [dec, dec_temp], [ra_low, ra_low_temp], [ra_upp, ra_upp_temp], [dec_low, dec_low_temp], [dec_upp, dec_upp_temp]]:
        adql = adql.replace(str2, str(str1))

    text_file = open(f'{query_dir}/{field_name}_wget.adql', "w")
    text_file.write('http://cda.cfa.harvard.edu/csccli/getProperties?query='+adql)
    text_file.close()
    

    # if operatin system is linux
    if os.name == 'posix':
        os.system("wget -O "+query_dir+'/'+field_name+".txt -i "+query_dir+'/'+field_name+"_wget.adql")
    elif os.name == 'nt':
        # read uri from the first line of the file, strip the newline character
        uri = open(query_dir+'/'+field_name+"_wget.adql", "r").readline().strip()
        # use requests to get the data
        r = requests.get(uri)
        # write the data to the file
        open(query_dir+'/'+field_name+".txt", "w").write(r.text)


    
    #df = pd.read_csv(f'{query_dir}/{field_name}.txt', header=154, sep='\t')

    return None

vizier_cols_dict = {
    'catalogs': {'CSC':'IX/57/csc2master','gaia':'I/355/gaiadr3', 'tmass':'II/246/out','allwise':'II/328/allwise','catwise':'II/365/catwise'},\
    #'search_radius':{'CSC':0.05/60,'gaia':1., 'tmass':1.,'allwise':2.,'catwise':1.},\
    'search_radius':{'CSC':0.05/60,'gaia':4., 'tmass':4.,'allwise':4.,'catwise':4.},\
    #'search_radius':{'CSC':0.05/60,'gaia':3., 'tmass':3.,'allwise':3.,'catwise':3.},\
    #'search_radius':{'CSC':0.01/60,'gaia':5., 'tmass':5.,'allwise':5.,'catwise':5.},
    #'search_radius':{'CSC':0.01/60,'gaia':10./60, 'tmass':10./60,'allwise':10./60,'catwise':10./60},
    'IX/57/csc2master':['_r','_RAJ2000','_DEJ2000','2CXO','RAICRS','DEICRS','r0','r1','PA','fe','fc',\
                       #'fp','fv','fst','fs','fa','fi','fr','fm','ff','fVi',\
                       #'F90b','F9h','F90m','F90s','F90u']
                       ],          
    'I/355/gaiadr3':['_r','_RAJ2000','_DEJ2000','Source','RA_ICRS','DE_ICRS','e_RA_ICRS','e_DE_ICRS','RADEcor',\
              'Plx','e_Plx','PM','pmRA','e_pmRA','pmDE','e_pmDE','pmRApmDEcor','epsi','amax','RUWE','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag',  
              #'BP-RP','BP-G','G-RP',\
              #'AllWISE','dAllWISE','f_AllWISE','AllWISEoid','2MASS','d2MASS','f_2MASS','2MASScoid'
              ],
    'II/246/out': ['_r','_RAJ2000','_DEJ2000','Date','JD','2MASS','RAJ2000','DEJ2000','errMaj','errMin','errPA',\
              'Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag'
              ],
    'II/328/allwise': ['_r','_RAJ2000','_DEJ2000','AllWISE','ID','RAJ2000','DEJ2000','eeMaj','eeMin','eePA',\
            'RA_pm','e_RA_pm','DE_pm','e_DE_pm','cosig_pm','pmRA','e_pmRA','pmDE','e_pmDE',\
              'W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag',
              #'2M',
              ],
    'II/365/catwise': ['_r','_RAJ2000','_DEJ2000','Name','ID','RA_ICRS','e_RA_ICRS','DE_ICRS','e_DE_ICRS','ePos',\
            'MJD','RAPMdeg','e_RAPMdeg','DEPMdeg','e_DEPMdeg','ePosPM','pmRA','e_pmRA','pmDE','e_pmDE','plx1','e_plx1',\
              'W1mproPM','W2mproPM','e_W1mproPM','e_W2mproPM',
              ],
}

def nway_mw_prepare_v1(ra_x, dec_x, X_name, mjd_dif=0.,catalog='gaia',data_dir='data',plot_density_curve=False,sigma=2):
    
    #catalog = 'I/355/gaiadr3'
        
    viz = Vizier(row_limit=-1,  timeout=5000, columns=vizier_cols_dict[vizier_cols_dict['catalogs'][catalog]],catalog=vizier_cols_dict['catalogs'][catalog])
          
    search_radius = vizier_cols_dict['search_radius'][catalog] # arcmin, we plot the density vs radius and see it starts to converge at around 4'
    
    query = viz.query_region(SkyCoord(ra=ra_x, dec=dec_x,
                        unit=(u.deg, u.deg),frame='icrs'),
                        radius=search_radius*60*u.arcsec)
                        #,column_filters={'Gmag': '<19'}
    
    query_res = query[0]

    df_q = query_res.to_pandas()
    df_q = df_q.sort_values(by='_r').reset_index(drop=True)
    #print(df_q.columns)
    #num = len(df_q)

    #'''
    #print(df_q.columns)
    if catalog == 'CSC':
        
        df_q['RA']  = df_q['RAICRS']
        df_q['DEC'] = df_q['DEICRS']
        
        
        df_q['ID'] = df_q.index + 1
        df_q['err_r0'] = df_q['r0']*sigma/2
        df_q['err_r1'] = df_q['r1']*sigma/2
    
        
        new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_2CXO','_r','fe','fc']]) # r0 is 95%, should be consistent with other PUs, 
        
        #new_t.write(f'./{data_dir}/{csc_name}_CSC.fits', overwrite=True)


    if catalog == 'gaia':
        #gaia_ref_mjd = 57388.
        #delta_yr = (ref_mjd - gaia_ref_mjd)/365.
        delta_yr = mjd_dif/365.
        df_q['RA']  = df_q['RA_ICRS']
        df_q['DEC'] = df_q['DE_ICRS']
        #df_q['RA']  = df_q.apply(lambda row:row.RA_ICRS+delta_yr*row.pmRA/(np.cos(row.DE_ICRS*np.pi/180.)*3.6e6),axis=1)
        #df_q['DEC'] = df_q.apply(lambda row:row.DE_ICRS+delta_yr*row.pmDE/3.6e6,axis=1)
        
        #df_q.loc[df_q['RA'].isnull(),'RA']   = df_q.loc[df_q['RA'].isnull(),'RA_ICRS']
        #df_q.loc[df_q['DEC'].isnull(),'DEC'] = df_q.loc[df_q['DEC'].isnull(),'DE_ICRS']
        #df_q['PU'] = df_q.apply(lambda r: max(r.e_RA_ICRS,r.e_DE_ICRS),axis=1)
        #df_q['PU'] = df_q.apply(lambda r: max(r.e_RA_ICRS,r.e_DE_ICRS)**2+(r.PM*mjd_difs[catalog]/365)**2+(r.e_PM_gaia*mjd_difs[catalog]/365)**2,axis=1)
        #print(df_q[['e_RA_ICRS','e_DE_ICRS','Plx','e_Plx','PM','epsi','amax','RUWE']].describe())
        
        # https://www.aanda.org/articles/aa/pdf/2018/08/aa32727-18.pdf eq. B.1-B.3
        
        df_q['C00'] = df_q['e_RA_ICRS'] * df_q['e_RA_ICRS']
        df_q['C01'] = df_q['e_RA_ICRS'] * df_q['e_DE_ICRS'] * df_q['RADEcor']
        df_q['C11'] = df_q['e_DE_ICRS'] * df_q['e_DE_ICRS']
        df_q['C33'] = df_q['e_pmRA']    * df_q['e_pmRA']
        df_q['C34'] = df_q['e_pmRA']    * df_q['e_pmDE'] * df_q['pmRApmDEcor']
        df_q['C44'] = df_q['e_pmDE']    * df_q['e_pmDE']
        df_q['sigma_pos'] = np.sqrt(0.5*(df_q.C00+df_q.C11) + 0.5*np.sqrt((df_q.C11-df_q.C00)**2+4*df_q.C01**2)) 
        df_q['sigma_pm']  = np.sqrt(0.5*(df_q.C33+df_q.C44) + 0.5*np.sqrt((df_q.C44-df_q.C33)**2+4*df_q.C34**2))
        df_q['PU_PM']   = df_q['PM']*delta_yr
        df_q['PU_ePM']  = df_q['sigma_pm']*delta_yr
        df_q['PU'] = sigma * np.sqrt(df_q['sigma_pos'].fillna(0.)**2+df_q['Plx'].fillna(0.)**2+df_q['e_Plx'].fillna(0.)**2 + (df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2+df_q['epsi'].fillna(0.)**2)/1e3
             #df_q['epsi'].fillna(0.)**2)/1e3
        #df_q['PU'] = df_q.apply(lambda r: max(r.PU_c,0.2), axis=1)
        #print(df_q[['sigma_pos','Plx','e_Plx','PM','sigma_pm','epsi','PU_c','PU']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','Source','_r','sigma_pos','Plx','e_Plx','PU_PM','PU_ePM','epsi','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag']])
        
    
    elif catalog == 'tmass':
        #df_q['MJD'] = df_q.apply(lambda r: Time(r.Date, format='isot').to_value('mjd', 'long') if pd.notnull(r.Date) else r, axis=1)
        #df_q['MJD'] = pd.to_numeric(df_q['MJD'], errors='coerce')
        #df_q['MJD'] = df_q['_tab1_36'] - 2400000.5
        
        #df_q['PU_PM_'+catalog] = df_q.apply(lambda row: row.PM_gaia*(row.MJD_2mass-X_mjd)/(365.*1e3),axis=1)
        #df_q['PU_e_PM_'+catalog] = df_q.apply(lambda row: row.e_PM_gaia*(row.MJD_2mass-X_mjd)*2/(365.*1e3),axis=1)
        
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']
        #df_q['PU'] = df_q['errMaj']
        #print(df_q[[]])
        #print(df_q[['MJD']])
        
        df_q['errMaj'] = df_q['errMaj'] * sigma
        df_q['errMin'] = df_q['errMin'] * sigma
        #print(df_q[['err0','err1']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','errMaj','errMin','errPA','_2MASS','_r','Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag',]])
        
    elif catalog == 'allwise':
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']
        
        #allwise_ref_mjd = 55400.
        #delta_yr = (ref_mjd - allwise_ref_mjd)/365.
        
        #df_q['PU_pm'] = (df_q['pmRA'].fillna(0.)*delta_yr)**2+(df_q['e_pmRA'].fillna(0.)*delta_yr)**2
        # Note concerning "proper" motions: according to AllWISE documentation, the motions measured are not proper motions 
        # because they are affected by parallax motions. More information is given in: http://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_6.html#How_To_Interpret
        df_q['eeMaj'] = df_q['eeMaj'] * sigma
        df_q['eeMin'] = df_q['eeMin'] * sigma
        
        #print(df_q[['eeMaj','eeMin','eePA','e_RA_pm','e_DE_pm']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','eeMaj','eeMin','eePA','AllWISE','_r','W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag']])
    
    elif catalog == 'catwise':
        # we use the proper motion and parallax from catwise 
        df_q['RA']  = df_q['RAPMdeg']
        df_q['DEC'] = df_q['DEPMdeg']
        #catwise_ref_mjd = 57170.
        #delta_yr = (ref_mjd - catwise_ref_mjd)/365.
        
        df_q['e_pos'] = df_q[['e_RAPMdeg', 'e_DEPMdeg']].max(axis=1)
        df_q['PM']   = np.sqrt((df_q['pmRA']*np.cos(df_q['DEPMdeg']*np.pi/180))**2+df_q['pmDE']**2)
        df_q['e_PM'] = np.sqrt((df_q['e_pmRA']*np.cos(df_q['DEPMdeg']*np.pi/180))**2+df_q['e_pmDE']**2)
        #df_q['PU_PM']   = df_q['PM']*delta_yr
        #df_q['PU_ePM']  = df_q['e_PM']*delta_yr
        df_q['PU'] = sigma * np.sqrt(df_q['e_pos'].fillna(0.)**2+df_q['plx1'].fillna(0.)**2+df_q['e_plx1'].fillna(0.)**2) 
            #(df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2)
        #print(df_q[['e_RAPMdeg','e_DEPMdeg','e_pos','PM','e_PM','plx1','e_plx1','PU_c']].describe())
        #df_q['PU'] = df_q.apply(lambda r: max(r.PU_c,0.3), axis=1)
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','Name','_r','e_pos','plx1','e_plx1','W1mproPM','W2mproPM','e_W1mproPM','e_W2mproPM']])
    
    
    #print(df_q['PU'].describe())
    #'''
    
    if plot_density_curve:
        df_q['n_match'] = df_q.index+1
        df_q['rho'] = df_q['n_match']/(np.pi*df_q['_r']**2)
        plt.plot(df_q['_r'], df_q['rho'])
        plt.yscale("log")
        #print(df_q[['n_match','_r','rho']])
        
    new_t.write(f'./{data_dir}/{X_name}_{catalog}.fits', overwrite=True)
    if catalog == 'CSC':
        area = 550./317000
    else:
        area = np.pi * (search_radius/60)**2
    #area = 550./317000
    os.system(f'python /Users/huiyang/Softwarses/nway-master/nway-write-header.py ./{data_dir}/{X_name}_{catalog}.fits {catalog} {area}')
    
    return len(df_q)
    
def nway_cross_matching_v1(TD, i, radius, query_dir, PU_col='r0',data_dir='data',explain=False,move=False,move_dir='check',rerun=False, sigma=2.):
    csc_name, ra, dec, r0 = TD.loc[i, 'name'][5:], TD.loc[i, 'ra'], TD.loc[i, 'dec'],  TD.loc[i, PU_col]
    print(csc_name, ra, dec)
    if path.exists(f'./{data_dir}/{csc_name}_nway.fits') == False or rerun==True:
        if path.exists(f'{query_dir}/{csc_name}.txt') == False:
            CSCviewsearch(csc_name, ra, dec, radius,query_dir,csc_version='2.0')
        df_r = pd.read_csv(f'{query_dir}/{csc_name}.txt', header=154, sep='\t')
        #print(len(df_r))
        #print(df_r.iloc[0]['gti_obs'])
        #t1 = Time(df_r.iloc[0]['gti_obs'], format='isot', scale='utc')
        #print(len(df_r))
        #print(df_r)
        df_r['mjd_dif'] = np.nan
        df_r['mjd_dif'] = df_r.apply(lambda r: abs(57388-Time(r['gti_obs'], format='isot', scale='utc').mjd),axis=1)
        mjd_dif_max = df_r['mjd_dif'].max()
        print('max dif:', mjd_dif_max/365.)

        #csc_name, CSC_id, r0 = df_r.loc[i, 'name'][5:], df_r.loc[i, 'ID'], TD_old.loc[i, 'r0']


        num_X = nway_mw_prepare_v1(ra, dec,  X_name=csc_name, mjd_dif=mjd_dif_max, catalog='CSC',data_dir=data_dir,sigma=sigma)

        nway_mw_prepare_v1(ra, dec,  X_name=csc_name, mjd_dif=mjd_dif_max, catalog='gaia',data_dir=data_dir,sigma=sigma)

        nway_mw_prepare_v1(ra, dec,  X_name=csc_name,catalog='tmass',data_dir=data_dir,sigma=sigma)

        nway_mw_prepare_v1(ra, dec,  X_name=csc_name, catalog='allwise',data_dir=data_dir,sigma=sigma)

        nway_mw_prepare_v1(ra, dec,  X_name=csc_name,catalog='catwise',data_dir=data_dir,sigma=sigma)

        os.system(f'python /Users/huiyang/Softwarses/nway-master/nway.py ./{data_dir}/{csc_name}_CSC.fits :err_r0:err_r1:PA \
              ./{data_dir}/{csc_name}_gaia.fits :PU ./{data_dir}/{csc_name}_tmass.fits :errMaj:errMin:errPA \
              ./{data_dir}/{csc_name}_allwise.fits :eeMaj:eeMin:eePA ./{data_dir}/{csc_name}_catwise.fits :PU \
              --out=./{data_dir}/{csc_name}_nway.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46') # 1.5 * r0 is 3-sigma
    
    if explain:

        os.system(f'python /Users/huiyang/Softwarses/nway-master/nway-explain.py ./{data_dir}/{csc_name}_nway.fits 1') 

    if move:
        Path(f'./{move_dir}/').mkdir(parents=True, exist_ok=True)
        os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./{move_dir}/')   
        os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./{move_dir}/')        
            
    return None

def nway_mw_prepare_v2(ra_x, dec_x, X_name, ref_mjd=np.array([57388.]),catalog='gaia',plot_density_curve=False,sigma=2):
    
    #catalog = 'I/355/gaiadr3'
    '''
    mjd_difs = {'gaia':X_mjd-57388.,'gaiadist':X_mjd-57388.,'2mass':max(abs(X_mjd-50600),(X_mjd-51955)),'catwise':X_mjd-57170.0,
              'unwise':max(abs(X_mjd-55203.),abs(X_mjd-55593.),abs(X_mjd-56627),abs(X_mjd-58088)),
              'allwise':max(abs(X_mjd-55203.),abs(X_mjd-55414.),abs(X_mjd-55468),abs(X_mjd-55593)),
              'vphas':max(abs(X_mjd-55923),abs(X_mjd-56536))
            }
    '''    
    viz = Vizier(row_limit=-1,  timeout=5000, columns=vizier_cols_dict[vizier_cols_dict['catalogs'][catalog]],catalog=vizier_cols_dict['catalogs'][catalog])
          
    search_radius = vizier_cols_dict['search_radius'][catalog] # arcmin, we plot the density vs radius and see it starts to converge at around 4'
    
    query = viz.query_region(SkyCoord(ra=ra_x, dec=dec_x,
                        unit=(u.deg, u.deg),frame='icrs'),
                        radius=search_radius*60*u.arcsec)
                        #,column_filters={'Gmag': '<19'}
    
    query_res = query[0]

    df_q = query_res.to_pandas()
    df_q = df_q.sort_values(by='_r').reset_index(drop=True)
    #print(df_q.columns)
    #num = len(df_q)

    #'''
    #print(df_q.columns)
    if catalog == 'CSC':
        
        df_q['RA']  = df_q['RAICRS']
        df_q['DEC'] = df_q['DEICRS']
        
        df_q['ID'] = df_q.index + 1
        df_q['err_r0'] = df_q['r0']*sigma/2
        df_q['err_r1'] = df_q['r1']*sigma/2
        
        new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_2CXO','_r','fe','fc']]) # r0 is 95%, should be consistent with other PUs, 
    
    if catalog == 'gaia':
        gaia_ref_mjd = 57388.
        delta_yr = max(abs((ref_mjd - gaia_ref_mjd)/365.))
        #delta_yr = mjd_dif/365.
        df_q['RA']  = df_q['RA_ICRS']
        df_q['DEC'] = df_q['DE_ICRS']
        #df_q['RA']  = df_q.apply(lambda row:row.RA_ICRS+delta_yr*row.pmRA/(np.cos(row.DE_ICRS*np.pi/180.)*3.6e6),axis=1)
        #df_q['DEC'] = df_q.apply(lambda row:row.DE_ICRS+delta_yr*row.pmDE/3.6e6,axis=1)
        
        #df_q.loc[df_q['RA'].isnull(),'RA']   = df_q.loc[df_q['RA'].isnull(),'RA_ICRS']
        #df_q.loc[df_q['DEC'].isnull(),'DEC'] = df_q.loc[df_q['DEC'].isnull(),'DE_ICRS']
        #df_q['PU'] = df_q.apply(lambda r: max(r.e_RA_ICRS,r.e_DE_ICRS),axis=1)
        #df_q['PU'] = df_q.apply(lambda r: max(r.e_RA_ICRS,r.e_DE_ICRS)**2+(r.PM*mjd_difs[catalog]/365)**2+(r.e_PM_gaia*mjd_difs[catalog]/365)**2,axis=1)
        #print(df_q[['e_RA_ICRS','e_DE_ICRS','Plx','e_Plx','PM','epsi','amax','RUWE']].describe())
        
        # https://www.aanda.org/articles/aa/pdf/2018/08/aa32727-18.pdf eq. B.1-B.3
        
        df_q['C00'] = df_q['e_RA_ICRS'] * df_q['e_RA_ICRS']
        df_q['C01'] = df_q['e_RA_ICRS'] * df_q['e_DE_ICRS'] * df_q['RADEcor']
        df_q['C11'] = df_q['e_DE_ICRS'] * df_q['e_DE_ICRS']
        df_q['C33'] = df_q['e_pmRA']    * df_q['e_pmRA']
        df_q['C34'] = df_q['e_pmRA']    * df_q['e_pmDE'] * df_q['pmRApmDEcor']
        df_q['C44'] = df_q['e_pmDE']    * df_q['e_pmDE']
        df_q['sigma_pos'] = np.sqrt(0.5*(df_q.C00+df_q.C11) + 0.5*np.sqrt((df_q.C11-df_q.C00)**2+4*df_q.C01**2)) 
        df_q['e_Pos'] = df_q['sigma_pos'].fillna(0.)/1e3
        df_q['sigma_pm']  = np.sqrt(0.5*(df_q.C33+df_q.C44) + 0.5*np.sqrt((df_q.C44-df_q.C33)**2+4*df_q.C34**2))
        df_q['e_PM'] = df_q['sigma_pm'].fillna(0.)/1e3
        df_q['PM'] = df_q['PM'].fillna(0.)/1e3
        df_q['epsi'] = df_q['epsi'].fillna(0.)/1e3
        df_q['Plx'] = df_q['Plx'].fillna(0.)/1e3
        df_q['e_Plx'] = df_q['e_Plx'].fillna(0.)/1e3

        df_q['PU'] = sigma * np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*delta_yr)**2+(df_q['e_PM']*delta_yr)**2+df_q['epsi']**2)
        
        #df_q['PU'] = sigma * np.sqrt((df_q['sigma_pos'].fillna(0.))**2+(df_q['Plx'].fillna(0.))**2+(df_q['e_Plx'].fillna(0.))**2 + (df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2+(df_q['epsi'].fillna(0.))**2)/1e3
             #df_q['epsi'].fillna(0.)**2)/1e3
        #df_q['PU'] = df_q.apply(lambda r: max(r.PU_c,0.2), axis=1)
        #print(df_q[['sigma_pos','Plx','e_Plx','PM','sigma_pm','epsi','PU_c','PU']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag']])

        df_q[['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi']].to_csv(f'./{data_dir}/{X_name}_gaia.csv',index=False)
    
    elif catalog == 'tmass':

        df_q['MJD'] = df_q.apply(lambda r: Time(r.Date, format='isot').to_value('mjd', 'long') if pd.notnull(r.Date) else r, axis=1)
        #df_q['MJD'] = pd.to_numeric(df_q['MJD'], errors='coerce')
        #df_q['MJD'] = df_q['_tab1_36'] - 2400000.5
        
        #df_q['PU_PM_'+catalog] = df_q.apply(lambda row: row.PM_gaia*(row.MJD_2mass-X_mjd)/(365.*1e3),axis=1)
        #df_q['PU_e_PM_'+catalog] = df_q.apply(lambda row: row.e_PM_gaia*(row.MJD_2mass-X_mjd)*2/(365.*1e3),axis=1)
        
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']

        #'''

        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            df_q['PU_gaia_2mass'] = np.sqrt((df_q['errMaj'].fillna(0))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*max(abs((df_q['MJD']-57388)/365)))**2+(df_q['e_PM']*max(abs((df_q['MJD']-57388)/365)))**2+df_q['epsi']**2)
            df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['MJD']-ref_mjd)/365.)),axis=1)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*df_q['mjd_dif_max'])**2+(df_q['e_PM']*df_q['mjd_dif_max'])**2+df_q['epsi']**2)
            df_q['err0'] = df_q.apply(lambda r:  sigma * np.sqrt(r['errMaj']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_2mass'] else sigma * r['errMaj'], axis=1)
            df_q['err1'] = df_q.apply(lambda r:  sigma * np.sqrt(r['errMin']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_2mass'] else sigma * r['errMin'], axis=1)

        else:
                
        #'''


            #df_q['PU'] = df_q['errMaj']
            #print(df_q[[]])
            #print(df_q[['MJD']])
            
            df_q['err0'] = df_q['errMaj'] * sigma
            df_q['err1'] = df_q['errMin'] * sigma
        
        #print(df_q[['err0','err1']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','err0','err1','errPA','_2MASS','_r','Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag']])
        #df_q[['RA','DEC','MJD','errMaj','errMin','errPA','_2MASS','_r']].to_csv(f'./{data_dir}/{X_name}_2mass.csv',index=False)
    elif catalog == 'allwise':
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']
        
        #allwise_ref_mjd = 55400.
        #delta_yr = (ref_mjd - allwise_ref_mjd)/365.
        
        #df_q['PU_pm'] = (df_q['pmRA'].fillna(0.)*delta_yr)**2+(df_q['e_pmRA'].fillna(0.)*delta_yr)**2
        # Note concerning "proper" motions: according to AllWISE documentation, the motions measured are not proper motions 
        # because they are affected by parallax motions. More information is given in: http://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_6.html#How_To_Interpret


        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            mjd_dif_max_gaia = max(abs(57388-55203.),abs(57388-55414.),abs(57388-55468),abs(57388-55593))/365
            mjd_dif_max_CSC = max(max(abs(ref_mjd-55203.)),max(abs(ref_mjd-55414.)),max(abs(ref_mjd-55468)),max(abs(ref_mjd-55593)))/365
            df_q['PU_gaia_allwise'] = np.sqrt((df_q['eeMaj'].fillna(0))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*mjd_dif_max_gaia)**2+(df_q['e_PM']*mjd_dif_max_gaia)**2+df_q['epsi']**2)
            #df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['MJD']-ref_mjd)/365.)),axis=1)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*mjd_dif_max_CSC)**2+(df_q['e_PM']*mjd_dif_max_CSC)**2+df_q['epsi']**2)
            df_q['err0'] = df_q.apply(lambda r:  sigma * np.sqrt(r['eeMaj']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_allwise'] else sigma * r['eeMaj'], axis=1)
            df_q['err1'] = df_q.apply(lambda r:  sigma * np.sqrt(r['eeMin']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_allwise'] else sigma * r['eeMin'], axis=1)

        else:
      
            df_q['err0'] = df_q['eeMaj'] * sigma
            df_q['err1'] = df_q['eeMin'] * sigma

        df_q = df_q.rename(columns={'eePA':'errPA'})    
        #print(df_q[['eeMaj','eeMin','eePA','e_RA_pm','e_DE_pm']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','err0','err1','errPA','AllWISE','_r','W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag']])
    
    elif catalog == 'catwise':
        # we use the proper motion and parallax from catwise 
        df_q['RA']  = df_q['RAPMdeg']
        df_q['DEC'] = df_q['DEPMdeg']
        #catwise_ref_mjd = 57170.
        #delta_yr = (ref_mjd - catwise_ref_mjd)/365.
        #print(df_q)

        df_q['e_Pos_catwise'] = df_q[['e_RAPMdeg', 'e_DEPMdeg']].max(axis=1)
        #print(df_q)

        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            df_q['PU_gaia_catwise'] = np.sqrt((df_q['e_Pos_catwise'].fillna(0))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*max(abs((df_q['_tab1_20']-57388)/365)))**2+(df_q['e_PM']*max(abs((df_q['_tab1_20']-57388)/365)))**2+df_q['epsi']**2)
            df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['_tab1_20']-ref_mjd)/365.)),axis=1)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*df_q['mjd_dif_max'])**2+(df_q['e_PM']*df_q['mjd_dif_max'])**2+df_q['epsi']**2)
            df_q['PU'] = df_q.apply(lambda r:  sigma * np.sqrt(r['e_Pos_catwise']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_catwise'] else sigma * r['e_Pos_catwise'], axis=1)

        else:
                  

        
        
            #df_q['PM']   = np.sqrt((df_q['pmRA']*np.cos(df_q['DEPMdeg']*np.pi/180))**2+df_q['pmDE']**2)
            #df_q['e_PM'] = np.sqrt((df_q['e_pmRA']*np.cos(df_q['DEPMdeg']*np.pi/180))**2+df_q['e_pmDE']**2)
            #df_q['PU_PM']   = df_q['PM']*delta_yr
            #df_q['PU_ePM']  = df_q['e_PM']*delta_yr
            df_q['PU'] = sigma * df_q['e_Pos_catwise']
                #(df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2)
        #print(df_q[['e_RAPMdeg','e_DEPMdeg','e_pos','PM','e_PM','plx1','e_plx1','PU_c']].describe())
        #df_q['PU'] = df_q.apply(lambda r: max(r.PU_c,0.3), axis=1)
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','Name','_r','W1mproPM','W2mproPM','e_W1mproPM','e_W2mproPM']])
    
    
    #print(df_q['PU'].describe())
    #'''
    
    new_t.write(f'./{data_dir}/{X_name}_{catalog}.fits', overwrite=True)

    if plot_density_curve:
        df_q['n_match'] = df_q.index+1
        df_q['rho'] = df_q['n_match']/(np.pi*df_q['_r']**2)
        plt.plot(df_q['_r'], df_q['rho'])
        plt.yscale("log")
        #print(df_q[['n_match','_r','rho']])
        
    
    if catalog == 'CSC':
        area = 550./317000
    else:
        area = np.pi * (search_radius/60)**2
    #area = 550./317000
    os.system(f'python /Users/huiyang/Softwarses/nway-master/nway-write-header.py ./{data_dir}/{X_name}_{catalog}.fits {catalog} {area}')
    
    return len(df_q)

def nway_cross_matching_v2(TD, i, radius, query_dir, explain=False,rerun=False, sigma=2.):
    csc_name, CSC_id, ra, dec, r0 = TD.loc[i, 'name'][5:], TD.loc[i, 'ID'], TD.loc[i, 'ra'], TD.loc[i, 'dec'],  TD.loc[i, 'r0']
    print(csc_name, ra, dec)
    if path.exists(f'./{data_dir}/{csc_name}_nway.fits') == False or rerun==True:
        if path.exists(f'{query_dir}/{csc_name}.txt') == False:
            CSCviewsearch(csc_name, ra, dec, radius,query_dir,csc_version='2.0')
        df_r = pd.read_csv(f'{query_dir}/{csc_name}.txt', header=154, sep='\t')
        
        #print(len(df_r))
        #print(df_r.iloc[0]['gti_obs'])
        #t1 = Time(df_r.iloc[0]['gti_obs'], format='isot', scale='utc')
        #print(len(df_r))
        #print(df_r)
        df_r['mjd'] = np.nan
        df_r['mjd'] = df_r.apply(lambda r: Time(r['gti_obs'], format='isot', scale='utc').mjd,axis=1)
        mjds = df_r['mjd'].values

        #csc_name, CSC_id, r0 = df_r.loc[i, 'name'][5:], df_r.loc[i, 'ID'], TD_old.loc[i, 'r0']


        num_X = nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='CSC',sigma=sigma)

        nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='gaia',sigma=sigma)

        nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='tmass',sigma=sigma)

        nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='allwise',sigma=sigma)

        nway_mw_prepare(ra, dec,  X_name=csc_name,ref_mjd=mjds, catalog='catwise',sigma=sigma)

        os.system(f'python /Users/huiyang/Softwarses/nway-master/nway.py ./{data_dir}/{csc_name}_CSC.fits :err_r0:err_r1:PA \
              ./{data_dir}/{csc_name}_gaia.fits :PU ./{data_dir}/{csc_name}_tmass.fits :err0:err0:errPA \
              ./{data_dir}/{csc_name}_allwise.fits :err0:err1:errPA ./{data_dir}/{csc_name}_catwise.fits :PU \
              --out=./{data_dir}/{csc_name}_nway.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46') # 1.5 * r0 is 3-sigma
    
    if explain:

        os.system(f'python /Users/huiyang/Softwarses/nway-master/nway-explain.py ./{data_dir}/{csc_name}_nway.fits 1') 

    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./LM-STARs_update_check/')   
    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./LM-STARs_update_check/')    
        
    return None


def nway_mw_prepare_v3(ra_x, dec_x, X_name, ref_mjd=np.array([57388.]),catalog='gaia',data_dir='data',plot_density_curve=False,sigma=2):
    
    #catalog = 'I/355/gaiadr3'
    '''
    mjd_difs = {'gaia':X_mjd-57388.,'gaiadist':X_mjd-57388.,'2mass':max(abs(X_mjd-50600),(X_mjd-51955)),'catwise':X_mjd-57170.0,
              'unwise':max(abs(X_mjd-55203.),abs(X_mjd-55593.),abs(X_mjd-56627),abs(X_mjd-58088)),
              'allwise':max(abs(X_mjd-55203.),abs(X_mjd-55414.),abs(X_mjd-55468),abs(X_mjd-55593)),
              'vphas':max(abs(X_mjd-55923),abs(X_mjd-56536))
            }
    '''    
    viz = Vizier(row_limit=-1,  timeout=5000, columns=vizier_cols_dict[vizier_cols_dict['catalogs'][catalog]],catalog=vizier_cols_dict['catalogs'][catalog])
          
    search_radius = vizier_cols_dict['search_radius'][catalog] # arcmin, we plot the density vs radius and see it starts to converge at around 4'
    
    query = viz.query_region(SkyCoord(ra=ra_x, dec=dec_x,
                        unit=(u.deg, u.deg),frame='icrs'),
                        radius=search_radius*60*u.arcsec)
                        #,column_filters={'Gmag': '<19'}
    
    query_res = query[0]

    df_q = query_res.to_pandas()
    df_q = df_q.sort_values(by='_r').reset_index(drop=True)
    #df_q = df_q[~np.isnan(df_q['_r'])].reset_index(drop=True)
    #print(df_q.dtypes)
    #print(df_q.columns)
    #num = len(df_q)

    #'''
    #print(df_q.columns)
    if catalog == 'CSC':
        
        df_q['RA']  = df_q['RAICRS']
        df_q['DEC'] = df_q['DEICRS']
        
        df_q['ID'] = df_q.index + 1
        df_q['err_r0'] = df_q['r0']*sigma/2
        df_q['err_r1'] = df_q['r1']*sigma/2
        
        new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_2CXO','_r','fe','fc']]) # r0 is 95%, should be consistent with other PUs, 
    
    if catalog == 'gaia':
        gaia_ref_mjd = 57388.
        delta_yr = max(abs((ref_mjd - gaia_ref_mjd)/365.))
        #delta_yr = mjd_dif/365.
        df_q['RA']  = df_q['RA_ICRS']
        df_q['DEC'] = df_q['DE_ICRS']
        #df_q['RA']  = df_q.apply(lambda row:row.RA_ICRS+delta_yr*row.pmRA/(np.cos(row.DE_ICRS*np.pi/180.)*3.6e6),axis=1)
        #df_q['DEC'] = df_q.apply(lambda row:row.DE_ICRS+delta_yr*row.pmDE/3.6e6,axis=1)
        
        #df_q.loc[df_q['RA'].isnull(),'RA']   = df_q.loc[df_q['RA'].isnull(),'RA_ICRS']
        #df_q.loc[df_q['DEC'].isnull(),'DEC'] = df_q.loc[df_q['DEC'].isnull(),'DE_ICRS']
        #df_q['PU'] = df_q.apply(lambda r: max(r.e_RA_ICRS,r.e_DE_ICRS),axis=1)
        #df_q['PU'] = df_q.apply(lambda r: max(r.e_RA_ICRS,r.e_DE_ICRS)**2+(r.PM*mjd_difs[catalog]/365)**2+(r.e_PM_gaia*mjd_difs[catalog]/365)**2,axis=1)
        #print(df_q[['e_RA_ICRS','e_DE_ICRS','Plx','e_Plx','PM','epsi','amax','RUWE']].describe())
        
        # https://www.aanda.org/articles/aa/pdf/2018/08/aa32727-18.pdf eq. B.1-B.3
        
        df_q['C00'] = df_q['e_RA_ICRS'] * df_q['e_RA_ICRS']
        df_q['C01'] = df_q['e_RA_ICRS'] * df_q['e_DE_ICRS'] * df_q['RADEcor']
        df_q['C11'] = df_q['e_DE_ICRS'] * df_q['e_DE_ICRS']
        df_q['C33'] = df_q['e_pmRA']    * df_q['e_pmRA']
        df_q['C34'] = df_q['e_pmRA']    * df_q['e_pmDE'] * df_q['pmRApmDEcor']
        df_q['C44'] = df_q['e_pmDE']    * df_q['e_pmDE']
        df_q['sigma_pos'] = np.sqrt(0.5*(df_q.C00+df_q.C11) + 0.5*np.sqrt((df_q.C11-df_q.C00)**2+4*df_q.C01**2)) 
        df_q['e_Pos'] = df_q['sigma_pos'].fillna(0.)/1e3
        df_q['sigma_pm']  = np.sqrt(0.5*(df_q.C33+df_q.C44) + 0.5*np.sqrt((df_q.C44-df_q.C33)**2+4*df_q.C34**2))
        df_q['e_PM'] = df_q['sigma_pm'].fillna(0.)/1e3
        df_q['PM'] = df_q['PM'].fillna(0.)/1e3
        df_q['epsi'] = df_q['epsi'].fillna(0.)/1e3
        df_q['Plx'] = df_q['Plx'].fillna(0.)/1e3
        df_q['e_Plx'] = df_q['e_Plx'].fillna(0.)/1e3

        df_q['PU'] = sigma * np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*delta_yr)**2+(df_q['e_PM']*delta_yr)**2+df_q['epsi']**2)
        
        #df_q['PU'] = sigma * np.sqrt((df_q['sigma_pos'].fillna(0.))**2+(df_q['Plx'].fillna(0.))**2+(df_q['e_Plx'].fillna(0.))**2 + (df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2+(df_q['epsi'].fillna(0.))**2)/1e3
             #df_q['epsi'].fillna(0.)**2)/1e3
        #df_q['PU'] = df_q.apply(lambda r: max(r.PU_c,0.2), axis=1)
        #print(df_q[['sigma_pos','Plx','e_Plx','PM','sigma_pm','epsi','PU_c','PU']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag']])
        #print(df_q[~df_q['RA'].isnull()][['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi']])
        df_q[['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi']].to_csv(f'./{data_dir}/{X_name}_gaia.csv',index=False)
    
    elif catalog == 'tmass':

        df_q['MJD'] = df_q.apply(lambda r: Time(r.Date, format='isot').to_value('mjd', 'long') if pd.notnull(r.Date) else r, axis=1)
        #df_q['MJD'] = pd.to_numeric(df_q['MJD'], errors='coerce')
        #df_q['MJD'] = df_q['_tab1_36'] - 2400000.5
        
        #df_q['PU_PM_'+catalog] = df_q.apply(lambda row: row.PM_gaia*(row.MJD_2mass-X_mjd)/(365.*1e3),axis=1)
        #df_q['PU_e_PM_'+catalog] = df_q.apply(lambda row: row.e_PM_gaia*(row.MJD_2mass-X_mjd)*2/(365.*1e3),axis=1)
        
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)

        #'''

        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            df_gaia = df_gaia[~df_gaia['RA'].isnull()].reset_index(drop=True)
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            #print(c, cata)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            df_q['PU_gaia_2mass'] = np.sqrt((df_q['errMaj'].fillna(0))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*max(abs((df_q['MJD']-57388)/365)))**2+(df_q['e_PM']*max(abs((df_q['MJD']-57388)/365)))**2+df_q['epsi']**2)
            df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['MJD']-ref_mjd)/365.)),axis=1)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*df_q['mjd_dif_max'])**2+(df_q['e_PM']*df_q['mjd_dif_max'])**2+df_q['epsi']**2)
            df_q['err0'] = df_q.apply(lambda r:  sigma * np.sqrt(r['errMaj']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_2mass'] else sigma * r['errMaj'], axis=1)
            df_q['err1'] = df_q.apply(lambda r:  sigma * np.sqrt(r['errMin']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_2mass'] else sigma * r['errMin'], axis=1)

        else:
                
        #'''


            #df_q['PU'] = df_q['errMaj']
            #print(df_q[[]])
            #print(df_q[['MJD']])
            
            df_q['err0'] = df_q['errMaj'] * sigma
            df_q['err1'] = df_q['errMin'] * sigma
        
        #print(df_q[['err0','err1']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','err0','err1','errPA','_2MASS','_r','Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag']])
        #df_q[['RA','DEC','MJD','errMaj','errMin','errPA','_2MASS','_r']].to_csv(f'./{data_dir}/{X_name}_2mass.csv',index=False)
    elif catalog == 'allwise':
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)
        
        #allwise_ref_mjd = 55400.
        #delta_yr = (ref_mjd - allwise_ref_mjd)/365.
        
        #df_q['PU_pm'] = (df_q['pmRA'].fillna(0.)*delta_yr)**2+(df_q['e_pmRA'].fillna(0.)*delta_yr)**2
        # Note concerning "proper" motions: according to AllWISE documentation, the motions measured are not proper motions 
        # because they are affected by parallax motions. More information is given in: http://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_6.html#How_To_Interpret


        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            df_gaia = df_gaia[~df_gaia['RA'].isnull()].reset_index(drop=True)
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            #print(c, cata)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            mjd_dif_max_gaia = max(abs(57388-55203.),abs(57388-55414.),abs(57388-55468),abs(57388-55593))/365
            mjd_dif_max_CSC = max(max(abs(ref_mjd-55203.)),max(abs(ref_mjd-55414.)),max(abs(ref_mjd-55468)),max(abs(ref_mjd-55593)))/365
            df_q['PU_gaia_allwise'] = np.sqrt((df_q['eeMaj'].fillna(0))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*mjd_dif_max_gaia)**2+(df_q['e_PM']*mjd_dif_max_gaia)**2+df_q['epsi']**2)
            #df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['MJD']-ref_mjd)/365.)),axis=1)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*mjd_dif_max_CSC)**2+(df_q['e_PM']*mjd_dif_max_CSC)**2+df_q['epsi']**2)
            df_q['err0'] = df_q.apply(lambda r:  sigma * np.sqrt(r['eeMaj']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_allwise'] else sigma * r['eeMaj'], axis=1)
            df_q['err1'] = df_q.apply(lambda r:  sigma * np.sqrt(r['eeMin']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_allwise'] else sigma * r['eeMin'], axis=1)

        else:
      
            df_q['err0'] = df_q['eeMaj'] * sigma
            df_q['err1'] = df_q['eeMin'] * sigma

        df_q = df_q.rename(columns={'eePA':'errPA'})    
        #print(df_q[['eeMaj','eeMin','eePA','e_RA_pm','e_DE_pm']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','err0','err1','errPA','AllWISE','_r','W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag']])
    
    elif catalog == 'catwise':
        # we use the proper motion and parallax from catwise 
        df_q['RA']  = df_q['RAPMdeg']
        df_q['DEC'] = df_q['DEPMdeg']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)
        #catwise_ref_mjd = 57170.
        #delta_yr = (ref_mjd - catwise_ref_mjd)/365.
        #print(df_q)
        #print(df_q)
        df_q['e_Pos_catwise'] = df_q[['e_RAPMdeg', 'e_DEPMdeg']].max(axis=1)

        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            df_gaia = df_gaia[~df_gaia['RA'].isnull()].reset_index(drop=True)
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            df_q['PU_gaia_catwise'] = np.sqrt((df_q['e_Pos_catwise'].fillna(0))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*max(abs((df_q['_tab1_20']-57388)/365)))**2+(df_q['e_PM']*max(abs((df_q['_tab1_20']-57388)/365)))**2+df_q['epsi']**2)
            df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['_tab1_20']-ref_mjd)/365.)),axis=1)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*df_q['mjd_dif_max'])**2+(df_q['e_PM']*df_q['mjd_dif_max'])**2+df_q['epsi']**2)
            df_q['PU'] = df_q.apply(lambda r:  sigma * np.sqrt(r['e_Pos_catwise']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_catwise'] else sigma * r['e_Pos_catwise'], axis=1)

        else:
                  

        
        
            #df_q['PM']   = np.sqrt((df_q['pmRA']*np.cos(df_q['DEPMdeg']*np.pi/180))**2+df_q['pmDE']**2)
            #df_q['e_PM'] = np.sqrt((df_q['e_pmRA']*np.cos(df_q['DEPMdeg']*np.pi/180))**2+df_q['e_pmDE']**2)
            #df_q['PU_PM']   = df_q['PM']*delta_yr
            #df_q['PU_ePM']  = df_q['e_PM']*delta_yr
            df_q['PU'] = sigma * df_q['e_Pos_catwise']
                #(df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2)
        #print(df_q[['e_RAPMdeg','e_DEPMdeg','e_pos','PM','e_PM','plx1','e_plx1','PU_c']].describe())
        #df_q['PU'] = df_q.apply(lambda r: max(r.PU_c,0.3), axis=1)
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','Name','_r','W1mproPM','W2mproPM','e_W1mproPM','e_W2mproPM']])
    
    
    #print(df_q['PU'].describe())
    #'''
    
    new_t.write(f'./{data_dir}/{X_name}_{catalog}.fits', overwrite=True)

    if plot_density_curve:
        df_q['n_match'] = df_q.index+1
        df_q['rho'] = df_q['n_match']/(np.pi*df_q['_r']**2)
        plt.plot(df_q['_r'], df_q['rho'])
        plt.yscale("log")
        #print(df_q[['n_match','_r','rho']])
        
    
    if catalog == 'CSC':
        area = 550./317000
    else:
        area = np.pi * (search_radius/60)**2
    #area = 550./317000
    os.system(f'python {nway_dir}nway-write-header.py ./{data_dir}/{X_name}_{catalog}.fits {catalog} {area}')
    
    return len(df_q)

def nway_cross_matching_v4(TD, i, radius, query_dir, PU_col='err_ellipse_r0',data_dir='data',explain=False,move=False,move_dir='check',rerun=False, sigma=2.):
    csc_name, ra, dec, r0 = TD.loc[i, 'name'][5:], TD.loc[i, 'ra'], TD.loc[i, 'dec'],  TD.loc[i, PU_col]#'r0']#err_ellipse_r0']#r0']
    #print(csc_name, ra, dec)
    try:
        if path.exists(f'./{data_dir}/{csc_name}_nway.fits') == False or rerun==True:
            print(csc_name, ra, dec)
            if path.exists(f'{query_dir}/{csc_name}.txt') == False:
                CSCviewsearch(csc_name, ra, dec, radius,query_dir,csc_version='2.0')
            df_r = pd.read_csv(f'{query_dir}/{csc_name}.txt', header=154, sep='\t')
            
            #print(len(df_r))
            #print(df_r.iloc[0]['gti_obs'])
            #t1 = Time(df_r.iloc[0]['gti_obs'], format='isot', scale='utc')
            #print(len(df_r))
            #print(df_r)
            df_r['mjd'] = np.nan
            df_r['mjd'] = df_r.apply(lambda r: Time(r['gti_obs'], format='isot', scale='utc').mjd,axis=1)
            mjds = df_r['mjd'].values
            print(mjds)
            #csc_name, CSC_id, r0 = df_r.loc[i, 'name'][5:], df_r.loc[i, 'ID'], TD_old.loc[i, 'r0']


            num_X = nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='CSC',data_dir=data_dir,sigma=sigma)

            nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='gaia',data_dir=data_dir,sigma=sigma)

            nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='tmass',data_dir=data_dir,sigma=sigma)

            nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='allwise',data_dir=data_dir,sigma=sigma)

            nway_mw_prepare(ra, dec,  X_name=csc_name,ref_mjd=mjds, catalog='catwise',data_dir=data_dir,sigma=sigma)

            os.system(f'python {nway_dir}nway.py ./{data_dir}/{csc_name}_CSC.fits :err_r0:err_r1:PA \
                ./{data_dir}/{csc_name}_gaia.fits :PU ./{data_dir}/{csc_name}_tmass.fits :err0:err0:errPA \
                ./{data_dir}/{csc_name}_allwise.fits :err0:err1:errPA ./{data_dir}/{csc_name}_catwise.fits :PU \
                --out=./{data_dir}/{csc_name}_nway.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46') # r0 is 2-sigma
                #--out=./{data_dir}/{csc_name}_nway.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46') # 1.5 * r0 is 3-sigma
        
        if explain:

            os.system(f'python {nway_dir}nway-explain.py ./{data_dir}/{csc_name}_nway.fits 1') 

        if move:
            Path(f'./{move_dir}/').mkdir(parents=True, exist_ok=True)
            os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./{move_dir}/')   
            os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./{move_dir}/')        
                

        #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./AGN_check/')   
        #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./AGN_check/')    

        #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./AGN_check/radiusr0/')   
        #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./AGN_check/radiusr0/')    
    except:
        print(csc_name) 
        with open(f'./error/{csc_name}.txt', 'w') as fp:
            pass
        pass
    return None


def newcsc_prepare(df_q,X_name,name_col='name',ra_col='ra', dec_col='dec',r0_col='r0',r1_col='r1',PA_col='PA',data_dir='data',sigma=2):

    df_q['_2CXO'] = df_q[name_col]
    df_q['RA']  = df_q[ra_col]
    df_q['DEC'] = df_q[dec_col]
    
    df_q['ID'] = df_q.index + 1
    df_q['err_r0'] = df_q[r0_col]*sigma/2
    df_q['err_r1'] = df_q[r1_col]*sigma/2
    df_q['PA'] = df_q[PA_col]
    
    new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_2CXO']]) # r0 is 95%, should be consistent with other PUs, 

    new_t.write(f'./{data_dir}/{X_name}_CSC.fits', overwrite=True)

    area = 550./317000
    
    os.system(f'python {nway_dir}nway-write-header.py ./{data_dir}/{X_name}_CSC.fits CSC {area}')
    
    return None

def nway_mw_prepare(ra_x, dec_x, X_name, ref_mjd=np.array([57388.]),catalog='gaia',data_dir='data',plot_density_curve=False,sigma=2):
    
    #catalog = 'I/355/gaiadr3'
    '''
    mjd_difs = {'gaia':X_mjd-57388.,'gaiadist':X_mjd-57388.,'2mass':max(abs(X_mjd-50600),(X_mjd-51955)),'catwise':X_mjd-57170.0,
              'unwise':max(abs(X_mjd-55203.),abs(X_mjd-55593.),abs(X_mjd-56627),abs(X_mjd-58088)),
              'allwise':max(abs(X_mjd-55203.),abs(X_mjd-55414.),abs(X_mjd-55468),abs(X_mjd-55593)),
              'vphas':max(abs(X_mjd-55923),abs(X_mjd-56536))
            }
    '''    
    viz = Vizier(row_limit=-1,  timeout=5000, columns=vizier_cols_dict[vizier_cols_dict['catalogs'][catalog]],catalog=vizier_cols_dict['catalogs'][catalog])
          
    search_radius = vizier_cols_dict['search_radius'][catalog] # arcmin, we plot the density vs radius and see it starts to converge at around 4'
    
    query = viz.query_region(SkyCoord(ra=ra_x, dec=dec_x,
                        unit=(u.deg, u.deg),frame='icrs'),
                        radius=search_radius*60*u.arcsec)
                        #,column_filters={'Gmag': '<19'}
    #print(catalog, ra_x, dec_x, search_radius*60)
    query_res = query[0]

    df_q = query_res.to_pandas()
    df_q = df_q.sort_values(by='_r').reset_index(drop=True)
    #df_q = df_q[~np.isnan(df_q['_r'])].reset_index(drop=True)
    #print(df_q.dtypes)
    #print(df_q.columns)
    #num = len(df_q)

    #'''
    #print(df_q.columns)
    if catalog == 'CSC':
        
        df_q['RA']  = df_q['RAICRS']
        df_q['DEC'] = df_q['DEICRS']
        
        df_q['ID'] = df_q.index + 1
        df_q['err_r0'] = df_q['r0']*sigma/2
        df_q['err_r1'] = df_q['r1']*sigma/2
        
        new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_2CXO','_r','fe','fc']]) # r0 is 95%, should be consistent with other PUs, 
    
    if catalog == 'gaia':
        gaia_ref_mjd = 57388.
        delta_yr = max(abs((ref_mjd - gaia_ref_mjd)/365.))
        #delta_yr = mjd_dif/365.
        df_q['RA']  = df_q['RA_ICRS']
        df_q['DEC'] = df_q['DE_ICRS']
        #df_q['RA']  = df_q.apply(lambda row:row.RA_ICRS+delta_yr*row.pmRA/(np.cos(row.DE_ICRS*np.pi/180.)*3.6e6),axis=1)
        #df_q['DEC'] = df_q.apply(lambda row:row.DE_ICRS+delta_yr*row.pmDE/3.6e6,axis=1)
        
        #df_q.loc[df_q['RA'].isnull(),'RA']   = df_q.loc[df_q['RA'].isnull(),'RA_ICRS']
        #df_q.loc[df_q['DEC'].isnull(),'DEC'] = df_q.loc[df_q['DEC'].isnull(),'DE_ICRS']
        #df_q['PU'] = df_q.apply(lambda r: max(r.e_RA_ICRS,r.e_DE_ICRS),axis=1)
        #df_q['PU'] = df_q.apply(lambda r: max(r.e_RA_ICRS,r.e_DE_ICRS)**2+(r.PM*mjd_difs[catalog]/365)**2+(r.e_PM_gaia*mjd_difs[catalog]/365)**2,axis=1)
        #print(df_q[['e_RA_ICRS','e_DE_ICRS','Plx','e_Plx','PM','epsi','amax','RUWE']].describe())
        
        # https://www.aanda.org/articles/aa/pdf/2018/08/aa32727-18.pdf eq. B.1-B.3
        
        df_q['C00'] = df_q['e_RA_ICRS'] * df_q['e_RA_ICRS']
        df_q['C01'] = df_q['e_RA_ICRS'] * df_q['e_DE_ICRS'] * df_q['RADEcor']
        df_q['C11'] = df_q['e_DE_ICRS'] * df_q['e_DE_ICRS']
        df_q['C33'] = df_q['e_pmRA']    * df_q['e_pmRA']
        df_q['C34'] = df_q['e_pmRA']    * df_q['e_pmDE'] * df_q['pmRApmDEcor']
        df_q['C44'] = df_q['e_pmDE']    * df_q['e_pmDE']
        df_q['sigma_pos'] = np.sqrt(0.5*(df_q.C00+df_q.C11) + 0.5*np.sqrt((df_q.C11-df_q.C00)**2+4*df_q.C01**2)) 
        df_q['e_Pos'] = df_q['sigma_pos'].fillna(0.)/1e3
        df_q['sigma_pm']  = np.sqrt(0.5*(df_q.C33+df_q.C44) + 0.5*np.sqrt((df_q.C44-df_q.C33)**2+4*df_q.C34**2))
        df_q['e_PM'] = df_q['sigma_pm'].fillna(0.)/1e3
        df_q['PM'] = df_q['PM'].fillna(0.)/1e3
        df_q['epsi'] = df_q['epsi'].fillna(0.)/1e3
        df_q['Plx'] = df_q['Plx'].fillna(0.)/1e3
        df_q['e_Plx'] = df_q['e_Plx'].fillna(0.)/1e3

        df_q['PU'] = sigma * np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*delta_yr)**2+(df_q['e_PM']*delta_yr)**2+df_q['epsi']**2)
        
        #df_q['PU'] = sigma * np.sqrt((df_q['sigma_pos'].fillna(0.))**2+(df_q['Plx'].fillna(0.))**2+(df_q['e_Plx'].fillna(0.))**2 + (df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2+(df_q['epsi'].fillna(0.))**2)/1e3
             #df_q['epsi'].fillna(0.)**2)/1e3
        #df_q['PU'] = df_q.apply(lambda r: max(r.PU_c,0.2), axis=1)
        #print(df_q[['sigma_pos','Plx','e_Plx','PM','sigma_pm','epsi','PU_c','PU']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag']])
        #print(df_q[~df_q['RA'].isnull()][['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi']])
        df_q[['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi']].to_csv(f'./{data_dir}/{X_name}_gaia.csv',index=False)
    
    elif catalog == 'tmass':

        df_q['MJD'] = df_q.apply(lambda r: Time(r.Date, format='isot').to_value('mjd', 'long') if pd.notnull(r.Date) else r, axis=1)
        #df_q['MJD'] = pd.to_numeric(df_q['MJD'], errors='coerce')
        #df_q['MJD'] = df_q['_tab1_36'] - 2400000.5
        
        #df_q['PU_PM_'+catalog] = df_q.apply(lambda row: row.PM_gaia*(row.MJD_2mass-X_mjd)/(365.*1e3),axis=1)
        #df_q['PU_e_PM_'+catalog] = df_q.apply(lambda row: row.e_PM_gaia*(row.MJD_2mass-X_mjd)*2/(365.*1e3),axis=1)
        
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)

        #'''

        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            df_gaia = df_gaia[~df_gaia['RA'].isnull()].reset_index(drop=True)
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            #print(c, cata)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            df_q['PU_gaia_2mass'] = np.sqrt((df_q['errMaj'].fillna(0))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*max(abs((df_q['MJD']-57388)/365)))**2+(df_q['e_PM']*max(abs((df_q['MJD']-57388)/365)))**2+df_q['epsi']**2)
            df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['MJD']-ref_mjd)/365.)),axis=1)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*df_q['mjd_dif_max'])**2+(df_q['e_PM']*df_q['mjd_dif_max'])**2+df_q['epsi']**2)
            df_q['err0'] = df_q.apply(lambda r:  sigma * np.sqrt(r['errMaj']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_2mass'] else sigma * r['errMaj'], axis=1)
            df_q['err1'] = df_q.apply(lambda r:  sigma * np.sqrt(r['errMin']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_2mass'] else sigma * r['errMin'], axis=1)

        else:
                
        #'''


            #df_q['PU'] = df_q['errMaj']
            #print(df_q[[]])
            #print(df_q[['MJD']])
            
            df_q['err0'] = df_q['errMaj'] * sigma
            df_q['err1'] = df_q['errMin'] * sigma
        
        #print(df_q[['err0','err1']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','err0','err1','errPA','_2MASS','_r','Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag']])
        #df_q[['RA','DEC','MJD','errMaj','errMin','errPA','_2MASS','_r']].to_csv(f'./{data_dir}/{X_name}_2mass.csv',index=False)
    elif catalog == 'allwise':
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)
        
        #allwise_ref_mjd = 55400.
        #delta_yr = (ref_mjd - allwise_ref_mjd)/365.
        
        #df_q['PU_pm'] = (df_q['pmRA'].fillna(0.)*delta_yr)**2+(df_q['e_pmRA'].fillna(0.)*delta_yr)**2
        # Note concerning "proper" motions: according to AllWISE documentation, the motions measured are not proper motions 
        # because they are affected by parallax motions. More information is given in: http://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_6.html#How_To_Interpret


        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            df_gaia = df_gaia[~df_gaia['RA'].isnull()].reset_index(drop=True)
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            #print(c, cata)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            mjd_dif_max_gaia = max(abs(57388-55203.),abs(57388-55414.),abs(57388-55468),abs(57388-55593))/365
            mjd_dif_max_CSC = max(max(abs(ref_mjd-55203.)),max(abs(ref_mjd-55414.)),max(abs(ref_mjd-55468)),max(abs(ref_mjd-55593)))/365
            df_q['PU_gaia_allwise'] = np.sqrt((df_q['eeMaj'].fillna(0))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*mjd_dif_max_gaia)**2+(df_q['e_PM']*mjd_dif_max_gaia)**2+df_q['epsi']**2)
            #df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['MJD']-ref_mjd)/365.)),axis=1)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*mjd_dif_max_CSC)**2+(df_q['e_PM']*mjd_dif_max_CSC)**2+df_q['epsi']**2)
            df_q['err0'] = df_q.apply(lambda r:  sigma * np.sqrt(r['eeMaj']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_allwise'] else sigma * r['eeMaj'], axis=1)
            df_q['err1'] = df_q.apply(lambda r:  sigma * np.sqrt(r['eeMin']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_allwise'] else sigma * r['eeMin'], axis=1)

        else:
      
            df_q['err0'] = df_q['eeMaj'] * sigma
            df_q['err1'] = df_q['eeMin'] * sigma

        df_q = df_q.rename(columns={'eePA':'errPA'})    
        #print(df_q[['eeMaj','eeMin','eePA','e_RA_pm','e_DE_pm']].describe())
        new_t = Table.from_pandas(df_q[['RA','DEC','err0','err1','errPA','AllWISE','_r','W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag']])
    
    elif catalog == 'catwise':
        # we use the proper motion and parallax from catwise 
        df_q['RA']  = df_q['RAPMdeg']
        df_q['DEC'] = df_q['DEPMdeg']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)
        df_q['PM_catwise'] = np.sqrt(df_q['pmRA'].fillna(0.)**2+df_q['pmDE'].fillna(0.)**2)
        df_q['e_PM_catwise'] = np.sqrt(df_q['e_pmRA'].fillna(0.)**2+df_q['e_pmDE'].fillna(0.)**2)
        #catwise_ref_mjd = 57170.
        #delta_yr = (ref_mjd - catwise_ref_mjd)/365.
        #print(df_q)
        #print(df_q)
        df_q['e_Pos_catwise'] = df_q[['e_RAPMdeg', 'e_DEPMdeg']].max(axis=1)
        df_q['mjd_dif_max'] = df_q.apply(lambda r:  max(abs((r['_tab1_20']-ref_mjd)/365.)),axis=1)
        df_q['PU_catwise'] = np.sqrt((df_q['e_Pos_catwise'].fillna(0))**2+(df_q['plx1'].fillna(0))**2+(df_q['e_plx1'].fillna(0))**2+(df_q['PM_catwise']*df_q['mjd_dif_max'])**2+(df_q['e_PM_catwise']*df_q['mjd_dif_max'])**2)
                

        if path.exists(f'./{data_dir}/{X_name}_gaia.csv') == True:
            df_gaia = pd.read_csv(f'./{data_dir}/{X_name}_gaia.csv')
            df_gaia = df_gaia[~df_gaia['RA'].isnull()].reset_index(drop=True)
            #df_gaia = df_gaia[df_gaia['_r']<gaia_search_radius].reset_index(drop=True)
            #if len(df_gaia)>0:
            c = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree)
            cata = SkyCoord(ra=df_gaia['RA']*u.degree, dec=df_gaia['DEC']*u.degree)
            idx, d2d, d3d = c.match_to_catalog_sky(cata)
            df_q['_r_gaia'] = d2d.arcsec
            for col in ['e_Pos', 'Plx', 'e_Plx', 'PM', 'e_PM', 'epsi']:
                df_q[col] = df_gaia.loc[idx, col].reset_index(drop=True)

            df_q['PU_gaia_catwise'] = np.sqrt((df_q['e_Pos_catwise'].fillna(0))**2+(df_q['plx1'].fillna(0))**2+(df_q['e_plx1'].fillna(0))**2+(df_q['PM_catwise']*max(abs((df_q['_tab1_20']-57388)/365)))**2+(df_q['e_PM_catwise']*max(abs((df_q['_tab1_20']-57388)/365)))**2+df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*max(abs((df_q['_tab1_20']-57388)/365)))**2+(df_q['e_PM']*max(abs((df_q['_tab1_20']-57388)/365)))**2+df_q['epsi']**2)
            df_q['PU_gaia_X'] = np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*df_q['mjd_dif_max'])**2+(df_q['e_PM']*df_q['mjd_dif_max'])**2+df_q['epsi']**2)
            df_q['PU'] = df_q.apply(lambda r:  sigma * np.sqrt(r['PU_catwise']**2+r['PU_gaia_X']**2) if r['_r_gaia']<= sigma * r['PU_gaia_catwise'] else sigma * r['PU_catwise'], axis=1)

        else:
                  

        
        
            #df_q['PM']   = np.sqrt((df_q['pmRA']*np.cos(df_q['DEPMdeg']*np.pi/180))**2+df_q['pmDE']**2)
            #df_q['e_PM'] = np.sqrt((df_q['e_pmRA']*np.cos(df_q['DEPMdeg']*np.pi/180))**2+df_q['e_pmDE']**2)
            #df_q['PU_PM']   = df_q['PM']*delta_yr
            #df_q['PU_ePM']  = df_q['e_PM']*delta_yr
            df_q['PU'] = sigma * df_q['PU_catwise']
                #(df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2)
        #print(df_q[['e_RAPMdeg','e_DEPMdeg','e_pos','PM','e_PM','plx1','e_plx1','PU_c']].describe())
        #df_q['PU'] = df_q.apply(lambda r: max(r.PU_c,0.3), axis=1)
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','Name','_r','W1mproPM','W2mproPM','e_W1mproPM','e_W2mproPM']])
    
    
    #print(df_q['PU'].describe())
    #'''
    
    new_t.write(f'./{data_dir}/{X_name}_{catalog}.fits', overwrite=True)

    if plot_density_curve:
        df_q['n_match'] = df_q.index+1
        df_q['rho'] = df_q['n_match']/(np.pi*df_q['_r']**2)
        plt.plot(df_q['_r'], df_q['rho'])
        plt.yscale("log")
        #print(df_q[['n_match','_r','rho']])
        
    
    if catalog == 'CSC':
        area = 550./317000
    else:
        area = np.pi * (search_radius/60)**2
    #area = 550./317000
    os.system(f'python {nway_dir}nway-write-header.py ./{data_dir}/{X_name}_{catalog}.fits {catalog} {area}')
    
    return len(df_q)

def nway_cross_matching(TD, i, radius, query_dir, name_col='name',ra_col='ra',dec_col='dec', ra_csc_col='ra',dec_csc_col='dec',PU_col='err_ellipse_r0',r0_col='r0',r1_col='r1',PA_col='PA',data_dir='data',explain=False,move=False,move_dir='check',rerun=False, sigma=2.,newcsc=False,per_file='txt'):
    csc_name, ra, dec,ra_csc,dec_csc, r0 = TD.loc[i, name_col][5:], TD.loc[i, ra_col], TD.loc[i, dec_col], TD.loc[i, ra_csc_col], TD.loc[i, dec_csc_col], TD.loc[i, PU_col]#'r0']#err_ellipse_r0']#r0']
    #print(csc_name, ra, dec)
    if path.exists(f'./{data_dir}/{csc_name}_nway.fits') == False or rerun==True:
        if type(per_file) == pd.DataFrame:
            df_r = per_file[per_file['name']==TD.loc[i, name_col]].reset_index(drop=True)   
        elif type(per_file) == str and per_file == 'txt':
            #print('txt')
            if path.exists(f'{query_dir}/{csc_name}.txt') == False:
                CSCviewsearch(csc_name, ra_csc, dec_csc, radius,query_dir,csc_version='2.0')
            df_r = pd.read_csv(f'{query_dir}/{csc_name}.txt', header=154, sep='\t')
        else:
            print('else')
              

        df_r['mjd'] = np.nan
        df_r['mjd'] = df_r.apply(lambda r: Time(r['gti_obs'], format='isot', scale='utc').mjd,axis=1)
        mjds = df_r['mjd'].values

        #csc_name, CSC_id, r0 = df_r.loc[i, 'name'][5:], df_r.loc[i, 'ID'], TD_old.loc[i, 'r0']

        if newcsc:
            newcsc_prepare(TD.iloc[[i]].reset_index(drop=True),X_name=csc_name,name_col=name_col,ra_col=ra_col, dec_col=dec_col,r0_col=r0_col,r1_col=r1_col,PA_col=PA_col,data_dir=data_dir,sigma=2)
        else:
            nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='CSC',data_dir=data_dir,sigma=sigma)

        
        for mw_cat in ['gaia', 'tmass', 'allwise', 'catwise']:
            if path.exists(f'./{data_dir}/{csc_name}_{mw_cat}.fits') == False:
                nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog=mw_cat,data_dir=data_dir,sigma=sigma)

        #nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='tmass',data_dir=data_dir,sigma=sigma)

        #nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='allwise',data_dir=data_dir,sigma=sigma)

        #nway_mw_prepare(ra, dec,  X_name=csc_name,ref_mjd=mjds, catalog='catwise',data_dir=data_dir,sigma=sigma)

        os.system(f'python {nway_dir}nway.py ./{data_dir}/{csc_name}_CSC.fits :err_r0:err_r1:PA \
              ./{data_dir}/{csc_name}_gaia.fits :PU ./{data_dir}/{csc_name}_tmass.fits :err0:err0:errPA \
              ./{data_dir}/{csc_name}_allwise.fits :err0:err1:errPA ./{data_dir}/{csc_name}_catwise.fits :PU \
              --out=./{data_dir}/{csc_name}_nway.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46') # r0 is 2-sigma
              #--out=./{data_dir}/{csc_name}_nway.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46') # 1.5 * r0 is 3-sigma
    
    if explain:

        os.system(f'python {nway_dir}nway-explain.py ./{data_dir}/{csc_name}_nway.fits 1') 

    if move:
        Path(f'./{move_dir}/').mkdir(parents=True, exist_ok=True)
        os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./{move_dir}/')   
        os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./{move_dir}/')        
            

    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./AGN_check/')   
    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./AGN_check/')    

    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./AGN_check/radiusr0/')   
    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./AGN_check/radiusr0/')    
        
    return csc_name


def nway_cross_matching_mag(TD, i, radius, query_dir, mag_catalogs=[], mags=[], hist_files=[], name_col='name',ra_col='ra',dec_col='dec', ra_csc_col='ra',dec_csc_col='dec',PU_col='err_ellipse_r0',r0_col='r0',r1_col='r1',PA_col='PA',data_dir='data',explain=False,move=False,move_dir='check',rerun=False, sigma=2.,newcsc=False,per_file='txt'):
    csc_name, ra, dec,ra_csc,dec_csc, r0 = TD.loc[i, name_col][5:], TD.loc[i, ra_col], TD.loc[i, dec_col], TD.loc[i, ra_csc_col], TD.loc[i, dec_csc_col], TD.loc[i, PU_col]#'r0']#err_ellipse_r0']#r0']
    #print(csc_name, ra, dec)
    if path.exists(f'./{data_dir}/{csc_name}_nway_mag.fits') == False or rerun==True:
        if type(per_file) == pd.DataFrame:
            df_r = per_file[per_file['name']==TD.loc[i, name_col]].reset_index(drop=True)   
        elif type(per_file) == str and per_file == 'txt':
            #print('txt')
            if path.exists(f'{query_dir}/{csc_name}.txt') == False:
                CSCviewsearch(csc_name, ra_csc, dec_csc, radius,query_dir,csc_version='2.0')
            df_r = pd.read_csv(f'{query_dir}/{csc_name}.txt', header=154, sep='\t')
        else:
            print('else')
              

        df_r['mjd'] = np.nan
        df_r['mjd'] = df_r.apply(lambda r: Time(r['gti_obs'], format='isot', scale='utc').mjd,axis=1)
        mjds = df_r['mjd'].values

        #csc_name, CSC_id, r0 = df_r.loc[i, 'name'][5:], df_r.loc[i, 'ID'], TD_old.loc[i, 'r0']

        if newcsc:
            newcsc_prepare(TD.iloc[[i]].reset_index(drop=True),X_name=csc_name,name_col=name_col,ra_col=ra_col, dec_col=dec_col,r0_col=r0_col,r1_col=r1_col,PA_col=PA_col,data_dir=data_dir,sigma=2)
        else:
            nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog='CSC',data_dir=data_dir,sigma=sigma)

        for mw_cat in ['gaia', 'tmass', 'allwise', 'catwise']:
            if path.exists(f'./{data_dir}/{csc_name}_{mw_cat}.fits') == False:
                nway_mw_prepare(ra, dec,  X_name=csc_name, ref_mjd=mjds, catalog=mw_cat,data_dir=data_dir,sigma=sigma)

        codes = f'python {nway_dir}nway.py ./{data_dir}/{csc_name}_CSC.fits :err_r0:err_r1:PA \
              ./{data_dir}/{csc_name}_gaia.fits :PU ./{data_dir}/{csc_name}_tmass.fits :err0:err0:errPA \
              ./{data_dir}/{csc_name}_allwise.fits :err0:err1:errPA ./{data_dir}/{csc_name}_catwise.fits :PU \
              --out=./{data_dir}/{csc_name}_nway_mag.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46' # r0 is 2-sigma
              #--out=./{data_dir}/{csc_name}_nway.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46') # 1.5 * r0 is 3-sigma
    
        for mag_cat, mag, hist_file in zip(mag_catalogs, mags, hist_files):
            codes = codes + f' --mag {mag_cat}:{mag} {hist_file}'

        #print(codes)
        os.system(codes)

    if explain:

        os.system(f'python {nway_dir}nway-explain.py ./{data_dir}/{csc_name}_nway_mag.fits 1') 

    if move:
        Path(f'./{move_dir}/').mkdir(parents=True, exist_ok=True)
        os.system(f'cp  ./{data_dir}/{csc_name}_nway_mag.fits_explain_1.pdf ./{move_dir}/')   
        os.system(f'cp  ./{data_dir}/{csc_name}_nway_mag.fits_explain_1_options.pdf ./{move_dir}/')        
            

    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./AGN_check/')   
    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./AGN_check/')    

    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1.pdf ./AGN_check/radiusr0/')   
    #os.system(f'cp  ./{data_dir}/{csc_name}_nway.fits_explain_1_options.pdf ./AGN_check/radiusr0/')    
        
    return csc_name

def update_nway(TD):

    nway_cols = {'gaia':['Gaia','GAIA_RA', 'GAIA_DEC',
       'GAIA_PU', 'GAIA_Source', 'GAIA__r', 'GAIA_e_Pos', 'GAIA_Plx',
       'GAIA_e_Plx', 'GAIA_PM', 'GAIA_e_PM', 'GAIA_epsi', 'GAIA_Gmag',
       'GAIA_BPmag', 'GAIA_RPmag', 'GAIA_e_Gmag', 'GAIA_e_BPmag'],
            'tmass':['TMASS_RA', 'TMASS_DEC', 'TMASS_err0', 'TMASS_err1',
       'TMASS_errPA', 'TMASS__2MASS', 'TMASS__r', 'TMASS_Jmag', 'TMASS_Hmag',
       'TMASS_Kmag', 'TMASS_e_Jmag', 'TMASS_e_Hmag', 'TMASS_e_Kmag'],
            'allwise':['ALLWISE_RA', 'ALLWISE_DEC', 'ALLWISE_err0', 'ALLWISE_err1',
       'ALLWISE_errPA', 'ALLWISE_AllWISE', 'ALLWISE__r', 'ALLWISE_W1mag',
       'ALLWISE_W2mag', 'ALLWISE_W3mag','ALLWISE_W4mag', 'ALLWISE_e_W1mag', 'ALLWISE_e_W2mag',
       'ALLWISE_e_W3mag', 'ALLWISE_e_W4mag'],
            'catwise':['CATWISE_RA', 'CATWISE_DEC',
       'CATWISE_PU', 'CATWISE_Name', 'CATWISE__r', 'CATWISE_W1mproPM',
       'CATWISE_W2mproPM', 'CATWISE_e_W1mproPM', 'CATWISE_e_W2mproPM']}

    nway_names = {'gaia':'GAIA_Source','tmass':'TMASS__2MASS','allwise':'ALLWISE_AllWISE','catwise':'CATWISE_Name'}

    TD_confused = TD[TD['CSC__2CXO'].isin(TD.loc[TD['match_flag']==2, 'CSC__2CXO'])].reset_index(drop=True)
    TD_Noconfused = TD[~TD['CSC__2CXO'].isin(TD.loc[TD['match_flag']==2, 'CSC__2CXO'])].reset_index(drop=True)
    #print(len(TD_confused),len(TD_Noconfused))

    TD_confused = TD_confused.sort_values(by=['CSC__2CXO','p_i'],ascending=False).reset_index(drop=True)


    TD_confused_clean = TD_confused.drop_duplicates(subset=['CSC__2CXO'])

    TD_confused_others = TD_confused[TD_confused['match_flag']==2].reset_index(drop=True)

    #print(TD_confused_clean['match_flag'].value_counts())


    #'''
    for src in TD_confused['CSC__2CXO'].unique():
        TD_src = TD_confused_others[TD_confused_others['CSC__2CXO']==src].reset_index(drop=True)
        #print(src, len(TD_src))
        for cat in ['gaia','tmass','allwise','catwise']:
            #print(cat)
            for i in range(len(TD_src)):

                if TD_confused_clean.loc[(TD_confused_clean['CSC__2CXO']==src), nway_names[cat]].values[0]=='-99':

                    index = TD_confused_clean[(TD_confused_clean['CSC__2CXO']==src)].index[0]

                    TD_confused_clean.loc[index, nway_cols[cat]] = TD_src.loc[i, nway_cols[cat]]

                else:
                    #print(f'break at {i}')
                    break
            

    TD_updated = pd.concat([TD_Noconfused, TD_confused_clean], ignore_index=True, sort=False)

    return TD_updated
                
        



