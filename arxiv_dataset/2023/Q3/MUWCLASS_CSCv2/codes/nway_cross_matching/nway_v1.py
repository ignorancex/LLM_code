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


    #df = pd.read_csv(f'{query_dir}/{field_name}.txt', header=15, sep='\t')
    df = pd.read_csv(f'{query_dir}/{field_name}.txt', header=154, sep='\t')

    return df

vizier_cols_dict = {
    'catalogs': {'CSC':'IX/57/csc2master','gaia':'I/355/gaiadr3', 'tmass':'II/246/out','allwise':'II/328/allwise','catwise':'II/365/catwise'},
    'search_radius':{'CSC':0.1/60,'gaia':5., 'tmass':5.,'allwise':5.,'catwise':5.},
    #'search_radius':{'CSC':5./60,'gaia':10./60, 'tmass':10./60,'allwise':10./60,'catwise':10./60},
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

def nway_mw_prepare(ra_x, dec_x, X_name, mjd_dif=0.,catalog='gaia',plot_density_curve=False):
    
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
        df_q['err_r0'] = df_q['r0']/2
        df_q['err_r1'] = df_q['r1']/2
    
        
        new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_2CXO','_r','fe','fc']]) # r0 is 95%, should be consistent with other PUs, 
        
        #new_t.write(f'./data/{csc_name}_CSC.fits', overwrite=True)


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
        df_q['PU'] = np.sqrt(df_q['sigma_pos'].fillna(0.)**2+df_q['Plx'].fillna(0.)**2+df_q['e_Plx'].fillna(0.)**2 + (df_q['PU_PM'].fillna(0.))**2+(df_q['PU_ePM'].fillna(0.))**2+df_q['epsi'].fillna(0.)**2)/1e3
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
        
        #df_q['err0'] = df_q.apply(lambda r: max(r.errMaj,0.3), axis=1)
        #df_q['err1'] = df_q.apply(lambda r: max(r.errMin,0.3), axis=1)
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
        #df_q['err0'] = df_q.apply(lambda r: max(r.eeMaj,0.3), axis=1)
        #df_q['err1'] = df_q.apply(lambda r: max(r.eeMin,0.3), axis=1)
        
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
        df_q['PU'] = np.sqrt(df_q['e_pos'].fillna(0.)**2+df_q['plx1'].fillna(0.)**2+df_q['e_plx1'].fillna(0.)**2) 
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
        
    new_t.write(f'./data/{X_name}_{catalog}.fits', overwrite=True)
    if catalog == 'CSC':
        area = 550./317000
    else:
        area = np.pi * (search_radius/60)**2
    #area = 550./317000
    os.system(f'python /CCAS/home/huiyang/softs/nway-master/nway-write-header.py ./data/{X_name}_{catalog}.fits {catalog} {area}')
    
    return len(df_q)
    
def nway_cross_matching(TD, i, radius, query_dir):
    csc_name, CSC_id, ra, dec, r0 = TD.loc[i, 'name'][5:], TD.loc[i, 'ID'], TD.loc[i, 'ra'], TD.loc[i, 'dec'],  TD.loc[i, 'r0']
    print(csc_name, ra, dec)
    if path.exists(f'./data/{csc_name}_nway.fits') == False:
        df_r = CSCviewsearch(csc_name, ra, dec, radius,query_dir,csc_version='2.0')
        #print(len(df_r))
        #print(df_r.iloc[0]['gti_obs'])
        #t1 = Time(df_r.iloc[0]['gti_obs'], format='isot', scale='utc')
        #print(len(df_r))
        #print(df_r)
        df_r['mjd_dif'] = np.nan
        df_r['mjd_dif'] = df_r.apply(lambda r: abs(57388-Time(r['gti_obs'], format='isot', scale='utc').mjd),axis=1)
        mjd_dif_max = df_r['mjd_dif'].max()

        #csc_name, CSC_id, r0 = df_r.loc[i, 'name'][5:], df_r.loc[i, 'ID'], TD_old.loc[i, 'r0']


        num_X = nway_mw_prepare(ra, dec,  X_name=csc_name, mjd_dif=mjd_dif_max, catalog='CSC')

        nway_mw_prepare(ra, dec,  X_name=csc_name, mjd_dif=mjd_dif_max, catalog='gaia')

        nway_mw_prepare(ra, dec,  X_name=csc_name,catalog='tmass')

        nway_mw_prepare(ra, dec,  X_name=csc_name, catalog='allwise')

        nway_mw_prepare(ra, dec,  X_name=csc_name,catalog='catwise')

        os.system(f'python /CCAS/home/huiyang/softs/nway-master/nway.py ./data/{csc_name}_CSC.fits :err_r0:err_r1:PA \
              ./data/{csc_name}_gaia.fits :PU ./data/{csc_name}_tmass.fits :errMaj:errMin:errPA \
              ./data/{csc_name}_allwise.fits :eeMaj:eeMin:eePA ./data/{csc_name}_catwise.fits :PU \
              --out=./data/{csc_name}_nway.fits --radius {min(r0*1.5, 3)} --prior-completeness 0.82:0.64:0.31:0.46') # 1.5 * r0 is 3-sigma
        
        return None
