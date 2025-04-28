import pandas as pd
import numpy as np
import copy

class Collaborator:
    def __init__(self):       
        self.SS_data = pd.read_csv('input_data/gmag_br_barySS.txt',sep = ' ')
        self.NST_data = pd.read_csv('input_data/gmag_br_baryNST.txt',sep = ' ')

    def collaborate(self,astrometry_df,catalogue = 'GDR2toGDR3',mode='SS',radec_unit = 'deg'):
        astrometry_df_this = copy.deepcopy(astrometry_df)
        if catalogue == 'GDR2toGDR3':
            if mode == 'SS':
                this_collaboration_data = self.SS_data
            else:
                this_collaboration_data = self.NST_data
        
            coeff = this_collaboration_data.loc[(this_collaboration_data['mag1']<=astrometry_df['mag'])&(this_collaboration_data['mag2']>astrometry_df['mag'])&(this_collaboration_data['br1']<=astrometry_df['br'])&(this_collaboration_data['br2']>astrometry_df['br']),'ex23 ey23 ez23 ox23 oy23 oz23 plx23 a3 b3 a2 b2 a1 b1'.split(' ')]
            #print(coeff)
            if len(coeff)!=1:
                print('mag,br out of range!')
                return {'ra':-99999999,'dec':-99999999,'plx':-99999999,'pmra':-99999999,'pmdec':-99999999}
        
            coeff = dict(coeff)
            #print(coeff['ex23'])
            coeff['ex23'] = -np.squeeze(coeff['ex23'].values)/206264.80624709636/1000 #mas->rad
            coeff['ey23'] = -np.squeeze(coeff['ey23'].values)/206264.80624709636/1000 #mas->rad
            coeff['ez23'] = -np.squeeze(coeff['ez23'].values)/206264.80624709636/1000 #mas->rad
            coeff['ox23'] = -np.squeeze(coeff['ox23'].values)/206264.80624709636/1000 #mas/yr ->rad/yr
            coeff['oy23'] = -np.squeeze(coeff['oy23'].values)/206264.80624709636/1000 #mas/yr ->rad/yr
            coeff['oz23'] = -np.squeeze(coeff['oz23'].values)/206264.80624709636/1000 #mas/yr ->rad/yr
            coeff['plx23'] =-np.squeeze(coeff['plx23'].values)
            coeff['a3'] = np.squeeze(coeff['a3'].values)
            coeff['a2'] = np.squeeze(coeff['a2'].values)
            coeff['a1'] = np.squeeze(coeff['a1'].values)
            coeff['b3'] = np.squeeze(coeff['b3'].values)
            coeff['b2'] = np.squeeze(coeff['b2'].values)
            coeff['b1'] = np.squeeze(coeff['b1'].values)
            beta = np.array([coeff['ex23'],coeff['ey23'],coeff['ez23'],coeff['ox23'],coeff['oy23'],coeff['oz23'],coeff['plx23']]).reshape(-1,1)
            Dt = -0.5
        if catalogue == 'HIPtoVLBI2015':
            beta = np.array([0,0,0,0.126/206264.80624709636/1000,-0.185/206264.80624709636/1000,-0.076/206264.80624709636/1000,0.089]).reshape(-1,1)
            Dt = 23.75
        if catalogue == 'VLBI2015toVLBI2020':
            beta = np.array([0.008/206264.80624709636/1000,0.015/206264.80624709636/1000,0/206264.80624709636/1000,0,0,0,0]).reshape(-1,1)
            Dt = 5.015      
        if catalogue == 'VLBI2020toGDR3':
            beta = np.array([0.226/206264.80624709636/1000,0.327/206264.80624709636/1000,0.168/206264.80624709636/1000,0.022/206264.80624709636/1000,0.065/206264.80624709636/1000,-0.016/206264.80624709636/1000,0]).reshape(-1,1)
            Dt = 4.015
        if catalogue == 'HIPtoGDR3':
            resVLBI15 = self.collaborate(astrometry_df,catalogue = 'HIPtoVLBI2015')
            resVLBI20 = self.collaborate(resVLBI15,catalogue = 'VLBI2015toVLBI2020')
            res = self.collaborate(resVLBI20,catalogue = 'VLBI2020toGDR3')
            return res

        if catalogue == 'GDR1toGDR3':
            if mode == 'NSTP2':
                beta = np.array([0/206264.80624709636/1000,-0.13/206264.80624709636/1000,-0.01/206264.80624709636/1000,0/206264.80624709636/1000,0/206264.80624709636/1000,0/206264.80624709636/1000,0]).reshape(-1,1)
            if mode == 'SSP5':
                beta = np.array([0.39/206264.80624709636/1000,-0.17/206264.80624709636/1000,0.12/206264.80624709636/1000,0.02/206264.80624709636/1000,-0.03/206264.80624709636/1000,0.02/206264.80624709636/1000,0]).reshape(-1,1)
            Dt = 1
            #return res
        
        if radec_unit== 'deg':
            ra = astrometry_df_this['ra']*np.pi/180         #deg->rad
            dec = astrometry_df_this['dec']*np.pi/180        #deg->rad
            try:
                pmra = astrometry_df_this['pmra']/206264.80624709636/1000     #mas/yr->rad/yr
                pmdec = astrometry_df_this['pmdec']/206264.80624709636/1000      #mas/yr->rad/yr
                have_pm = 1
                if astrometry_df_this['pmra'] == -99999999 or astrometry_df_this['pmra'] == -99999999:
                    have_pm = 0
            except:
                pmra = 0
                pmdec = 0
                have_pm = 0
            try:
                plx = astrometry_df_this['plx']
                have_Plx = 1
                if plx == -99999999:
                    plx = 0
                    have_Plx = 0
            except:
                plx = 0
                have_Plx = 0
        
        Kappa = np.array(
            [
                [np.cos(ra)*np.sin(dec),np.sin(ra)*np.sin(dec),-np.cos(dec),Dt*np.cos(ra)*np.sin(dec),Dt*np.sin(ra)*np.sin(dec),-Dt*np.cos(dec),0],
                [-np.sin(ra),np.cos(ra),0,-Dt*np.sin(ra),Dt*np.cos(ra),0,0],
                [0,0,0,0,0,0,1],
                [0,0,0,np.cos(ra)*np.sin(dec),np.sin(ra)*np.sin(dec),-np.cos(dec),0],
                [0,0,0,-np.sin(ra),np.cos(ra),0,0]
            ]
        )
        astro_origin = np.array([ra*np.cos(dec),dec,plx,pmra,pmdec]).reshape(-1,1)

        collaborated_astrometry = np.squeeze(astro_origin-Kappa.dot(beta))
        res = {'ra':collaborated_astrometry[0]/np.cos(dec)*180/np.pi,'dec':collaborated_astrometry[1]*180/np.pi,'plx':collaborated_astrometry[2]*have_Plx-(1-have_Plx)*99999999,'pmra':have_pm*collaborated_astrometry[3]*206264.80624709636*1000-(1-have_pm)*99999999,'pmdec':have_pm*collaborated_astrometry[4]*206264.80624709636*1000-(1-have_pm)*99999999}
        return res