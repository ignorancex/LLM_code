# from __future__ import print_function
# from __future__ import division # undo if in Python 2 
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import time
import sys
import copy
#Quick fudge to make import from ../Scripts work
import sys
sys.path.append('../Scripts')
import string

import ClassCosmicIntegrator  as CI #Given settings and redshifts returns rates (2D arrays) Loads the data
from PostProcessingScripts import * 
from ClassFormationChannels_5mainchannels import * 
from formation_channels import * 

import pandas as pd
from astropy import units as u
from astropy import constants as const





MSSFRnameslist = []
MSSFRnameslist.append('000') # add phenomenological 

for ind_GSMF, GSMF in enumerate(GSMFs):
    ind_y = ind_GSMF + 1
    for ind_MZ, MZ in enumerate(MZs):
        ind_z = ind_MZ +1
        for ind_SFR, SFR in enumerate(SFRs):
            ind_x = ind_SFR+1

            MSSFRnameslist.append('%s%s%s'%(ind_x, ind_y, ind_z))


GSMFs = [1,2,3]
SFRs = [1,2,3]
MZs=[1,2,3]


MSSFRnameslistCSV = []
MSSFRnameslistCSV.append('.0.0.0') # add phenomenological 


for ind_GSMF, GSMF in enumerate(GSMFs):
    ind_y = ind_GSMF + 1
    for ind_MZ, MZ in enumerate(MZs):
        ind_z = ind_MZ +1

        for ind_SFR, SFR in enumerate(SFRs):
            ind_x = ind_SFR+1            




            MSSFRnameslistCSV.append('.%s.%s.%s'%(ind_x, ind_y, ind_z))




def writeToRatesFile_MRR_Spins(BPSmodelName='Z', DCOtype='BHNS', spin_threshold=0.05):
    """writes NS-BH rate to CSV file for all models"""

    if DCOtype=='BHNS':
        DCOname='BHNS'
    elif DCOtype=='BBH':
        DCOname='BHBH'
    elif DCOtype=='BNS':
        DCOname='NSNS'


    # path for files 
    path_dir = '/Volumes/Andromeda2/DATA/AllDCO_bugfix/'
    path_ = path_dir
    path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
    path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
            
    # read in data 
    fdata = h5.File(path, 'r')
    fDCO      = fdata['doubleCompactObjects'] # hdf5 file with the DCO information
    fSN       = fdata['supernovae']  # hdf5 file with the SN information
    #
    M1 = fDCO['M1'][...].squeeze()   # Compact object mass [Msun] of the initially more massive star
    M2 = fDCO['M2'][...].squeeze()  # Compact object mass [Msun] of the initially less massive star

    print('using indices')
    seedsDCO = fDCO['seed'][...].squeeze()  # get the seeds in the DCO file 
    seedsSN = fSN['randomSeed'][...].squeeze()    # get the seeds in the SN file 
    indices = np.sort(np.unique(seedsSN[1::2], return_index=True)[1])
    maskSNdco = np.in1d(seedsSN,  seedsDCO) # mask in the SNe files the SNe that correspond to our DCO
    whichSN = fSN['whichStar'][...].squeeze()[maskSNdco]  # this is 1 if the initially primary star goes SN and 2 if the secondary goes supernova     
    whichSN2 = whichSN[1::2][indices]

    # either SN2 = primary (1) and M1 is > M2, or SN2 = secondary & M1 < M2 
    # this takes into account (first term) rejuvenation 
    MRR_mask = ((whichSN2==1) & (M1>M2) ) | ((whichSN2==2) & (M1<M2)) 

    del M1
    del M2
    del whichSN2
    del whichSN
    del maskSNdco
    del indices
    del seedsSN
    del seedsDCO
    del fDCO
    del fSN

    fdata.close()



    spin = COspin(data_path=path, state='he_depletion')  # set class 
    spin.setCOMPASData() # reads in the COMPAS DCO parameters 
    spinMZAMS1, spinMZAMS2  = spin.BaveraSpin()


    spinLVKM1, spinLVKM2 = np.zeros_like(spinMZAMS1), np.zeros_like(spinMZAMS1)
    spinLVKM1[MRR_mask] = spinMZAMS2[MRR_mask]  # MRR so M1 comes from M2ZAMS, we assign it spin from M2ZAMS
    spinLVKM1[~MRR_mask] = spinMZAMS1[~MRR_mask]  # no MRR so M1 comes from M1ZAMS, we assign it spin from M1ZAMS
    spinLVKM2[MRR_mask] = spinMZAMS1[MRR_mask]   # MRR so M2 comes from M1ZAMS, we assign it spin from M1ZAMS
    spinLVKM2[~MRR_mask] = spinMZAMS2[~MRR_mask]   # no MRR so M2 comes from M2ZAMS, we assign it spin from M2ZAMS     

    # spin_threshold = 0.05 # definition of "spinning BH"
    mask_LVKM1_spinning = (spinLVKM1 > spin_threshold ) 
    mask_LVKM2_spinning = (spinLVKM2 > spin_threshold ) # definition of "spinning BH"
    mask_anySpin = (spinLVKM1 > spin_threshold ) | (spinLVKM2 > spin_threshold )


    # get intrinsic weights
    fparam_intrinsic = 'weights_intrinsic'
    # get detected weights
    fparam_detected = 'weights_detected'




    ####################################################
    ######### ITERATE  OVER  MSSFR  MODELS #############
    ####################################################
    intrinsicRates = np.zeros(len(MSSFRnameslist))
    detectedRates = np.zeros(len(MSSFRnameslist))
    # namesEMlist = []
    intrinsicRates_spinning = np.zeros(len(MSSFRnameslist))
    detectedRates_spinning = np.zeros(len(MSSFRnameslist))

    intrinsicRates_spinning_LVKM1 = np.zeros(len(MSSFRnameslist))
    detectedRates_spinning_LVKM1 = np.zeros(len(MSSFRnameslist))
    intrinsicRates_spinning_LVKM2 = np.zeros(len(MSSFRnameslist))
    detectedRates_spinning_LVKM2 = np.zeros(len(MSSFRnameslist))


    fdata = h5.File(path)

    DCOSeeds = fdata['doubleCompactObjects']['seed'][...].squeeze()

    for ind_mssfr, mssfr in enumerate(MSSFRnameslist):

        weightheader = 'w_' + mssfr
        # print(ind_mssfr, weightheader)
        w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()
        w_det = fdata[fparam_detected][weightheader][...].squeeze()

        # ALL BBH RATES 
        intrinsicRates[ind_mssfr] = np.sum(w_int)
        detectedRates[ind_mssfr] = np.sum(w_det)
        
        # MASS RATIO REVERSAL RATE ALL CHANNELS 
        intrinsicRates_spinning[ind_mssfr] = np.sum(w_int[mask_anySpin])
        detectedRates_spinning[ind_mssfr] = np.sum(w_det[mask_anySpin])

        intrinsicRates_spinning_LVKM1[ind_mssfr] = np.sum(w_int[mask_LVKM1_spinning])
        detectedRates_spinning_LVKM1[ind_mssfr] = np.sum(w_det[mask_LVKM1_spinning])       
        intrinsicRates_spinning_LVKM2[ind_mssfr] = np.sum(w_int[mask_LVKM2_spinning])
        detectedRates_spinning_LVKM2[ind_mssfr] = np.sum(w_det[mask_LVKM2_spinning])     



    fdata.close()  

    stringgg =  'MRR_Spins'
    if spin_threshold!=0.05:
        stringgg =  'MRR_Spins' +  '_threshold_%s_'%spin_threshold

    df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
    namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
    nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'
    namez0_spinAny = BPSmodelName + 'All: at least one spin >%s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
    nameObs_spinAny = BPSmodelName + 'All: at least one spin >%s observed (design LVK) [yr^{-1}]'%spin_threshold

    namez0_spinning_LVKM1 = BPSmodelName + 'LVKM1 spinning > %s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
    nameObs_spinning_LVKM1 = BPSmodelName + 'LVKM1 spinning > %s observed (design LVK) [yr^{-1}]'%spin_threshold
    namez0_spinning_LVKM2= BPSmodelName + 'LVKM2 spinning > %s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
    nameObs_spinning_LVKM2 = BPSmodelName + 'LVKM2 spinning > %s observed (design LVK) [yr^{-1}]'%spin_threshold



    df[namez0]                      = intrinsicRates
    df[nameObs]                     = detectedRates
    df[namez0_spinAny]              = intrinsicRates_spinning
    df[nameObs_spinAny]             = detectedRates_spinning

    df[namez0_spinning_LVKM1]       = intrinsicRates_spinning_LVKM1
    df[nameObs_spinning_LVKM1]      = detectedRates_spinning_LVKM1
    df[namez0_spinning_LVKM2]       = intrinsicRates_spinning_LVKM2
    df[nameObs_spinning_LVKM2]      = detectedRates_spinning_LVKM2
 


    df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


    fdata.close()







def initialize_CSV_files_MRRspins(DCOname='BHBH', spin_threshold=0.05):

    # namesEMlist=[]

    iii=0
    

    # CREATE PANDAS FILE 
    nModels=26
    BPSnameslist = list(string.ascii_uppercase)[0:nModels]

    NAMES = []
    stringgg =  'MRR_Spins'
    if spin_threshold!=0.05:
        stringgg =  'MRR_Spins' +  '_threshold_%s_'%spin_threshold

    for ind_l, BPSmodelName in enumerate(BPSnameslist):
        # str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
        # str_obs = str(L + ' observed (design LVK) [yr^{-1}]')

        namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
        nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'
        namez0_spinAny = BPSmodelName + 'All: at least one spin >%s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
        nameObs_spinAny = BPSmodelName + 'All: at least one spin >%s observed (design LVK) [yr^{-1}]'%spin_threshold

        namez0_spinning_LVKM1 = BPSmodelName + 'LVKM1 spinning > %s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
        nameObs_spinning_LVKM1 = BPSmodelName + 'LVKM1 spinning > %s observed (design LVK) [yr^{-1}]'%spin_threshold
        namez0_spinning_LVKM2= BPSmodelName + 'LVKM2 spinning > %s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
        nameObs_spinning_LVKM2 = BPSmodelName + 'LVKM2 spinning > %s observed (design LVK) [yr^{-1}]'%spin_threshold

        NAMES.append(namez0)
        NAMES.append(nameObs)

        NAMES.append(namez0_spinAny)
        NAMES.append(nameObs_spinAny)

        NAMES.append(namez0_spinning_LVKM1)
        NAMES.append(nameObs_spinning_LVKM1)
        NAMES.append(namez0_spinning_LVKM2)
        NAMES.append(nameObs_spinning_LVKM2)




    datas=[]

    for i in range(len(BPSnameslist)):
        for ii in range(4):
            datas.append(np.zeros_like(MSSFRnameslist))
            datas.append(np.zeros_like(MSSFRnameslist))
        
        
    df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
    df.columns =   df.columns.map(str)
    df.index.names = ['xyz']
    df.columns.names = ['m']

    # print(df) 

    df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')







def initialize_CSV_files_MRR_nonMRR_ratio(DCOname='BHBH', spin_threshold=0.05, GWname='GW150629'):

    # namesEMlist=[]

    iii=0
    

    # CREATE PANDAS FILE 
    nModels=26
    BPSnameslist = list(string.ascii_uppercase)[0:nModels]

    NAMES = []
    stringgg =  'MRR_nonMRR_ratio_' + GWname 


    for ind_l, BPSmodelName in enumerate(BPSnameslist):
        # str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
        # str_obs = str(L + ' observed (design LVK) [yr^{-1}]')

        namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
        nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'
        
        namez0_MRR = BPSmodelName + 'MRR: matches GW contour intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
        nameObs_MRR = BPSmodelName + 'MRR: matches GW contour observed (design LVK) [yr^{-1}]'
        namez0_nonMRR = BPSmodelName + 'nonMRR: matches GW contour intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
        nameObs_nonMRR = BPSmodelName + 'nonMRR: matches GW contour observed (design LVK) [yr^{-1}]'
        
        namez0_MRR_spin= BPSmodelName + 'MRR: matches GW contour spin2 < %s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
        nameObs_MRR_spin = BPSmodelName + 'MRR: matches GW contour spin2 < %s observed (design LVK) [yr^{-1}]'%spin_threshold
        namez0_nonMRR_spin= BPSmodelName + 'nonMRR: matches GW contour spin1 < %s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
        nameObs_nonMRR_spin = BPSmodelName + 'nonMRR: matches GW contour spin1 < %s observed (design LVK) [yr^{-1}]'%spin_threshold



        NAMES.append(namez0)
        NAMES.append(nameObs)

        NAMES.append(namez0_MRR)
        NAMES.append(nameObs_MRR)
        NAMES.append(namez0_nonMRR)
        NAMES.append(nameObs_nonMRR)

        NAMES.append(namez0_MRR_spin)
        NAMES.append(nameObs_MRR_spin)
        NAMES.append(namez0_nonMRR_spin)
        NAMES.append(nameObs_nonMRR_spin)




    datas=[]

    for i in range(len(BPSnameslist)):
        for ii in range(5):
            datas.append(np.zeros_like(MSSFRnameslist))
            datas.append(np.zeros_like(MSSFRnameslist))
        
        
    df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
    df.columns =   df.columns.map(str)
    df.index.names = ['xyz']
    df.columns.names = ['m']

    # print(df) 

    df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')





def GW_credible_intervals(GW_name, mode):

    # GW_list = ['GW151226','GW170729', 'GW190517_055101', 'GW190412','GW191109_010717'  ,'GW191103_012549', 'GW191126_115259']

    
    dfCSVname= '/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/GWTC_posterior_samples/' 
    dfCSVname_ = dfCSVname + 'posteriorSamples_' + GW_name  + '.csv' 


    df = pd.read_csv(dfCSVname_, index_col=0, skiprows=[1])
    mass_ratio = df['M2']/df['M1']
    total_mass = df['M2'] + df['M1']
    spin1 = df['spin1'] 
    spin2 = df['spin2'] 
    
    chi_eff = df['chi_eff']
    chirp_mass = chirpmass(df['M2'], df['M1'])
    y_quantiles = [0.05, 0.5, 0.95]

    if mode=='normal':
    
        total_mass_CI = weighted_quantile(values=total_mass, quantiles=y_quantiles)
        mass1_CI = weighted_quantile(values=df['M1'], quantiles=y_quantiles)
        mass2_CI = weighted_quantile(values=df['M2'], quantiles=y_quantiles)
        chirp_mass_CI = weighted_quantile(values=chirp_mass, quantiles=y_quantiles)
        mass_ratio_CI = weighted_quantile(values=mass_ratio, quantiles=y_quantiles)
        spin1_CI = weighted_quantile(values=spin1, quantiles=y_quantiles)    
        spin2_CI = weighted_quantile(values=spin2, quantiles=y_quantiles)    
        chi_eff_quantiles = weighted_quantile(values=chi_eff, quantiles=y_quantiles)    

    elif mode=='spin1_is_zero':
        mask_spin = (abs(spin1)<0.05) & (spin2>0.05) # non MRR, spin 2 is the spinning one, we want spin1 to be zero 
        total_mass_CI = weighted_quantile(values=total_mass[mask_spin], quantiles=y_quantiles)
        mass1_CI = weighted_quantile(values=df['M1'][mask_spin], quantiles=y_quantiles)
        mass2_CI = weighted_quantile(values=df['M2'][mask_spin], quantiles=y_quantiles)
        chirp_mass_CI = weighted_quantile(values=chirp_mass[mask_spin], quantiles=y_quantiles)
        mass_ratio_CI = weighted_quantile(values=mass_ratio[mask_spin], quantiles=y_quantiles)
        spin1_CI = weighted_quantile(values=spin1[mask_spin], quantiles=y_quantiles)    
        spin2_CI = weighted_quantile(values=spin2[mask_spin], quantiles=y_quantiles)    
        chi_eff_quantiles = weighted_quantile(values=chi_eff[mask_spin], quantiles=y_quantiles)   

    elif  mode=='spin2_is_zero':
        mask_spin = (abs(spin2)<0.05) & (spin1>0.05)  # MRR spin1 is the spinning one, we want the other one to be zero
        total_mass_CI = weighted_quantile(values=total_mass[mask_spin], quantiles=y_quantiles)
        mass1_CI = weighted_quantile(values=df['M1'][mask_spin], quantiles=y_quantiles)
        mass2_CI = weighted_quantile(values=df['M2'][mask_spin], quantiles=y_quantiles)
        chirp_mass_CI = weighted_quantile(values=chirp_mass[mask_spin], quantiles=y_quantiles)
        mass_ratio_CI = weighted_quantile(values=mass_ratio[mask_spin], quantiles=y_quantiles)
        spin1_CI = weighted_quantile(values=spin1[mask_spin], quantiles=y_quantiles)    
        spin2_CI = weighted_quantile(values=spin2[mask_spin], quantiles=y_quantiles)    
        chi_eff_quantiles = weighted_quantile(values=chi_eff[mask_spin], quantiles=y_quantiles)   

    return total_mass_CI, mass1_CI, mass2_CI, chirp_mass_CI, mass_ratio_CI, spin1_CI, spin2_CI, chi_eff_quantiles
    



def writeToRatesFile_MRR_nonMRR_ratio(BPSmodelName='Z', DCOtype='BHNS', spin_threshold=0.05, GWnameList=['GW150629']):
    """writes NS-BH rate to CSV file for all models"""


    if DCOtype=='BHNS':
        DCOname='BHNS'
    elif DCOtype=='BBH':
        DCOname='BHBH'
    elif DCOtype=='BNS':
        DCOname='NSNS'


    # path for files 
    path_dir = '/Volumes/Andromeda2/DATA/AllDCO_bugfix/'
    path_ = path_dir
    path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
    path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
            
    # read in data 
    fdata = h5.File(path, 'r')
    fDCO      = fdata['doubleCompactObjects'] # hdf5 file with the DCO information
    fSN       = fdata['supernovae']  # hdf5 file with the SN information
    #
    M1 = fDCO['M1'][...].squeeze()   # Compact object mass [Msun] of the initially more massive star
    M2 = fDCO['M2'][...].squeeze()  # Compact object mass [Msun] of the initially less massive star

    print('using indices')
    seedsDCO = fDCO['seed'][...].squeeze()  # get the seeds in the DCO file 
    seedsSN = fSN['randomSeed'][...].squeeze()    # get the seeds in the SN file 
    indices = np.sort(np.unique(seedsSN[1::2], return_index=True)[1])
    maskSNdco = np.in1d(seedsSN,  seedsDCO) # mask in the SNe files the SNe that correspond to our DCO
    whichSN = fSN['whichStar'][...].squeeze()[maskSNdco]  # this is 1 if the initially primary star goes SN and 2 if the secondary goes supernova     
    whichSN2 = whichSN[1::2][indices]

    # either SN2 = primary (1) and M1 is > M2, or SN2 = secondary & M1 < M2 
    # this takes into account (first term) rejuvenation 
    MRR_mask = ((whichSN2==1) & (M1>M2) ) | ((whichSN2==2) & (M1<M2)) 

    M1LVK, M2LVK = obtainM1BHandM2BHassymetric(M1, M2)
    chirp_mass = chirpmass(M1LVK, M2LVK)
    mass_ratio_LVK =  M2LVK/M1LVK

    del M1
    del M2
    del whichSN2
    del whichSN
    del maskSNdco
    del indices
    del seedsSN
    del seedsDCO
    del fDCO
    del fSN

    fdata.close()



    spin = COspin(data_path=path, state='he_depletion')  # set class 
    spin.setCOMPASData() # reads in the COMPAS DCO parameters 
    spinMZAMS1, spinMZAMS2  = spin.BaveraSpin()


    spinLVKM1, spinLVKM2 = np.zeros_like(spinMZAMS1), np.zeros_like(spinMZAMS1)
    spinLVKM1[MRR_mask] = spinMZAMS2[MRR_mask]  # MRR so M1 comes from M2ZAMS, we assign it spin from M2ZAMS
    spinLVKM1[~MRR_mask] = spinMZAMS1[~MRR_mask]  # no MRR so M1 comes from M1ZAMS, we assign it spin from M1ZAMS
    spinLVKM2[MRR_mask] = spinMZAMS1[MRR_mask]   # MRR so M2 comes from M1ZAMS, we assign it spin from M1ZAMS
    spinLVKM2[~MRR_mask] = spinMZAMS2[~MRR_mask]   # no MRR so M2 comes from M2ZAMS, we assign it spin from M2ZAMS     

    chi_eff = ((spinLVKM1*M1LVK) + (spinLVKM2*M2LVK)) / (M1LVK + M2LVK)


    # spin_threshold = 0.05 # definition of "spinning BH"





    # get intrinsic weights
    fparam_intrinsic = 'weights_intrinsic'
    # get detected weights
    fparam_detected = 'weights_detected'







    fdata = h5.File(path)

    DCOSeeds = fdata['doubleCompactObjects']['seed'][...].squeeze()

     

    for ind_GW, GWname in enumerate(GWnameList):  
        print(GWname)


        ####################################################
        ######### ITERATE  OVER  MSSFR  MODELS #############
        ####################################################
        intrinsicRates = np.zeros(len(MSSFRnameslist))
        detectedRates = np.zeros(len(MSSFRnameslist))
        # namesEMlist = []

        intrinsicRates_MRR = np.zeros(len(MSSFRnameslist))
        detectedRates_MRR = np.zeros(len(MSSFRnameslist))
        intrinsicRates_nonMRR = np.zeros(len(MSSFRnameslist))
        detectedRates_nonMRR = np.zeros(len(MSSFRnameslist))

        intrinsicRates_MRR_spin = np.zeros(len(MSSFRnameslist))
        detectedRates_MRR_spin = np.zeros(len(MSSFRnameslist))
        intrinsicRates_nonMRR_spin = np.zeros(len(MSSFRnameslist))
        detectedRates_nonMRR_spin = np.zeros(len(MSSFRnameslist))




        total_mass_CI, mass1_CI, mass2_CI, chirp_mass_CI, mass_ratio_CI, spin1_CI, spin2_CI, chi_eff_CI = GW_credible_intervals(GWname, mode='normal')
        # mask_GW =  (total_mass_CI[0]<=(M1LVK+M2LVK))  & ((M1LVK+M2LVK)<=total_mass_CI[2])  &  (chirp_mass_CI[0]<=chirp_mass) & (chirp_mass<=chirp_mass_CI[2])  & (mass_ratio_CI[0]<= mass_ratio_LVK) & (mass_ratio_LVK<=mass_ratio_CI[2]) & (chi_eff_CI[0]<=chi_eff) & (chi_eff<=chi_eff_CI[2]) & (mass1_CI[0]<= M1LVK) & (M1LVK<=mass1_CI[2]) & (mass2_CI[0]<= M2LVK) & (M2LVK<=mass2_CI[2])
        mask_GW =   (chirp_mass_CI[0]<=chirp_mass) & (chirp_mass<=chirp_mass_CI[2])  & (mass_ratio_CI[0]<= mass_ratio_LVK) & (mass_ratio_LVK<=mass_ratio_CI[2]) & (chi_eff_CI[0]<=chi_eff) 

        mask_GW_MRR = (mask_GW==1) & (MRR_mask==1)
        mask_GW_nonMRR = (mask_GW==1) & (MRR_mask==0)


        total_mass_CI, mass1_CI, mass2_CI, chirp_mass_CI, mass_ratio_CI, spin1_CI, spin2_CI, chi_eff_CI = GW_credible_intervals(GWname, mode='spin1_is_zero')
        # mask_spin1_zero =  (total_mass_CI[0]<=(M1LVK+M2LVK))  & ((M1LVK+M2LVK)<=total_mass_CI[2])  &  (chirp_mass_CI[0]<=chirp_mass) & (chirp_mass<=chirp_mass_CI[2]) & (mass_ratio_CI[0]<= mass_ratio_LVK) & (mass_ratio_LVK<=mass_ratio_CI[2]) & (chi_eff_CI[0]<=chi_eff) & (chi_eff<=chi_eff_CI[2]) & (mass1_CI[0]<= M1LVK) & (M1LVK<=mass1_CI[2]) & (mass2_CI[0]<= M2LVK) & (M2LVK<=mass2_CI[2])# &  (spin2_CI[0]<=spinLVKM2 ) & (spinLVKM2<=spin2_CI[2]) 
        mask_spin1_zero =  (chirp_mass_CI[0]<=chirp_mass) & (chirp_mass<=chirp_mass_CI[2]) & (mass_ratio_CI[0]<= mass_ratio_LVK) & (mass_ratio_LVK<=mass_ratio_CI[2]) & (chi_eff_CI[0]<=chi_eff) # 

        total_mass_CI, mass1_CI, mass2_CI, chirp_mass_CI, mass_ratio_CI, spin1_CI, spin2_CI, chi_eff_CI = GW_credible_intervals(GWname, mode='spin2_is_zero')
        
        mask_spin2_zero =  (chirp_mass_CI[0]<=chirp_mass) & (chirp_mass<=chirp_mass_CI[2]) & (mass_ratio_CI[0]<= mass_ratio_LVK) & (mass_ratio_LVK<=mass_ratio_CI[2]) & (chi_eff_CI[0]<=chi_eff) 
        # mask_spin2_zero =  (total_mass_CI[0]<=(M1LVK+M2LVK))  & ((M1LVK+M2LVK)<=total_mass_CI[2])  &  (chirp_mass_CI[0]<=chirp_mass) & (chirp_mass<=chirp_mass_CI[2]) & (mass_ratio_CI[0]<= mass_ratio_LVK) & (mass_ratio_LVK<=mass_ratio_CI[2]) & (chi_eff_CI[0]<=chi_eff) & (chi_eff<=chi_eff_CI[2]) & (mass1_CI[0]<= M1LVK) & (M1LVK<=mass1_CI[2]) & (mass2_CI[0]<= M2LVK) & (M2LVK<=mass2_CI[2]) # &  (spin1_CI[0]<=spinLVKM1 ) & (spinLVKM1<=spin1_CI[2]) 


        mask_spin2zero_MRR = (mask_spin2_zero==1) & (MRR_mask==1)
        mask_spin1zero_nonMRR = (mask_spin1_zero==1) & (MRR_mask==0)



        for ind_mssfr, mssfr in enumerate(MSSFRnameslist):

            weightheader = 'w_' + mssfr
            # print(ind_mssfr, weightheader)
            w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()
            w_det = fdata[fparam_detected][weightheader][...].squeeze()

            # ALL BBH RATES 
            intrinsicRates[ind_mssfr] = np.sum(w_int)
            detectedRates[ind_mssfr] = np.sum(w_det)
            
            # MASS RATIO REVERSAL RATE ALL CHANNELS 
            intrinsicRates_MRR[ind_mssfr] = np.sum(w_int[mask_GW_MRR])
            detectedRates_MRR[ind_mssfr] = np.sum(w_det[mask_GW_MRR])
            intrinsicRates_nonMRR[ind_mssfr] = np.sum(w_int[mask_GW_nonMRR])
            detectedRates_nonMRR[ind_mssfr] = np.sum(w_det[mask_GW_nonMRR])

            intrinsicRates_MRR_spin[ind_mssfr] = np.sum(w_int[mask_spin2zero_MRR])
            detectedRates_MRR_spin[ind_mssfr] = np.sum(w_det[mask_spin2zero_MRR])       
            intrinsicRates_nonMRR_spin[ind_mssfr] = np.sum(w_int[mask_spin1zero_nonMRR])
            detectedRates_nonMRR_spin[ind_mssfr] = np.sum(w_det[mask_spin1zero_nonMRR])     



    

        stringgg =  'MRR_nonMRR_ratio_' + GWname 

        df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
        namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
        nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'

        namez0_MRR = BPSmodelName + 'MRR: matches GW contour intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
        nameObs_MRR = BPSmodelName + 'MRR: matches GW contour observed (design LVK) [yr^{-1}]'
        namez0_nonMRR = BPSmodelName + 'nonMRR: matches GW contour intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
        nameObs_nonMRR = BPSmodelName + 'nonMRR: matches GW contour observed (design LVK) [yr^{-1}]'
        
        namez0_MRR_spin= BPSmodelName + 'MRR: matches GW contour spin2 < %s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
        nameObs_MRR_spin = BPSmodelName + 'MRR: matches GW contour spin2 < %s observed (design LVK) [yr^{-1}]'%spin_threshold
        namez0_nonMRR_spin= BPSmodelName + 'nonMRR: matches GW contour spin1 < %s intrinsic (z=0) [Gpc^{-3} yr^{-1}]'%spin_threshold
        nameObs_nonMRR_spin = BPSmodelName + 'nonMRR: matches GW contour spin1 < %s observed (design LVK) [yr^{-1}]'%spin_threshold



        df[namez0]                      = intrinsicRates
        df[nameObs]                     = detectedRates

        df[namez0_MRR]              = intrinsicRates_MRR
        df[nameObs_MRR]             = detectedRates_MRR
        df[namez0_nonMRR]              = intrinsicRates_nonMRR
        df[nameObs_nonMRR]             = detectedRates_nonMRR

        df[namez0_MRR_spin]          = intrinsicRates_MRR_spin
        df[nameObs_MRR_spin]         = detectedRates_MRR_spin
        df[namez0_nonMRR_spin]       = intrinsicRates_nonMRR_spin
        df[nameObs_nonMRR_spin]      = detectedRates_nonMRR_spin
     


        df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')







    fdata.close()






def createEmptyCSVplaceholderMetallicity(DCOtype='BBH'):


   


    DCOname=dictDCOtypeDCOlabel[DCOtype]
  
    channel_names = ['total', 'MRR',  'nonMRR', 'MRR_spin', 'nonMRR_spin']



    NAMES = []
    # stringgg = 'GW190814rate'

    for ind_m, m_ in enumerate(BPSnameslist):
        for ind_c, c_ in enumerate(channel_names):
            str_ = m_ + ' ' + c_ + '  [Msun^{-1}]'

            NAMES.append(str_)

            
            


    datas=[]
    nMetallicities = 53
    Zlist=['0_0001','0_00011', '0_00012', '0_00014', '0_00016', '0_00017',\
    '0_00019', '0_00022', '0_00024', '0_00027', '0_0003', '0_00034',\
    '0_00037', '0_00042', '0_00047', '0_00052', '0_00058', '0_00065',\
    '0_00073', '0_00081', '0_0009', '0_00101', '0_00113', '0_00126',\
    '0_0014', '0_00157', '0_00175', '0_00195', '0_00218', '0_00243',\
    '0_00272', '0_00303', '0_00339', '0_00378', '0_00422', '0_00471',\
    '0_00526', '0_00587', '0_00655', '0_00732', '0_00817', '0_00912',\
    '0_01018', '0_01137', '0_01269', '0_01416', '0_01581', '0_01765', '0_01971', '0_022', '0_0244', '0_02705', '0_03']

    for i in range(len(NAMES)):
        datas.append(np.zeros(nMetallicities))
        # datas.append(np.zeros(nMetallicities))
        
    
    df = pd.DataFrame(data=datas, index=NAMES, columns=Zlist).T
    df.columns =   df.columns.map(str)
    df.index.names = ['Z_i']
    df.columns.names = ['model']

    df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/MRRformationRatesperMetallicity_'+DCOname+ '_' +  '.csv')

    return 






def writeFormationRatesAndChannelsToFile_MRR_PerMetallicity(DCOtype='BBH', pathCOMPASOutput='/Volumes/Andromeda2/DATA/AllDCO_bugfix/',spin_threshold=0.05):
    
    
    # BPSnameslist = list(string.ascii_uppercase)[0:nBPSmodels]   
    channel_names = ['total', 'MRR',  'nonMRR', 'MRR_spin', 'nonMRR_spin']
    # temp = range(nModels+3)
    DCOname=dictDCOtypeDCOlabel[DCOtype]
    
 

    print('now at DCO type  ', DCOtype)
    for ind_m, bps_model in enumerate(BPSnameslist[:]):      
        print()
        print('now at bps label', bps_model)
        print('now at model ', alphabetDirDict[bps_model])
            
        # set always optimistic CE false, unless we are doing the optimistic variation
        OPTIMISTIC=False
        if (bps_model=='F') or (bps_model=='K'):
            OPTIMISTIC=True
            print('doing optimistic version of %s'%alphabetDirDict[bps_model])
            
        # path to datafile 
        # path = pathCOMPASOutput+alphabetDirDict[bps_model] + '/' + 'COMPASCompactOutput_'+DCOtype +'_'+bps_model+'.h5'
        path = pathCOMPASOutput+alphabetDirDict[bps_model] + '/' + 'COMPASOutput.h5'
            
        #But I want only within Hubble time 
        Data            = CC.COMPASData(path=path, lazyData=True, Mlower=5., \
                         Mupper=150, binaryFraction=1)
        Data.setCOMPASDCOmask(types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC)
        Data.setCOMPASData()
        
        metallicities = Data.metallicitySystems
        seeds    = Data.seeds[Data.Hubble==True]
        weights = Data.weight

        Data_totalMassEvolvedPerZ = Data.totalMassEvolvedPerZ
        Data_metallicityGrid = Data.metallicityGrid
        del Data 

        BPSmodelName = bps_model
        # path for files 
        path_dir = '/Volumes/Andromeda2/DATA/AllDCO_bugfix/'
        path_ = path_dir
        path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
        path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
                
        # read in data 
        fdata = h5.File(path, 'r')
        fDCO      = fdata['doubleCompactObjects'] # hdf5 file with the DCO information
        fSN       = fdata['supernovae']  # hdf5 file with the SN information
        #
        M1 = fDCO['M1'][...].squeeze()   # Compact object mass [Msun] of the initially more massive star
        M2 = fDCO['M2'][...].squeeze()  # Compact object mass [Msun] of the initially less massive star

        print('using indices')
        seedsDCO = fDCO['seed'][...].squeeze()  # get the seeds in the DCO file 
        seedsSN = fSN['randomSeed'][...].squeeze()    # get the seeds in the SN file 
        indices = np.sort(np.unique(seedsSN[1::2], return_index=True)[1])
        maskSNdco = np.in1d(seedsSN,  seedsDCO) # mask in the SNe files the SNe that correspond to our DCO
        whichSN = fSN['whichStar'][...].squeeze()[maskSNdco]  # this is 1 if the initially primary star goes SN and 2 if the secondary goes supernova     
        whichSN2 = whichSN[1::2][indices]

        # either SN2 = primary (1) and M1 is > M2, or SN2 = secondary & M1 < M2 
        # this takes into account (first term) rejuvenation 
        MRR_mask = ((whichSN2==1) & (M1>M2) ) | ((whichSN2==2) & (M1<M2)) 

        # M1LVK, M2LVK = obtainM1BHandM2BHassymetric(M1, M2)
        # chirp_mass = chirpmass(M1LVK, M2LVK)
        # mass_ratio_LVK =  M2LVK/M1LVK

        del M1
        del M2
        del whichSN2
        del whichSN
        del maskSNdco
        del indices
        del seedsSN
        # del seedsDCO
        del fDCO
        del fSN

        fdata.close()


        spin = COspin(data_path=path, state='he_depletion')  # set class 
        spin.setCOMPASData() # reads in the COMPAS DCO parameters 
        spinMZAMS1, spinMZAMS2  = spin.BaveraSpin()


        spinLVKM1, spinLVKM2 = np.zeros_like(spinMZAMS1), np.zeros_like(spinMZAMS1)
        spinLVKM1[MRR_mask] = spinMZAMS2[MRR_mask]  # MRR so M1 comes from M2ZAMS, we assign it spin from M2ZAMS
        spinLVKM1[~MRR_mask] = spinMZAMS1[~MRR_mask]  # no MRR so M1 comes from M1ZAMS, we assign it spin from M1ZAMS
        spinLVKM2[MRR_mask] = spinMZAMS1[MRR_mask]   # MRR so M2 comes from M1ZAMS, we assign it spin from M1ZAMS
        spinLVKM2[~MRR_mask] = spinMZAMS2[~MRR_mask]   # no MRR so M2 comes from M2ZAMS, we assign it spin from M2ZAMS     

        # chi_eff = ((spinLVKM1*M1LVK) + (spinLVKM2*M2LVK)) / (M1LVK + M2LVK)



        mask_spin2zero_MRR = (spinLVKM1>spin_threshold) & (MRR_mask==1)
        mask_spin1zero_nonMRR = (spinLVKM2>spin_threshold) & (MRR_mask==0)
        # dictChannelsBHNS = { 'classic':seedsClassic, \
        #                     'immediate CE':seedsSingleCE,\
        #                          'stable B no CEE':seedsOnlyStableMT, \
        #                      r'double-core CE':seedsDoubleCE,  \
        #                         'other':seedsOther\
        #                        }


        listt=[0.0001, 0.00011, 0.00012, 0.00014, 0.00016, 0.00017,\
               0.00019, 0.00022, 0.00024, 0.00027, 0.0003, 0.00034, \
               0.00037, 0.00042, 0.00047, 0.00052, 0.00058, 0.00065,\
               0.00073, 0.00081, 0.0009, 0.00101, 0.00113, 0.00126,\
               0.0014, 0.00157, 0.00175, 0.00195, 0.00218, 0.00243, \
               0.00272, 0.00303, 0.00339, 0.00378, 0.00422, 0.00471, \
               0.00526, 0.00587, 0.00655, 0.00732, 0.00817, 0.00912, \
               0.01018, 0.01137, 0.01269, 0.01416, 0.01581, 0.01765, 0.01971, 0.022, 0.0244, 0.02705, 0.03]

                 
        formationRateTotal           = np.zeros(len(listt))  
        formationRateMRR             = np.zeros(len(listt)) 
        formationRatenonMRR          = np.zeros(len(listt)) 
        formationRateMRR_spin        = np.zeros(len(listt)) 
        formationRatenonMRR_spin       = np.zeros(len(listt)) 
        # formationRateOther           = np.zeros(len(listt)) 

        # print('#Z =',len(Data.metallicityGrid))
        for nrZ, Z in enumerate(listt):
            # this if and else statement is a little hack. Data.metallicityGrid might not contains some metallicities since
            # it is based on the systems in the hdf5 file, but since the big Data files only contain the DCOs, it can be that a certain metallciity point
            # has 0 DCOs and thats what the data.metallicityGrid is based on            
            if Z in Data_metallicityGrid:
                maskZ = (metallicities == Z)
                formationRateTotal[nrZ] = np.sum(weights[maskZ]) # //floor weights
                # print('total 1 =',formationRateTotal[nrZ])
                # mask different channels
                InMRR       = np.in1d(seeds, seedsDCO[MRR_mask])
                InnonMRR  = np.in1d(seeds, seedsDCO[~MRR_mask])
                InMRR_spin       = np.in1d(seeds, seedsDCO[mask_spin2zero_MRR])
                InnonMRR_spin       = np.in1d(seeds, seedsDCO[mask_spin1zero_nonMRR])
                # InOther         = np.in1d(seeds, seedsOther)
                


                maskMRR         = (metallicities == Z) & (InMRR==1)
                masknonMRR    = (metallicities == Z) & (InnonMRR==1)
                maskMRR_spin        = (metallicities == Z) & (InMRR_spin==1)
                masknonMRR_spin       = (metallicities == Z) & (InnonMRR_spin==1)
                # maskOther           = (metallicities == Z) & (InOther==1)
                # del InClassic
                # del InOnlyStableMT
                # del InSingleCE
                # del InDoubleCE
                # del InOther


                # print('4')
                formationRateMRR[nrZ]         = np.sum(weights[maskMRR])
                formationRatenonMRR[nrZ]    = np.sum(weights[masknonMRR])
                formationRateMRR_spin[nrZ]        = np.sum(weights[maskMRR_spin]) 
                formationRatenonMRR_spin[nrZ]        = np.sum(weights[masknonMRR_spin])
                # formationRateOther[nrZ]           = np.sum(weights[maskOther])
            else:
                formationRateTotal[nrZ]           = 0
                formationRateMRR[nrZ]         = 0
                formationRatenonMRR[nrZ]    = 0
                formationRateMRR_spin[nrZ]        = 0
                formationRatenonMRR_spin[nrZ]        = 0
                # formationRateOther[nrZ]           = 0           

        # del seedsMRR
        # del seedsnonMR
        # del seedsSingleCE
        # del seedsDoubleCE
        # del seedsOther
        # del seeds           
        # mask the Z that are in the grid           
        maskZgridinZlist = np.in1d(listt, Data_metallicityGrid)

        formationRateTotal[maskZgridinZlist] = np.divide(formationRateTotal[maskZgridinZlist], Data_totalMassEvolvedPerZ) + 0 #lowerY        
        formationRateMRR[maskZgridinZlist] = np.divide(formationRateMRR[maskZgridinZlist], Data_totalMassEvolvedPerZ)
        formationRatenonMRR[maskZgridinZlist] = np.divide(formationRatenonMRR[maskZgridinZlist], Data_totalMassEvolvedPerZ)
        formationRateMRR_spin[maskZgridinZlist] = np.divide(formationRateMRR_spin[maskZgridinZlist], Data_totalMassEvolvedPerZ)
        formationRatenonMRR_spin[maskZgridinZlist] = np.divide(formationRatenonMRR_spin[maskZgridinZlist], Data_totalMassEvolvedPerZ)
        # formationRateOther[maskZgridinZlist] = np.divide(formationRateOther[maskZgridinZlist], Data_totalMassEvolvedPerZ)

        df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/MRRformationRatesperMetallicity_' +DCOname+ '_' +  '.csv', index_col=0)
        # namez0 = bps_model +' total  [Msun^{-1}]'
        for ind_c, c_ in enumerate(channel_names):
            str_ = bps_model + ' ' + c_ + '  [Msun^{-1}]'

            # total rates 
            if c_=='total':             
                df[str_] = formationRateTotal 
            elif c_=='MRR':
                df[str_] = formationRateMRR
            elif c_=='nonMRR':
                df[str_] = formationRatenonMRR
            elif c_=='MRR_spin':
                df[str_] = formationRateMRR_spin
            elif c_=='nonMRR_spin':
                df[str_] = formationRatenonMRR_spin
            # elif c_=='V_other':
            #     df[str_] = formationRateOther

        df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/MRRformationRatesperMetallicity_'+DCOname+ '_' +  '.csv')


    print('finished')

    return






def initialize_CSV_files_MRRformationChannels(DCOname='BHBH'):

    """" code used to initialize file for formation channel summary data used for Figure 1 in MRR paper """

    iii=0
    

    # CREATE PANDAS FILE 
    nModels=26
    BPSnameslist = list(string.ascii_uppercase)[0:nModels]

    NAMES = []
    stringgg =  'MRR_FormationChannels'


    headerDict_intrinsic = { 5:'MRR Channel VI intrinsic (z=0) [Gpc^{-3} yr^{-1}]',  6:'MRR Channel VII intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 7:'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 8:'All MRR intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 0:'MRR channel V intrinsic (z=0) [Gpc^{-3} yr^{-1}]',  1:'MRR channel I intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 2:'MRR channel II intrinsic (z=0) [Gpc^{-3} yr^{-1}]',3:'MRR channel III intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 4:'MRR channel IV intrinsic (z=0) [Gpc^{-3} yr^{-1}]'}
    headerDict_observed  = { 5:'MRR channel VI observed (design LVK) [yr^{-1}]',     6:'MRR channel VII observed (design LVK) [yr^{-1}]',    7:'All observed (design LVK) [yr^{-1}]',    8:'All MRR observed (design LVK) [yr^{-1}]', 0:'MRR channel V observed (design LVK) [yr^{-1}]', 1:'MRR channel I observed (design LVK) [yr^{-1}]', 2:'MRR channel II observed (design LVK) [yr^{-1}]', 3:'MRR channel III observed (design LVK) [yr^{-1}]', 4:'MRR channel IV observed (design LVK) [yr^{-1}]'}    
    enumerate_list = range(9)
    
    for ind_l, BPSmodelName in enumerate(BPSnameslist):
        for ind_c, Channel in enumerate(enumerate_list):
            namez0 = BPSmodelName + '_' + headerDict_intrinsic[Channel]
            nameObs = BPSmodelName + '_' + headerDict_observed[Channel]            
            NAMES.append(namez0)
            NAMES.append(nameObs)

    datas=[]

    for i in range(len(BPSnameslist)):
        for ii in range(9):
            datas.append(np.zeros_like(MSSFRnameslist))
            datas.append(np.zeros_like(MSSFRnameslist))
        
        
    df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
    df.columns =   df.columns.map(str)
    df.index.names = ['xyz']
    df.columns.names = ['m']

    # print(df) 

    df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')








def writeToRatesFile_MRR_FormationChannels(BPSmodelName='Z', DCOtype='BHNS'):
    """writes NS-BH rate to CSV file for all models"""

    if DCOtype=='BHNS':
        DCOname='BHNS'
    elif DCOtype=='BBH':
        DCOname='BHBH'
    elif DCOtype=='BNS':
        DCOname='NSNS'


    # path for files 
    path_dir = '/Volumes/Andromeda2/DATA/AllDCO_bugfix/'
    path_ = path_dir + alphabetDirDict[BPSmodelName] +'/'
    path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
            
    # read in data 
    fdata = h5.File(path, 'r')
    # obtain DCO masses
    M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
    M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
    MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)

    whichSN = fdata['supernovae']['whichStar'][...].squeeze() 

    DCOseeds = fdata['doubleCompactObjects']['seed'][...].squeeze()
    seedsSN = fdata['supernovae']['randomSeed'][...].squeeze()
    
    u, indices = np.unique(seedsSN, return_index=True)
    whichSN = fdata['supernovae']['whichStar'][...].squeeze()[indices] # get whichStar for first SN 
    
    mask_temp = ((whichSN==2) & (M1>M2) )
    mask_2 = ((whichSN==1) & (M1<M2))
    print('nr of weird reversals = %s'%np.sum(mask_temp))
    print('nr of normal reversals = %s'%np.sum(mask_2))

    maskMRR = ((whichSN==2) & (M1>M2) ) | ((whichSN==1) & (M1<M2) ) 

    del M1
    del M2
    del u
    del indices 



    # get intrinsic weights
    fparam_intrinsic = 'weights_intrinsic'
    # get detected weights
    fparam_detected = 'weights_detected'


    ####################################################
    ######### ITERATE  OVER  MSSFR  MODELS #############
    ####################################################

    fdata = h5.File(path)

    # DCOSeeds = fdata['doubleCompactObjects']['seed'][...].squeeze()
    seeds = fdata['doubleCompactObjects']['seed'][...].squeeze()
    channels = identify_formation_channels(seeds=seeds, file=fdata)
    headerDict_intrinsic = { 5:'MRR Channel VI intrinsic (z=0) [Gpc^{-3} yr^{-1}]',  6:'MRR Channel VII intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 7:'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 8:'All MRR intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 0:'MRR channel V intrinsic (z=0) [Gpc^{-3} yr^{-1}]',  1:'MRR channel I intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 2:'MRR channel II intrinsic (z=0) [Gpc^{-3} yr^{-1}]',3:'MRR channel III intrinsic (z=0) [Gpc^{-3} yr^{-1}]', 4:'MRR channel IV intrinsic (z=0) [Gpc^{-3} yr^{-1}]'}
    headerDict_observed  = { 5:'MRR channel VI observed (design LVK) [yr^{-1}]',     6:'MRR channel VII observed (design LVK) [yr^{-1}]',    7:'All observed (design LVK) [yr^{-1}]',    8:'All MRR observed (design LVK) [yr^{-1}]', 0:'MRR channel V observed (design LVK) [yr^{-1}]', 1:'MRR channel I observed (design LVK) [yr^{-1}]', 2:'MRR channel II observed (design LVK) [yr^{-1}]', 3:'MRR channel III observed (design LVK) [yr^{-1}]', 4:'MRR channel IV observed (design LVK) [yr^{-1}]'}    
    enumerate_list = range(9)

    stringgg =  'MRR_FormationChannels'
    df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)

    
    for nrC, Channel in enumerate(enumerate_list):  
        #Get the seeds that relate to sorted indices
        mask_C  = (channels==Channel) & (maskMRR==1)
        
    
        intrinsicRates = np.zeros(len(MSSFRnameslist))
        detectedRates = np.zeros(len(MSSFRnameslist))       


        for ind_mssfr, mssfr in enumerate(MSSFRnameslist):
            weightheader = 'w_' + mssfr
            w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()
            w_det = fdata[fparam_detected][weightheader][...].squeeze()

            if Channel==enumerate_list[-2]: 
                # TOTAL RATE
                intrinsicRates[ind_mssfr] = np.sum(w_int)
                detectedRates[ind_mssfr]  = np.sum(w_det)  
            elif Channel==enumerate_list[-1]: 
                # TOTAL RATE ALL 
                intrinsicRates[ind_mssfr] = np.sum(w_int[maskMRR])
                detectedRates[ind_mssfr]  = np.sum(w_det[maskMRR])  
            else:
                # CHANNEL RATE 
                intrinsicRates[ind_mssfr] = np.sum(w_int[mask_C])
                detectedRates[ind_mssfr]  = np.sum(w_det[mask_C])   



        namez0  = BPSmodelName + '_' + headerDict_intrinsic[Channel]
        nameObs = BPSmodelName + '_' + headerDict_observed[Channel]
        df[namez0] = intrinsicRates
        df[nameObs] = detectedRates


    df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/MRR_Project/dataFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


    fdata.close()

    return 










GW_MRR_list = ['GW151226','GW170729', 'GW190517_055101', 'GW190412','GW191109_010717'  ,'GW191103_012549', 'GW191126_115259']


GWTC1 = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729',  'GW170809', 'GW170814', 'GW170818', 'GW170823'] # 'GW170817', 
#     GWTC2 = ['GW190408_181802','GW190412','GW190413_052954','GW190413_134308','GW190421_213856',\
#     'GW190424_180648','GW190503_185404','GW190512_180714',\
#     'GW190513_205428','GW190514_065416','GW190517_055101','GW190519_153544','GW190521_074359',\
#     'GW190521','GW190527_092055','GW190602_175927','GW190620_030421','GW190630_185205','GW190701_203306',\
#     'GW190706_222641','GW190707_093326','GW190708_232457','GW190720_000836',\
#     'GW190727_060333','GW190728_064510','GW190731_140936','GW190803_022701','GW190828_063405',\
#     'GW190828_065509','GW190910_112807','GW190915_235702','GW190924_021846','GW190929_012149',\
#      'GW190930_133541', 'GW190425', 'GW190814', 'GW190426_152155']
    
GWTC2 = [ 'GW190408_181802', 'GW190412', 'GW190413_052954', 'GW190413_134308', 'GW190421_213856', 'GW190424_180648', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428', 'GW190514_065416', 'GW190517_055101', 'GW190519_153544', 'GW190521', 'GW190521_074359', 'GW190527_092055', 'GW190602_175927', 'GW190620_030421', 'GW190630_185205', 'GW190701_203306', 'GW190706_222641', 'GW190707_093326', 'GW190708_232457', 'GW190719_215514', 'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190731_140936', 'GW190803_022701', 'GW190814', 'GW190828_063405', 'GW190828_065509', 'GW190910_112807', 'GW190915_235702', 'GW190924_021846', 'GW190929_012149', 'GW190930_133541'] #, 'GW191103_012549', 'GW191105_143521', 'GW191109_010717', 'GW191113_071753', 'GW191126_115259', 'GW191127_050227', 'GW191129_134029', 'GW191204_110529', 'GW191204_171526', 'GW191215_223052', 'GW191216_213338', 'GW191222_033537', 'GW191230_180458', 'GW200112_155838', 'GW200128_022011', 'GW200129_065458', 'GW200202_154313', 'GW200208_130117', 'GW200208_222617', 'GW200209_085452', 'GW200210_092255', 'GW200216_220804', 'GW200219_094415', 'GW200220_061928', 'GW200220_124850', 'GW200224_222234', 'GW200225_060421', 'GW200302_015811', 'GW200306_093714', 'GW200308_173609', 'GW200311_115853', 'GW200316_215756', 'GW200322_091133']
#     GWTC3 = ['GW191103_012549', 'GW191126_115259','GW191109_010717' ]    
GWTC3 = ['GW191103_012549', 'GW191105_143521', 'GW191109_010717', 'GW191113_071753', 'GW191126_115259', 'GW191127_050227', 'GW191129_134029', 'GW191204_110529', 'GW191204_171526', 'GW191215_223052', 'GW191216_213338', 'GW191222_033537', 'GW191230_180458', 'GW200112_155838', 'GW200128_022011', 'GW200129_065458', 'GW200202_154313', 'GW200208_130117', 'GW200208_222617', 'GW200209_085452', 'GW200210_092254', 'GW200216_220804', 'GW200219_094415', 'GW200220_061928', 'GW200220_124850', 'GW200224_222234', 'GW200225_060421', 'GW200302_015811', 'GW200306_093714', 'GW200308_173609', 'GW200311_115853', 'GW200316_215756', 'GW200322_091133']             
    


BBHlist_GWTC_123 = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729', 'GW170809', \
                    'GW170814', 'GW170818', 'GW170823', 'GW190408_181802', 'GW190412', 'GW190413_052954',\
                    'GW190413_134308', 'GW190421_213856', 'GW190424_180648', 'GW190503_185404', 'GW190512_180714', \
                    'GW190513_205428', 'GW190514_065416', 'GW190517_055101', 'GW190519_153544', 'GW190521', \
                    'GW190521_074359', 'GW190527_092055', 'GW190602_175927', 'GW190620_030421', 'GW190630_185205', \
                    'GW190701_203306', 'GW190706_222641', 'GW190707_093326', 'GW190708_232457', 'GW190719_215514',\
                    'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190731_140936', 'GW190803_022701',\
                    'GW190814', 'GW190828_063405', 'GW190828_065509', 'GW190910_112807', 'GW190915_235702', \
                    'GW190924_021846', 'GW190929_012149', 'GW190930_133541', 'GW191103_012549', 'GW191105_143521',\
                    'GW191109_010717', 'GW191113_071753', 'GW191126_115259', 'GW191127_050227', 'GW191129_134029', \
                    'GW191204_110529', 'GW191204_171526', 'GW191215_223052', 'GW191216_213338', 'GW191222_033537', \
                    'GW191230_180458', 'GW200112_155838', 'GW200128_022011', 'GW200129_065458', 'GW200202_154313', \
                    'GW200208_130117', 'GW200208_222617', 'GW200209_085452', 'GW200210_092254', 'GW200216_220804', \
                    'GW200219_094415', 'GW200220_061928', 'GW200220_124850', 'GW200224_222234', 'GW200225_060421',\
                    'GW200302_015811', 'GW200306_093714', 'GW200308_173609', 'GW200311_115853', 'GW200316_215756', 'GW200322_091133']






# INITIALIZE
INITIALIZE_GENERAL = False # True #False #True #False#True #False
INITIALIZE_lightestBHfirst = False #True
INITIALIZE_MRR_FormationChannels = True #False
INITIALIZE_MRR_Spins = False
INITIALIZE_runMRR_nonMRR_ratio = False
# 
spin_threshold=0.1
INITIALIZE_perMetallicity = False



if INITIALIZE_perMetallicity==True:
    createEmptyCSVplaceholderMetallicity(DCOtype='BBH')



if INITIALIZE_GENERAL==True:
    # initialize_CSV_files_general(DCOname='BHNS')
    initialize_CSV_files_general(DCOname='BHBH')
    # initialize_CSV_files_general(DCOname='NSNS')


if INITIALIZE_runMRR_nonMRR_ratio==True:
    for GWname in BBHlist_GWTC_123:
        print('initializing ', GWname)
        # initialize_CSV_files_general(DCOname='BHNS')
        initialize_CSV_files_MRR_nonMRR_ratio(DCOname='BHBH', spin_threshold=spin_threshold, GWname=GWname)



if INITIALIZE_lightestBHfirst==True:
    # initialize_CSV_files_general(DCOname='BHNS')
    initialize_CSV_files_lightestBHfirst(DCOname='BHBH')
    # initialize_CSV_files_general(DCOname='NSNS')

if INITIALIZE_MRR_FormationChannels==True:
    # initialize MRR formation channel file 
    initialize_CSV_files_MRRformationChannels(DCOname='BHBH')


if INITIALIZE_MRR_Spins==True:
    # initialize MRR formation channel file 
    initialize_CSV_files_MRRspins(DCOname='BHBH', spin_threshold=spin_threshold)

#### RUN different simulation summaries : 
runMejecta = False 
runFormationChannels =False
runNSBH = False
runGeneralBHNS = False
runGeneralBHBH = False
runGeneralNSNS = False


runLightestFormsFirst=False
runMRR_FormationChannels = True
runMRR_Spins = False 
runMRR_nonMRR_ratio = False
runMRR_perMetallicity = False




if runMRR_perMetallicity==True:
    writeFormationRatesAndChannelsToFile_MRR_PerMetallicity(DCOtype='BBH', pathCOMPASOutput='/Volumes/Andromeda2/DATA/AllDCO_bugfix/', spin_threshold=spin_threshold)



if runMRR_FormationChannels==True:
    for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
        print(BPS)
        for DCOtype in ['BBH']:
            print('at DCOtype =', DCOtype)
            writeToRatesFile_MRR_FormationChannels(BPSmodelName=BPS, DCOtype=DCOtype)
            print('done with ', BPS)






if runMRR_nonMRR_ratio==True:
    for BPS in [  'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
    # for BPS in ['C']:
    # for BPS in ['N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
        print('inside')
        print(BPS)
        for DCOtype in ['BBH']:
            print('at DCOtype =', DCOtype)
            # for GWname in GW_MRR_list:
            # print('at GW =', GWname)
            writeToRatesFile_MRR_nonMRR_ratio(BPSmodelName=BPS, DCOtype=DCOtype, spin_threshold=spin_threshold, GWnameList=BBHlist_GWTC_123)

            print('done with ', BPS)
            print('----------------')
            print()





if runMRR_Spins==True:
    for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
        print(BPS)
        for DCOtype in ['BBH']:
            writeToRatesFile_MRR_Spins(BPSmodelName=BPS, DCOtype=DCOtype, spin_threshold=spin_threshold)




if runLightestFormsFirst==True:
    for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
        print(BPS)
        for DCOtype in ['BBH']:
            print('at DCOtype =', DCOtype)
            writeToRatesFile_lightestFormsFirst(BPSmodelName=BPS, DCOtype=DCOtype)
            print('done with ', BPS)



if runGeneralBHBH ==True:
    for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
        print(BPS)
        for DCOtype in ['BBH']:
            print('at DCOtype =', DCOtype)
            writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
            print('done with ', BPS)


