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



####################################
#path to the data




def writeToRatesFile_GENERAL(BPSmodelName='Z', DCOtype='BHNS'):






	# DCOtype='BHNS'

	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'



	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	# nModels=15
	# BPSnameslist = list(string.ascii_uppercase)[0:nModels]
	# modelDirList = ['fiducial', 'massTransferEfficiencyFixed_0_25', 'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75', \
	#                'unstableCaseBB', 'alpha0_5', 'alpha2_0', 'fiducial', 'rapid', 'maxNSmass2_0', 'maxNSmass3_0', 'noPISN',  'ccSNkick_100km_s', 'ccSNkick_30km_s', 'noBHkick' ]

	# alphabetDirDict =  {BPSnameslist[i]: modelDirList[i] for i in range(len(BPSnameslist))}



	#####



	path_ = path_dir
	path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
	path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
			



	# read in data 
	fdata = h5.File(path)
	
	# # obtain BH and NS masses
	# M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	# M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	# MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)
	# del M1
	# del M2


	# get intrinsic weights

	fparam_intrinsic = 'weights_intrinsic'
	# get detected weights

	fparam_detected = 'weights_detected'


	####################################################
	######### ITERATE  OVER  MSSFR  MODELS #############
	####################################################
	intrinsicRates = np.zeros(len(MSSFRnameslist))
	detectedRates = np.zeros(len(MSSFRnameslist))
	namesEMlist = []



	print(MSSFRnameslist)

	for ind_mssfr, mssfr in enumerate(MSSFRnameslist):
#             print('mssfr =',ind_mssfr)
		weightheader = 'w_' + mssfr
		print(ind_mssfr, weightheader)
		w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()
		w_det = fdata[fparam_detected][weightheader][...].squeeze()

		intrinsicRates[ind_mssfr] = np.sum(w_int)
		detectedRates[ind_mssfr] = np.sum(w_det)



	   


	stringgg =  'AllDCOsimulation'

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


	fdata.close() 

	return




def writeToRatesFile_FormationChannels(BPSmodelName='Z', DCOtype='BHNS'):






	# DCOtype='BHNS'

	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'



	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	path_ = path_dir
	path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
	path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
			
	# read in data 
	fdata = h5.File(path)

	# set optimistic true if that is the variation (H) 
	OPTIMISTIC=False
	if BPSmodelName=='H':
		OPTIMISTIC=True 



	# get formation channel Seeds!
	seedsPercentageClassic, seedsPercentageOnlyStableMT = returnSeedsPercentageClassicAndOnlyStableMT(pathCOMPASOutput=path_,\
	                                types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC, \
	                                binaryFraction=1)
	seedsClassic, percentageClassic = seedsPercentageClassic
	seedsOnlyStableMT, percentageOnlyStableMT = seedsPercentageOnlyStableMT



	seedsDoubleCE, percentageDoubleCE = returnSeedsPercentageDoubleCoreCEE(pathCOMPASOutput=path_,\
	                                types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC, \
	                                binaryFraction=1)


	seedsSingleCE, percentageSingleCE = returnSeedsPercentageSingleCoreCEE(pathCOMPASOutput=path_,\
	                                types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC, \
	                                binaryFraction=1)



	seedschannels = [seedsClassic, seedsOnlyStableMT, seedsSingleCE, seedsDoubleCE]

	seedsOther, percentageOther = returnSeedsPercentageOther(pathCOMPASOutput=path_,\
	                                types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC, \
	                                binaryFraction=1, channelsSeedsList=seedschannels)




	dictChannelsBHNS = { 'classic':seedsClassic, \
	                    'immediate CE':seedsSingleCE,\
	                         'stable B no CEE':seedsOnlyStableMT, \
	                     r'double-core CE':seedsDoubleCE,  \
	                        'other':seedsOther\
	                       }


	dictPercentages = { 'classic':percentageClassic, \
	                    'immediate CE':percentageSingleCE,\
	                         'stable B no CEE':percentageOnlyStableMT, \
	                     r'double-core CE':percentageDoubleCE,  \
	                        'other':percentageOther\
	                       } 




	# # obtain BH and NS masses
	# M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	# M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	# MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)
	# del M1
	# del M2


	# get intrinsic weights

	fparam_intrinsic = 'weights_intrinsic'
	# get detected weights

	fparam_detected = 'weights_detected'


	####################################################
	######### ITERATE  OVER  MSSFR  MODELS #############
	####################################################
	intrinsicRates = np.zeros(len(MSSFRnameslist))
	detectedRates = np.zeros(len(MSSFRnameslist))
	namesEMlist = []


	intrinsicRates_I = np.zeros(len(MSSFRnameslist))
	detectedRates_I = np.zeros(len(MSSFRnameslist))
	intrinsicRates_II = np.zeros(len(MSSFRnameslist))
	detectedRates_II = np.zeros(len(MSSFRnameslist))
	intrinsicRates_III = np.zeros(len(MSSFRnameslist))
	detectedRates_III = np.zeros(len(MSSFRnameslist))
	intrinsicRates_IV = np.zeros(len(MSSFRnameslist))
	detectedRates_IV = np.zeros(len(MSSFRnameslist))
	intrinsicRates_V = np.zeros(len(MSSFRnameslist))
	detectedRates_V = np.zeros(len(MSSFRnameslist))

	DCOSeeds = fdata['doubleCompactObjects']['seed'][...].squeeze()

	for ind_mssfr, mssfr in enumerate(MSSFRnameslist):

		# print('mssfr =',ind_mssfr, 'mssfr= ', mssfr)
		weightheader = 'w_' + mssfr
		# print(ind_mssfr, weightheader)
		w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()
		w_det = fdata[fparam_detected][weightheader][...].squeeze()

		intrinsicRates[ind_mssfr] = np.sum(w_int)
		detectedRates[ind_mssfr] = np.sum(w_det)
		
		for nrC, Channel in enumerate(dictChannelsBHNSList):
                        
#             #Get the seeds that relate to sorted indices
			seedsInterest = dictChannelsBHNS[Channel]
			mask_C = np.in1d(DCOSeeds, np.array(seedsInterest))
			if Channel=='classic':
				intrinsicRates_I[ind_mssfr] = np.sum(w_int[mask_C])
				detectedRates_I[ind_mssfr] = np.sum(w_det[mask_C])
			elif Channel=='stable B no CEE':
				intrinsicRates_II[ind_mssfr] = np.sum(w_int[mask_C])
				detectedRates_II[ind_mssfr] = np.sum(w_det[mask_C])
			elif Channel=='immediate CE':
				intrinsicRates_III[ind_mssfr] = np.sum(w_int[mask_C])
				detectedRates_III[ind_mssfr] = np.sum(w_det[mask_C])
			elif Channel=='double-core CE':
				intrinsicRates_IV[ind_mssfr] = np.sum(w_int[mask_C])
				detectedRates_IV[ind_mssfr] = np.sum(w_det[mask_C])
			elif Channel=='other':
				intrinsicRates_V[ind_mssfr] = np.sum(w_int[mask_C])
				detectedRates_V[ind_mssfr] = np.sum(w_det[mask_C])



	   

	stringgg =  'AllDCOsimulation_formation_channels'

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'

	namez0_I = BPSmodelName + 'channel I intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_I = BPSmodelName + 'channel I observed (design LVK) [yr^{-1}]'
	namez0_II = BPSmodelName + 'channel II intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_II = BPSmodelName + 'channel II observed (design LVK) [yr^{-1}]'
	namez0_III = BPSmodelName + 'channel III intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_III = BPSmodelName + 'channel III observed (design LVK) [yr^{-1}]'
	namez0_IV = BPSmodelName + 'channel IV intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_IV = BPSmodelName + 'channel IV observed (design LVK) [yr^{-1}]'
	namez0_V = BPSmodelName + 'channel V intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_V = BPSmodelName + 'channel V observed (design LVK) [yr^{-1}]'



	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates

	df[namez0_I] = intrinsicRates_I
	df[nameObs_I] = detectedRates_I
	df[namez0_II] = intrinsicRates_II
	df[nameObs_II] = detectedRates_II
	df[namez0_III] = intrinsicRates_III
	df[nameObs_III] = detectedRates_III 
	df[namez0_IV] = intrinsicRates_IV
	df[nameObs_IV] = detectedRates_IV 
	df[namez0_V] = intrinsicRates_V
	df[nameObs_V] = detectedRates_V  


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


	fdata.close() 

	return









#### FUNCTIONS TO INITIALIZE CSV FILES

def initialize_CSV_files_general(DCOname='BHNS'):



	namesEMlist=[]


	iii=0
	

	# CREATE PANDAS FILE 
	nModels=26
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	NAMES = []
	stringgg = 'AllDCOsimulation'

	for ind_l, L in enumerate(BPSnameslist):
		str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
		str_obs = str(L + ' observed (design LVK) [yr^{-1}]')
		NAMES.append(str_z0)
		NAMES.append(str_obs)
		
		


	datas=[]

	for i in range(len(BPSnameslist)):
		datas.append(np.zeros_like(MSSFRnameslist))
		datas.append(np.zeros_like(MSSFRnameslist))
		
		
	df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
	df.columns =   df.columns.map(str)
	df.index.names = ['xyz']
	df.columns.names = ['m']

	# print(df) 

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')



def writeToRatesFile_lightestFormsFirst(BPSmodelName='Z', DCOtype='BHNS'):
	"""writes NS-BH rate to CSV file for all models"""

	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'



	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	path_ = path_dir
	path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
	path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
			
	# read in data 
	fdata = h5.File(path)

	# set optimistic true if that is the variation (H) 
	OPTIMISTIC=False
	if BPSmodelName=='H':
		OPTIMISTIC=True 

	# obtain BH and NS masses
	M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)




	whichSN = fdata['supernovae']['whichStar'][...].squeeze() 
	maskSNnot3 = (whichSN ==1) | (whichSN==2)


	DCOseeds = fdata['doubleCompactObjects']['seed'][...].squeeze()
	seedsSN = fdata['supernovae']['randomSeed'][...].squeeze()
	
	u, indices = np.unique(seedsSN, return_index=True)
	# uniqueSN = seedsSN[indices]

	# bools = np.in1d(uniqueSN, DCOseeds)



	# maskSNspecial = (bools==1) & (maskSNnot3==1)

	# print(len(uniqueSN), )
	print(len(fdata['supernovae']['whichStar'][...].squeeze()))
	print(seedsSN)



	whichSN = fdata['supernovae']['whichStar'][...].squeeze()[indices] # get whichStar for first SN 

	print(len(seedsSN), len(set(seedsSN)))
	print(whichSN)
	print(set(whichSN))
	print(len(whichSN), len(M1), len(M2))

	
	mask_temp = ((whichSN==2) & (M1>M2) )
	mask_2 = ((whichSN==1) & (M1<M2))
	print('nr of weird reversals = %s'%np.sum(mask_temp))
	print('nr of normal reversals = %s'%np.sum(mask_2))





	maskNSBH = ((whichSN==2) & (M1>M2) ) | ((whichSN==1) & (M1<M2) ) 



	del M1
	del M2




	# get intrinsic weights

	fparam_intrinsic = 'weights_intrinsic'
	# get detected weights

	fparam_detected = 'weights_detected'


	####################################################
	######### ITERATE  OVER  MSSFR  MODELS #############
	####################################################
	intrinsicRates = np.zeros(len(MSSFRnameslist))
	detectedRates = np.zeros(len(MSSFRnameslist))
	namesEMlist = []

	for ind_mssfr, mssfr in enumerate(MSSFRnameslist):
#             print('mssfr =',ind_mssfr)
		weightheader = 'w_' + mssfr
		w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()[maskNSBH]
		w_det = fdata[fparam_detected][weightheader][...].squeeze()[maskNSBH]

		intrinsicRates[ind_mssfr] = np.sum(w_int)
		detectedRates[ind_mssfr] = np.sum(w_det)



	   


	stringgg =  'lightestFormsFirst'

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')
	fdata.close() 
	return



#### FUNCTIONS TO INITIALIZE CSV FILES

def initialize_CSV_files_lightestBHfirst(DCOname='BHNS'):



	namesEMlist=[]


	iii=0
	

	# CREATE PANDAS FILE 
	nModels=26
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	NAMES = []
	stringgg =  'lightestFormsFirst'

	for ind_l, L in enumerate(BPSnameslist):
		str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
		str_obs = str(L + ' observed (design LVK) [yr^{-1}]')
		NAMES.append(str_z0)
		NAMES.append(str_obs)
		
		


	datas=[]

	for i in range(len(BPSnameslist)):
		datas.append(np.zeros_like(MSSFRnameslist))
		datas.append(np.zeros_like(MSSFRnameslist))
		
		
	df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
	df.columns =   df.columns.map(str)
	df.index.names = ['xyz']
	df.columns.names = ['m']

	# print(df) 

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')




#####################




def initialize_CSV_files_MRRformationChannels(DCOname='BHBH'):

	# namesEMlist=[]

	iii=0
	

	# CREATE PANDAS FILE 
	nModels=26
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	NAMES = []
	stringgg =  'MRR_FormationChannels'

	for ind_l, BPSmodelName in enumerate(BPSnameslist):
		# str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
		# str_obs = str(L + ' observed (design LVK) [yr^{-1}]')

		namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'
		namez0_MRR = BPSmodelName + 'All MRR intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_MRR = BPSmodelName + 'All MRR observed (design LVK) [yr^{-1}]'

		namez0_I = BPSmodelName + 'MRR channel I intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_I = BPSmodelName + 'MRR channel I observed (design LVK) [yr^{-1}]'
		namez0_II = BPSmodelName + 'MRR channel II intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_II = BPSmodelName + 'MRR channel II observed (design LVK) [yr^{-1}]'
		namez0_III = BPSmodelName + 'MRR channel III intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_III = BPSmodelName + 'MRR channel III observed (design LVK) [yr^{-1}]'
		namez0_IV = BPSmodelName + 'MRR channel IV intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_IV = BPSmodelName + 'MRR channel IV observed (design LVK) [yr^{-1}]'
		namez0_V = BPSmodelName + 'MRR channel V intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_V = BPSmodelName + 'MRR channel V observed (design LVK) [yr^{-1}]'

		NAMES.append(namez0)
		NAMES.append(nameObs)

		NAMES.append(namez0_MRR)
		NAMES.append(nameObs_MRR)

		NAMES.append(namez0_I)
		NAMES.append(nameObs_I)
		NAMES.append(namez0_II)
		NAMES.append(nameObs_II)
		NAMES.append(namez0_III)
		NAMES.append(nameObs_III)
		NAMES.append(namez0_IV)
		NAMES.append(nameObs_IV)
		NAMES.append(namez0_V)
		NAMES.append(nameObs_V)


		
		


	datas=[]

	for i in range(len(BPSnameslist)):
		for ii in range(7):
			datas.append(np.zeros_like(MSSFRnameslist))
			datas.append(np.zeros_like(MSSFRnameslist))
		
		
	df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
	df.columns =   df.columns.map(str)
	df.index.names = ['xyz']
	df.columns.names = ['m']

	# print(df) 

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')







def writeToRatesFile_MRR_FormationChannels(BPSmodelName='Z', DCOtype='BHNS'):
	"""writes NS-BH rate to CSV file for all models"""

	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'


	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	path_ = path_dir
	path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
	path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
			
	# read in data 
	fdata = h5.File(path)


	# set optimistic true if that is the variation (H) 
	OPTIMISTIC=False
	if (BPSmodelName=='F') | (BPSmodelName=='K') :
		OPTIMISTIC=True 

	# obtain DCO masses
	M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)


	whichSN = fdata['supernovae']['whichStar'][...].squeeze() 
	maskSNnot3 = (whichSN ==1) | (whichSN==2)


	DCOseeds = fdata['doubleCompactObjects']['seed'][...].squeeze()
	seedsSN = fdata['supernovae']['randomSeed'][...].squeeze()
	
	u, indices = np.unique(seedsSN, return_index=True)

	print(len(fdata['supernovae']['whichStar'][...].squeeze()))
	print(seedsSN)


	whichSN = fdata['supernovae']['whichStar'][...].squeeze()[indices] # get whichStar for first SN 

	print(len(seedsSN), len(set(seedsSN)))
	print(whichSN)
	print(set(whichSN))
	print(len(whichSN), len(M1), len(M2))

	
	mask_temp = ((whichSN==2) & (M1>M2) )
	mask_2 = ((whichSN==1) & (M1<M2))
	print('nr of weird reversals = %s'%np.sum(mask_temp))
	print('nr of normal reversals = %s'%np.sum(mask_2))




	maskMRR = ((whichSN==2) & (M1>M2) ) | ((whichSN==1) & (M1<M2) ) 



	del M1
	del M2


	print('at formation channel calc.')
	fdata.close()

	# get formation channel Seeds!
	seedsPercentageClassic, seedsPercentageOnlyStableMT = returnSeedsPercentageClassicAndOnlyStableMT(pathCOMPASOutput=path,\
	                                types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC, \
	                                binaryFraction=1)
	seedsClassic, percentageClassic = seedsPercentageClassic
	seedsOnlyStableMT, percentageOnlyStableMT = seedsPercentageOnlyStableMT



	seedsDoubleCE, percentageDoubleCE = returnSeedsPercentageDoubleCoreCEE(pathCOMPASOutput=path,\
	                                types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC, \
	                                binaryFraction=1)


	seedsSingleCE, percentageSingleCE = returnSeedsPercentageSingleCoreCEE(pathCOMPASOutput=path,\
	                                types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC, \
	                                binaryFraction=1)



	seedschannels = [seedsClassic, seedsOnlyStableMT, seedsSingleCE, seedsDoubleCE]

	seedsOther, percentageOther = returnSeedsPercentageOther(pathCOMPASOutput=path,\
	                                types=DCOtype,  withinHubbleTime=True, optimistic=OPTIMISTIC, \
	                                binaryFraction=1, channelsSeedsList=seedschannels)

	print('00')


	dictChannelsBHNS = { 'classic':seedsClassic, \
	                    'immediate CE':seedsSingleCE,\
	                         'stable B no CEE':seedsOnlyStableMT, \
	                     r'double-core CE':seedsDoubleCE,  \
	                        'other':seedsOther\
	                       }


	dictPercentages = { 'classic':percentageClassic, \
	                    'immediate CE':percentageSingleCE,\
	                         'stable B no CEE':percentageOnlyStableMT, \
	                     r'double-core CE':percentageDoubleCE,  \
	                        'other':percentageOther\
	                       }




	# get intrinsic weights

	fparam_intrinsic = 'weights_intrinsic'
	# get detected weights

	fparam_detected = 'weights_detected'


	####################################################
	######### ITERATE  OVER  MSSFR  MODELS #############
	####################################################



	####################################################
	######### ITERATE  OVER  MSSFR  MODELS #############
	####################################################
	intrinsicRates = np.zeros(len(MSSFRnameslist))
	detectedRates = np.zeros(len(MSSFRnameslist))
	# namesEMlist = []
	intrinsicRates_MRR = np.zeros(len(MSSFRnameslist))
	detectedRates_MRR = np.zeros(len(MSSFRnameslist))

	intrinsicRates_I = np.zeros(len(MSSFRnameslist))
	detectedRates_I = np.zeros(len(MSSFRnameslist))
	intrinsicRates_II = np.zeros(len(MSSFRnameslist))
	detectedRates_II = np.zeros(len(MSSFRnameslist))
	intrinsicRates_III = np.zeros(len(MSSFRnameslist))
	detectedRates_III = np.zeros(len(MSSFRnameslist))
	intrinsicRates_IV = np.zeros(len(MSSFRnameslist))
	detectedRates_IV = np.zeros(len(MSSFRnameslist))
	intrinsicRates_V = np.zeros(len(MSSFRnameslist))
	detectedRates_V = np.zeros(len(MSSFRnameslist))

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

	    intrinsicRates_MRR[ind_mssfr] = np.sum(w_int[maskMRR])
	    detectedRates_MRR[ind_mssfr] = np.sum(w_det[maskMRR])

	    
	    for nrC, Channel in enumerate(dictChannelsBHNSList):
	                    
	#             #Get the seeds that relate to sorted indices
	        seedsInterest = dictChannelsBHNS[Channel]
	        mask_temp = np.in1d(DCOSeeds, np.array(seedsInterest))
	        mask_C = (mask_temp==1) & (maskMRR==1)
	        if Channel=='classic':
	            intrinsicRates_I[ind_mssfr] = np.sum(w_int[mask_C])
	            detectedRates_I[ind_mssfr] = np.sum(w_det[mask_C])
	        elif Channel=='stable B no CEE':
	            intrinsicRates_II[ind_mssfr] = np.sum(w_int[mask_C])
	            detectedRates_II[ind_mssfr] = np.sum(w_det[mask_C])
	        elif Channel=='immediate CE':
	            intrinsicRates_III[ind_mssfr] = np.sum(w_int[mask_C])
	            detectedRates_III[ind_mssfr] = np.sum(w_det[mask_C])
	        elif Channel=='double-core CE':
	            intrinsicRates_IV[ind_mssfr] = np.sum(w_int[mask_C])
	            detectedRates_IV[ind_mssfr] = np.sum(w_det[mask_C])
	        elif Channel=='other':
	            intrinsicRates_V[ind_mssfr] = np.sum(w_int[mask_C])
	            detectedRates_V[ind_mssfr] = np.sum(w_det[mask_C])




	fdata.close()  

	stringgg =  'MRR_FormationChannels'

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'
	namez0_MRR = BPSmodelName + 'All MRR intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_MRR = BPSmodelName + 'All MRR observed (design LVK) [yr^{-1}]'

	namez0_I = BPSmodelName + 'MRR channel I intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_I = BPSmodelName + 'MRR channel I observed (design LVK) [yr^{-1}]'
	namez0_II = BPSmodelName + 'MRR channel II intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_II = BPSmodelName + 'MRR channel II observed (design LVK) [yr^{-1}]'
	namez0_III = BPSmodelName + 'MRR channel III intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_III = BPSmodelName + 'MRR channel III observed (design LVK) [yr^{-1}]'
	namez0_IV = BPSmodelName + 'MRR channel IV intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_IV = BPSmodelName + 'MRR channel IV observed (design LVK) [yr^{-1}]'
	namez0_V = BPSmodelName + 'MRR channel V intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_V = BPSmodelName + 'MRR channel V observed (design LVK) [yr^{-1}]'



	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates
	df[namez0_MRR] = intrinsicRates_MRR
	df[nameObs_MRR] = detectedRates_MRR

	df[namez0_I] = intrinsicRates_I
	df[nameObs_I] = detectedRates_I
	df[namez0_II] = intrinsicRates_II
	df[nameObs_II] = detectedRates_II
	df[namez0_III] = intrinsicRates_III
	df[nameObs_III] = detectedRates_III 
	df[namez0_IV] = intrinsicRates_IV
	df[nameObs_IV] = detectedRates_IV 
	df[namez0_V] = intrinsicRates_V
	df[nameObs_V] = detectedRates_V  


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


	fdata.close()













def writeToRatesFile_MRR_Spins(BPSmodelName='Z', DCOtype='BHNS'):
	"""writes NS-BH rate to CSV file for all models"""

	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'


	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	path_ = path_dir
	path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
	path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
			
	# read in data 
	fdata = h5.File(path)


	# set optimistic true if that is the variation (H) 
	OPTIMISTIC=False
	if (BPSmodelName=='F') | (BPSmodelName=='K') :
		OPTIMISTIC=True 

	# obtain DCO masses
	M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)


	whichSN = fdata['supernovae']['whichStar'][...].squeeze() 
	maskSNnot3 = (whichSN ==1) | (whichSN==2)


	DCOseeds = fdata['doubleCompactObjects']['seed'][...].squeeze()
	seedsSN = fdata['supernovae']['randomSeed'][...].squeeze()
	
	u, indices = np.unique(seedsSN, return_index=True)




	whichSN = fdata['supernovae']['whichStar'][...].squeeze()[indices] # get whichStar for first SN 


	
	mask_temp = ((whichSN==2) & (M1>M2) )
	mask_2 = ((whichSN==1) & (M1<M2))





	MRR_mask = ((whichSN==2) & (M1>M2) ) | ((whichSN==1) & (M1<M2) ) 



	del M1
	del M2


	print('at spin calc.')
	fdata.close()



	spin = COspin(data_path=path)  # set class 
	spin.setCOMPASData() # reads in the COMPAS DCO parameters 
	spinMZAMS1, spinMZAMS2  = spin.BaveraSpin()


	spinLVKM1, spinLVKM2 = np.zeros(len(spinMZAMS1)), np.zeros(len(spinMZAMS1))
	spinLVKM1[MRR_mask] = spinMZAMS2[MRR_mask]  # MRR so M1 comes from M2ZAMS, we assign it spin from M2ZAMS
	spinLVKM1[~MRR_mask] = spinMZAMS1[~MRR_mask]  # no MRR so M1 comes from M1ZAMS, we assign it spin from M1ZAMS
	spinLVKM2[MRR_mask] = spinMZAMS1[MRR_mask]   # MRR so M2 comes from M1ZAMS, we assign it spin from M1ZAMS
	spinLVKM2[~MRR_mask] = spinMZAMS2[~MRR_mask]   # no MRR so M2 comes from M2ZAMS, we assign it spin from M2ZAMS     


	mask_LVKM1_spinning = (spinLVKM1 > 0.05) # definition of "spinning BH"
	mask_LVKM2_spinning = (spinLVKM2 > 0.05) # definition of "spinning BH"
	mask_anySpin = (spinLVKM1 > 0.05) | (spinLVKM2 > 0.05)


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

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'
	namez0_spinAny = BPSmodelName + 'All: at least one spin >0.05 intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_spinAny = BPSmodelName + 'All: at least one spin >0.05 observed (design LVK) [yr^{-1}]'

	namez0_spinning_LVKM1 = BPSmodelName + 'LVKM1 spinning > 0.05 intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_spinning_LVKM1 = BPSmodelName + 'LVKM1 spinning > 0.05 observed (design LVK) [yr^{-1}]'
	namez0_spinning_LVKM2= BPSmodelName + 'LVKM2 spinning > 0.05 intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs_spinning_LVKM2 = BPSmodelName + 'LVKM2 spinning > 0.05 observed (design LVK) [yr^{-1}]'



	df[namez0] 						= intrinsicRates
	df[nameObs] 					= detectedRates
	df[namez0_spinAny] 				= intrinsicRates_spinning
	df[nameObs_spinAny] 			= detectedRates_spinning

	df[namez0_spinning_LVKM1]  		= intrinsicRates_spinning_LVKM1
	df[nameObs_spinning_LVKM1] 		= detectedRates_spinning_LVKM1
	df[namez0_spinning_LVKM2] 		= intrinsicRates_spinning_LVKM2
	df[nameObs_spinning_LVKM2] 		= detectedRates_spinning_LVKM2
 


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


	fdata.close()







def initialize_CSV_files_MRRspins(DCOname='BHBH'):

	# namesEMlist=[]

	iii=0
	

	# CREATE PANDAS FILE 
	nModels=26
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	NAMES = []
	stringgg =  'MRR_Spins'

	for ind_l, BPSmodelName in enumerate(BPSnameslist):
		# str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
		# str_obs = str(L + ' observed (design LVK) [yr^{-1}]')

		namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'
		namez0_spinAny = BPSmodelName + 'All: at least one spin >0.05 intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_spinAny = BPSmodelName + 'All: at least one spin >0.05 observed (design LVK) [yr^{-1}]'

		namez0_spinning_LVKM1 = BPSmodelName + 'LVKM1 spinning > 0.05 intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_spinning_LVKM1 = BPSmodelName + 'LVKM1 spinning > 0.05 observed (design LVK) [yr^{-1}]'
		namez0_spinning_LVKM2= BPSmodelName + 'LVKM2 spinning > 0.05 intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs_spinning_LVKM2 = BPSmodelName + 'LVKM2 spinning > 0.05 observed (design LVK) [yr^{-1}]'

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

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')











# INITIALIZE
INITIALIZE_GENERAL = False # True #False #True #False#True #False
INITIALIZE_lightestBHfirst = False #True
INITIALIZE_MRR_FormationChannels = False
INITIALIZE_MRR_Spins = False



if INITIALIZE_GENERAL==True:
	# initialize_CSV_files_general(DCOname='BHNS')
	initialize_CSV_files_general(DCOname='BHBH')
	# initialize_CSV_files_general(DCOname='NSNS')

if INITIALIZE_lightestBHfirst==True:
	# initialize_CSV_files_general(DCOname='BHNS')
	initialize_CSV_files_lightestBHfirst(DCOname='BHBH')
	# initialize_CSV_files_general(DCOname='NSNS')

if INITIALIZE_MRR_FormationChannels==True:
	# initialize MRR formation channel file 
	initialize_CSV_files_MRRformationChannels(DCOname='BHBH')


if INITIALIZE_MRR_Spins==True:
	# initialize MRR formation channel file 
	initialize_CSV_files_MRRspins(DCOname='BHBH')

#### RUN different simulation summaries : 
runMejecta = False 
runFormationChannels =False 
runNSBH = False
runGeneralBHNS = False
runGeneralBHBH = False
runGeneralNSNS = False
  
runLightestFormsFirst=False
runMRR_FormationChannels = False
runMRR_Spins = True




if runMRR_Spins==True:
	for BPS in [ 'P', 'Q', 'R', 'S', 'T' ]:
		print(BPS)
		for DCOtype in ['BBH']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_MRR_Spins(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)



if runMRR_FormationChannels==True:
	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
		print(BPS)
		for DCOtype in ['BBH']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_MRR_FormationChannels(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)




# if runMRR_FormationChannels==True:
# 	for BPS in ['F', 'H', 'K']:
# 		print(BPS)
# 		for DCOtype in ['BBH']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_MRR_FormationChannels(BPSmodelName=BPS, DCOtype=DCOtype)
# 			print('done with ', BPS)



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

	# # INITIALIZE FILE 
	# namesEMlist=[]

	# DCOname ='BHBH'
	# iii=0
	

	# # CREATE PANDAS FILE 
	# nModels=26
	# BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	# NAMES = []
	# stringgg = 'lightestFormsFirst'

	# for ind_l, L in enumerate(BPSnameslist):
	# 	str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
	# 	str_obs = str(L + ' observed (design LVK) [yr^{-1}]')
	# 	NAMES.append(str_z0)
	# 	NAMES.append(str_obs)
		
		


	# datas=[]

	# for i in range(len(BPSnameslist)):
	# 	datas.append(np.zeros_like(MSSFRnameslist))
	# 	datas.append(np.zeros_like(MSSFRnameslist))
		
		
	# df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
	# df.columns =   df.columns.map(str)
	# df.index.names = ['.x.y.z']
	# df.columns.names = ['m']

	# # print(df) 

	# df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')

	# # run calculation 
	# for BPS in   ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' ]:
	# 	print(BPS)
	# 	for DCOtype in ['BBH']:
	# 		print('at DCOtype =', DCOtype)
	# 		writeToRatesFile_lightestFormsFirst(BPSmodelName=BPS, DCOtype=DCOtype)
	# 		print('done with ', BPS)














# if runMejecta ==True:
# 	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' ]:
# 		print(BPS)
# 		for DCOtype in ['BHNS']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_Mejecta(BPSmodelName=BPS)
# 			print('done with ', BPS)




# if runFormationChannels ==True:
# 	# for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O' ]:
# 	# 	print(BPS)
# 	# 	for DCOtype in ['BNS']:
# 	# 		print('at DCOtype =', DCOtype)
# 	# 		writeToRatesFile_FormationChannels(BPSmodelName=BPS, DCOtype='BNS')
# 	# 		print('done with ', BPS)


# 	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' ]:
# 		print(BPS)
# 		for DCOtype in ['BBH']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_FormationChannels(BPSmodelName=BPS, DCOtype='BBH')
# 			print('done with ', BPS)

# if runNSBH ==True:
# 	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' ]:
# 		print(BPS)
# 		for DCOtype in ['BHNS']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_NSBH(BPSmodelName=BPS)
# 			print('done with ', BPS)




# if runGeneralBHNS ==True:
# 	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' , 'R', 'S', 'T']:
# 		print(BPS)
# 		for DCOtype in ['BHNS']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
# 			print('done with ', BPS)


# if runGeneralNSNS ==True:
# 	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
# 		print(BPS)
# 		for DCOtype in ['BBH']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
# 			print('done with ', BPS)

# if runGeneralBHBH ==True:
# 	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
# 		print(BPS)
# 		for DCOtype in ['BNS']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
# 			print('done with ', BPS)














# Models to RUN 

# May 20: I am updating my data with the AllDCO focused runs :-) 

# this is an overwrite with better data (old ones are in BHNS copy)


# for DCOtype in ['BHNS', 'BBH', 'BNS']:
# 	print('at DCOtype =', DCOtype)
# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# 	modelname = 'A'
# 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)


# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# 	modelname = 'B'
# 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)


# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/zeroBHkick/'
# 	modelname = 'G'
# 	writeToRatesFile(modelname=modelname, pa thCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)

# INITIALIZE_FormationChannels = False
# INITIALIZE_NSBH= False #False#True
# INITIALIZE=False #False #True 
# INITIALIZE_GW190814 = False
# INITIALIZE_EM =False

# # INITIALIZE_NSBH= True #False#True
# # INITIALIZE=True #False #True 
# # INITIALIZE_GENERAL = True #False#True #False


# # ['.0.0.0', '.1.1.1', '.2.1.1', '.3.1.1', '.1.1.2', '.2.1.2', '.3.1.2', '.1.1.3', '.2.1.3', '.3.1.3', '.1.2.1', '.2.2.1', '.3.2.1', '.1.2.2', '.2.2.2', '.3.2.2', '.1.2.3', '.2.2.3', '.3.2.3', '.1.3.1', '.2.3.1', '.3.3.1', '.1.3.2', '.2.3.2', '.3.3.2', '.1.3.3', '.2.3.3', '.3.3.3']


# if INITIALIZE_FormationChannels==True:

# 	# namesEMlist=[]

# 	DCOname ='NSNS' 
# 	iii=0
	

# 	# CREATE PANDAS FILE 
# 	nModels=26
# 	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

# 	NAMES = []
# 	stringgg =  'AllDCOsimulation_formation_channels'

# 	for ind_l, BPSmodelName in enumerate(BPSnameslist):
# 		# str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
# 		# str_obs = str(L + ' observed (design LVK) [yr^{-1}]')

# 		namez0 = BPSmodelName + 'All intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
# 		nameObs = BPSmodelName + 'All observed (design LVK) [yr^{-1}]'

# 		namez0_I = BPSmodelName + 'channel I intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
# 		nameObs_I = BPSmodelName + 'channel I observed (design LVK) [yr^{-1}]'
# 		namez0_II = BPSmodelName + 'channel II intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
# 		nameObs_II = BPSmodelName + 'channel II observed (design LVK) [yr^{-1}]'
# 		namez0_III = BPSmodelName + 'channel III intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
# 		nameObs_III = BPSmodelName + 'channel III observed (design LVK) [yr^{-1}]'
# 		namez0_IV = BPSmodelName + 'channel IV intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
# 		nameObs_IV = BPSmodelName + 'channel IV observed (design LVK) [yr^{-1}]'
# 		namez0_V = BPSmodelName + 'channel V intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
# 		nameObs_V = BPSmodelName + 'channel V observed (design LVK) [yr^{-1}]'

# 		NAMES.append(namez0)
# 		NAMES.append(nameObs)

# 		NAMES.append(namez0_I)
# 		NAMES.append(nameObs_I)
# 		NAMES.append(namez0_II)
# 		NAMES.append(nameObs_II)
# 		NAMES.append(namez0_III)
# 		NAMES.append(nameObs_III)
# 		NAMES.append(namez0_IV)
# 		NAMES.append(nameObs_IV)
# 		NAMES.append(namez0_V)
# 		NAMES.append(nameObs_V)


		
		


# 	datas=[]

# 	for i in range(len(BPSnameslist)):
# 		for ii in range(6):
# 			datas.append(np.zeros_like(MSSFRnameslist))
# 			datas.append(np.zeros_like(MSSFRnameslist))
		
		
# 	df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
# 	df.columns =   df.columns.map(str)
# 	df.index.names = ['xyz']
# 	df.columns.names = ['m']

# 	# print(df) 

# 	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')





# if INITIALIZE_GW190814==True:

# 	for dcotype in ['NSNS', 'BHBH', 'BHNS']:

# 		namesEMlist=[]

# 		DCOname=dcotype
# 		iii=0
		

# 		# CREATE PANDAS FILE 
# 		nModels=26
# 		BPSnameslist = list(string.ascii_uppercase)[0:nModels]

# 		NAMES = []
# 		stringgg = 'GW190814rate'

# 		for ind_l, L in enumerate(BPSnameslist):
# 			str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
# 			str_obs = str(L + ' observed (design LVK) [yr^{-1}]')
# 			NAMES.append(str_z0)
# 			NAMES.append(str_obs)
			
			


# 		datas=[]

# 		for i in range(len(BPSnameslist)):
# 			datas.append(np.zeros_like(MSSFRnameslist))
# 			datas.append(np.zeros_like(MSSFRnameslist))
			
			
# 		df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
# 		df.columns =   df.columns.map(str)
# 		df.index.names = ['xyz']
# 		df.columns.names = ['m']

# 		# print(df) 

# 		df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')





# if INITIALIZE_NSBH==True:


# 	namesEMlist=[]

# 	DCOname ='BHNS'
# 	iii=0
	

# 	# CREATE PANDAS FILE 
# 	nModels=26
# 	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

# 	NAMES = []
# 	stringgg = 'NSBH'

# 	for ind_l, L in enumerate(BPSnameslist):
# 		str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
# 		str_obs = str(L + ' observed (design LVK) [yr^{-1}]')
# 		NAMES.append(str_z0)
# 		NAMES.append(str_obs)
		
		


# 	datas=[]

# 	for i in range(len(BPSnameslist)):
# 		datas.append(np.zeros_like(MSSFRnameslist))
# 		datas.append(np.zeros_like(MSSFRnameslist))
		
		
# 	df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
# 	df.columns =   df.columns.map(str)
# 	df.index.names = ['.x.y.z']
# 	df.columns.names = ['m']

# 	# print(df) 

# 	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')









# # #### INITIALIZE::: 
# if INITIALIZE_EM==True:

# 	namesEMlist=[]

# 	DCOname ='BHNS'
# 	iii=0
# 	for ind_chi, chi in enumerate([0.0, .5, 'Qin']):
# 		# print(chi)
# 		iii+=1
# 		BH_chi   = chi 
# 		for ind_Rns, NSradii in enumerate([11.5,13.0]):
# 			iii+=1
# 			Rns = NSradii
# 			# if ind_mssfr ==0:
# 			# 	# print(chi)
# 			stringg = 'Rns_'+ str(NSradii) + 'km_' + 'spinBH_' + str(chi) 
# 			namesEMlist.append(stringg)


# 			# CREATE PANDAS FILE 
# 			nModels=26
# 			BPSnameslist = list(string.ascii_uppercase)[0:nModels]

# 			NAMES = []

# 			for ind_l, L in enumerate(BPSnameslist):
# 				str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
# 				str_obs = str(L + ' observed (design LVK) [yr^{-1}]')
# 				NAMES.append(str_z0)
# 				NAMES.append(str_obs)
				


# 			datas=[]

# 			for i in range(len(BPSnameslist)):
# 				datas.append(np.zeros_like(MSSFRnameslist))
# 				datas.append(np.zeros_like(MSSFRnameslist))
				
# 			print(MSSFRnameslist)
# 			df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
# 			df.columns =   df.columns.map(str)
# 			df.index.names = ['xyz']
# 			df.columns.names = ['m']

# 			# print(df) 

# 			df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringg + '.csv')


# # print(namesEMlist)


# # for DCOtype in ['BHNS']:

# # 	for Rns in enumerate()

# # 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/zeroBHkick/'
# # 	modelname = 'G'
# # 	print('modelname')
# # 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)

# # 	print('at DCOtype =', DCOtype)
# # 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# # 	modelname = 'A'
# # 	print('modelname')
# # 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)


# # 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# # 	modelname = 'B'
# # 	print('modelname')
# # 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)




# # for DCOtype in ['BHNS']:

# # 	for Rns in enumerate()

# # 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/zeroBHkick/'
# # 	modelname = 'G'
# # 	print('modelname')
# # 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)

# # 	print('at DCOtype =', DCOtype)
# # 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# # 	modelname = 'A'
# # 	print('modelname')
# # 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)


# # 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# # 	modelname = 'B'
# # 	print('modelname')
# # 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)





# # pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/alpha0_5/'
# # modelname, DCOtype = 'M', 'BNS'
# # writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)

# # DCOtype='BHNS'
# # writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)

# # DCOtype='BBH'
# # writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)



# # pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/alpha2_0/'
# # modelname, DCOtype = 'N', 'BNS'
# # writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)

# # DCOtype='BHNS'
# # writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)

# # DCOtype='BBH'
# # writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)



