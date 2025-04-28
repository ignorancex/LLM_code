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


def writeToRatesFile_Mejecta(BPSmodelName='Z'):
	DCOtype='BHNS'

	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'



	# # path for files 
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

			


	fdata = h5.File(path)
	
	# obtain BH and NS masses
	M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)
	# del M1
	# del M2



	seedsSN = fdata['supernovae']['randomSeed'][...].squeeze()
	# get only SN seeds for DCOs 
	# maskSNdco = np.in1d(seedsSN,  Data.seeds[mask][maskZ]) 
	whichSN = fdata['supernovae']['whichStar'][...].squeeze()
	whichSN1 = whichSN[::2] # get whichStar for first SN 


	separationPreSN = fdata['supernovae']['separationBefore'][...].squeeze()
	separationPreSN2 = separationPreSN[1::2] # in Rsun. 

	maskNSBH = ((whichSN1==2) & (M1>M2) ) | ((whichSN1==1) & (M1<M2) ) 

	# get intrinsic weights

	fparam_intrinsic = 'weights_intrinsic'
	# get detected weights
	fparam_detected = 'weights_detected'


	####################################################
	######### ITERATE  OVER  MSSFR  MODELS #############
	####################################################
	intrinsicRates = np.zeros((6, len(MSSFRnameslist)))
	detectedRates = np.zeros((6, len(MSSFRnameslist)))
	namesEMlist = []

	for ind_mssfr, mssfr in enumerate(MSSFRnameslist):
#             print('mssfr =',ind_mssfr)
		weightheader = 'w_' + mssfr
		w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()
		w_det = fdata[fparam_detected][weightheader][...].squeeze()


		
		iii=0



		# needed for Qin spin model 


		for ind_chi, chi in enumerate([0.0, .5, 'Qin']):
			if chi=='Qin':
				# Qin 2018 spin model 
				BH_chi = QinBHspinmodel(separationPreSN2, M1, M2, maskNSBH)
			else:
				BH_chi   = chi * np.ones_like(w_int)

			for ind_Rns, NSradii in enumerate([11.5,13.0]):
				
				Rns = NSradii
				if ind_mssfr ==0:
					stringg = 'Rns_'+ str(NSradii) + 'km_' + 'spinBH_' + str(chi) 
					namesEMlist.append(stringg)

				NS_radii = Rns * np.ones_like(w_int)
				Mej = calculateEjectedMassMerger(m_ns=MNS, r_ns=NS_radii, m_bh=MBH, Xeff=BH_chi)
				mask_EM = (Mej>0)


				intrinsicRates[iii][ind_mssfr] = np.sum(w_int[mask_EM])
				detectedRates[iii][ind_mssfr] = np.sum(w_det[mask_EM])
				iii+=1


	   


	# rates0 =  df[name0]
	for iii in range(6):
		# print(iii)
		# print(shape(intrinsicRates))

		df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + namesEMlist[iii] + '.csv', index_col=0)
		namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

		df[namez0] = intrinsicRates[iii]
		df[nameObs] = detectedRates[iii] 


		df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + namesEMlist[iii] + '.csv')

	return



def writeToRatesFile_GW(BPSmodelName='Z', DCOtype='BHNS', GWname='GW200105'):



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
	
	
	# obtain BH and NS masses
	M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)
	Mchirp = chirpmass(MBH, MNS)


	# 90% confidence intervals:
	if GWname=='GW200105': 
		maskGW = ((MBH <= (8.9+1.2))  & (MBH>=(8.9-1.5))) & ((MNS <= (1.9+0.3))  & (MNS>=(1.9-0.2))) & ((Mchirp>=(3.41-0.07)) & (Mchirp<=(3.41+0.08))) & ((MNS/MBH>=(0.26-0.1)) & (MNS/MBH<=(0.26+0.35))) & (((MBH+MNS)>=(10.9-1.2)) & ((MBH+MNS)<=(10.9+1.1)))
	elif GWname=='GW200115':
		maskGW = ((MBH <= (5.7+1.8))  & (MBH>=(5.7-2.1))) & ((MNS <= (1.5+0.7))  & (MNS>=(1.5-0.3))) & ((Mchirp>=(2.42-0.07)) & (Mchirp<=(2.42+0.05))) & ((MNS/MBH>=(0.26-0.1)) & (MNS/MBH<=(0.26+0.35))) & (((MBH+MNS)>=(7.1 -1.4)) & ((MBH+MNS)<=(7.1 +1.5)))
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



	print(MSSFRnameslist)

	for ind_mssfr, mssfr in enumerate(MSSFRnameslist):
#             print('mssfr =',ind_mssfr)
		weightheader = 'w_' + mssfr
		print(ind_mssfr, weightheader)
		w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()
		w_det = fdata[fparam_detected][weightheader][...].squeeze()

		# mask for the GW event 
		intrinsicRates[ind_mssfr] = np.sum(w_int[maskGW])
		detectedRates[ind_mssfr] = np.sum(w_det[maskGW])



	   


	stringgg =  GWname

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


	fdata.close() 

	return






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

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
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


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


	fdata.close() 

	return






def writeToRatesFile_NSBH(BPSmodelName='Z'):
	"""writes NS-BH rate to CSV file for all models"""


	DCOtype='BHNS'
	DCOname='BHNS'





	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	# nModels=15
	# BPSnameslist = list(string.ascii_uppercase)[0:nModels]
	# modelDirList = ['fiducial', 'massTransferEfficiencyFixed_0_25', 'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75', \
	#                'unstableCaseBB', 'alpha0_5', 'alpha2_0', 'fiducial', 'rapid', 'maxNSmass2_0', 'maxNSmass3_0', 'noPISN',  'ccSNkick_100km_s', 'ccSNkick_30km_s', 'noBHkick' ]

	# alphabetDirDict =  {BPSnameslist[i]: modelDirList[i] for i in range(len(BPSnameslist))}

	path_ = path_dir
	path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
	path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'



	fdata = h5.File(path)
	
	# obtain BH and NS masses
	M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)


	whichSN = fdata['supernovae']['whichStar'][...].squeeze()[::2] # get whichStar for first SN 
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



	   


	stringgg =  'NSBH'

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')

	return





def writeToRatesFile_GW190814(BPSmodelName='Z', DCOtype='BNS'):
	print('NEED TO UPDATE THIS FUNCTION')

	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'


	# constants
	Zsolar=0.0142
	nModels = 12
	BPScolors       = sns.color_palette("husl", nModels)
	lw = 3.5
	Virgo         = '/Volumes/Virgo/DATA/BHNS/'
	VirgoAllDCO = '/Volumes/Virgo/DATA/AllDCO/'
	AndromedaBHNS = '/Volumes/Andromeda/DATA/BHNS/'
	AndromedaAllDCO  = '/Volumes/Andromeda/DATA/AllDCO/'

	alphabet = list(string.ascii_uppercase)
	BPSnameslist = alphabet[:nModels]

	BPSdir = ['fiducial/', 'fiducial/', 'alpha0_5/', 'alpha2_0/', 'unstableCaseBB/', 'rapid/', 'zeroBHkick/', 'massTransferEfficiencyFixed_0_25/', 'massTransferEfficiencyFixed_0_5/', 'massTransferEfficiencyFixed_0_75/', 'ccSNkick_100km_s/', 'ccSNkick_30km_s/']

	dictBPSnameToDir   = dict(zip(BPSnameslist, BPSdir))    
	dictBPSnameToColor = dict(zip(BPSnameslist, BPScolors))


	# READ IN DATA 
	if BPSmodelName in ['A', 'B', 'C',  'D', 'E', 'F', 'G', 'H',  'I' ,'J', 'K', 'L']:
		path1 = AndromedaAllDCO
		path1 = path1 + dictBPSnameToDir[BPSmodelName]


		path = path1 + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'
			
#             print(path)


	else:
		print('error: path does not exist')
#             print('given path:', path)




	fdata = h5.File(path)
	
	# obtain BH and NS masses
	M1 = fdata['doubleCompactObjects']['M1'][...].squeeze()
	M2 = fdata['doubleCompactObjects']['M2'][...].squeeze()
	MBH, MNS = obtainM1BHandM2BHassymetric(M1, M2)


	# 90% confidence intervals:
	maskGW190412 = (((MBH <= (23.2+1.1))  & (MBH>=23.2-1.0)) & ((MNS <= (2.85))  & (MNS>=2.35))) 

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
		w_int = fdata[fparam_intrinsic][weightheader][...].squeeze()[maskGW190412]
		w_det = fdata[fparam_detected][weightheader][...].squeeze()[maskGW190412]

		intrinsicRates[ind_mssfr] = np.sum(w_int)
		detectedRates[ind_mssfr] = np.sum(w_det)



	   


	stringgg =  'GW190814rate'

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2//Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')

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

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')



def initialize_CSV_files_GW(DCOname='BHNS', GWname='GW200115'):



	namesEMlist=[]


	iii=0
	

	# CREATE PANDAS FILE 
	nModels=26
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	NAMES = []
	stringgg = GWname

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

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')







#####################







# INITIALIZE
INITIALIZE_GENERAL = False #False #True #False#True #False
INITIALIZE_GW = True


if INITIALIZE_GW==True:
	initialize_CSV_files_GW(DCOname='BHNS', GWname='GW200105')
	initialize_CSV_files_GW(DCOname='BHNS', GWname='GW200115')


if INITIALIZE_GENERAL==True:
	initialize_CSV_files_general(DCOname='BHNS')
	initialize_CSV_files_general(DCOname='BHBH')
	initialize_CSV_files_general(DCOname='NSNS')



#### RUN different simulation summaries : 
runMejecta = False 
runFormationChannels =False 
runNSBH = False
runGeneralBHNS = False # True
runGeneralBHBH = False # True
runGeneralNSNS = False # True
runGW =True 



if runGW ==True:
	print('running GW BHNS')
	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' , 'R', 'S', 'T']:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_GW(BPSmodelName=BPS, DCOtype=DCOtype, GWname='GW200115')
			print('done with ', BPS)

	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' , 'R', 'S', 'T']:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_GW(BPSmodelName=BPS, DCOtype=DCOtype, GWname='GW200105')
			print('done with ', BPS)


if runMejecta ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' ]:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_Mejecta(BPSmodelName=BPS)
			print('done with ', BPS)




if runFormationChannels ==True:
	# for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O' ]:
	# 	print(BPS)
	# 	for DCOtype in ['BNS']:
	# 		print('at DCOtype =', DCOtype)
	# 		writeToRatesFile_FormationChannels(BPSmodelName=BPS, DCOtype='BNS')
	# 		print('done with ', BPS)


	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' ]:
		print(BPS)
		for DCOtype in ['BBH']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_FormationChannels(BPSmodelName=BPS, DCOtype='BBH')
			print('done with ', BPS)

if runNSBH ==True:
	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' ]:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_NSBH(BPSmodelName=BPS)
			print('done with ', BPS)




# if runGW200115 ==True:
# 	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' , 'R', 'S', 'T']:
# 		print(BPS)
# 		for DCOtype in ['BHNS']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_GW200115(BPSmodelName=BPS, DCOtype=DCOtype)
# 			print('done with ', BPS)

# if runGW200105 ==True:
# 	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' , 'R', 'S', 'T']:
# 		print(BPS)
# 		for DCOtype in ['BHNS']:
# 			print('at DCOtype =', DCOtype)
# 			writeToRatesFile_GW200105(BPSmodelName=BPS, DCOtype=DCOtype)
# 			print('done with ', BPS)



if runGeneralBHNS ==True:
	for BPS in ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' , 'R', 'S', 'T']:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)


if runGeneralNSNS ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
		print(BPS)
		for DCOtype in ['BBH']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)

if runGeneralBHBH ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T' ]:
		print(BPS)
		for DCOtype in ['BNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)














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

INITIALIZE_FormationChannels = False
INITIALIZE_NSBH= False #False#True
INITIALIZE=False #False #True 
INITIALIZE_GW190814 = False
INITIALIZE_EM =False

# INITIALIZE_NSBH= True #False#True
# INITIALIZE=True #False #True 
# INITIALIZE_GENERAL = True #False#True #False


# ['.0.0.0', '.1.1.1', '.2.1.1', '.3.1.1', '.1.1.2', '.2.1.2', '.3.1.2', '.1.1.3', '.2.1.3', '.3.1.3', '.1.2.1', '.2.2.1', '.3.2.1', '.1.2.2', '.2.2.2', '.3.2.2', '.1.2.3', '.2.2.3', '.3.2.3', '.1.3.1', '.2.3.1', '.3.3.1', '.1.3.2', '.2.3.2', '.3.3.2', '.1.3.3', '.2.3.3', '.3.3.3']


if INITIALIZE_FormationChannels==True:

	# namesEMlist=[]

	DCOname ='NSNS' 
	iii=0
	

	# CREATE PANDAS FILE 
	nModels=26
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	NAMES = []
	stringgg =  'AllDCOsimulation_formation_channels'

	for ind_l, BPSmodelName in enumerate(BPSnameslist):
		# str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
		# str_obs = str(L + ' observed (design LVK) [yr^{-1}]')

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

		NAMES.append(namez0)
		NAMES.append(nameObs)

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
		for ii in range(6):
			datas.append(np.zeros_like(MSSFRnameslist))
			datas.append(np.zeros_like(MSSFRnameslist))
		
		
	df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
	df.columns =   df.columns.map(str)
	df.index.names = ['xyz']
	df.columns.names = ['m']

	# print(df) 

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')





if INITIALIZE_GW190814==True:

	for dcotype in ['NSNS', 'BHBH', 'BHNS']:

		namesEMlist=[]

		DCOname=dcotype
		iii=0
		

		# CREATE PANDAS FILE 
		nModels=26
		BPSnameslist = list(string.ascii_uppercase)[0:nModels]

		NAMES = []
		stringgg = 'GW190814rate'

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

		df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')





if INITIALIZE_NSBH==True:


	namesEMlist=[]

	DCOname ='BHNS'
	iii=0
	

	# CREATE PANDAS FILE 
	nModels=26
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	NAMES = []
	stringgg = 'NSBH'

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
	df.index.names = ['.x.y.z']
	df.columns.names = ['m']

	# print(df) 

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')



#%% here 


# print(namesEMlist)


# for DCOtype in ['BHNS']:

# 	for Rns in enumerate()

# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/zeroBHkick/'
# 	modelname = 'G'
# 	print('modelname')
# 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)

# 	print('at DCOtype =', DCOtype)
# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# 	modelname = 'A'
# 	print('modelname')
# 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)


# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# 	modelname = 'B'
# 	print('modelname')
# 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)




# for DCOtype in ['BHNS']:

# 	for Rns in enumerate()

# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/zeroBHkick/'
# 	modelname = 'G'
# 	print('modelname')
# 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)

# 	print('at DCOtype =', DCOtype)
# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# 	modelname = 'A'
# 	print('modelname')
# 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)


# 	pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/fiducial/'
# 	modelname = 'B'
# 	print('modelname')
# 	writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=True)





# pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/alpha0_5/'
# modelname, DCOtype = 'M', 'BNS'
# writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)

# DCOtype='BHNS'
# writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)

# DCOtype='BBH'
# writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)



# pathCOMPASOutput = '/Volumes/Andromeda/DATA/AllDCO/alpha2_0/'
# modelname, DCOtype = 'N', 'BNS'
# writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)

# DCOtype='BHNS'
# writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)

# DCOtype='BBH'
# writeToRatesFile(modelname=modelname, pathCOMPASOutput=pathCOMPASOutput, DCOtype=DCOtype, Optmistic=False)






# #### INITIALIZE::: 
if INITIALIZE_EM==True:

	namesEMlist=[]

	DCOname ='BHNS'
	iii=0
	for ind_chi, chi in enumerate([0.0, .5, 'Qin']):
		# print(chi)
		iii+=1
		BH_chi   = chi 
		for ind_Rns, NSradii in enumerate([11.5,13.0]):
			iii+=1
			Rns = NSradii
			# if ind_mssfr ==0:
			# 	# print(chi)
			stringg = 'Rns_'+ str(NSradii) + 'km_' + 'spinBH_' + str(chi) 
			namesEMlist.append(stringg)


			# CREATE PANDAS FILE 
			nModels=26
			BPSnameslist = list(string.ascii_uppercase)[0:nModels]

			NAMES = []

			for ind_l, L in enumerate(BPSnameslist):
				str_z0 = str(L + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]')
				str_obs = str(L + ' observed (design LVK) [yr^{-1}]')
				NAMES.append(str_z0)
				NAMES.append(str_obs)
				


			datas=[]

			for i in range(len(BPSnameslist)):
				datas.append(np.zeros_like(MSSFRnameslist))
				datas.append(np.zeros_like(MSSFRnameslist))
				
			print(MSSFRnameslist)
			df = pd.DataFrame(data=datas, index=NAMES, columns=MSSFRnameslistCSV).T
			df.columns =   df.columns.map(str)
			df.index.names = ['xyz']
			df.columns.names = ['m']

			# print(df) 

			df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_2/rates_MSSFR_Models_'+DCOname+ '_' + stringg + '.csv')



