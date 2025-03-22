
# from __future__ import division # un comment if you use python 2 !
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import time
import sys
import copy
#Quick fudge to make import from ../Scripts work

sys.path.append('../Scripts')


import gc


import ClassCosmicIntegrator  as CI #Given settings and redshifts returns rates (2D arrays) Loads the data
# import coencodeVarious        as CV
from PostProcessingScripts import * 
import ClassCOMPAS     as CC ###
import ClassFormationChannels as FC 
from ClassFormationChannels_5mainchannels import * 

import pandas as pd
from astropy import units as u
from astropy import constants as const


dictDCOtypeDCOlabel = {'BBH':'BHBH', 'BNS':'NSNS', 'BHNS':'BHNS'}



def createEmptyCSVplaceholder(DCOtype='BBH', nBPSmodels=15):


   


	DCOname=dictDCOtypeDCOlabel[DCOtype]
	BPSnameslist = list(string.ascii_uppercase)[0:nBPSmodels]   
	channel_names = ['total', 'I_classic', 'II_only_stable_MT', 'III_single_core_CE', 'IV_double_core_CE', 'V_other']



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

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_1/formationRatesTotalAndPerChannel_'+DCOname+ '_' +  '.csv')

	return 





def writeFormationRatesAndChannelsToFile(DCOtype='BBH', \
	pathCOMPASOutput='/Volumes/Andromeda/DATA/AllDCO_bugfix/',\
	 alphabetDirDict=[], nBPSmodels=15):
	
	
	BPSnameslist = list(string.ascii_uppercase)[0:nBPSmodels]   
	channel_names = ['total', 'I_classic', 'II_only_stable_MT', 'III_single_core_CE', 'IV_double_core_CE', 'V_other']
	# temp = range(nModels+3)
	DCOname=dictDCOtypeDCOlabel[DCOtype]
	
 

	print('now at DCO type  ', DCOtype)
	# for ind_m, bps_model in enumerate(BPSnameslist):  	
	for ind_m, bps_model in enumerate(['F', 'H', 'K']):    

		print()
		print('now at bps label,' bps_model)
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


		seedschannels = [seedsClassic, seedsOnlyStableMT, seedsSingleCE, seedsDoubleCE, seedsOther]




		dictChannelsBHNS = { 'classic':seedsClassic, \
							'immediate CE':seedsSingleCE,\
								 'stable B no CEE':seedsOnlyStableMT, \
							 r'double-core CE':seedsDoubleCE,  \
								'other':seedsOther\
							   }




		listt=[0.0001, 0.00011, 0.00012, 0.00014, 0.00016, 0.00017,\
			   0.00019, 0.00022, 0.00024, 0.00027, 0.0003, 0.00034, \
			   0.00037, 0.00042, 0.00047, 0.00052, 0.00058, 0.00065,\
			   0.00073, 0.00081, 0.0009, 0.00101, 0.00113, 0.00126,\
			   0.0014, 0.00157, 0.00175, 0.00195, 0.00218, 0.00243, \
			   0.00272, 0.00303, 0.00339, 0.00378, 0.00422, 0.00471, \
			   0.00526, 0.00587, 0.00655, 0.00732, 0.00817, 0.00912, \
			   0.01018, 0.01137, 0.01269, 0.01416, 0.01581, 0.01765, 0.01971, 0.022, 0.0244, 0.02705, 0.03]

				 
		formationRateTotal           = np.zeros(len(listt))  
		formationRateClassic         = np.zeros(len(listt)) 
		formationRateOnlyStableMT    = np.zeros(len(listt)) 
		formationRateSingleCE        = np.zeros(len(listt)) 
		formationRateDoubleCE        = np.zeros(len(listt)) 
		formationRateOther           = np.zeros(len(listt)) 

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
				InClassic       = np.in1d(seeds, seedsClassic)
				InOnlyStableMT  = np.in1d(seeds, seedsOnlyStableMT)
				InSingleCE      = np.in1d(seeds, seedsSingleCE)
				InDoubleCE      = np.in1d(seeds, seedsDoubleCE)
				InOther         = np.in1d(seeds, seedsOther)
				


				maskClassic         = (metallicities == Z) & (InClassic==1)
				maskOnlyStableMT    = (metallicities == Z) & (InOnlyStableMT==1)
				maskSingleCE        = (metallicities == Z) & (InSingleCE==1)
				maskDoubleCE        = (metallicities == Z) & (InDoubleCE==1)
				maskOther           = (metallicities == Z) & (InOther==1)
				del InClassic
				del InOnlyStableMT
				del InSingleCE
				del InDoubleCE
				del InOther


				# print('4')
				formationRateClassic[nrZ]         = np.sum(weights[maskClassic])
				formationRateOnlyStableMT[nrZ]    = np.sum(weights[maskOnlyStableMT])
				formationRateSingleCE[nrZ]        = np.sum(weights[maskSingleCE]) 
				formationRateDoubleCE[nrZ]        = np.sum(weights[maskDoubleCE])
				formationRateOther[nrZ]           = np.sum(weights[maskOther])
			else:
				formationRateTotal[nrZ] 		  = 0
				formationRateClassic[nrZ]         = 0
				formationRateOnlyStableMT[nrZ]    = 0
				formationRateSingleCE[nrZ]        = 0
				formationRateDoubleCE[nrZ]        = 0
				formationRateOther[nrZ]           = 0        	

		del seedsClassic
		del seedsOnlyStableMT
		del seedsSingleCE
		del seedsDoubleCE
		del seedsOther
		del seeds	        
		# mask the Z that are in the grid	        
		maskZgridinZlist = np.in1d(listt, Data_metallicityGrid)

		formationRateTotal[maskZgridinZlist] = np.divide(formationRateTotal[maskZgridinZlist], Data_totalMassEvolvedPerZ) + 0 #lowerY        
		formationRateClassic[maskZgridinZlist] = np.divide(formationRateClassic[maskZgridinZlist], Data_totalMassEvolvedPerZ)
		formationRateOnlyStableMT[maskZgridinZlist] = np.divide(formationRateOnlyStableMT[maskZgridinZlist], Data_totalMassEvolvedPerZ)
		formationRateSingleCE[maskZgridinZlist] = np.divide(formationRateSingleCE[maskZgridinZlist], Data_totalMassEvolvedPerZ)
		formationRateDoubleCE[maskZgridinZlist] = np.divide(formationRateDoubleCE[maskZgridinZlist], Data_totalMassEvolvedPerZ)
		formationRateOther[maskZgridinZlist] = np.divide(formationRateOther[maskZgridinZlist], Data_totalMassEvolvedPerZ)

		df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_1/formationRatesTotalAndPerChannel_'+DCOname+ '_' +  '.csv', index_col=0)
		# namez0 = bps_model +' total  [Msun^{-1}]'
		for ind_c, c_ in enumerate(channel_names):
			str_ = bps_model + ' ' + c_ + '  [Msun^{-1}]'

			# total rates 
			if c_=='total':         	
				df[str_] = formationRateTotal 
			elif c_=='I_classic':
				df[str_] = formationRateClassic
			elif c_=='II_only_stable_MT':
				df[str_] = formationRateOnlyStableMT
			elif c_=='III_single_core_CE':
				df[str_] = formationRateSingleCE
			elif c_=='IV_double_core_CE':
				df[str_] = formationRateDoubleCE
			elif c_=='V_other':
				df[str_] = formationRateOther


		df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/summary_data_Fig_1/formationRatesTotalAndPerChannel_'+DCOname+ '_' +  '.csv')


	print('finished')

	return


import string



print('number of Models = ', nModels)

# if you run this script for the first time, please run this by setting INITIALIZE = True
INITIALIZE=False

if INITIALIZE == True:
	createEmptyCSVplaceholder(DCOtype='BHNS', nBPSmodels=nModels)
	createEmptyCSVplaceholder(DCOtype='BNS',  nBPSmodels=nModels)
	createEmptyCSVplaceholder(DCOtype='BBH',  nBPSmodels=nModels)


print('do not forget to first Initialize if this is the first time you run this script')


# The things below are imported from PostProcessing.py
# nModels=17
# BPSnameslist = list(string.ascii_uppercase)[0:nModels]
# modelDirList = ['fiducial', 'massTransferEfficiencyFixed_0_25', 'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75', \
#                'unstableCaseBB', 'alpha0_5', 'alpha2_0', 'fiducial', 'rapid', 'maxNSmass2_0', 'maxNSmass3_0', 'noPISN',  'ccSNkick_100km_s', 'ccSNkick_30km_s', 'noBHkick', 'wolf_rayet_multiplier_0_1', 'wolf_rayet_multiplier_5' ]
# alphabetDirDict =  {BPSnameslist[i]: modelDirList[i] for i in range(len(BPSnameslist))}


writeFormationRatesAndChannelsToFile(DCOtype='BHNS', \
	pathCOMPASOutput='/Volumes/Andromeda/DATA/AllDCO_bugfix/',\
	 alphabetDirDict=alphabetDirDict, nBPSmodels=nModels)


writeFormationRatesAndChannelsToFile(DCOtype='BNS', \
	pathCOMPASOutput='/Volumes/Andromeda/DATA/AllDCO_bugfix/',\
	 alphabetDirDict=alphabetDirDict, nBPSmodels=nModels)

writeFormationRatesAndChannelsToFile(DCOtype='BBH', \
	pathCOMPASOutput='/Volumes/Andromeda/DATA/AllDCO_bugfix/',\
	 alphabetDirDict=alphabetDirDict, nBPSmodels=nModels)

# plotFormationRatePerZ(pathCOMPASOutput='/Volumes/Andromeda/DATA/AllDCO_bugfix/', alphabetDirDict=alphabetDirDict)    
	
	
