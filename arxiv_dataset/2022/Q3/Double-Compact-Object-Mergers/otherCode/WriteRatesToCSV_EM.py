# from __future__ import print_function
from __future__ import division
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

# import ClassCosmicIntegrator  as CI #Given settings and redshifts returns rates (2D arrays) Loads the data
# import coencodeVarious        as CV
from PostProcessingScripts import * 
from ClassFormationChannels_5mainchannels import * 


import pandas as pd


from astropy import units as u
from astropy import constants as const


# MSSFRnameslist = []
# MSSFRnameslist.append('000') # add phenomenological 

# for ind_SFR, SFR in enumerate(SFRs):
# 	ind_x = ind_SFR+1
# 	for ind_GSMF, GSMF in enumerate(GSMFs):
# 		ind_y = ind_GSMF + 1
# 		for ind_MZ, MZ in enumerate(MZs):
# 			ind_z = ind_MZ +1

# 			MSSFRnameslist.append('%s%s%s'%(ind_x, ind_y, ind_z))


# MSSFRnameslistCSV = ['.0.0.0', '.1.1.1', '.1.1.2', '.1.1.3', '.1.2.1', '.1.2.2', '.1.2.3', '.1.3.1', '.1.3.2', '.1.3.3', '.2.1.1', '.2.1.2', '.2.1.3', '.2.2.1', '.2.2.2', '.2.2.3', '.2.3.1', '.2.3.2', '.2.3.3', '.3.1.1', '.3.1.2', '.3.1.3', '.3.2.1', '.3.2.2', '.3.2.3', '.3.3.1', '.3.3.2', '.3.3.3']


# OLD:


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

# def calculateRisco(m_bhtemp, Xefftemp):
# 	# this is prograde orbit
# 	# see also https://duetosymmetry.com/tool/kerr-isco-calculator/

# 	# everything in cgs
# 	c = 2.99792458E10 #[cm s-1] 
# 	G = 6.67259E-8   
# 	Msun = 1.99E33 # gr
# 	Rsun = 6.96E10 # cm     
	
# 	factorFront =   ((G*m_bhtemp)/c**2) #m_bhtemp #s
	
# 	Z1 = 1 + (1 - Xefftemp**2)**(1/3) * ((1 + Xefftemp)**(1/3) + (1 - Xefftemp)**(1/3) )
# 	Z2 = np.sqrt((3* Xefftemp**2 + Z1**2))
	
# 	Risco = factorFront * (3 + Z2 - np.sqrt((3-Z1)*(3+Z1 +2*Z2)))
# 	return Risco



# def calculateEjectedMassMerger(m_ns, r_ns, m_bh, Xeff ):
#  # from 1807.00011, Eq 4 
# 	# returns M_rem in solar masses 
# 	# input r and m in solar masses and R sun. Xeff in [0,1] (symmetric) 
# 	# RNS in km
	
	
# 	# everything in cgs
# 	c = 2.99792458E10 #[cm s-1] 
# 	G = 6.67259E-8   
# 	Msun = 1.99E33 # gr
# 	Rsun = 6.96E10 # cm         
	
	
# 	# convert to cgs
# 	r_ns  = r_ns*0.1*10**6 #np.asarray([1.E6]* len(m_ns)) # to cm
# 	m_ns_cgs = Msun * m_ns
# 	m_bh_cgs = Msun * m_bh
	
	
# 	alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761
# 	C_NS = G * m_ns_cgs / (r_ns * c**2)
	
# 	R_isco = calculateRisco(m_bh_cgs, Xeff)
	
# 	R_isco_norm  = R_isco / (m_bh_cgs * (G/c**2)) 
	
# 	Q = m_bh_cgs / m_ns_cgs
	
# 	eta = Q / (1 + Q)**2
	
# 	FirstTerm  = alpha*(1 - 2*C_NS) / eta**(1/3)
# 	SecondTerm = beta* R_isco_norm * C_NS / eta 
	
# 	A = np.asarray(FirstTerm - SecondTerm + gamma)
# 	B = np.zeros_like(m_ns_cgs)
	
# 	Mrem_model = np.maximum(A,B)**(delta)
	
# 	Mrem_model /= Msun # in solar masses 
	
# 	# and the true M remnant mass (not normalized and in solar masses =)
# 	Mrem_solar = Mrem_model * m_ns_cgs  
# 	return Mrem_solar # in [Msun]

# def QinBHspinmodel(separationPreSN2, M1, M2, maskNSBH):
#     # returns spins from Qin et al + 2018 model 
    
#     # start with all zeseparationDCOFormationarationDCOFormationBH spins
#     BHspins = np.zeros_like(separationPreSN2)
    
#     # now add spins for NS-BH following Qin et al 2018:
#     # this is based on the separation prior to the second SN  
#     PeriodPreSN2 = convert_a_to_P_circular(separation=separationPreSN2*u.Rsun, M1=M1*u.Msun, M2=M2*u.Msun)
#     PeriodPreSN2 = PeriodPreSN2.to(u.d).value # in days 
    
#     # only do non zero spins
#     # first mask super tight NSBH that will get spin 1
#     maskNSBHChi1 = (np.log10(PeriodPreSN2) < -0.3) & (maskNSBH ==1)
#     BHspins[maskNSBHChi1] = np.ones(np.sum(maskNSBHChi1)) # fill with ones
# #     print('#total, = ', len(maskNSBHChi1))
# #     print('# with Chi = 1, = ', np.sum(maskNSBHChi1))
    
#     # now the variable spin
#     maskNSBHChi_var = (np.log10(PeriodPreSN2) > -0.3) &  (np.log10(PeriodPreSN2) < 0.3)  &(maskNSBH ==1)
#     m_, c_ = -5./3, 0.5 # from Qin + 2018 
#     spins_var =  m_ * np.log10(PeriodPreSN2[maskNSBHChi_var])  + c_   
#     BHspins[maskNSBHChi_var] = spins_var
# #     print('# with Chi var = ', np.sum(maskNSBHChi_var))
    
#     return BHspins
    
    
    
    
    
    


# Rns = 11.5 # in km 
# r_ns = np.asarray([Rns]*len(m1bh))
# for ind_chi, chi_bh in enumerate(listXbh):


# Mej = calculateEjectedMassMerger(m_ns=1.4, r_ns=Rns, m_bh=10, Xeff=0)
# print(Mej)

# NSmasses = np.linspace(1,2.5,10000)
# Niter = 1000
# # BH_chi = 0
# Arrays_minNSmassEjecta = [] # _Rns11chi0 _Rns13chi0 Rns11chi1 Rns13chi1
# for ind_chi, chi in enumerate([0.0, 0.5]):
# 	BH_chi   = chi * np.ones_like(NSmasses)
# 	for ind_Rns, NSradii in enumerate([11.5,13.0]):
# 		Rns = NSradii
# #         BH_chi=chi
# 		minNSmassEjecta = []
# 		for ind_bh, BHmass in enumerate(np.linspace(2.5, 20, Niter)):

# 			BHmasses = BHmass*np.ones_like(NSmasses)
# 			NS_radii = Rns * np.ones_like(NSmasses)
			
# 			Mej = calculateEjectedMassMerger(m_ns=NSmasses, r_ns=NS_radii, m_bh=BHmasses, Xeff=BH_chi)

# 			maskEjecta = (Mej > 0)
# 			# if there are solutions with Mej >0, append the first solution (with min BH mass)
# 			if len(NSmasses[maskEjecta]):
# 				minNSmassEjecta.append(NSmasses[maskEjecta][-1])
# 		#         print(minNSmassEjecta[-1])
# 			else:
# 				minNSmassEjecta.append(-1) # just append a non physical value that should not show up on plot
# 		print('R_ns, chi =', Rns, BH_chi )
# 		Arrays_minNSmassEjecta.append(minNSmassEjecta)
	
	




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



	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	nModels=15
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]
	modelDirList = ['fiducial', 'massTransferEfficiencyFixed_0_25', 'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75', \
	               'unstableCaseBB', 'alpha0_5', 'alpha2_0', 'fiducial', 'rapid', 'maxNSmass2_0', 'maxNSmass3_0', 'noPISN',  'ccSNkick_100km_s', 'ccSNkick_30km_s', 'noBHkick' ]

	alphabetDirDict =  {BPSnameslist[i]: modelDirList[i] for i in range(len(BPSnameslist))}



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

        # labelMej = []
        # for ind_chi, chi in enumerate([0.0, .5, 'Qin']):
        #     if chi=='Qin':
        #         # Qin 2018 spin model 
        #         BH_chi = QinBHspinmodel(separationPreSN2, M1, M2, maskNSBH)
        #     else:    
        #         BH_chi   = chi * np.ones_like(MNS)
            
        #     for ind_Rns, NSradii in enumerate([11.5,13.0]):
        #         Rns = NSradii
        #         NS_radii = Rns * np.ones_like(MNS)

        #         Mej = calculateEjectedMassMerger(m_ns=MNS, r_ns=NS_radii, m_bh=MBH, Xeff=BH_chi)

        #         maskEjecta = (Mej > 0)
                
        #         rate_Z[ii]+= np.sum(weightsALL[maskEjecta])
        #         labelMej.append('chi = ' + str(chi) + ' Rns = ' + str(NSradii) +' km')
        #         ii+=1
                    

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

		df = pd.read_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + namesEMlist[iii] + '.csv', index_col=0)
		namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
		nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

		df[namez0] = intrinsicRates[iii]
		df[nameObs] = detectedRates[iii] 


		df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + namesEMlist[iii] + '.csv')
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
	nModels=15
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]
	modelDirList = ['fiducial', 'massTransferEfficiencyFixed_0_25', 'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75', \
	               'unstableCaseBB', 'alpha0_5', 'alpha2_0', 'fiducial', 'rapid', 'maxNSmass2_0', 'maxNSmass3_0', 'noPISN',  'ccSNkick_100km_s', 'ccSNkick_30km_s', 'noBHkick' ]

	alphabetDirDict =  {BPSnameslist[i]: modelDirList[i] for i in range(len(BPSnameslist))}



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

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


	fdata.close() 

	return




def writeToRatesFile_FormationChannels(BPSmodelName='Z', DCOtype='BHNS'):




	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'



	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	nModels=15
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]
	modelDirList = ['fiducial', 'massTransferEfficiencyFixed_0_25', 'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75', \
	               'unstableCaseBB', 'alpha0_5', 'alpha2_0', 'fiducial', 'rapid', 'maxNSmass2_0', 'maxNSmass3_0', 'noPISN',  'ccSNkick_100km_s', 'ccSNkick_30km_s', 'noBHkick' ]

	alphabetDirDict =  {BPSnameslist[i]: modelDirList[i] for i in range(len(BPSnameslist))}



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

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
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


	df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')


	fdata.close() 

	return






def writeToRatesFile_NSBH(BPSmodelName='Z'):
	"""writes NS-BH rate to CSV file for all models"""


	DCOname=dic





	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'
	nModels=15
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]
	modelDirList = ['fiducial', 'massTransferEfficiencyFixed_0_25', 'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75', \
	               'unstableCaseBB', 'alpha0_5', 'alpha2_0', 'fiducial', 'rapid', 'maxNSmass2_0', 'maxNSmass3_0', 'noPISN',  'ccSNkick_100km_s', 'ccSNkick_30km_s', 'noBHkick' ]

	alphabetDirDict =  {BPSnameslist[i]: modelDirList[i] for i in range(len(BPSnameslist))}

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

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')
	fdata.close() 
	return




def writeToRatesFile_lightestFormsFirst(BPSmodelName='Z', DCOtype='BHNS'):
	"""writes NS-BH rate to CSV file for all models"""

	if DCOtype=='BHNS':
		DCOname='BHNS'
	elif DCOtype=='BBH':
		DCOname='BHBH'
	elif DCOtype=='BNS':
		DCOname='NSNS'




	print('nmodels =', nModels)
	# path for files 
	path_dir = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'

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



	   


	stringgg =  'lightestFormsFirst'

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')
	fdata.close() 
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

	df = pd.read_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv', index_col=0)
	namez0 = BPSmodelName + ' intrinsic (z=0) [Gpc^{-3} yr^{-1}]'
	nameObs = BPSmodelName + ' observed (design LVK) [yr^{-1}]'

	df[namez0] = intrinsicRates
	df[nameObs] = detectedRates 


	df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/ratesCSVfiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg  + '.csv')
	fdata.close() 
	return







#### RUN different simulation summaries : 

runMejecta = False
runFormationChannels =False
runNSBH = False
runGeneralBHNS = False
runGeneralBHBH = False
runGeneralNSNS = False
runLightestFormsFirst = True 
  




if runLightestFormsFirst==True:

	# INITIALIZE FILE 
	namesEMlist=[]

	DCOname ='BHBH'
	iii=0
	

	# CREATE PANDAS FILE 
	nModels=26
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]

	NAMES = []
	stringgg = 'lightestFormsFirst'

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

	df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/lightestBHformsFirst/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')

	# run calculation 
	for BPS in  BPSnameslist:
		print(BPS)
		for DCOtype in ['BHBH']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_lightestFormsFirst(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)








if runMejecta ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O' ]:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_Mejecta(BPSmodelName=BPS)
			print('done with ', BPS)




if runFormationChannels ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O' ]:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_FormationChannels(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)


	# for BPS in  [ 'O' ]:
	# 	print(BPS)
	# 	for DCOtype in ['BBH']:
	# 		print('at DCOtype =', DCOtype)
	# 		writeToRatesFile_FormationChannels(BPSmodelName=BPS, DCOtype='BBH')
	# 		print('done with ', BPS)

if runNSBH ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O' ]:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_NSBH(BPSmodelName=BPS)
			print('done with ', BPS)




if runGeneralBHNS ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O' ]:
		print(BPS)
		for DCOtype in ['BHNS']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)


if runGeneralNSNS ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O' ]:
		print(BPS)
		for DCOtype in ['BBH']:
			print('at DCOtype =', DCOtype)
			writeToRatesFile_GENERAL(BPSmodelName=BPS, DCOtype=DCOtype)
			print('done with ', BPS)

if runGeneralBHBH ==True:
	for BPS in  ['A','B',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O' ]:
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



# INITIALIZE_FormationChannels = True
# INITIALIZE_NSBH= True #False #False#True
# INITIALIZE=True #False #False #True 
# INITIALIZE_GENERAL = True #False #False #True #False#True #False
# INITIALIZE_GW190814 = True #False
# INITIALIZE_EM =True #False


INITIALIZE_FormationChannels = False
INITIALIZE_NSBH= False #False #False#True
INITIALIZE=False #False #False #True 
INITIALIZE_GENERAL =  False #False #False #False #True #False#True #False
INITIALIZE_GW190814 = False #False
INITIALIZE_EM =False #False


# ['.0.0.0', '.1.1.1', '.2.1.1', '.3.1.1', '.1.1.2', '.2.1.2', '.3.1.2', '.1.1.3', '.2.1.3', '.3.1.3', '.1.2.1', '.2.2.1', '.3.2.1', '.1.2.2', '.2.2.2', '.3.2.2', '.1.2.3', '.2.2.3', '.3.2.3', '.1.3.1', '.2.3.1', '.3.3.1', '.1.3.2', '.2.3.2', '.3.3.2', '.1.3.3', '.2.3.3', '.3.3.3']


if INITIALIZE_FormationChannels==True:

	# namesEMlist=[]

	DCOname ='BHNS' 
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

	df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')





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

		df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')





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

	df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')






if INITIALIZE_GENERAL==True:

	namesEMlist=[]

	DCOname ='BHNS' 
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

	df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringgg + '.csv')





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

			df.to_csv('/Users/floorbroekgaarden/Projects/BlackHole-NeutronStar/csvFiles/rates_MSSFR_Models_'+DCOname+ '_' + stringg + '.csv')


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



