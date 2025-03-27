#pathToData = './COMPASOutput.h5'


import sys

import h5py  as h5   #for handling data format
import numpy as np  #for array handling
import os           #For checking existence data
import WriteH5File



def reduceH5file(pathToData, pathToNewData):

    # read data
    Data  = h5.File(pathToData)
    print("The main files I have at my disposal are:\n",list(Data.keys()))


    # print("The main files I have at my disposal are:\n",list(Data['formationChannels'].keys()))

	# Which Files do I want?
	# options: ['RLOF', 'XRayBinaries', 'commonEnvelopes', 'cppSource', 'doubleCompactObjects', 'formationChannels', 'pulsarEvolution', 'runtimes', 'supernovae', 'systems'])
    filesOfInterest   = {1:'doubleCompactObjects',2:'systems',\
	                     3:'supernovae', 4:'formationChannels'}

	# #Give a list of columns you want, if you want all, say ['All']
	# columnsOfInterest = {1:['All'],\
	#                      2:['All'],\
	#                      3:['SEED', 'MZAMS_1', 'MZAMS_2']}
    columnsOfInterest =	   {1:[  'M1', 'M1ZAMS', 'M2', 'M2ZAMS', 'Metallicity1',  \
							     'mergesInHubbleTimeFlag', 'optimisticCEFlag',   'seed', \
							    'separationDCOFormation', 'separationInitial', 'stellarType1', 'stellarType2', 'tc', 'tform', 'thetaSupernova1', 'weight'],\
							# 2:['ID', 'Metallicity1', 'Metallicity2', 'SEED', 'disbound', 'eccentricity',  'mass1', 'mass2', 'meanAnomaly1', 'meanAnomaly2', 'omega1', 'omega2', 'phi1', 'phi2', 'rk1', 'rk2', 'samplingPhase', 'separation', 'stellar_merger', 'theta1', 'theta2', 'weight'],\
							3:['MassCOCoreSN', 'MassCoreSN', 'MassStarCompanion', 'MassStarSN',  'kickVelocity', \
								   'randomSeed',  'separationAfter', 'separationBefore',  'whichStar'],\
							4:['All'] \
							}

	# #example of the seeds dictionary the actual one will be defined later
	# seedsOfInterest   = {1:None,\
	#                      2:None,\
	#                      3:None}
    
    seedsDCO = Data['doubleCompactObjects']['seed'][()]
    # seedsSystems = Data['systems']['SEED'][()]
    seedsSN = Data['supernovae']['randomSeed'][()]
    seedsFC = Data['formationChannels']['m_randomSeed'][()]


    seedsOfInterest   = {1:seedsDCO,\
                          # 2:seedsSystems,\
                          3:seedsSN,\
                          4:seedsFC}

    WriteH5File.reduceH5(pathToOld = pathToData, pathToNew = pathToNewData,\
                     dictFiles=filesOfInterest, dictColumns=columnsOfInterest, dictSeeds=seedsOfInterest)



if __name__ == "__main__":
    pathToData = (sys.argv[1])
    pathToNewData = (sys.argv[2])
    
#    print('test')
    reduceH5file(pathToData, pathToNewData)




