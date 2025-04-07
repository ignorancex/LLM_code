#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:07:54 2020

@author: remy.benmessaoud
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from my_functions import evaluation_functions as myEval
import pandas as pd
import os.path
import os
from math import isnan
import xlsxwriter
import pickle
#++++++=================================================
saveOpt = True
sph = 5 * 60 # 90 min
sop = 30 * 60 #  min
nRuns = 5
postProcessWindow = 14
thresh = 0.83 * postProcessWindow 
autorejcetFactor = 1.45
preIctal = 10
sustain = True
sustainPoints = 4
autoreject = True
tl = False
scoreType = 'contrast'

filtOpt = True
freqProp = 0.03
filtOrder = 20
#h = signal.firwin(filtOrder, freqProp)

"""       LDA SVC KNN """
weightsAbs = np.array([ 4 , 8 , 6])
weights = weightsAbs/np.sum(weightsAbs)
########################################
nHyper = 10
nThresh = 20
threshSpace = np.linspace(0.4 , 0.95 , num = nThresh )
postProcWinSpace = np.arange(7 , 7 + nHyper)
autorejcetFactorSpace = np.linspace(1.35 , 2 , num = nHyper )
sustainPointsSpace = np.arange(1 , 8)
alphaSpace = np.linspace(0.15 , 0.7 , num = nHyper )
n_estimatorsADASpace = np.array([50 , 60 , 70 , 80 , 90 , 100 , 110 , 120 , 130 , 140])
alphaMLPSpace = np.logspace(-5 , 1 , num = nHyper )
########################################


subjects2test = np.arange(1 , 25)
#subjects2test = np.array([1 , 2 , 3 , 4  , 5 , 6 , 7 , 8 , 9 , 10 , 11  , 14 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24])

subjects2compareWith = np.array([1 , 2 , 3 , 4  , 5 , 6 , 7 , 8 , 9 , 10 , 11  , 14 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24])
subjects2compareWithInds = (subjects2compareWith - np.ones(subjects2compareWith.shape)).astype(int)

nSubs=len(subjects2test)
# deal with path
currentPath = os.getcwd()
cDir = os.path.basename(currentPath)
if not(cDir == 'neuroProject'):
    newPath = os.path.join(currentPath , 'neuroProject')
    if os.path.isdir(newPath):
        os.chdir(newPath)
    else:
        raise ValueError(' problem of path can not find neuroPpoject directory \nHere is the current path : '\
                         .format(currentPath))
#=============================================================================
classifiers = ['svc' , 'rfc' , 'svc+rfc']
classifiers = ['rfc5feats' ]
nClassifiers = len(classifiers)
classifier = 'mlp'
#classifier = 'lda'
#tlOpt = [False , True]
#tlOpt = [True ]
nWin = len(postProcWinSpace)
nThresh = len(threshSpace)
nSustain  = len(sustainPointsSpace)

archiveSn = np.zeros((nClassifiers , nHyper , nWin , nThresh , nSustain))
archivecFPR = np.zeros((nClassifiers , nHyper , nWin , nThresh , nSustain))
subarchiveSn = np.zeros((nClassifiers , nHyper , nWin , nThresh , nSustain))
subarchivecFPR = np.zeros((nClassifiers , nHyper , nWin , nThresh , nSustain))
#
#archiveSnSubtot = np.zeros((nHyper , nWin , nThresh , nSustain))
#archivecFPRSubtot = np.zeros((nHyper , nWin , nThresh , nSustain))

for kHyper in  np.arange(0 , nHyper):
    autorejcetFactor = autorejcetFactorSpace[kHyper]
    #autorejcetFactor = 1.45
    print("autorejcetFactor = ".format(autorejcetFactor))
    n_estimatorsADA = n_estimatorsADASpace[kHyper]
    alphaMLP = alphaMLPSpace[kHyper]
    alpha = 0.31
#    subFolder = os.path.join(currentPath , "performanceEvaluationResults" , "autoRejectFactor_{}".format(autorejcetFactor) )
    
#    knnParam = knnParamSpace[kHyper]
#    RFestimators = RFestimatorsSpace[kHyper]
#    CregulSVC = CregulSVCSpace[kHyper]
    subFolder = os.path.join(currentPath , "perfDataTest5featsResults" , "autoReject_{}".format(autorejcetFactor) )
    
    subFolderEx = os.path.exists(subFolder)
    if not(subFolderEx):
        os.mkdir(subFolder)
    for kWin in range(nWin):                            ############## change here
        postProcessWindow = postProcWinSpace[kWin]
        for kSustain in range(nSustain):                    ############## change here
            sustainPoints = sustainPointsSpace[kSustain]
            for kThresh in range(nThresh):
                threshProportion = threshSpace[kThresh]
                thresh = threshProportion * postProcessWindow 
    
                # Create a Pandas Excel writer using XlsxWriter as the engine.
#                targetFileSpecBold = "cohort{}_runs_{}_sph_{}_sop_{}_preIctal_{}_postProc_{:.2f}_{:.2f}_TL_{}_score{}_sustain_{}_{}_autoreject_{:.2f}_VoteWithFilter_weights468.xlsx".format(nSubs  ,\
#                        nRuns , int(sph/60), int(sop/60) , preIctal , thresh , postProcessWindow , tl , scoreType, sustain , \
#                        sustainPoints , autorejcetFactor , weightsAbs[0] , weightsAbs[1] , weightsAbs[2] )
                
                targetFileSpecBold =\
"cohort{}_runs_{}_sph_{}_sop_{}_preIctal_{}_postProc_{:.2f}_{:.2f}_TL_{}_score{}_sustain_{}_{}_autoReject_{}_RF5feats.xlsx".format(\
       nSubs  , nRuns , int(sph/60), int(sop/60) , preIctal , thresh , postProcessWindow , tl , scoreType, sustain , \
                        sustainPoints , autorejcetFactor  )
                
                
                file_nameBold = os.path.join(subFolder, targetFileSpecBold )
                
                writer = pd.ExcelWriter(file_nameBold)#, engine='xlsxwriter')
                
                # Get the xlsxwriter workbook and worksheet objects.
                workbook  = writer.book       
                
                for clfInd in range(nClassifiers):
                    clf = classifiers[clfInd] 
                    sensitivityList = list()
                    specificityList = list()
                    predTimesList = list()
                    FPRList = list()
                    cFPRList = list()
                    pValList = list()
                       
                    patientNamesList = list()
                    numberSeizuresList = list()
                    interIctalHoursList = list()
                    badsPropList = list()
                    
                    totsensitivityList = list()
                    totspecificityList = list()
                    totFPRList = list()
                    totcFPRList = list()
                    totPredTime = 0
                    totPredTimeStd = 0
                    
                    
                    availableSubs = 0
                    totSeizures = 0
                    totinterIctalHours = 0 
                    
                    for kSub in range(nSubs):
                        sub = subjects2test[kSub]
                        #print('sub = {}'.format(sub))
#                        targetFile = os.path.join(currentPath \
#                    ,"performanceEvaluationData", 'sub{}'.format(sub), "sub_{}sop_{}_sph_{}_postProc_{:.2f}_{:.2f}_TL_{}_score_{}_sustain_{}_{}_autoreject_{}_VoteWithFilter_weights468.pkl".format(\
#                        sub , int(sop/60) , int(sph/60) , thresh , postProcessWindow , tl , scoreType , sustain , sustainPoints ,\
#                        autorejcetFactor ))
                        
                        targetFile = os.path.join(currentPath ,"perfDataTest5feats", 'sub{}'.format(sub),\
            "sub_{}sop_{}_sph_{}_postProc_{:.2f}_{:.2f}_TL_{}_score_{}_sustain_{}_{}_autoreject_{}_RF5feats.pkl".format(\
                        sub , int(sop/60) , int(sph/60) , thresh , postProcessWindow , tl , scoreType , sustain , sustainPoints ,\
                        autorejcetFactor ))

                        if os.path.exists(targetFile):
                            f = open(targetFile,"rb")
                            perfDict = pickle.load(f)
                            f.close()
                        else:
                            print(targetFile)
                            raise ValueError('data file not found ')
                            
                           
                        sensitivity = perfDict["sensitivity"]
                        specificity = perfDict["specificity"]
                        predTimes = perfDict["predTimes"]
                        FPR = perfDict["FPR"]
                        cFPR = perfDict["cFPR"]
                        
                        totsensitivityList.append(sensitivity)
                        totspecificityList.append(specificity)
                        totFPRList.append(FPR)
                        totcFPRList.append (cFPR)
                        
                        pVal = perfDict["pVal"]
                        sensitivityStd = perfDict["sensitivityStd"]
                        specificityStd = perfDict["specificityStd"]
                        predTimesStd = perfDict["predTimesStd"]
                        FPRStd = perfDict["FPRStd"]
                        cFPRStd = perfDict["cFPRStd"]
                        
                        interIctalHours = perfDict["interIctalHours"]
                        interIctalHoursList.append("{:.2f}".format(interIctalHours))
                        nSeizures = perfDict["numberSeizures"]
                        nOrigSeizures = perfDict["numberSeizuresOrig"]
                        totSeizures = totSeizures + nSeizures
                        totinterIctalHours = totinterIctalHours + interIctalHours
                        patientNamesList.append(perfDict["patientName"])
                        badsPropList.append("{:.2f}".format(perfDict["badsProp"]))
                        
                        if nSeizures != nOrigSeizures:
                            numberSeizuresList.append("{} ({})".format(nSeizures , nOrigSeizures))
                        else:
                            numberSeizuresList.append("{}".format(nSeizures))
                        #    print("=============== Results for subject {} ===========\nSensitivity = {} ± {}\nFPR = {} ± {}\ncorrFPR = {}±{}\nprediction time = {}±{} min\npValue = {}\n==========================================".\
                        #          format(sub , mSn , sSn , mFPR , sFPR , mCorrFPR , sCorrFPR , mPredTime / 60 , sPredTime/60 , mpVal))
                        
                        sensitivityList.append("{:.2f} ± {:.2f}".format(sensitivity[0 , clfInd] , sensitivityStd[0 , clfInd]))
                        specificityList.append("{:.2f} ± {:.2f}".format(specificity[0 , clfInd] , specificityStd[0 , clfInd]))            
                        if not(isnan(predTimes[0 , clfInd])):
                            predTimesList.append("{:.2f} ± {:.2f}".format(predTimes[0 , clfInd] , predTimesStd[0 , clfInd]))
                        else:
                            predTimesList.append("-")
                            
                        FPRList.append("{:.3f} ± {:.3f}".format(FPR[0 , clfInd] , FPRStd[0 , clfInd]))
                        cFPRList.append("{:.3f} ± {:.3f}".format(cFPR[0 , clfInd]  , cFPRStd[0 , clfInd] ))
                        pValList.append("{:.2e}".format(pVal[0 , clfInd]))
                
                        if not(isnan(predTimes[0 , clfInd])):
                            totPredTime =totPredTime +  predTimes[0 , clfInd]
                            totPredTimeStd =totPredTimeStd +  predTimesStd[0 , clfInd]
                            availableSubs = availableSubs + 1
                            
                    ###################################################################################
                    
                    totsensitivity =   np.vstack(totsensitivityList)  
                    totspecificity = np.vstack(totspecificityList)
                #    print(totsensitivity)
                #    print(np.mean(totsensitivity[: , clfInd]))
                    
                    FPR = np.vstack(totFPRList) 
                    cFPR = np.vstack(totcFPRList) 
                    """ Now we add the average row"""
                    nSubs = len(subjects2test)
                    
                    patientNamesList.append("Total")
                    numberSeizuresList.append("{}".format(totSeizures))
                    interIctalHoursList.append("{:.2f}".format(totinterIctalHours))
                    badsPropList.append("-")
                    
                    sensitivityList.append("{:.2f} ± {:.2f}".format(np.mean(totsensitivity[: , clfInd]) , np.std(totsensitivity[: , clfInd])))
                    specificityList.append("{:.2f} ± {:.2f}".format(np.mean(totspecificity[: , clfInd]) , np.std(totspecificity[: , clfInd])))
                    
                    predTimesList.append("{:.2f} ± {:.2f}".format(totPredTime/availableSubs  ,  totPredTimeStd/availableSubs) )
                    
                    FPRList.append("{:.3f} ± {:.3f}".format(np.mean(FPR[: , clfInd]) , np.std(FPR[: , clfInd])))
                    cFPRList.append("{:.3f} ± {:.3f}".format(np.mean(cFPR[: , clfInd]) , np.std(cFPR[: , clfInd])))
                    
                    
                    totpVal = myEval.getPValue(np.mean(FPR[: , clfInd]) , sop/3600 , int(np.round(np.mean(totsensitivity[: , clfInd])/100  * totSeizures/ nSubs))  , int(np.round(totSeizures/ nSubs)))
                    pValList.append("{:.2e}".format(totpVal))
                    
                    if True:
                        archiveSn[clfInd , kHyper , kWin , kThresh , kSustain] = np.mean(totsensitivity[: , clfInd])
                        archivecFPR[clfInd , kHyper , kWin , kThresh , kSustain] = np.mean(cFPR[: , clfInd])

                    ###################################### Do the same with subTotal ###############################################
                    patientNamesList.append("Subtotal")
                    numberSeizuresList.append("-")
                    interIctalHoursList.append("-".format(totinterIctalHours))
                    badsPropList.append("-")
                    
                    sensitivityList.append("{:.2f} ± {:.2f}".format(np.mean(totsensitivity[subjects2compareWithInds , clfInd]) , np.std(totsensitivity[subjects2compareWithInds , clfInd])))
                    specificityList.append("{:.2f} ± {:.2f}".format(np.mean(totspecificity[subjects2compareWithInds , clfInd]) , np.std(totspecificity[subjects2compareWithInds , clfInd])))
                    
                    predTimesList.append("-")
                    
                    FPRList.append("{:.3f} ± {:.3f}".format(np.mean(FPR[subjects2compareWithInds , clfInd]) , np.std(FPR[subjects2compareWithInds , clfInd])))
                    cFPRList.append("{:.3f} ± {:.3f}".format(np.mean(cFPR[subjects2compareWithInds , clfInd]) , np.std(cFPR[subjects2compareWithInds , clfInd])))
                    
                    
                    totpVal = myEval.getPValue(np.mean(FPR[: , clfInd]) , sop/3600 , int(np.round(np.mean(totsensitivity[subjects2compareWithInds , clfInd])/100  * 64))  , 64)
                    pValList.append("{:.2e}".format(totpVal))
                    
                    if True:
                        subarchiveSn[clfInd , kHyper , kWin , kThresh , kSustain] = np.mean(totsensitivity[subjects2compareWithInds , clfInd])
                        subarchivecFPR[clfInd , kHyper , kWin , kThresh , kSustain] = np.mean(cFPR[subjects2compareWithInds , clfInd])
                    #####################################################################################
                    
                    table = {"Patient" : patientNamesList , "#Seizures" : numberSeizuresList , "Interictal hours" :  interIctalHoursList ,\
                             "Artifacts (%)" : badsPropList , "Sensitivity (%)" : sensitivityList , "Specificity (%)" : specificityList \
                             , "Pred.Time (min)" : predTimesList ,"FPR (/h)" : FPRList , "corrFPR (/h)" : cFPRList , "p" : pValList}
                    #print(table)
                    df = pd.DataFrame(table)
                    #print(df)
                #    if tl:
                #        targetFileSpec = "cohort{}_clf_{}_runs_{}_sph_{}_sop_{}_preIctal_{}_postWin_{}_TransferLearning".format(nSubs , clf \
                #                                , nRuns , int(sph/60), int(sop/60) , preIctal , postProcessWindow)
                #    else:
                #        targetFileSpec = "cohort{}_clf_{}_runs_{}_sph_{}_sop_{}_preIctal_{}_postWin_{}".format(nSubs , clf \
                #                                , nRuns , int(sph/60), int(sop/60)  , preIctal , postProcessWindow)
                ##    file_name = os.path.join(myPath , targetFileSpec )
                    ############################################
                    sheetName = '{}'.format(clf)
                    worksheet = workbook.add_worksheet(sheetName)
                
                    # Add a header format.
                    header_format = workbook.add_format({'bold': True})
                    
                    
                    # Write the column headers with the defined format.
                    for j, val in enumerate(df.columns.values):
                        worksheet.write(0, j, val, header_format)
                        
                            
                    # Write the rows
                    nRows = len(df.values)
                    for i, row in enumerate(df.values):
                        for j, val in enumerate(row):
                            if i== nRows - 1 : worksheet.write(i + 1, j, val, header_format)
                            else: worksheet.write(i + 1, j, val)
                
                #        # Close the Pandas Excel writer and output the Excel file.
                #        if saveOpt:
                #            df.to_excel(writer, sheet_name=sheetName)
                    
                #    if saveOpt :
                #        df.to_pickle(file_name + ".pkl")  # where to save it, usually as a .pkl
                #        df.to_csv(path_or_buf = file_name + ".csv" , index=False)
                        
                # Close the Pandas Excel writer and output the Excel file.
                if saveOpt:
                #        df.to_excel(writer)
                    writer.save()
                    #print('df saved')

for clfInd in range(nClassifiers):
    strct2save = {"Sn" : archiveSn[clfInd , : , : , : , :]  , "FPR" : archivecFPR[clfInd , : , : , : , :] , \
                  "subSn" : subarchiveSn[clfInd , : , : , : , :]  , "subFPR" : subarchivecFPR[clfInd , : , : , : , :]}
    structurePath = "Results/archives_Sn_FPR_{}_newGrid_cohort20.pkl".format(classifiers[clfInd])
    f = open(structurePath,"wb")
    pickle.dump(strct2save,f)
    f.close()
    print("archives saved")
    
    """============== plotting ============"""



