import random
import numpy as np
import pandas as pd
import pickle
import neurokit2 as nk
from DataManager import DataManager
import json
from ManageCONST import readCONST
import Logger

def FeatureExtraction(PPG_data, sampleRate=1000):
    """Extract features from PPG data."""
    PPG = np.array(PPG_data)

    # Preprocess the data
    processed_data, _ = nk.bio_process(ppg=PPG, sampling_rate=sampleRate)
    heartRate = processed_data['PPG_Rate'].mean()
    print("Heart Rate:", heartRate)
    Logger.log("Heart Rate", heartRate)

    # Compute relevant features
    features = nk.bio_analyze(processed_data, sampling_rate=sampleRate)

    # Select the best features
    bestFeatures = [
        'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD',
        'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20',
        'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn',
        'HRV_SD1', 'HRV_SD2', 'HRV_S', 'HRV_CVI', 'HRV_CSI_Modified', 
        'HRV_PIP', 'HRV_IALS', 'HRV_PSS', 'HRV_PI', 'HRV_SD1d', 
        'HRV_SD1a', 'HRV_SD2d', 'HRV_SD2a', 'HRV_SDNNd', 'HRV_SDNNa', 
        'HRV_DFA_alpha1', 'HRV_ShanEn', 'HRV_HFD', 'HRV_KFD'
    ]

    chosenFeatures = features[bestFeatures]
    return chosenFeatures

def PredictStressLevel(ppg):
    """Predict the stress level based on PPG data."""
    features = FeatureExtraction(ppg)
    CONST = readCONST()

    # Load scaler and model
    filenameScaler = CONST["PredictionModel"]["filename_scaler"]
    loadedScaler = pickle.load(open(filenameScaler, 'rb'))
    featuresScaled = loadedScaler.transform(features.values)

    filenameModel = CONST["PredictionModel"]["filename_model"]
    loadedModel = pickle.load(open(filenameModel, 'rb'))

    # Predict stress level
    result = loadedModel.predict_proba(featuresScaled)
    stressLevel = round(result[0][1] * 10)  # Scale to 0-10
    return stressLevel

def GetStressLevel():
    """Get the current stress level from PPG data."""
    CONST = readCONST()
    signalLength = CONST["SignalLength"]
    samplingRate = CONST["Bitalino"]["samplingRate"]

    dataManager = DataManager()
    ppg = dataManager.loadData(signalLength)
    print("Length of PPG data:", len(ppg), "Expected:", signalLength)

    if len(ppg) >= signalLength * samplingRate:
        stressLevel = PredictStressLevel(ppg)
        print("Predicted Stress Level:", stressLevel)
    else:
        stressLevel = -10
        print("Length of PPG signal is less than expected:", signalLength)

    Logger.log("StressLevel", stressLevel)
    return stressLevel
