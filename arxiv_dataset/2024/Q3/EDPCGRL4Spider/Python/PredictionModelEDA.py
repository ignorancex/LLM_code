import random
import numpy as np
import pandas as pd
import pickle
import neurokit2 as nk
from DataManager import DataManager
import json
from ManageCONST import readCONST
import Logger
from statistics import mean
import math
  
def changeRange(OldValue, OldMax, OldMin, NewMax, NewMin):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin  
    return NewValue

def predictStressLevel(EDA):
  CONST = readCONST()
  # EDA is between [base_EDA-20] > stress level between [0-10]
  print("EDA", EDA)
  stress_level = changeRange(mean(EDA), CONST["EDA"]["max_EDA"], CONST["EDA"]["base_EDA"], 10, 0)
  if stress_level < 0:
        return 0
  return round(stress_level)

def readBaseEDA():
    CONST = readCONST()
    relaxing_duration = CONST["Relaxing_Duration"]
    dataManagerObj = DataManager()
    EDA = dataManagerObj.loadData(relaxing_duration)
    print("len:", len(EDA), relaxing_duration)
    if (len(EDA) >= (relaxing_duration - 20)):
        base_EDA = min(EDA)
    else: 
        return 0
    return base_EDA


def getStressLevel():
  CONST = readCONST()
  SignalLength = CONST["SignalLengthEDA"] - 5

  dataManagerObj = DataManager()
  EDA = dataManagerObj.loadData(SignalLength)
  print("len:", len(EDA), SignalLength)
  if (len(EDA) >= SignalLength):
      stress_level = predictStressLevel(EDA)
      print(stress_level)
  else:
      stress_level = -10
      print("length of EDA signal is less than " , SignalLength)
  Logger.log("StressLevel" , stress_level)
  return stress_level