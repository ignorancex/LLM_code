import calendar

import pandas as pd
import csv

import time
from scipy.stats import *
import Constants
from os import path

# def getPeriods(p):
#     releaseDates =  pd.read_csv('./data/release_info/'+p+".csv")['releases'].values.tolist()
#
#     if len(releaseDates) > Constants.CONSIDER_LAST_N_RELEASES:
#         return releaseDates[ len(releaseDates) - Constants.CONSIDER_LAST_N_RELEASES : ]
#     else:
#         return releaseDates

def toEpoch(humanDate):

    return calendar.timegm(time.strptime(humanDate, '%Y-%m-%d %H:%M:%S'))

def normalize(column):

    if len(column) <= 0:
        return -1

    minVal = min(column)
    maxVal = max(column)

    # print(minVal, maxVal)
    normCol = []

    index = 0
    for v in column:
        # print('v', v)
        normCol.append((v - minVal) / ((maxVal - minVal) + 0.000000001))
        index += 1

    # print(normCol)
    return normCol

# def fileExists(filePath):
#     print(len(pd.read_csv(filePath)))
#     return path.exists(filePath)

def percentage(numer, denom):

    if denom > 0:
        return float(float(numer)*100/float(denom))
    else:
        return 0


def writeRow(filename, rowEntry):

    with open(filename, newline='', mode='a') as status_file:
        writer = csv.writer(status_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(rowEntry)

def computeCorrelation(X, Y):

    # print(X, Y)
    pearResult = pearsonr(X, Y)
    pearRho = pearResult[0]
    pearPValue = pearResult[1]

    spearResult = spearmanr(X, Y)
    spearRho = spearResult[0]
    spearPValue = spearResult[1]

    return [pearRho, pearPValue, spearRho,spearPValue]
# if __name__ == '__main__':
#     # print(getPeriods('nova'))
#
#     print(fileExists('./results/project_apollo_results.csv'))