# -*- coding: utf-8 -*-
"""
BFCR Internal Anomaly Detection Algorithm

This code takes as inputs a 1-D array or list of data, and optionally a real 
number -sigma_power-, and determines whether or not points in the input
data, excluding the first and last points, are outliers using the BFCR internal
anomaly detection algorithm. It assumes the deviations from the BFCR trend 
lines follow a normal distribution.

@author: Josef M Sabuda
"""

if __name__ == "__main__":
    #Example data
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.array([40987.,22962.,26380.,19968.,23014.,19141.,25967.,44778.,20572.,20311.,97003.,62177.,49415.,49049.,70141.,15263.,29883.])
    
def BFCR_interal_anomaly_detection(data,sigma_power=None):
    #Import numpy
    import numpy as np
    #Import BFCR algorithm
    from BFCR import BFCR
    
    #Check if sigma_power was prescribed
    if sigma_power is None:
        sigma_power = 4 #default value used in the paper
    
    #Calculate BFCR trend 1
    BFCR_trend = BFCR(data,sigma_power)
    
    #Compute the population statistics
    deviations = abs(BFCR_trend-data)
    mean_dev = np.mean(deviations)
    std = np.std(deviations)
    
    #Loop through the input data, excluding the first and last points,
    #and look for outliers
    anomalies = []
    for i in range(1,len(data)-1):
        if (deviations[i]-mean_dev)/std >= 2:
                anomalies.append(i)
    
    #Return indices of any found outliers
    return(anomalies)

if __name__ == "__main__":
    from BFCR import BFCR
    BFCR_trend = BFCR(data)
    x1 = np.linspace(0,len(data)-1,len(data))
    anomalies = BFCR_interal_anomaly_detection(data)
    
    #Plot of the algorithm on the provided example data
    plt.figure(1)
    plt.plot(data,'k.',label='Input data')
    plt.plot(BFCR_trend,'r',label='BFCR Trend')
    if anomalies != []:
        plt.plot(anomalies,data[anomalies],'rs',markerfacecolor='none',label='Outlier(s)')
    plt.xticks(x1)
    plt.legend()
    
    
    
    
    
    
    