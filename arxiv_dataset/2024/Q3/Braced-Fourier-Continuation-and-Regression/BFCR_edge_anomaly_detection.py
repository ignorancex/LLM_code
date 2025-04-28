# -*- coding: utf-8 -*-
"""
BFCR Edge Anomaly Detection Algorithm

This code takes as inputs a 1-D array or list of data, and optionally a real 
number -sigma_power-, as well as an indicator variable, 
Screen_Internal_Outliers, which can be set to True or False but by default is 
False. With these inputs, the code determines whether or not the last point in 
the input data is an outlier using the BFCR edge anomaly detection algorithm. 
It assumes the deviations from the BFCR trend lines follow a normal 
distribution.

If Screen_Internal_Outliers is set to True, the algorithm will first screen for 
and filter out the influence of any internal outlier points to remove their 
influence from the population statistics in the algorithm.

@author: Josef M Sabuda
"""

if __name__ == "__main__":
    #Example data
    import numpy as np
    import matplotlib.pyplot as plt
    #data  = np.array([4900,5300,3000,3400,5600,7100,4100,6000,6100,4000,2100,2900,3200,3300,2500,6200])
    data = np.array([.75,.81,.83,.82,1.02,1.09,1.1,.83,1.05,1.15,1.09,1.14,1.05,1.01,1,1.22])
    
def BFCR_edge_anomaly_detection(data,sigma_power=None,Screen_Internal_Outliers=False):
    #Import numpy
    import numpy as np
    #Import BFCR algorithm
    from BFCR import BFCR
    
    #Check if sigma_power was prescribed
    if sigma_power is None:
        sigma_power = 4 #default value used in the paper
    
    #Calculate BFCR trend 1
    BFCR_trend_1 = BFCR(data[0:-1],sigma_power)
    
    #Compute the population statistics
    deviations = abs(BFCR_trend_1-data[0:-1])
    mean_dev = np.mean(deviations)
    std = np.std(deviations)
    
    if Screen_Internal_Outliers != False:
        #Screen out internal outliers
        deviations_indices = [-1]*len(deviations)
        d_count = 0
        #Loop through points
        for i in range(len(deviations)):
            if abs(deviations[i]-mean_dev)/std >= 2:
                continue
            else:
                deviations_indices[d_count] = i
                d_count+=1
        #Filter outliers from deviations
        deviations = [deviations[k] for k in deviations_indices[0:d_count]]
        #Recalculate populaion staistics without the influence of the outliers
        mean_dev = np.mean(deviations)
        std = np.std(deviations)
    
    #Calculate BFCR trend 2
    BFCR_trend_2 = BFCR(data,sigma_power)
    
    #Compute the sample to compare to the population statistics
    deviation = abs(BFCR_trend_2[-1]-data[-1])
    
    #Compare the sample to the population statistics
    anomaly = -1
    if (deviation-mean_dev)/std >= 2:
            anomaly = len(data)-1
    
    #Return the index of the last point if it is an outlier, otherwise
    #return -1
    return(anomaly)

if __name__ == "__main__":
    from BFCR import BFCR
    BFCR_trend_1 = BFCR(data[0:-1])
    BFCR_trend_2 = BFCR(data)
    x1 = np.linspace(0,len(data)-1,len(data))
    x2 = x1[0:-1]
    anomaly = BFCR_edge_anomaly_detection(data,sigma_power=None,Screen_Internal_Outliers=True)
    
    #Plot of the algorithm on the provided example data
    plt.figure(1)
    plt.plot(data,'k.',label='Input data')
    plt.plot(BFCR_trend_1,'r',label='BFCR Trend 1')
    plt.plot(BFCR_trend_2,'b',label='BFCR Trend 2')
    if anomaly != -1:
        plt.plot(anomaly,data[anomaly],'rs',markerfacecolor='none',label='Outlier')
    plt.xticks(x1)
    plt.legend()