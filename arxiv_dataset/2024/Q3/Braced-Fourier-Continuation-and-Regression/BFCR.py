# -*- coding: utf-8 -*-
"""
BFCR trend line

This code takes as inputs a 1-D array or list of data, and optionally a real 
number -sigma_power-, to output a trend line via the Braced Fourier 
Continuation for Regression (BFCR) algorithm.

@author: Josef M Sabuda
"""

if __name__ == "__main__":
    #Example data
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.array([40987.,22962.,26380.,19968.,23014.,19141.,25967.,44778.,20572.,20311.,97003.,62177.,49415.,49049.,70141.,15263.,9883.])
    data = data/np.max(data)
    
def BFCR(data,sigma_power=None):
    import numpy as np
    #Import relevant precomputed data
    from BFCR_precomputed_data import BFCR_precomputed_data
    
    #Check if sigma_power was prescribed
    if sigma_power is None:
        sigma_power=4 #default value used in the paper
        
    #Ensure input data is a numpy array
    data = np.array(data)
    
    #Find length of input data for optimization
    len_data = len(data)
    
    ###Add synthetic data to the input data
    #Pull precomputed data
    left_bracing_data,right_bracing_data,matrix_left,matrix_right = BFCR_precomputed_data()
    
    #Rescale both the precomputed bracing and continuation data based on the 
    #projected next value in the dataset as determined by the average of
    #two lines of best fit projected forward
    Q, R = np.linalg.qr(np.vstack([np.linspace(0,2,3), np.ones(3)]).T)
    Q = Q.T #take the transpose
    
    #Initialize solutions to the linear least squares problems
    solutions = np.zeros((2, 2))
    
    #Left bracing data point
    targets = np.column_stack((data[3:0:-1], data[2::-1]))
    for i in range(2):
        solutions[:, i] = np.linalg.solve(R, Q @ targets[:, i])
    m, c = solutions[:, 0]  # First solution
    m2, c2 = solutions[:, 1]  # Second solution
    data_left_pt = .5*((m*4+c)+(m2*3+c2)) #Average of forward projections of these lines
    
    #Right bracing data point
    targets = np.column_stack((data[-4:-1], data[-3:]))
    for i in range(2):
        solutions[:, i] = np.linalg.solve(R, Q @ targets[:, i])
    m, c = solutions[:, 0]  # First solution
    m2, c2 = solutions[:, 1]  # Second solution
    data_right_pt = .5*((m*4+c)+(m2*3+c2)) #Average of forward projections of these lines
    
    #Calculate scaling multipliers
    left_mult = data_left_pt/left_bracing_data[-1]
    right_mult = data_right_pt/right_bracing_data[0]
    
    #Build the continuation Af
    Af = matrix_left*left_mult+matrix_right*right_mult
    
    #Concatenate the synthetic data with the input data to create continued data
    data_cont = np.concatenate((left_bracing_data*left_mult,data,right_bracing_data*right_mult,Af),axis=0)
    ###
    
    #Subtract the mean from the continued data
    cont_data_mean = np.mean(data_cont)
    data_cont -= cont_data_mean
    
    #Calculate the fft of the continued data
    coeffs = np.fft.rfft(data_cont)
    
    #Apply sigma approximation to the coeffs before reconstruction
    len_coeffs = len(coeffs) #For efficiency
    coeffs *= [np.sinc(i/len_coeffs)**sigma_power for i in range(len_coeffs)]  
    
    #Reconstruct the sigma approximated continued data
    #The number 51 comes from the parameters chosen in the FC process,
    #namely d=12 and C=27; 2*d+C = 2*12+27 = 51
    BFCR_trend = np.fft.irfft(coeffs,len_data+51)
    
    #Trim synthetic data points from the reconstruction
    #The number is 12 here because d=12 was used in the FC process
    BFCR_trend = BFCR_trend[12:len_data+12]
    
    #Add back the mean to the reconstruction
    BFCR_trend += cont_data_mean
    
    #Shift the reconstruction by the average difference between the reconstruction
    #and the original signal to eliminate residual error from the neglected modes
    #to arrive at the final trend
    BFCR_trend -= np.mean(BFCR_trend-data)
    
    return(BFCR_trend)
    
if __name__ == "__main__":
    #Calculate BFCR regression
    BFCR_trend = BFCR(data)
    #plot data and BFCR regression
    plt.figure(1)
    plt.plot(data,'k.',label='Input data')
    plt.plot(BFCR_trend,'b',label='BFCR Trend')
    plt.legend()
