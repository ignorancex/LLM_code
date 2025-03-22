import numpy as np 
import dirichlet 

def dirichlet_moments(samples):
    # first compute moments using dirichlet package
    means = dirichlet.mle(samples)
    size = samples.shape[1] # check this
    
    summed = np.sum(means)
    cov = np.zeros((size, size))
    div = (summed**2) * (summed + 1)
    for i in range(size):
        cov[i,i] = means[i] * (summed - means[i]) / div
        for j in range(i, size):
            cc = - means[i] * means[j] / div 
            cov[i,j] = cc
            cov[j,i] = cc
    
    return means/summed, cov  