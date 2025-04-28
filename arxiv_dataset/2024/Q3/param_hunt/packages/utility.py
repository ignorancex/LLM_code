import numpy as np
import os

# regularize the logarithm of the posterior by setting the minimum value to pref
# already performs min max scaling internally
# assumes already correct shape (Nsam, 2, Nlin)
def reg_log(x, pref):
    y=np.copy(x)
    y[y<pref]=pref
    y = (y-pref)/(np.max(y, axis=(1,2)).reshape(-1,1,1)-pref)
    return y

# function that creates a new directory with incremental name
def uniquify(path):
    foldername = path
    counter = 0
    path = foldername + str(counter) 

    while os.path.exists(path):
        counter += 1
        path = foldername + str(counter) 

    return path

