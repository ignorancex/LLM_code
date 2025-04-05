## Code to obtain the SVD-UAP from a stored list of perturbation vectors

import numpy as np
from scipy.sparse.linalg import svds

sample_size = 1024 
t = np.load("imagenet_1024_samples_deepfool_vectors.npy") # load the stored list of perturbation vectors eg. Gradient, FGSM, DeepFool
t = t[:sample_size*3] 
t = np.reshape(t, (sample_size*3, 50176))
d = np.zeros([sample_size, 150528])

cnt = 0
for i in range(0, sample_size*3, 3):
    d[cnt] = np.reshape(t[i:i+3], 150528)
    cnt += 1

## Normalize the vectors
for i in range(sample_size):
    d[i] = d[i]/np.linalg.norm(d[i], ord=2)


## Can use different packages to obtain the singular vectors
#tu, ts, tvh =  np.linalg.svd(d)
tu, ts, tvh =  svds(d)

np.save("imagenet_1024_samples_deepfool_svd_tvh.npy", tvh[0]) # store the entire right singular vectors or only the top singular vector which is to be used.
