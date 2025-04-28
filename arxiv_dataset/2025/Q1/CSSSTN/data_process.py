import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.io
from skimage import transform
if __name__ == '__main__':

    # Set the size of EEGData to be processed
    len = 2200
    for sub in range(10):
        # Location of the original EEGData file
        mat_file = scipy.io.loadmat(f'')

        data = mat_file['data'][:len]
        label = mat_file['label'].squeeze()[:len]

        # Correct the situation where label=2 in tsinghua data is not a target
        label = (label == 1).astype(int)

        # Save label file
        np.save(f'', label)

        # Select the wavelet function, which can be adjusted according to the actual situation
        wavelet = 'morl'
        # Scale range
        scales = np.arange(2, 30)

        res = np.empty((len, data.shape[1], 64, 64))
        for i in range(len):
            for j in range(data.shape[1]):
                coefficients, _ = pywt.cwt(data[i][j], scales, wavelet)
                res[i, j, :, :] = transform.resize(coefficients, (64, 64), order=1)

        # Save data file location
        np.save(f'', res)

