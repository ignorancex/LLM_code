import scipy.io as sio
import numpy as np

def load_mat_params(filename):
    mat_dict = sio.loadmat(filename)
    params = mat_dict['output']
    num_problems = params.shape[1]
    output = params[0,:]
    for i in range(num_problems):
        output['wg'][i]=np.transpose(output['wg'][i])


    return output #should have fields wg, alphag (sometimes), pigy, numg, pg, with a list of values for each problem, and perhaps also
                #test accuracy, train accuracy, pg


