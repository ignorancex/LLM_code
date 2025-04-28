# module1.py
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import eval_legendre
import scipy.io as sio
import os




path =  '../main/muNCC/'

PCNum = sio.loadmat(path +'PCNum.mat')['PCNum']
PCNum = PCNum[0][0]

basis_indices = []
pce_coefficients  = []

for i in range (1,PCNum+1):

    # Load the MATLAB file containing the basis indices
    mat_cont = sio.loadmat(path + f'Indices_{i}.mat')['Indices']
    basis_indices.append(mat_cont.toarray())

    # Load the MATLAB file containing the coefficient
    mat_Coef = sio.loadmat(path + f'Coef_{i}.mat')['Coef']
    pce_coefficients.append(mat_Coef)

MyPCEs = list(zip(pce_coefficients, basis_indices))    
    
for i in range (0,PCNum):
    pce_coefficients[i], basis_indices[i] = MyPCEs[i] 
    
PCBasis = sio.loadmat(path + 'PCBasis.mat')['PCBasis']
PCMean = sio.loadmat(path +'PCMean.mat')['PCMean']



x_min = np.array([0.04, 0.20, 0.92, 0.61, -1.30, -10, 1.7e-9])
x_max = np.array([0.06, 0.34, 1.00, 0.73, -0.7, -5, 2.5e-9])


def map_input(x, x_min, x_max):
    return (2 * (x - x_min) / (x_max - x_min)) - 1


def eval_model_single(x, MyPCE):
    coefficient, indices = MyPCE
    basis_functions_eval = np.ones(
        len(coefficient)
    )  # Initialize to ones because the first basis function is always 1.
    for i in range(indices.shape[0]):  # loop over each term in the PCE
        for j in range(indices.shape[1]):  # loop over each dimension
            basis_functions_eval[i] *= np.sqrt(2 * indices[i, j] + 1) * eval_legendre(
                indices[i, j], x[j]
            )
    return basis_functions_eval @ coefficient


def uq_evalModel(x_input, MyPCEs):
    # Map the input to the range [-1, 1]
    x_input_mapped = map_input(x_input, x_min, x_max)

    eval_model_vec = np.vectorize(
        eval_model_single, excluded=["MyPCE"], signature="(m)->()", otypes=[float]
    )
    return eval_model_vec(x_input_mapped, MyPCE=MyPCEs)


def get_mu(x, z):
    # turn x dictionary to an array: x_array
    x_array = list(x.values())
    
    # remove the third element of the array
    del x_array[2]  # removing the third row

    lambda_PCEs = []

    for i in range (0, PCNum):
        lambda_PCEs.append(uq_evalModel(x_array, MyPCEs[i]))

    lambda_PCEs = np.array(lambda_PCEs)

    D_projback = lambda_PCEs.T @ PCBasis.T
    D_recovered_PCE = PCMean + D_projback
    Emulated = 10**(D_recovered_PCE)
    
    redshifts = [
    0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
    0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
    0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36,
    0.38, 0.4, 0.44, 0.5, 0.54, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0,
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3
    ]


    reshape_emulated = np.reshape(Emulated, (len(redshifts), 1000))
    
    z_mapping = {value: index for index, value in enumerate(redshifts)}

    if z in z_mapping:
        return reshape_emulated[z_mapping[z], :]
    
    elif min(redshifts) < z < max(redshifts):
        f = interp1d(redshifts, reshape_emulated, axis=0, kind='linear')
        return f(z)
    
    else:
        raise ValueError("Invalid input for redshift. Please enter a value between 0 and 3 ")

