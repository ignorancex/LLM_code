# module1.py
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.special import eval_legendre
import scipy.io as sio
import os




path = path =  '/home/ahmad/CODE/CODE/muCC/'

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



kk_tr = np.loadtxt('wavenumbers.txt')

def get_mu(x, z, kk=None):
    # Validate redshifts
    if isinstance(z, list):
        if any(z_val < 0 or z_val > 3 for z_val in z):
            raise ValueError("Invalid input for redshift. Please enter a value between 0 and 3")
    else:
        if z < 0 or z > 3:
            raise ValueError("Invalid input for redshift. Please enter a value between 0 and 3")

    # Set kk to kk_tr if not provided
    if kk is None:
        kk = kk_tr
    else:
        # Check kk values against kk_tr bounds and warn if necessary
        if any(k < min(kk_tr) for k in kk):
            warnings.warn("The CLODE emulator is designed to emulate the mu function for wavenumbers within the range [0.016 h/Mpc to 9.4 h/Mpc]. "
                          "The wavenumbers you have requested include values lower than the minimum threshold of 0.016 h/Mpc. "
                          "These values will be clipped to the minimum threshold.", Warning)
        if any(k > max(kk_tr) for k in kk):
            warnings.warn("The CLODE emulator is designed to emulate the mu function for wavenumbers within the range [0.016 h/Mpc to 9.4 h/Mpc]. "
                          "The wavenumbers you have requested include values higher than the maximum threshold of 9.4 h/Mpc. "
                          "These values will be clipped to the maximum threshold.", Warning)
        # Clip kk values to ensure they are within the bounds of kk_tr
        # kk = np.clip(kk, np.min(kk_tr), np.max(kk_tr))

    # Convert x dictionary into an array, removing the third element
    x_array = list(x.values())
    del x_array[2]  # removing the third row

    lambda_PCEs = [uq_evalModel(x_array, MyPCEs[i]) for i in range(0, PCNum)]
    lambda_PCEs = np.array(lambda_PCEs)

    D_projback = lambda_PCEs.T @ PCBasis.T
    D_recovered_PCE = PCMean + D_projback
    Emulated = 10 ** (D_recovered_PCE)

    redshifts = [
        0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
        0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36,
        0.38, 0.4, 0.44, 0.5, 0.54, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3
    ]
    reshape_emulated = np.reshape(Emulated, (len(redshifts), len(kk_tr)))

    def calculate_for_single_z(z_val):
        if z_val in redshifts:
            outcome_at_z = reshape_emulated[redshifts.index(z_val), :]
        else:
            f = interp1d(redshifts, reshape_emulated, axis=0, kind='linear')
            outcome_at_z = f(z_val)

        # Interpolate outcomes based on kk_tr directly without needing to clip kk again
        f_kk = interp1d(kk_tr, outcome_at_z, kind="cubic", bounds_error=False, fill_value=(outcome_at_z[0], outcome_at_z[-1]))
        outcome_at_kk = f_kk(kk)
        return outcome_at_kk

    # Handle if z is a list or a single value
    if isinstance(z, list):
        outcomes = [calculate_for_single_z(z_val) for z_val in z]
    else:
        outcomes = calculate_for_single_z(z)

    return kk, outcomes


