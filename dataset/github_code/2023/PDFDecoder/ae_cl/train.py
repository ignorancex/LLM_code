import os
import sys
import pathlib
import pickle

import numpy as np

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
utils_dir = parent_dir + '/utils'
model_dir = parent_dir + '/ae_cl'
input_dir = parent_dir + '/input_data'
sys.path.insert(1, utils_dir)
sys.path.insert(2, model_dir)
sys.path.insert(3, input_dir)

from pdf_gen import PDFDataSet
from ensemble import Ensemble

if __name__ == '__main__':
    num_runs = 100
    latent_size = 32

    new_dir_name = f'ae_cl_l{latent_size}_data'
    new_dir = pathlib.Path(input_dir, new_dir_name)
    new_dir.mkdir(parents=True, exist_ok=True)

    # Create data class
    data = PDFDataSet()

    # Create central dataset no noise
    data.create_data(xmin=1e-2,
                     num_params=5,
                     num_pdfs=10_000,
                     divx_minus=0,
                     divx_plus=0,
                     test_size=0.3,
                     err_param=0.,
                     pmerr_factor=0.,
                     latent_size=latent_size)

    central_train, central_valid, central_test = data.get_pdfs()

    # Add noise to central dataset
    data.create_data(xmin=1e-2,
                     num_params=5,
                     num_pdfs=10_000,
                     divx_minus=0,
                     divx_plus=0,
                     test_size=0.3,
                     err_param=0.01,
                     pmerr_factor=10.,
                     latent_size=latent_size)

    train, valid, test = data.get_pdfs()

    # Scale the data with standard scaler
    data.scale_data()
    x_train, x_valid, x_test = data.get_pdfs()

    # Get central moments unscaled
    train_moments, valid_moments, test_moments = data.get_moments()

    # Scale the moments with standard scaler
    data.scale_moments()
    y_train, y_valid, y_test = data.get_moments()

    # Standard Deviations
    u_plus_noise_train = data.uplus_std[data.train_indices]
    d_plus_noise_train = data.dplus_std[data.train_indices]
    u_minus_noise_train = data.uminus_std[data.train_indices]
    d_minus_noise_train = data.dminus_std[data.train_indices]

    u_plus_noise_valid = data.uplus_std[data.valid_indices]
    d_plus_noise_valid = data.dplus_std[data.valid_indices]
    u_minus_noise_valid = data.uminus_std[data.valid_indices]
    d_minus_noise_valid = data.dminus_std[data.valid_indices]

    u_plus_noise_test = data.uplus_std[data.test_indices]
    d_plus_noise_test = data.dplus_std[data.test_indices]
    u_minus_noise_test = data.uminus_std[data.test_indices]
    d_minus_noise_test = data.dminus_std[data.test_indices]

    # Track noise on central dataset
    train_noise_array = np.hstack([d_plus_noise_train,
                                   u_plus_noise_train,
                                   u_minus_noise_train,
                                   d_minus_noise_train])
    valid_noise_array = np.hstack([d_plus_noise_valid,
                                   u_plus_noise_valid,
                                   u_minus_noise_valid,
                                   d_minus_noise_valid])
    test_noise_array = np.hstack([d_plus_noise_test,
                                  u_plus_noise_test,
                                  u_minus_noise_test,
                                  d_minus_noise_test])

    np.save(new_dir / 'pdf_noise_train.npy', train_noise_array)
    np.save(new_dir / 'pdf_noise_valid.npy', valid_noise_array)
    np.save(new_dir / 'pdf_noise_test.npy', test_noise_array)

    np.save(new_dir / 'central_pdf_train.npy', central_train)
    np.save(new_dir / 'central_pdf_valid.npy', central_valid)
    np.save(new_dir / 'central_pdf_test.npy', central_test)

    np.save(new_dir / 'noisy_pdf_train.npy', train)
    np.save(new_dir / 'noisy_pdf_valid.npy', valid)
    np.save(new_dir / 'noisy_pdf_test.npy', test)

    np.save(new_dir / 'moments_train.npy', train_moments)
    np.save(new_dir / 'moments_valid.npy', valid_moments)
    np.save(new_dir / 'moments_test.npy', test_moments)

    with open(new_dir / 'pdf_scaler.pkl', 'wb') as f:
        pickle.dump(data.sc, f)
    with open(new_dir / 'moment_scaler.pkl', 'wb') as g:
        pickle.dump(data.moment_sc, g)

    ae_cl_ens = Ensemble(latent_size)
    ae_cl_ens.train([x_train, x_valid], [y_train, y_valid], num_runs)
    ae_cl_ens.get_ensemble(x_test)
