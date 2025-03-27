from run_moped import output
import numpy as np
import os

try:
    os.stat('moped_input')
except:
    os.system('cosmosis moped_params.ini')

parameters = {
    'cosmological_parameters': ['omch2', 'ombh2', 'h0', 'n_s', 's_8_input'],
    'halo_model_parameters': ['A'],
    'intrinsic_alignment_parameters': ['A'],
    'nofz_shifts': ['uncorr_bias_1', 'uncorr_bias_2',
                    'uncorr_bias_3', 'uncorr_bias_4', 'uncorr_bias_5'],
    'shear_c_bias': ['delta_c']
}

transform = 'comp'
output_dir = 'moped_output'
Weight, TransformedData, TransformedCovariance = output(transform, parameters)

try:
    os.stat(output_dir)
except:
    os.mkdir(output_dir)

np.savetxt(output_dir+'/'+transform+'_weight.txt', Weight)
np.savetxt(output_dir+'/'+transform+'_data.txt', TransformedData)
np.savetxt(output_dir+'/'+transform+'_cov.txt', TransformedCovariance)
