import h5py
import numpy as np
import matplotlib.pyplot as plt
filename = "/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta/output_alpha_CE_0.1_beta_0.25/COMPAS_Output_Weighted.h5"

with h5py.File(filename, "r") as f:
    dco_seeds = f['BSE_Double_Compact_Objects']['SEED']

    dco_mask = np.isin(f['BSE_System_Parameters']['SEED'], dco_seeds)

    CH_on_MS = f['BSE_System_Parameters']['CH_on_MS(1)'][...][dco_mask]

    print(len(CH_on_MS[CH_on_MS != 0]) / len(dco_seeds))

    print(f['BSE_System_Parameters']['CH_on_MS(1)'][...][f['BSE_System_Parameters']['CH_on_MS(1)'][...] != 0])
    print('1')

    


filename2 = '/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta/output_alpha_CE_10.0_beta_0.25/Pickled_Rates/MSSFR_111/BBH/rates_and_bootstraps_by_formation_channel.h5'

# with h5py.File(filename2, "r") as f2:
#     print(f2['(I) Classic']['total_merger_rates'][...])

