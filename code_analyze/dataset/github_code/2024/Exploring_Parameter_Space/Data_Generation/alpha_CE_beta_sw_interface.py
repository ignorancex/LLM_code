import os, sys
import pandas as pd
import re
import shutil
import time
import numpy as np
from sqlalchemy import null
from stroopwafel import sw, classes, prior, sampler, distributions, constants, utils
import argparse
import h5py
from custom_sw_interface import SwInterface


def main():



    
    # # ALPHA_CE
    # for filename in os.listdir('/Users/adamboesky/Research/PRISE/exploring_parameter_space/common_envelope_alpha_config_files'):
    #     val = filename[:-5][-4:].replace('_','')
    #     ce_alpha_interface = SwInterface('common_envelope_alpha_config_files/' + filename, 'common_envelope_alpha', val, num_systems=1000000, num_per_core=100000, num_cores=10)
    #     ce_alpha_interface.run_sw()




    # ALPHA_CE AND BETA
    for filename in os.listdir('/Users/adamboesky/Research/PRISE/exploring_parameter_space/Configuration_Files/common_envelope_alpha_mass_transfer_fa_config_files'):
        
        # Get the valeus from each filename
        vals = []
        for piece in filename[:-5].split('_'):
            try:
                float(piece)
                vals.append(piece)
            except ValueError:
                pass
        
        # Run the stroopwafel sampling
        ce_alpha_interface = SwInterface('common_envelope_alpha_mass_transfer_fa_config_files/' + filename, 'alpha_CE', vals[0], output_dir_name='sw_weights_test', param2='beta', val2=vals[1], num_systems=10000, num_per_core=1000, num_cores=10)
        ce_alpha_interface.run_sw()


if __name__ == '__main__':
    sys.exit(main())

