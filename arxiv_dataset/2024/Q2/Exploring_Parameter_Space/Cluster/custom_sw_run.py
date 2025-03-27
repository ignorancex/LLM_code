import os
import sys

# Append paths to access the scripts we need
sys.path.append('/n/home04/aboesky/berger/Exploring_Parameter_Space/')

from custom_sw_interface import SwInterface


def get_env_params():

    # Get the config file name
    config_filepath = os.environ.get('CONFIG_FILEPATH')     # name of the config file

    # Get the output directory name
    output_dirname = os.environ.get('OUTPUT_DIRNAME')

    # Get numbers related to quantity of systems and computer
    num_systems = os.environ.get('NUM_SYSTEMS')
    num_cores = os.environ.get('NUM_CORES')
    num_per_core = os.environ.get('NUM_PER_CORE')

    # Get variable labels
    param1_label = os.environ.get('PARAM1_LABEL')               # the label of the first variable
    param2_label = os.environ.get('PARAM2_LABEL')               # the label of the second variable
    # Get variable values
    param1_val = os.environ.get('PARAM1_VAL')                   # the value of the first variable
    param2_val = os.environ.get('PARAM2_VAL')                   # the value of the second variable

    return config_filepath, output_dirname, num_systems, num_cores, num_per_core, param1_label, param2_label, param1_val, param2_val
    


def main():
    
    # Get the environment variables just declared by the sbatch submission script
    config_filepath, output_dirname, num_systems, num_cores, num_per_core, param1_label, param2_label, param1_val, param2_val = get_env_params()

    # Declare a SwInterface
    ce_alpha_interface = SwInterface(config_filepath, param1_label, param1_val, output_dir_name=output_dirname, param2=param2_label, val2=param2_val, num_systems=num_systems, num_per_core=num_per_core, num_cores=num_cores)
    ce_alpha_interface.run_sw()


if __name__ == '__main__':
    sys.exit(main())