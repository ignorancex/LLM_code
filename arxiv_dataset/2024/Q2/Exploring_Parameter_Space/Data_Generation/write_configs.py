import os
import sys
from sqlalchemy import null
import yaml
import shutil
import numpy as np

COMPAS_ROOT_DIR = os.getenv('COMPAS_ROOT_DIR')
sys.path.append(os.path.dirname(COMPAS_ROOT_DIR))


class ConfigWriter:
    """
    This class writes yaml files to make compas configuration for a number of varied parameters easier

    ...

    Attributes
    ----------
    config : dict
        a dictionary whose keys are the parameters and whose values are the parameter value of the configuration file

    param : str
        the parameter that we wish to vary

    param2 : str
        optional second parameter to vary with the first

    choice_type : str
        the ChoiceType section that the parameter will be under in our configuration file
    
    choice_type2 : str
        the ChoiceType section that the optional second parameter will be under in our configuration file
    
    default_config_filename : str
        the basic configuration file that the user will be varying; this is necessary to vary parameters that depend on other options

    Methods
    -------
    write_configs(param, vals, vals2 = False, filepath='/Users/adamboesky/Research/PRISE/exploring_parameter_space')
        Writes a number of configuration files with the varied parameters in a folder in the given file path. If the user passes
        in a value for vals2, the function will write configuration files varying both parameters together over a grid.
    """

    def __init__(self, param, param2 = False, choice_type='numericalChoices', choice_type2='numericalChoices', default_config_filename='compasConfigDefault.yaml'):

        self.param = param
        self.param_name = param[2:].replace('-', '_')

        if isinstance(param2, str):
            self.param2 = param2
            self.param2_name = param2[2:].replace('-', '_')

        self.choice_type = choice_type
        self.choice_type2 = choice_type2

        # Get the default config file from the compas directory
        with open('Configuration_Files/Defaults/' + default_config_filename, "r") as default_yaml:
            self.config = yaml.load(default_yaml, Loader=yaml.FullLoader)


    def write_configs(self, vals, vals2=False, path='/Users/adamboesky/Research/PRISE/exploring_parameter_space/Configuration_Files'):
        
        # Create folder to dump yaml files into
        if not vals2:
            dir_path = os.path.join(path, self.param_name + '_config_files')
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                os.mkdir(dir_path)
            else:
                os.mkdir(dir_path)
        else:
            dir_path = os.path.join(path, self.param_name + '_' + self.param2_name + '_config_files')
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                os.mkdir(dir_path)
            else:
                os.mkdir(dir_path)

        # Vary over each parameter
        if not vals2:
            for val in list(vals):
                self.config[self.choice_type][self.param] = float(val)

                # Dump the config files into the folder
                if isinstance(val, str):
                    with open(dir_path + '/config_' + self.param_name + '_' + val + '.yaml', 'w') as file:
                        yaml.dump(self.config, file)
                else:
                    with open(dir_path + '/config_' + self.param_name + '_' + str(val)[:4] + '.yaml', 'w') as file:
                        yaml.dump(self.config, file)
        else: 

            # Vary over the grid of parameters
            for val in list(vals):
                for val2 in list(vals2):
                    self.config[self.choice_type][self.param] = float(val)
                    self.config[self.choice_type2][self.param2] = float(val2)

                    # Dump the config files into the folder
                    if isinstance(val, str) and isinstance(val2, str):
                        with open(dir_path + '/config_' + self.param_name + '_' + val + '_' + self.param2_name + '_' + val2 + '.yaml', 'w') as file:
                            yaml.dump(self.config, file)
                    elif isinstance(val, str) and not isinstance(val2, str):
                        with open(dir_path + '/config_' + self.param_name + '_' + val + '_' + self.param2_name + '_' + str(val2)[:4] + '.yaml', 'w') as file:
                            yaml.dump(self.config, file)
                    elif not isinstance(val2, str) and isinstance(val2, str):
                        with open(dir_path + '/config_' + self.param_name + '_' + str(val)[:4] + '_' + self.param2_name + '_' + val2 + '.yaml', 'w') as file:
                            yaml.dump(self.config, file)
                    else:
                        with open(dir_path + '/config_' + self.param_name + '_' + str(val)[:4] + '_' + self.param2_name + '_' + str(val2)[:4] + '.yaml', 'w') as file:
                            yaml.dump(self.config, file)



def main():

    # SINGLE PARAMETER VARIATIONS

    # # Vary alpha_CE
    # alpha_configs = ConfigWriter('--common-envelope-alpha')
    # alpha_configs.write_configs([0.1, 0.5, 2.0, 10.0])

    # DOUBLE PARAMETER VARIATIONS

    # Vary alpha_CE and beta at the same time
    alpha_configs = ConfigWriter('--common-envelope-alpha', param2='--mass-transfer-fa', default_config_filename='config_beta_continuous.yaml')
    alpha_configs.write_configs([0.1, 0.5, 2.0, 10.0], [0.25, 0.5, 0.75])


if __name__ == '__main__':
    sys.exit(main())
