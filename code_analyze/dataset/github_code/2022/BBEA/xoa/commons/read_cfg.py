import os

from .logger import *

def read_config(cfg_file_name, path='run_conf/'):
    try:
        # file existence test
        if not os.path.exists(path):
            raise ValueError("Run config path not existed: {}".format(path))

        if not os.path.exists(path + cfg_file_name):
            if os.path.exists(path + cfg_file_name + ".yaml"):
                cfg_file_name += ".yaml"
            elif os.path.exists(path + cfg_file_name + ".yml"):
                cfg_file_name += ".yml"
            elif os.path.exists(path + cfg_file_name + ".json"):
                cfg_file_name += ".json"

        with open(path + cfg_file_name) as cfg_file:
            if ".yaml" in cfg_file_name or ".yml" in cfg_file_name:
                import yaml
                if hasattr(yaml, 'FullLoader'): 
                    json_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
                else:
                    json_dict = yaml.load(cfg_file) # python 2.x support
            elif ".json" in cfg_file_name:
                import json
                json_dict = json.load(cfg_file)
            else:
                raise ValueError("Invalid run config format")
            return json_dict   

    except Exception as ex:
        error('Exception on loading run configuration: {}'.format(ex))
        raise ValueError('Reading {} file failed.'.format(cfg_file_name))


def validate_config(config):
    # validate config schema
    try:
        if type(config) is not dict:
            return False

        if "arms" in config:
            if len(config["arms"]) == 0:
                error("no arm is listed.")
                return False
            else:
                for arm in config["arms"]:
                    if not "model" in arm.keys():
                        raise ValueError("no mode attribute in bandit")
                    if not "acq_func" in arm.keys():
                        raise ValueError("no spec attribute  in bandit")
                    # TODO: validate value

        if "bandits" in config:
            if len(config["bandits"]) == 0:
                error("no bandit is listed.")
                return False
            else:
                for bandit in config["bandits"]:
                    if not "mode" in bandit.keys():
                        raise ValueError("no model attribute in bandit")
                    if not "spec" in bandit.keys():
                        raise ValueError("no strategy attribute  in bandit")
                    # TODO: validate value
        return True            
    except:
        error("invalid BO configuration: {}".format(config))
        return False