import yaml
import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def read_config(args):
    dataset_name=args.dataset_name.replace('-', '_')
    yml_path="{}/{}_{}.yml".format(args.config_root_path,args.model_name,dataset_name)
    with open(yml_path, 'r') as f:
        default_arg = yaml.safe_load(f)
    args = dict2namespace({**vars(args), **default_arg})
    return args