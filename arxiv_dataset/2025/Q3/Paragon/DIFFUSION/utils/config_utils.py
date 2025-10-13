import configparser
import os

def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def get_path(config, path_name):
    path = config['PATHS'][path_name]
    return path.format(**config['DEFAULT'])

def process_paths(config):
    model_name = config['DEFAULT']['model_name']
    dataset = config['DEFAULT']['dataset_of_Rec']
    denoise = config['DEFAULT']['denoise']

    config['PATHS']['base_folder'] = config['PATHS']['base_folder'].replace('model_name', model_name).replace('dataset', dataset)
    config['PATHS']['original_model_path'] = config['PATHS']['original_model_path'].replace('model_name', model_name).replace('dataset', dataset)
    config['PATHS']['diffusion_model_path'] = config['PATHS']['diffusion_model_path'].replace('model_name', model_name).replace('dataset', dataset).replace('denoise', denoise)
    config['PATHS']['reconstruct_dir'] = config['PATHS']['reconstruct_dir'].replace('model_name', model_name).replace('dataset', dataset).replace('denoise', denoise)

    return config