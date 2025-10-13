import os
import json
import yaml


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getstate__ = dict
    __setstate__ = dict.update


def get_config(model_config: str, dataset_config: str = None, trainer_config: str = None):
    """Get the full configuration file

    Params:
        model_config (str): path to the model config file
        dataset_config (str): path to the dataset config file (optional)
        trainer_config (str): path to the trainer config file (optional)

    Returns:
        DotDict: merged configuration
    """

    config = load_config(model_config)

    if dataset_config is not None:
        config.dataset_config = dataset_config

    if trainer_config is not None:
        config.trainer_config = trainer_config

    config.dataset = load_config(config.dataset_config)
    config = merge_config(config, load_config(config.trainer_config))

    return config


def load_config(file: str):
    """Load the configuration files"""

    assert os.path.exists(file), f'File {file} does not exist!'
    with open(file, 'r') as f:
        ext = os.path.splitext(os.path.basename(file))[-1]
        if ext == '.json':
            config = json.load(f)
        elif ext == '.yaml':
            config = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            raise ValueError(f'Invalid file type: {file}')

    def recursive_convert(data):
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = recursive_convert(v)
            return DotDict(data)
        if isinstance(data, list):
            return [recursive_convert(i) for i in data]
        if isinstance(data, tuple):
            return tuple(recursive_convert(i) for i in data)
        return data

    return recursive_convert(config)


def merge_config(d1, d2):
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            merge_config(d1[key], value)
        else:
            d1[key] = value
    return d1


def print_arguments(args, length=100, sep=': ', delim=' | '):
    """Print the arguments in a nice format

    Params:
        args (dict): arguments
        length (int): maximum length of each line
        sep (str): separator between key and value
        delim (str): delimiter between lines
    """

    text = []
    for key in args.keys():
        text.append('{}{}{}'.format(key, sep, args[key]))

    cl = 0
    for n, line in enumerate(text):
        if cl + len(line) > length:
            print('')
            cl = 0
        print(line, end='')
        cl += len(line)
        if n != len(text) - 1:
            print(delim, end='')
            cl += len(delim)
    print('')
