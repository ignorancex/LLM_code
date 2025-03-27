# Several extensive operations for python list.
import math, json, logging

from tqdm import tqdm
from functools import reduce


def add(a, b):
    return a + b


def mean(iter):
    return reduce(add, iter)/len(iter)


def lst_add_lst(list1, list2):
    return [sum(x) for x in zip(list1, list2)]


def lst_divide(lst, denominator):
    if isinstance(denominator, list):
        assert len(lst) == len(denominator)
        return [x/y for x, y in zip(lst, denominator)]
    return [x/denominator for x in lst]


# How to print formated logs via logger and format definitions.
def print_performances(logger, procedure, lr = None, num_format = None, **kwargs):
    if num_format is None or len(num_format) != len(kwargs):
        logger.exception('Bad num_format dictoinary.')

    info = f'{procedure:12}' + (f' ,lr: {lr:8.5f}' if lr else '')
    for key in kwargs.keys():
        info += ' ,' + key + ': {' + key + num_format[key] + '}'
    logger.info(info.format_map(kwargs))


# Read and convert a json file into a dict object.
def read_json(json_path):
    with open(json_path, 'r') as f:
        a = json.load(f)
    return a


# Help construct the output dir name using model hyperparameters.
def suffix(opt, *args):
    output = []
    for item in args:
        output.append(getattr(opt, item))
    
    output = "_".join(map(str, output))
    
    return output


# General evaluation procedure.
def evaluation(data, model, model_class, device, output_length, desc):
    sum_ = [0] * output_length
    dataset_size = len(data)
    
    for minibatch in tqdm(data, desc):
        batch_sum = model_class.evaluation_step(model, minibatch, device)
        sum_ = lst_add_lst(sum_, lst_divide(batch_sum, dataset_size))

    return sum_


# extract dataset name from the input string
# eg: 'dataset_name_new_v2'
def restore_dataset_name(name):
    name = name.strip('v123456789')
    name = name[:-1]
    if name.endswith('_new'):
        name = name[:-4]
    if name.endswith('_continuous'):
        name = name[:-11]
    return name


class Metric():
    '''
    A Metric handler.
    1. metric_number: How many metric do you have?
    2. smaller_is_better: If model performance is better with lower metric value, you should set it to true. Otherwise, it is false.
    If smaller_is_better is set, its length must match argument 'metric_number'.
    '''
    def __init__(self, metric_number, smaller_is_better = None):
        self.metric_number = metric_number
        self.map = {True:1, False: -1}
        self.best_metric = [math.inf] * self.metric_number
        if smaller_is_better is None:
            self.mask = [1] * self.metric_number
        else:
            assert len(smaller_is_better) == self.metric_number
            self.mask = [self.map[item] for item in smaller_is_better]
    
    def compare(self, input_metric):
        assert len(input_metric) == len(self.mask)
        tmp = lst_divide(input_metric, self.mask)
        output = True

        for input_number, recorded in zip(tmp, self.best_metric):
            if input_number > recorded:
                output = False
                break
        
        if output:
            self.best_metric = input_metric
        
        return output
    
    def show(self):
        return self.best_metric


# add a prefix for all keys in a dict.
# wandb use only
def add_prefix_to_keys(dct, temp):
    tmp_dct = dict(dct)
    del tmp_dct['num_format']
    result = {temp + str(key): item for key, item in tmp_dct.items()}
    return result


# A more neat way to print hyperparameters:
def print_args(opt):
    output = '\nAll hyperparameters:\n'
    for key, value in opt.__dict__.items():
        output += str(key) + ': ' + str(value) + '\n'

    return output


'''
Logger settings
'''

'''
def getEventLogger(name, root):
    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.DEBUG)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d]: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger


def getFileLogger(name, file, root):
    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.DEBUG)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.FileHandler(file, mode = 'w')
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger


def getLogger(name = None, file = None, root = True):
    \'''
    Get normal loggers or file loggers.

    Args:
    name: The name of a generated logger
    file: print all logs into the file if set.
    \'''
    if file:
        return getFileLogger(name, file, root)
    else:
        return getEventLogger(name, root)
'''

def getEventLogger(name, root):
    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.DEBUG)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d]: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger


def getLogger(name = None, root = True):
    '''
    Get normal loggers or file loggers.

    Args:
    name: The name of a generated logger
    file: print all logs into the file if set.
    '''

    return getEventLogger(name, root)