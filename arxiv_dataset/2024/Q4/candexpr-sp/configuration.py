
import os
from itertools import chain
from importlib import import_module
import argparse
import shutil
# from functools import lru_cache
from functools import cache

# from accelerate.utils.dataclasses import DistributedType

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.lazy import LazyProxy
# from dhnamlib.pylib.decoration import Register
from dhnamlib.pylib.decoration import variable  # , deprecated
from dhnamlib.pylib.filesys import json_skip_types
from dhnamlib.pylib.filesys import (
    json_pretty_save, InvalidObjectSkippingJSONEncoder,
    mkpdirs_unless_exist, make_logger, NoLogger, mkloc_unless_exist, get_new_path_with_number)
# from dhnamlib.pylib.filesys import pickle_load
# from dhnamlib.pylib.iteration import rcopy
from dhnamlib.pylib.iteration import distinct_pairs, not_none_valued_pairs
from dhnamlib.pylib.text import parse_bool
# from dhnamlib.pylib.package import import_from_module
from dhnamlib.pylib.version_control import get_git_hash
from dhnamlib.pylib.hflib.acceleration import XAccelerator, NoAccelerator, set_seed_randomly
from dhnamlib.pylib.package import ModuleAccessor
from dhnamlib.pylib.structure import AttrDict

from utility.time import initial_date_str

from splogic.utility.acceleration import set_accelerator_initializer, accelerator
from splogic.utility.tqdm import set_tqdm_use_initializer
from splogic.utility.logging import set_logger_initializer

from meta_configuration import get_default_domain_name, NO_DOMAIN_NAME


_DEBUG = False

# @cache
# def _get_domain():
@LazyProxy
def domain():

    cmd_arg_dict = _parse_cmd_args()
    if cmd_arg_dict['domain_name'] is NO_DOMAIN_NAME:
        domain = AttrDict(configuration=AttrDict(config=dict()))
    else:
        domain = ModuleAccessor('domain.{}'.format(cmd_arg_dict['domain_name']))

    return domain


@set_accelerator_initializer
def _make_accelerator():
    # kwargs = dict(split_batches=True, even_batches=False)
    kwargs = dict(split_batches=False, even_batches=True)  # default options for Accelerator
    xaccelerator = XAccelerator(**kwargs)

    if xaccelerator.accelerating:
        set_seed_randomly()
        accelerator = xaccelerator
    else:
        # assert config.device in ['cuda', 'cpu']
        # accelerator = XAccelerator(**kwargs,
        #                            cpu=(config.device == 'cpu'))
        accelerator = NoAccelerator(config.device)

    return accelerator


set_tqdm_use_initializer(lambda: config.using_tqdm)


@set_logger_initializer
def _make_logger():
    if accelerator.is_local_main_process:
        if _is_training_run_mode(config.run_mode):
            log_file_path = os.path.join(config.model_learning_dir_path, f'{initial_date_str}_{config.run_mode}.log')
            mkpdirs_unless_exist(log_file_path)
        else:
            log_file_path = None

        logger = make_logger(
            name=config.run_mode,
            log_file_path=log_file_path)
    else:
        logger = NoLogger()

    return logger


def _get_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


ALL_RUN_MODES = [
    'train-default', 'train-for-multiple-decoding-strategies', 'test-on-val-set', 'test-on-test-set',
    'oracle-test-on-val-set', 'search-train'
    # 'retrain', 'finetune',
]

TRAINING_RUN_MODES = [
    'train-default', 'train-for-multiple-decoding-strategies', 'search-train'
]


def _is_valid_run_mode(run_mode):
    # return run_mode in domain.configuration.ALL_RUN_MODES
    return run_mode in ALL_RUN_MODES


def _is_training_run_mode(run_mode):
    assert _is_valid_run_mode(config.run_mode)
    # return run_mode in domain.configuration.TRAINING_RUN_MODES
    return run_mode in TRAINING_RUN_MODES


def _get_git_hash():
    try:
        return get_git_hash()
    except FileNotFoundError:
        return None


# # coc: configuration object collection
# coc = Environment(
#     domain=LazyEval(_get_domain),
# )

@cache
def _parse_cmd_args():
    parser = argparse.ArgumentParser(description='Semantic parsing')
    parser.add_argument('--domain', dest='domain_name', help='a domain name', choices=['kqapro', 'overnight'], default=get_default_domain_name())
    parser.add_argument('--config', dest='config_module', help='a config module (e.g. config.kqapro.test_general)')
    parser.add_argument('--extra-config', dest='extra_config_modules', help='an additional config module(s) which can overwrite other configurations. When more than one module is passed, the modules are separated by commas.')
    parser.add_argument('--model-learning-dir', dest='model_learning_dir_path', help='a path to the directory of learning')
    parser.add_argument('--model-path', dest='model_path', help='a path to the directory of a checkpoint')
    # parser.add_argument('--model-dir-name', dest='model_dir_name', help='the name of the directory of a model')
    # parser.add_argument('--run_mode', dest='run_mode', help='an execution run_mode', choices=['train', 'test'])
    parser.add_argument('--using-tqdm', dest='using_tqdm', type=parse_bool, default=True, help='whether using tqdm')

    args, unknown = parser.parse_known_args()
    cmd_arg_dict = dict(not_none_valued_pairs(vars(args).items()))
    cmd_arg_dict['remaining_cmd_args'] = unknown
    return cmd_arg_dict


def has_model_learning_dir_path():
    cmd_arg_dict = _parse_cmd_args()
    return 'model_learning_dir_path' in cmd_arg_dict


@cache
def _get_specific_config_module():
    cmd_arg_dict = _parse_cmd_args()
    if cmd_arg_dict.get('config_module') is not None:
        specific_config_module = import_module(cmd_arg_dict['config_module'])
    else:
        specific_config_module = None
    return specific_config_module


@cache
def _get_extra_config_modules():
    cmd_arg_dict = _parse_cmd_args()
    if cmd_arg_dict.get('extra_config_modules') is not None:
        # module_names = [name.strip() for name in cmd_arg_dict['extra_config_modules'].split(',')]
        module_names = [name.strip() for name in cmd_arg_dict['extra_config_modules'].split('|')]
        _extra_config_modules = tuple(map(import_module, module_names))
    else:
        _extra_config_modules = None
    return _extra_config_modules


_default_config = Environment(
    device=LazyEval(_get_device),
    ignoring_parsing_errors=True,
    git_hash=_get_git_hash(),
    debug=_DEBUG,
    decoding_speed_optimization=True,
)


@variable
def config():
    cmd_arg_dict = _parse_cmd_args()

    domain_config = domain.configuration.config

    specific_config_module = _get_specific_config_module()
    specific_config = (dict() if specific_config_module is None else
                       specific_config_module.config)

    extra_config_modules = _get_extra_config_modules()
    extra_configs = ([] if extra_config_modules is None else
                     [module.config for module in extra_config_modules])

    config = Environment(
        chain(
            distinct_pairs(chain(
                _default_config.items(),
                domain_config.items(),
                specific_config.items(),
                cmd_arg_dict.items())),
            *(config.items() for config in extra_configs)
        ))
    return config


@variable
def config_path_dict():
    specific_config_module = _get_specific_config_module()
    specific_config_module_file_path = (None if specific_config_module is None else
                                        specific_config_module.__file__)
    extra_config_modules = _get_extra_config_modules()
    extra_config_module_file_paths = ([] if extra_config_modules is None else
                                      [module.__file__ for module in extra_config_modules])
    return dict(
        general=__file__,
        specific=specific_config_module_file_path,
        extra=extra_config_module_file_paths
    )


def _config_to_json_dict(config):
    return dict([key, value] for key, value in config.items()
                # if not isinstance(value, (LazyEval, Register))
                if not isinstance(value, (LazyEval,)))


@accelerator.within_local_main_process
def save_config_info(dir_path):
    json_dict = dict(config.items())
    config_info_path = get_new_path_with_number(
        os.path.join(dir_path, f'config-info-{config.run_mode}'),
        starting_num=1, no_first_num=True
    )
    mkloc_unless_exist(config_info_path)
    json_pretty_save(json_dict, os.path.join(config_info_path, 'config.json'), cls=InvalidObjectSkippingJSONEncoder)
    json_pretty_save(config_path_dict, os.path.join(config_info_path, 'config-path.json'))
    shutil.copytree('./config', os.path.join(config_info_path, 'config'))
    shutil.copyfile('./configuration.py', os.path.join(config_info_path, 'configuration.py'))
    shutil.copyfile(f'./domain/{config.domain_name}/configuration.py',
                    os.path.join(config_info_path, f'{config.domain_name}_configuration.py'))


# if 'run_mode' in config and _is_training_run_mode(config.run_mode):
#     save_config_info(config.model_learning_dir_path)
