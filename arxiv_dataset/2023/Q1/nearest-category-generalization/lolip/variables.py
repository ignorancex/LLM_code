import logging
from functools import partial

import numpy as np
from autovar import AutoVar
from autovar.base import RegisteringChoiceType, register_var, VariableClass
from autovar.hooks import check_result_file_exist, save_result_to_file
from autovar.hooks import create_placeholder_file, remove_placeholder_if_error
from autovar.hooks import default_get_file_name as get_file_name

from .datasets import DatasetVarClass
from .models import ModelVarClass
from .attacks import AttackVarClass

auto_var = AutoVar(
    logging_level=logging.INFO,
    before_experiment_hooks=[
        partial(check_result_file_exist, get_name_fn=get_file_name),
        partial(create_placeholder_file, get_name_fn=get_file_name),
    ],
    after_experiment_hooks=[
        partial(save_result_to_file, get_name_fn=get_file_name),
        partial(remove_placeholder_if_error, get_name_fn=get_file_name),
    ],
    settings={
        'file_format': 'pickle',
        'server_url': '',
        'result_file_dir': './results/'
    }
)

class NormVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which distance measure to use for attack."""
    var_name = "norm"

    @register_var()
    @staticmethod
    def inf(auto_var):
        """L infinity norm"""
        return np.inf

    @register_var(argument='2')
    @staticmethod
    def l2(auto_var):
        """L2 norm"""
        return 2

    @register_var(argument='1')
    @staticmethod
    def l1(auto_var):
        """L1 norm"""
        return 1

auto_var.add_variable_class(NormVarClass())
auto_var.add_variable_class(DatasetVarClass())
auto_var.add_variable_class(ModelVarClass())
auto_var.add_variable_class(AttackVarClass())
auto_var.add_variable('random_seed', int)
auto_var.add_variable('eps', float)
auto_var.add_variable('optimizer', str)
auto_var.add_variable('learning_rate', float, default=1e-2)
auto_var.add_variable('batch_size', int, default=64)
auto_var.add_variable('momentum', float, default=0.9)
auto_var.add_variable('epochs', int, default=2)
auto_var.add_variable('weight_decay', float, default=0.)

auto_var.add_variable('vae_arch', str, default="")
auto_var.add_variable('vae_model_path', str, default="")

auto_var.add_variable('drocc_arch', str, default="")
auto_var.add_variable('lamda', float, default=0.5)
auto_var.add_variable('radius', float, default=-1.)
auto_var.add_variable('ascent_num_steps', int, default=50)
