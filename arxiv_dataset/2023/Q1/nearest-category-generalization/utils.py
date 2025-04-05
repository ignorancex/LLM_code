import os
import json
import logging
import gc
from typing import List, Dict, Union, Callable, Any, Optional

import joblib
from mkdir_p import mkdir_p
import numpy as np
import pandas as pd

from lolip.variables import auto_var, get_file_name
from autovar import AutoVar
from autovar.hooks import get_ext

import experiments

logging.basicConfig(level=0)
tex_base = "./tex_files"

def _register_experiment(auto_var, exp_name):
    mkdir_p(f"./results/{exp_name}")
    auto_var.register_experiment(f'{exp_name}', getattr(experiments, f'run_{exp_name}'),
            {'file_format': 'pickle', 'result_file_dir': f'./results/{exp_name}'})

def setup_experiments(auto_var):
    _register_experiment(auto_var, 'out_of_sample')
    _register_experiment(auto_var, 'oos_repr')
    _register_experiment(auto_var, 'get_preds')
    _register_experiment(auto_var, 'ood_robustness_correct')

class Experiments():
    name: str
    experiment_fn: Callable[[AutoVar], Any]
    grid_params: Union[List[Dict[str, str]], Dict[str, str]]
    run_param: Dict[str, Any]

    def __init__(self):
        pass

    def __call__(self):
        return self.experiment_fn, self.name, self.grid_params, self.run_param

class ExpExperiments(Experiments):
    def __new__(cls, *args, **kwargs):
        # if attribute is function it will pass self as one of its argument
        cls.run_param = {'verbose': 1, 'n_jobs': 4,}
        return Experiments.__new__(cls, *args, **kwargs)

def get_result(auto_var, columns=None):
    file_name = get_file_name(auto_var)
    file_format = auto_var.settings['file_format']
    file_path = os.path.join(auto_var.settings['result_file_dir'],
                             f"{file_name}.{get_ext(file_format)}")
    if not os.path.exists(file_path):
        print("Missing %s" % file_path)
        return None
    try:
        if file_format == 'json':
            with open(file_path, "r") as f:
                tret = json.load(f)
        elif file_format == 'pickle':
            with open(file_path, "rb") as f:
                tret = joblib.load(f)
        else:
            raise ValueError(f"Not supported file format {file_format}")
    except Exception as e:
        print("problem with %s" % file_path)
        raise e

    ret = dict()
    if columns is not None:
        for j, col in enumerate(columns):
            if isinstance(col, tuple):
                current = tret[col]
                for col in column:
                    if isinstance(current, dict):
                        if col not in current:
                            current = np.nan
                            break
                    elif isinstance(current, list):
                        if col >= len(current):
                            current = np.nan
                            break
                    else:
                        break
                    current = current[col]
                ret[col] = current
            elif col not in tret:
                ret[col] = np.nan
            else:
                ret[col] = tret[col]
    else:
        ret = tret
    return ret

def params_to_dataframe(grid_param, columns: List[Union[tuple, str]],
                        proc_fns: Optional[List[Union[Callable]]] = None,
                        file_format=None, result_file_dir=None, verbose=0,
                        logging_level=logging.INFO):
    #{'file_format': 'pickle', 'result_file_dir': './results/normal'}
    auto_var.set_logging_level(logging_level)
    if file_format is not None:
        auto_var.settings['file_format'] = file_format
    if result_file_dir is not None:
        auto_var.settings['result_file_dir'] = result_file_dir
    params, loaded_results = auto_var.run_grid_params(
            get_result, grid_param, with_hook=False, verbose=verbose, n_jobs=1)
    results = loaded_results
    if proc_fns is not None:
        assert len(proc_fns) == len(columns)

    params, results = zip(*[(params[i], results[i]) for i in range(len(params)) if results[i]])
    params, results = list(params), list(results)
    for i, _ in enumerate(params):
        for j, column in enumerate(columns):
            if isinstance(column, tuple):
                current = results[i]
                for col in column:
                    if isinstance(current, dict):
                        if col not in current:
                            current = np.nan
                            break
                    elif isinstance(current, list):
                        if col >= len(current):
                            current = np.nan
                            break
                    else:
                        break
                    current = current[col]
                if proc_fns is not None and j < len(proc_fns) and proc_fns[j] is not None:
                    current = proc_fns[j](current)
                params[i][column] = current
            elif column not in results[i]:
                params[i][column] = np.nan
            else:
                params[i][column] = results[i][column]

    df = pd.DataFrame(params)

    del params
    del results
    gc.collect()

    return df

def set_plot(fig, ax, ord=np.inf):
    fig.autofmt_xdate()
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0.)
    ax.legend(prop={'size': 16}, loc='upper right', frameon=True)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_ylabel('Accuracy', fontsize=20)
    xlabel = 'Perturbation distance'
    if ord == np.inf:
        ax.set_xlabel(xlabel + ' (Linf)', fontsize=20)
    else:
        ax.set_xlabel(xlabel, fontsize=20)

def union_param_key(grid_param, key):
    if isinstance(grid_param, list):
        ret = []
        for g in grid_param:
            for v in g[key]:
                if v not in ret:
                    ret.append(v)
        return ret
    else:
        return grid_param[key]
