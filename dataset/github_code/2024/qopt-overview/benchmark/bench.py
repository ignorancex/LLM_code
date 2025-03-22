"""Implements abstract interfaces for instance storage and QPU solvers."""
import glob
import json
import logging
import os
import traceback
from abc import abstractmethod
from datetime import datetime

import numpy as np

__version__ = '2.0.2'
__debug_mode__ = False


class DataBase():
    """Implements storage/access to instances to be solved on QPUs."""
    def __init__(self, directory, file_ext='.json', filter_fun=None, sort_fun=None):
        self.directory = directory
        self.file_ext = file_ext
        self.filter_fun = filter_fun
        self.sort_fun = sort_fun
        self.db = {}
        self._gather_data()

    def _gather_data(self):
        self.db.clear()
        self._load_map = {}
        glob_request = f'{self.directory}/*{self.file_ext}'
        for path in glob.glob(glob_request):
            with open(path, 'r') as fh:
                data = json.load(fh)
                if self.filter_fun is None or self.filter_fun(data):
                    _, filename = os.path.split(path)
                    instance_id = data['description']['instance_id']
                    assert instance_id not in self.db
                    self.db[instance_id] = data
                    self._load_map[filename] = instance_id
        if self.sort_fun is not None:
            self.db = dict(sorted([(instance_id, data) for instance_id, data in self.db.items()],
                                  key=lambda item: self.sort_fun(item[0], item[1])))

    def _preprocess_qubo_data(self, Q, P, a, b):
        if a is None or b is None:
            return Q, P
        else:
            assert a == 1 and b == 1
            qubo_matrix = Q / 2 + np.diag(P)
            scaling_factor = a / np.max(np.abs(qubo_matrix)) ** b
            Q_scaled = Q * scaling_factor
            P_scaled = P * scaling_factor
            return Q_scaled, P_scaled

    def get_data(self, instance_id):
        for data in self.db.values():
            if data['description']['instance_id'] == instance_id:
                return data
        raise ValueError

    def get_qubo_data(self, instance_id, a, b):
        qubo_data = self.db[instance_id]
        # get_data, Const is ignored
        Q = np.array(qubo_data['Q'])
        P = np.array(qubo_data['P'])
        return self._preprocess_qubo_data(Q, P, a, b)

    def get_instance_ids(self):
        return (key for key in self.db.keys())

    def get_filename_from_instance_id(self, instance_id):
        load_map_inv = {instance_id: filename for (filename, instance_id) in self._load_map.items()}
        return load_map_inv[instance_id]


class Solver():
    """Implements abstract QPU Solver interface.

    See device-specific code in :py:mod:`bench_qubo`.
    """
    def __init__(self, name, db):
        self.name = name
        self.db = db
        self.logger = None

    @abstractmethod
    def _run(self, instance_id, *args):
        raise NotImplementedError

    @abstractmethod
    def get_data(self, instance_id, *args):
        raise NotImplementedError

    def _stringify_keys(self, d):
        # Convert a dict's keys to strings if they are not. (https://stackoverflow.com/questions/12734517/json-dumping-a-dict-throws-typeerror-keys-must-be-a-string)
        d_new = dict()
        for key in d.keys():
            value = d[key]
            if isinstance(value, dict):
                new_value = self._stringify_keys(value)
            elif isinstance(value, list) or isinstance(value, tuple):
                new_value = [self._stringify_keys(v) if isinstance(v, dict) else v for v in value]
                if isinstance(value, tuple):
                    new_value = tuple(new_value)
            else:
                new_value = value
            if not isinstance(key, str) and not isinstance(key, int) and not isinstance(key, float) and not isinstance(
                    key, bool) and key is not None:
                try:
                    new_key = str(key)
                except Exception:
                    try:
                        new_key = repr(key)
                    except Exception as e:
                        raise e
            else:
                new_key = key
            d_new[new_key] = new_value
        return d_new

    def _save(self, path, data):
        with open(path, 'w') as fh:
            try:
                new_dict = self._stringify_keys(data)
            except Exception as e:
                new_dict = dict(str_data=str(data), exception=e)  # fallback
                if __debug_mode__:
                    raise e
            json.dump(new_dict, fh, default=str)

    def _setup_logger(self, log_file, file_mode='w', log_name=__name__, log_level=logging.DEBUG, save_logger=True):
        logger = logging.getLogger(log_name)
        logger.setLevel(log_level)
        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s %(name)-6s %(levelname)-6s %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        if save_logger:
            self.logger = logger
        return logger

    def _run_multiple(self, run_list, save_file_path, logger):
        num_success = 0
        if logger is not None:
            logger.info(f'started {len(run_list)} run(s).')
        for run_idx, (instance_id, args) in enumerate(run_list):
            num_success += self.run(f'{save_file_path}_{run_idx:04d}', logger, instance_id, *args)['success']
            if logger is not None:
                logger.info(f'run {run_idx + 1}/{len(run_list)} complete.')
        if logger is not None:
            logger.info(f'completed {len(run_list)} run(s): {num_success} successful.')

    def run(self, save_file_path, logger, instance_id, *args):
        if logger is not None:
            logger.info(f'run start: instance_id={instance_id}, args={args}.')
        start_timestamp = datetime.now().timestamp()
        try:
            solver = self._run(instance_id, *args)
            success = True
        except Exception as e:
            fail_timestamp = datetime.now().timestamp()
            solver = dict(exception=e, traceback=traceback.format_exc(), fail_timestamp=fail_timestamp)
            success = False
            if __debug_mode__:
                raise e
        end_timestamp = datetime.now().timestamp()
        result = dict(instance_id=instance_id,
                      instance_filename=self.db.get_filename_from_instance_id(instance_id),
                      solver_name=self.name,
                      args=[str(arg) for arg in args],
                      timestamps=dict(start=start_timestamp, end=end_timestamp),
                      solver=solver,
                      success=success,
                      version=__version__)
        if save_file_path is not None:
            self._save(f'{save_file_path}.json', result)
        if logger is not None:
            logger.info(f'run finished: {"success" if success else "failure"}, saved to "{save_file_path}.json".')
        return result

    def run_multiple(self, run_list, save_file_path=None, logger=None):
        self._run_multiple(run_list, save_file_path, logger)

    def benchmark(self, run_list, save_file_path):
        logger = self._setup_logger(f'{save_file_path}.log')
        self._run_multiple(run_list, save_file_path, logger)
