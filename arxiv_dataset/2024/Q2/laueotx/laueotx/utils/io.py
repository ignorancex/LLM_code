import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, numpy as np, pickle, importlib, h5py
from . import logging as utils_logging
LOGGER = utils_logging.get_logger(__file__)

def is_remote(path):
    return '@' in path and ':/' in path

def robust_makedirs(path):

    if is_remote(path):
        LOGGER.info('Creating remote directory {}'.format(path))
        host, path = path.split(':')
        cmd = 'ssh {} "mkdir -p {}"'.format(host, path)
        subprocess.call(shlex.split(cmd))

    elif not os.path.isdir(path):
        try:
            os.makedirs(path)
            LOGGER.info('Created directory {}'.format(path))
        except FileExistsError as err:
            LOGGER.error(f'already exists {path}')

def robust_remove(dirpath):

    if os.path.isdir(dirpath):
        LOGGER.info(f'removing {dirpath}')
        shutil.rmtree(dirpath)
    else:
        LOGGER.error(f'dir {dirpath} not found')

def get_abs_path(path):

    if '@' in path and ':/' in path:
        abs_path = path

    elif os.path.isabs(path):
        abs_path = path

    else:
        if 'SUBMIT_DIR' in os.environ:
            parent = os.environ['SUBMIT_DIR']
        else:
            parent = os.getcwd()

        abs_path = os.path.join(parent, path)

    return abs_path

def read_yaml(filename):

    import yaml
    with open(filename, 'r') as fobj:
        d = yaml.load(fobj, Loader=yaml.FullLoader)
    LOGGER.debug('read yaml {}'.format(filename))
    return d

def write_yaml(filename, d):

    import yaml

    with open(filename, 'w') as f:
        stream = yaml.dump(d, default_flow_style=False, width=float("inf"))
        f.write(stream.replace('\n- ', '\n\n- '))

    LOGGER.debug('wrote yaml {}'.format(filename))


def write_to_pickle(filepath, obj):

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    LOGGER.info(f'wrote {filepath}')


def read_from_pickle(filepath):

    with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    LOGGER.debug(f'read {filepath}')
    return obj

def parse_params_ot(pars_str):

    dict_pars = {}
    for p in pars_str.split(','):
        key, val = p.split('=')
        dict_pars[key] = float(val)

    return dict_pars

def read_config(conf, args):

    conf = read_yaml(conf)
        
    # config defaults

    conf.setdefault('solver', {})
    conf['solver'].setdefault('method_single', 'softslacks')
    conf['solver'].setdefault('method_ot', 'softslacks')
    conf['solver'].setdefault('params_ot', {})
    conf['solver']['params_ot'].setdefault('unbalanced_kappa_obs', 0.1)
    conf['solver']['params_ot'].setdefault('unbalanced_lambda_mod', 0.1)
    conf['solver']['params_ot'].setdefault('partial_fracmass', 0.9)
    conf['solver']['params_ot'].setdefault('softslacks_nsig_outliers', 3.)
    conf['solver']['params_ot'].setdefault('softslacks_frac_outliers', 0.05)
    conf['solver']['params_ot'].setdefault('softslacks_frac_unmatched', 0.4)

    # command line overrides

    grainspotter_overrides = ['solver', 'loss_nestedopt']
    for key in grainspotter_overrides:
        if hasattr(args, key):
            if getattr(args, key) is not None:
                conf['grainspotter'][key] = getattr(args, key)


    solver_overrides = {'method_ot': 'method_ot', 'outliers_m': 'outliers_m', 'outliers_tau': 'outliers_m'}
    for k, v in solver_overrides.items():
        if hasattr(args, v):
            if getattr(args, v) is not None:
                conf['solver'][k] = getattr(args,v)

    general_overrides = {'noise_sigma': 'noise_sig', 'frac_outliers': 'frac_outliers'}
    for k, v in general_overrides.items():
        if hasattr(args, v):
            conf[k] = getattr(args,v)

    if hasattr(args, 'params_ot'):
        if args.params_ot is not None:
            params_ot = parse_params_ot(args.params_ot)
            conf['solver']['params_ot'].update(params_ot)

    return conf

def write_arrays(fname, open_mode='w', *args, **kwargs):

    kw_compress = dict(compression='lzf', shuffle=True)

    j = 0
    with h5py.File(fname, open_mode) as f:
        for i, a in enumerate(args):
            if str(i) in f.keys(): 
                del(f[str(i)])
            f.create_dataset(name=f'{i:03d}', data=np.atleast_1d(a), **kw_compress)
            j+=1
        for key, val in kwargs.items():
            if key in f.keys(): 
                del(f[key])
            f.create_dataset(name=key, data=np.atleast_1d(val), **kw_compress)
            j+=1

    LOGGER.info(f'wrote {fname} with {j} arrays')

def read_arrays(fname, as_list=False):

    out = {}
    with h5py.File(fname, 'r') as f:
        for k in f.keys():
            out[k] = np.array(f[k])

    if as_list:
        keys = np.sort(list(out.keys()))
        return [v for k, v in out.items()]
    
    else:
        return out