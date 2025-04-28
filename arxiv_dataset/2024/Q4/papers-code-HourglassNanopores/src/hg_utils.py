# hourglass nanopores analysis helper utilities

import os
import subprocess, sys
import numpy as np
import json

def get_path(cmd, hint=''):
    """Get the path to a needed executable. """
    try:
        my_env = os.environ.copy()
        my_env["PATH"] = f"{sys.exec_prefix}/bin:{my_env['PATH']}"
        return os.path.dirname(subprocess.check_output(['which', cmd], env=my_env).decode('ascii').strip())
    except subprocess.CalledProcessError:
        return input(f"Enter path to `{cmd}` executables ({hint}): ")

def lab(_ΔR):
    """Dictionary key generator. """
    return f'dR_eq_{_ΔR:3.1f}'

def base_dir(_ΔR,cylinder=True, raw=False):
    """Generate various directory where data is stored/loaded from."""
    if raw:
        base = f'{raw_data_dir}/{lab(_ΔR)}/OUTPUT/MERGED'
    else:
        base = f'{data_dir}/{lab(_ΔR)}'
        
    if cylinder:
        base += '/CYLINDER'
    return base

def shifted_ave(y,Δy,ȳ,Δȳ):
    """The error in a shifted mean from https://en.wikipedia.org/wiki/Propagation_of_uncertainty"""

    f = y/ȳ - 1.0
    
    # First we compute the error in the numerator y-ȳ
    num = y-ȳ
    Δnum = np.sqrt(Δy**2 + Δȳ**2)

    # now we computer the error in (y-ȳ)/ȳ
    Δf = np.abs(f)*np.sqrt((Δnum/num)**2 + (Δȳ/ȳ)**2) 
    
    return f, Δf

# we load relavent local path information from a file if it exists, if not, we query the user
try:
    with open('../include/local.json', 'r') as f:
        local = json.load(f)
        pimc_bin_path = local['pimc_bin_path']
        data_dir = local['data_dir']
        gnu_parallel = local['gnu_parallel']
        raw_data_dir = local['raw_data_dir']

except:
    pimc_bin_path = get_path('merge.py',hint='(see e.g. https://github.com/DelMaestroGroup/pimcscripts)')
    gnu_parallel = get_path('parallel', hint='(e.g. /localb∈paral≤l/local/bin/parallel): ')
    data_dir = input("Enter path to merged QMC Data (enter for default ..data../data): ") or '../data'
    raw_data_dir = input("Enter path to raw QMC data (enter for default ..dataqmc../data/qmc: ") or '../data/qmc'
    
    local = {'pimc_bin_path':pimc_bin_path, 'data_dir':data_dir, 'gnu_parallel':gnu_parallel, 
            'raw_data_dir':raw_data_dir}

    # Make sure we strip any extraneous path separators
    for path_name, path in local.items():
        if path.endswith(os.sep):
            local[path_name] = path[:-1]
            
    with open('../include/local.json', 'w') as f:
        json.dump(local, f, ensure_ascii=True, indent=4)
