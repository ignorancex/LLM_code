import pandas as pd
import numpy as np
from collections import OrderedDict
from astropy import units as u, constants as cst

"""
Little script to grab all the results of spectral fits and make a nice latex table
(9 fits total)
"""

def get_chains(gravity=False, sinfoni=False, model='bt-settl-cifist', gp=False, teff_prior=None):
    
    header = ['teff', 'logg']
    if model == 'exo-rem':
        header += ['co', 'feh']
    header += ['rad','plx']
    name = ''
    if gravity:
        name += 'gravity_'
    if sinfoni:
        name += 'sinfoni_'
    name += 'jwst_'
    if not gp:
        name += 'nosphereGP_'
    else:
        header += ['gp_length_scale','gp_amp']
    if sinfoni:
        header += ['rv_sinfoni']
    if teff_prior is not None:
        name += '{}_'.format(teff_prior)
    header += ['']
    name += model

    chains_file = 'results/{}/multinest/post_equal_weights.dat'.format(name)
    chains = pd.read_csv(chains_file, delim_whitespace=True, names=header)

    return chains

models = OrderedDict()
models['\\texttt{BT-Settl} (G only, yes GP)'] = get_chains(gravity=True, gp=True, teff_prior='teffhi')
models['\\texttt{BT-Settl} (G only, yes GP) (teff lo)'] = get_chains(gravity=True, gp=True, teff_prior='tefflo')

models['\\texttt{BT-Settl} (Si only, yes GP)'] = get_chains(sinfoni=True, gp=True, teff_prior='teffhi')
models['\\texttt{BT-Settl} (Si only, yes GP) (teff lo)'] = get_chains(sinfoni=True, gp=True, teff_prior='tefflo')

models['\\texttt{BT-Settl} (G only, no GP)'] = get_chains(gravity=True, teff_prior='teffhi')
models['\\texttt{BT-Settl} (G only, no GP) (teff lo)'] = get_chains(gravity=True, teff_prior='tefflo')

models['\\texttt{BT-Settl} (Si only, no GP)'] = get_chains(sinfoni=True, teff_prior='teffhi')
models['\\texttt{BT-Settl} (Si only, no GP) (teff lo)'] = get_chains(sinfoni=True, teff_prior='tefflo')

models['\\texttt{Exo-REM} (Si+G, yes GP)'] = get_chains(model='exo-rem', sinfoni=True, gravity=True, gp=True, teff_prior=None)

def format_post(array, decimals=0):
    quantiles = np.quantile(array, [.16,.5,.84])

    median = np.round(quantiles[1], decimals=decimals)
    plus = np.round(quantiles[2] - quantiles[1], decimals=decimals)
    minus = np.round(quantiles[1] - quantiles[0], decimals=decimals)
    if decimals <= 0:
        plus = int(plus)
        minus = int(minus)
        median = int(median)
    if plus == minus:
        return '${}\\pm{}$'.format(median, minus)
    else:
        return '${}^{{+{}}}_{{-{}}}$'.format(median, plus, minus)

def calc_lum(radius, teff):
    lum_planet = (
        4.0 * np.pi *
        (radius * cst.R_jup)**2 *
        cst.sigma_sb * 
        (teff * u.K)**4 / 
        cst.L_sun
    )

    return np.log10(lum_planet)

for model_name in models.keys():

    print_model_name = model_name
    if '(teff lo)' in model_name:
        print_model_name = ''
    
    if not 'no GP' in model_name:
        gp_amp = format_post(models[model_name].gp_amp.values, decimals=1)
        gp_len = format_post(np.exp(models[model_name].gp_length_scale.values), decimals=1)
    else:
        gp_amp = '--'
        gp_len = '--'
    if 'Exo-REM' not in model_name:
        co = '--'
        feh = '--'
    else:
        co = format_post(models[model_name].co.values, decimals=2)
        feh = format_post(models[model_name].feh.values, decimals=3)
    if 'Si' in model_name:
        rv = format_post(models[model_name].rv_sinfoni.values, decimals=0)
    else:
        rv = '--'

    lum = calc_lum(models[model_name].rad.values, models[model_name].teff.values)
    print(
        '{} & {} & {} & {} & {} & {} & {}  & {} & {} & {}\\\\'.format(
            print_model_name,
            format_post(models[model_name].teff.values),
            format_post(models[model_name].logg.values, decimals=2),
            format_post(models[model_name].plx.values, decimals=2),
            gp_len, gp_amp, co, feh, 
            format_post(models[model_name].rad.values, decimals=2), 
            rv,
            format_post(lum, decimals=2),
        )
    )


