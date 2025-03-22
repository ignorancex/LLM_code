import os 
import pandas as pd
import numpy as np


from . import physical_params as pp
from . import utils

def read_planet_data(filepath, completo=True):

    columns = ['System', 'Mp', 'Mperr', 'Rp', 'Rperr', \
                    'Mstar', 'Mstarerr', 'Teq', 'sma', \
                    'Tstar', 'Tstarerr', 'Rstar', 'Rstarerr']
    data = pd.read_csv(filepath, names=columns)

    if completo:
        # compute stellar luminosity as computed by COMPLETO
        Msample = pp.compute_mstar_completo_given_observables(
                    data['Tstar'], data['Tstarerr'], \
                    data['Rstar'], data['Rstarerr'], data['Mstar'][0])
        med, errl, erru    = utils.inference(Msample)
        mstar_completo     = med
        mstar_completo_err = ( errl + erru ) / 2.

        # append the stellar mass + err as computed by COMPLETO to the catalog
        data['Mstar_completo']    = mstar_completo
        data['Mstarerr_completo'] = mstar_completo_err

        return data

    return data


def get_catalog(filepath, filename, download_catalog=False, clean_data=True, filter_data=True):
    '''
    Load TEPCAT catalog

    Args:
        filepath         : the path to the catalog
        filename         : name of the catalog
        download_catalog : if True, download catalog! 
        clean_data       : if True, select only planets with measured masses, radii, and semimajor axis (i.e. != -1)
        filter_data      : if True, select only planets with:
                           0.37 < Mp / MJ < 2.6
                           4000 < Tstar/K < 7000 
                           logg > 4.0
                           0.01 < semimajor axis (AU) < 0.1

    Returns:
        System, Mp, Mperr, Rp, Rperr, Mstar, Mstarerr 
    '''
    if download_catalog is True:
        os.system('wget http://www.astro.keele.ac.uk/jkt/tepcat/allplanets-ascii.txt -P data/')
    
    # original columns from the TEPCAT catalog
    columns = ["System", "Tstar", "Tstaru", "Tstarl", "FeH", "FeHu", "FeHl", "Mstar",
           "Mstaru", "Mstarl", "Rstar", "Rstaru", "Rstarl", "logg",
           "loggu", "loggl", "rhos", "rhosu", "rhosl", "period", "ecc",
           "eccu", "eccl", "sma", "smau", "smal", "Mp", "mpu", "mpl", "Rp",
           "rpu", "rpl", "gravityp", "gravitypu", "gravitypl", "rhop", 
           "rhopu", "rhopl", "teq", "tequ", "teql", "discovery-ref", "recent-ref"]
    
    catalog = pd.read_csv(filepath+filename, sep = '\s+', names = columns, skiprows=1)

    # these are the only columns this function returns
    selected_columns = ['System', 'Mp', 'Mperr', 'Rp', 'Rperr', \
                    'Mstar', 'Mstarerr', 'Teq', 'sma', 'smaerr', \
                    'Tstar', 'Tstarerr', 'Rstar', 'Rstarerr']

    if clean_data is True:
        catalog = catalog.loc[ (catalog['Mp'] > 0) & (catalog['Rp'] > 0) & (catalog['sma'] > 0)]

        # compute symmetric uncertainties
        catalog['Mperr']    = ( catalog['mpu'] + catalog['mpl'] ) / 2.
        catalog['Rperr']    = ( catalog['rpu'] + catalog['rpl'] ) / 2.
        catalog['Mstarerr'] = ( catalog['Mstaru'] + catalog['Mstarl'] ) / 2.
        catalog['smaerr']   = ( catalog['smau'] + catalog['smal'] ) / 2.
        catalog['Tstarerr'] = ( catalog['Tstaru'] + catalog['Tstarl'] ) / 2.
        catalog['Rstarerr'] = ( catalog['Rstaru'] + catalog['Rstarl'] ) / 2.
        catalog['Feherr']   = ( catalog['FeHu'] + catalog['FeHl'] ) / 2.

        # compute Teq
        catalog['Teq'] = pp.compute_teq(catalog['sma'], catalog['Rstar'], catalog['Tstar'])


        if not filter_data:
            return catalog[selected_columns].reset_index(drop=True)

        else:
            # filter data 
            catalog = catalog.loc[ (catalog['Mp'] >= 0.37) & (catalog['Mp'] <= 13) \
                        & (catalog['Tstar'] > 4000) & (catalog['Tstar'] < 7000) \
                        & (catalog['logg'] >= 4) \
                        & (catalog['sma'] > 0.01) & (catalog['sma'] < 0.1)]

            return catalog[selected_columns].reset_index(drop=True)

