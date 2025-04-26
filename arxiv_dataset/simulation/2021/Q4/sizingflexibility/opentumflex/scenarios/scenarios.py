"""
The "scenarios.py" generate diverse scenarios to simplify the modeling procedure
"""

__author__ = "Babu Kumaran Nalini" "Michel Zadé" "Zhengjie You"
__copyright__ = "2020 TUM-EWK"
__credits__ = []
__license__ = "GPL v3.0"
__version__ = "1.0"
__maintainer__ = "Babu Kumaran Nalini"
__email__ = "babu.kumaran-nalini@tum.de"
__status__ = "Development"

from opentumflex.configuration.devices import create_device

# Input only scenario. Read all parameters from input file with no modifications
def scenario_fromfile(ems):
    """ change the device parameters and obtain the input time series
        according to customized scenario from spreadsheet(xlsx/csv)
        this function is only a placeholder, which serves as tag for selector in run_scenario.py

    Args:
        - ems: ems model instance
        
    """
    return ems

def scenario_flexsize(ems, i, j):
    siz_pv = [0, 1.62, 3.24, 4.86, 6.48, 8.1]
    siz_bat = [0, 4.5, 9, 13.5, 18, 22.5]
    ems['devices'].update(create_device(device_name='pv', minpow=0.5, maxpow=siz_pv[i], eta=0.95))
    ems['devices'].update(create_device(device_name='bat', minpow=0, maxpow=siz_bat[j]/2, stocap=siz_bat[j], init_soc=50, eta=0.95))
    # constant and dual pricing scenario (default is variable pricing)
    # ems['fcst']['ele_price_in'] = [0.28]*ems['time_data']['nsteps']
    # ems['fcst']['ele_price_in'] = [0.23]*24+[0.29]*60 + [0.23]*12
    return ems


