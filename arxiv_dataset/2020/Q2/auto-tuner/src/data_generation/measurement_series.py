"""
Author: Renato Durrer
Created: 01.04.2019

File used for measurements.
"""
from Labber import ScriptTools
import os
from src.utils.measurement_funcs import measure
from itertools import product


setup = {
    'LRG': [-0.65, -0.67],
    'LTG': [-1.17, -1.18],
    'RTG': [-1.38],
    'WGL': [-0.69],
    'RPG': [-40e-3]
}

PG1 = {
    'name': 'LPG0',
    'mean': -330e-3,
    'start': -180e-3,
    'stop': -470e-3,
    'step': 1e-3
}

PG2 = {
    'name': 'MPG0',
    'mean': -410e-3,
    'start': -260e-3,
    'stop': -550e-3,
    'step': 1e-3
}

# create configurations
configs = []
keys, values = zip(*setup.items())
for v in product(*values):
    config = dict(zip(keys, v))
    config['WGR'] = config['WGL']
    configs.append(config)


ScriptTools.setExePath(r'C:\\Program Files (x86)\\Labber\\Program')
sPath = os.path.abspath(r'C:\\Users\\Measure2\\Desktop\\measurement_series\\charge transitions')
path_in = os.path.join(sPath, 'coarse_msm_config2.hdf5')


# Measure the configuration
for k in range(132, 136):
    print('Currently measuring:\n'
          'Configuration: {}\n'
          'Number: {}'.format(configs[k-132], k))
    path_out = os.path.join(sPath, 'test\\{}.hdf5'.format(k))
    Measurement = ScriptTools.MeasurementObject(
        path_in,
        path_out
    )
    path = os.path.join(sPath, 'msm5\\{}'.format(k))
    measure(PG1, PG2, Measurement, path, calibrate=True, config=configs[k-132])
