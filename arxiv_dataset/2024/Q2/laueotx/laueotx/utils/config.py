import os, sys, warnings,numpy as np, time, itertools
from laueotx.utils import logging as utils_logging
from laueotx.peaks_data import PeaksData

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)

LOGGER = utils_logging.get_logger(__file__)


def get_angles(conf_angles):

	if type(conf_angles) is list:
		return np.array(conf_angles)
	elif type(conf_angles) is dict:
		return np.arange(conf_angles['start'], conf_angles['end'], conf_angles['step'])


def read_peaks_data(conf, beamline):

    lpd = []
    for p in conf['data']['peaks']:

        pd = PeaksData(detector_type=p['detector'],
                       beamline=beamline, 
                       filepath=os.path.join(conf['data']['path_root'], p['filepath']),
                       apply_coord_shift=p['apply_coord_shift'],
                       flip_x_coord=p['flip_x_coord'])
        lpd.append(pd)


    return lpd


