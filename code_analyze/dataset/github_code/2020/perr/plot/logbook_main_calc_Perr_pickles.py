"""
Log book of pickles saved to disk
"""

from __future__ import print_function, division


def get_pickle_fname():

    # Run A1: Quadratic bias model
    # M=12.8-13.8, Ngrid=64, seeds 400,403
    #fname = 'main_calc_Perr_2019_May_10_14:50:00_time1557499800.dill' # old
    #fname = 'main_calc_Perr_2019_May_15_17:39:57_time1557941997.dill'

    # Run A2: Include cubic operators
    fname = 'main_calc_Perr_2020_Feb_26_22:19:40_time1582755580.dill'


    plot_opts = {}
    return {'fname': fname, 'plot_opts': plot_opts}
