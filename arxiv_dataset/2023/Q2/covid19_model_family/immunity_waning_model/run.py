"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""


import os
import sys

from config import Config
from result_exporter import ResultExporter
from result_plotter import ResultPlotter
from simulation import Simulation
import datetime as dt

if __name__=='__main__':
    """
    Runs and plots the simulation for given config files
    """
    nowStamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    fl = sys.argv[1]
    if os.path.isdir(fl):
        files = [os.path.join(fl,x) for x in os.listdir(fl) if x.startswith('config') and x.endswith('json')]
    elif fl.endswith('.json'):
        files = [fl]
    else:
        raise RuntimeError('Cannot run simulation. Specified config file or folder is not valid')

    for filename in files:
        config = Config(filename,nowStamp) #new config instance
        s = Simulation(config) #initialize simulation
        result = s.run() #run simulation
        del(s) #free RAM space for plots

        # export to csv
        filename = ResultExporter().export_to_csv(config, result)

        # plot result in various ways
        rp = ResultPlotter()
        rp.load_result(filename)
        rp.set_plotPDF(config.plotPdfs)
        rp.set_darkBG(False)
        rp.set_dpi(200)
        filenamePrefix = config.scenario
        rp.plot_cases_by_variant(os.path.join(config.get_result_folder(), filenamePrefix + '.png'),False)
        rp.plot_cases_by_variant(os.path.join(config.get_result_folder(), filenamePrefix + '_newconfirmed.png'),True)
        for target in ['DELTA','OMICRON_BA1','OMICRON_BA2','OMICRON_BA5','HOSPITALISATION']:
            rp.plot_immunity(config,os.path.join(config.get_result_folder(), filenamePrefix + '_'+target+'_immunity.png'),
                             os.path.join(config.get_result_folder(), filenamePrefix + '_' + target + '_immunityLoss.png'),
                                     target)
        rp.plot_reinfections(os.path.join(config.get_result_folder(), config.scenario + '_reinfections_bubbles_absolute.png'),False)
        rp.plot_reinfections(os.path.join(config.get_result_folder(), config.scenario + '_reinfections_bubbles_relative.png'),True)