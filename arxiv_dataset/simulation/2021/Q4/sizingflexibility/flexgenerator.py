"""
__author__ = "Babu Kumaran Nalini"
__copyright__ = "2021 TUM-EWK"
__credits__ = []
__license__ = "GPL v3.0"
__version__ = "1.0"
__maintainer__ = "Babu Kumaran Nalini"
__email__ = "babu.kumaran-nalini@tum.de"
__status__ = "Development"
"""


from analysis.date_incrementer import date_incrementer
import opentumflex
import os
import tqdm
import time
import pickle

# File path instructions
_base_dir = os.path.abspath(os.getcwd())
_input_file = r'\input\input_data_'
_output_dir = r'\output'
path_input_data = _base_dir + _input_file
path_results = _base_dir + _output_dir

# Initialize
days = 365

for cpv in range(6):
    for cbat in range(6):
        ems = [{} for sub in range(days)]
        for day in tqdm.tqdm(range(days)):   
            ems[day] = opentumflex.run_scenario(opentumflex.scenario_flexsize,       # Select scenario from scenario.py
                                            path_input=path_input_data,              # Input path
                                            path_results=path_results,               # Output path
                                            date=date_incrementer(day),              # Start date
                                            solver='glpk',                           # Select solver
                                            time_limit=50,                           # Time limit to solve the optimization
                                            save_opt_res=False,                      # Save optimization results
                                            show_opt_balance=False,                  # Plot energy balance
                                            show_opt_soc=False,                      # Plot optimized SOC plan
                                            show_flex_res=False,                     # Show flexibility plots
                                            show_aggregated_flex=False,              # Plot aggregated flex
                                            show_aggregated_flex_price='bar',        # Plot aggregated price as bar/scatter
                                            save_flex_offers=False,                  # Save flexibility offers in comax/alf format
                                            convert_input_tocsv=False,               # Save .xlsx file to .csv format
                                            cpv=cpv, cbat=cbat,                      # Select PV and Bat properties
                                            troubleshooting=False)                   # Troubleshooting on/off
              
        # create a general dictionary to store all required values
        dict_ems = {}
        dict_ems['fcst'] = ems[0]['fcst']
        dict_ems['optplan'] = ems[0]['optplan']
        dict_ems['pvflex'] = {'Neg_P': ems[0]['flexopts']['pv']['Neg_P'].to_list(), 
                              'Neg_E': ems[0]['flexopts']['pv']['Neg_E'].to_list()} 
        dict_ems['batflex'] = {'Neg_P':ems[0]['flexopts']['bat']['Neg_P'].to_list(), 
                               'Neg_E':ems[0]['flexopts']['bat']['Neg_E'].to_list(), 
                               'Pos_P':ems[0]['flexopts']['bat']['Pos_P'].to_list(), 
                               'Pos_E':ems[0]['flexopts']['bat']['Pos_E'].to_list()}
        
        for i in range(1,len(ems)):
            for key in dict_ems['fcst']:
                dict_ems['fcst'][key] = dict_ems['fcst'][key] + ems[i]['fcst'][key]
            for key in dict_ems['optplan']:    
                dict_ems['optplan'][key] = dict_ems['optplan'][key] + ems[i]['optplan'][key]
            for key in dict_ems['pvflex']:
                dict_ems['pvflex'][key] = dict_ems['pvflex'][key] + ems[i]['flexopts']['pv'][key].to_list()
            for key in dict_ems['batflex']:    
                dict_ems['batflex'][key] = dict_ems['batflex'][key] + ems[i]['flexopts']['bat'][key].to_list()
        
        file_to_write = open("var/ems"+str(cpv)+str(cbat)+".pickle", "wb")
        pickle.dump(dict_ems, file_to_write)
        
        # print when all the ems for the days is generated
        print('\nEMS '+str(cpv)+str(cbat)+' generated')