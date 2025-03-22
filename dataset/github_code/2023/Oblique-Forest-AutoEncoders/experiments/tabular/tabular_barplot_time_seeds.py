################################################################################################### 
#                               Load modules
###################################################################################################
import sys
module_path = "./misc/"
sys.path.insert(0, module_path)

import pandas as pd 
from tqdm import tqdm

from datasets import load_dataset
from models import run_tabular_model_with_time
from utils import make_barplot_time, makedir, to_json




################################################################################################### 
#                               Config
###################################################################################################
opts_seeds = {
    'name': 'seeds',
    'random_state': None,
    'train_path': 'D:/seeds',
    'test_path': None,
    'train_filename': 'seeds', 
    'test_filename': None,
    'test_size': 0.25,
    'verbose': True,
}


opts_model = {
    'n_estimators': 200,
    'max_samples': 0.5,
    'max_features': 0.75,
    'bootstrap': True,
    'bootstrap_features': False,
    'oob_score': False,
    'warm_start': False,
    'n_jobs': -1,
    'bounds_constraints': None,
    'verbose_model': False,
}


opts_hhcart = {
    'hhcart_method': 'eig',
    'hhcart_tau': 1e-4,
    'max_depth': 3,
    'min_samples': 2,
}

opts_randcart = {
    'max_depth': 3,
    'min_samples': 2,
    'compare_with_cart': False,
}


opts_simulations = {
    'n_runs': 10,
}



################################################################################################### 
#                               Function
###################################################################################################
def run_tabular_barplot_time_seeds(folder, folder_plots, filename):

    dataset = load_dataset(opts_seeds)
    results = pd.DataFrame(columns = ['time', 'time_category', 'method'])
    method_list = ["HHCART", "RandCART"]

    to_print_in_json = [opts_model, opts_seeds]
    to_print_in_json.append({'n_runs': opts_simulations['n_runs']})


    for i in range(opts_simulations["n_runs"]):

        iterator = tqdm(method_list, total = len(method_list), 
            desc = '[method = %s] ' % ("HHCART"), 
            bar_format = "{desc} : {percentage:.2f}% | {bar} | [{n_fmt} / {total_fmt}]", position = 0, mininterval = 10)

        for oblique_method in iterator:

            current_opts_model = opts_model.copy()
            current_opts_model["oblique_method"] = oblique_method
            
            current_opts_estimator = opts_hhcart if oblique_method == "HHCART" else opts_randcart                
            current_opts = [current_opts_model, current_opts_estimator, opts_seeds]

            if i == opts_simulations["n_runs"] - 1:
                to_print_in_json.append(current_opts_estimator)

            current_test_params = run_tabular_model_with_time(dataset, current_opts)
            current_fit_time, current_decode_time = current_test_params[2], current_test_params[3]

            results.loc[len(results)] = [current_fit_time, 'Fit', oblique_method]
            results.loc[len(results)] = [current_decode_time, 'Decode', oblique_method]

            iterator.set_description('[method = %s]' % (oblique_method), refresh = False)


    to_json(folder, filename + "--settings", to_print_in_json)
    make_barplot_time(results, ["time_category", "time", "method"], 
                            path = folder + folder_plots + filename + "--plot", save = True)




################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "tabular_barplot_time_seeds/"
path_name_plots = "plots/" 

filename_results = "tabular_barplot_time_seeds"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_tabular_barplot_time_seeds(path_results, path_name_plots, filename_results)

