################################################################################################### 
#                               Load modules
###################################################################################################
import sys
module_path = "./misc/"
sys.path.insert(0, module_path)

import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from datasets import load_dataset
from models import run_tabular_model
from utils import plot_results_tabular_misc, makedir, to_json



################################################################################################### 
#                               Config
###################################################################################################
opts_diabetes = {
    'name': 'diabetes',
    'random_state': None,
    'test_size': 0.25,
    'verbose': True,
    'N_train': None,
    'N_test': 10,
}


opts_model = {
    'n_estimators': 50,
    'max_samples': 0.5,
    'max_features': 0.75,
    'bootstrap': True,
    'bootstrap_features': False,
    'oob_score': False,
    'warm_start': False,
    'n_jobs': -1,
    'bounds_constraints': None,
    'oblique_method': 'HHCART',
    'verbose_model': False,
}


opts_hhcart = {
    'hhcart_method': 'proj',
    'hhcart_tau': 1e-4,
    'max_depth': 3,
    'min_samples': 2,
}


opts_simulations = {
    'n_runs': 10,
    'n_train_list': list(range(-20, 220, 20))[2:],
}


if opts_model["oblique_method"] == "HHCART":
    opts_estimator = opts_hhcart
else:
    raise ValueError("The oblique tree must be of type HHCART !")



################################################################################################### 
#                               Function
###################################################################################################
def run_tabular_n_train_vs_mse_diabetes(folder, folder_plots, filename):
    dataset = load_dataset(opts_diabetes)

    mean_results, std_results = [], []
    to_print_in_json = [opts_model, opts_estimator, opts_diabetes]
    to_print_in_json.append({'n_runs': opts_simulations['n_runs']})


    for i in opts_simulations["n_train_list"]:
        opts = [opts_model, opts_estimator, opts_diabetes]

        iterator = tqdm(range(opts_simulations["n_runs"]), total = len(range(opts_simulations["n_runs"])), 
        desc = '[n_runs = %d] ' % (0), 
        bar_format = "{desc} : {percentage:.2f}% | {bar} | [{n_fmt} / {total_fmt}]", position = 0, mininterval = 10)      


        mse_list = []

        for count_run in iterator:
            test_params = run_tabular_model(dataset, opts, n_train = i, n_test = opts_diabetes['N_test'])
            current_mse = mean_squared_error(test_params[1], test_params[0])
            mse_list.append(current_mse)
            iterator.set_description('[n_runs = %d]' % (count_run), refresh = False)
        
        mean_results.append( np.mean(mse_list) )
        std_results.append( np.std(mse_list) )

        to_print_in_json.append({'N_train': i, 'N_test': opts_diabetes['N_test']})


    to_json(folder, filename + "--settings", to_print_in_json)

    title_list = ["n_train", "mse"]
    plot_results_tabular_misc(opts_simulations["n_train_list"], mean_results, std_results, title_list, 
                                    path = folder + folder_plots + filename + "--plot", save = True)






################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "tabular_n_train_vs_mse_diabetes/"
path_name_plots = "plots/" 

filename_results = "tabular_n_train_vs_mse_diabetes"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_tabular_n_train_vs_mse_diabetes(path_results, path_name_plots, filename_results)

