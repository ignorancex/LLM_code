################################################################################################### 
#                               Load modules
###################################################################################################
import sys
module_path = "./misc/"
sys.path.insert(0, module_path)

import pandas as pd 
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from datasets import load_dataset
from models import run_tabular_model
from utils import make_boxplot_estimators, makedir, to_json



################################################################################################### 
#                               Config
###################################################################################################
opts_htru2 = {
    'name': 'htru2',
    'random_state': None,
    'train_path': 'D:/htru2',
    'test_path': None,
    'train_filename': 'HTRU_2', 
    'test_filename': None,
    'test_size': 0.5,
    'N_train': 50,
    'N_test': 10,
    'verbose': True,
}


opts_model = {
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
    'estimators': list(range(1, 30, 3))[1:],
}



################################################################################################### 
#                               Function
###################################################################################################
def run_tabular_boxplot_estimators_htru2(folder, folder_plots, filename):

    dataset = load_dataset(opts_htru2)
    results = pd.DataFrame(columns = ['mse', 'method', 'n_estimators'])

    to_print_in_json = [opts_model, opts_htru2]
    to_print_in_json.append({'n_runs': opts_simulations['n_runs']})


    for i in range(opts_simulations["n_runs"]):

        iterator = tqdm(opts_simulations["estimators"], total = len(opts_simulations["estimators"]), 
            desc = '[n_estim = %d] ' % (0), 
            bar_format = "{desc} : {percentage:.2f}% | {bar} | [{n_fmt} / {total_fmt}]", position = 0, mininterval = 10)

        for estim in iterator:
            for oblique_method in ["HHCART", "RandCART"]:

                current_opts_model = opts_model.copy()
                current_opts_model["n_estimators"] = estim
                current_opts_model["oblique_method"] = oblique_method
                
                current_opts_estimator = opts_hhcart if oblique_method == "HHCART" else opts_randcart                
                current_opts = [current_opts_model, current_opts_estimator, opts_htru2]

                if i == opts_simulations["n_runs"] - 1:
                    to_print_in_json.append({'n_estimators': current_opts_model["n_estimators"]})
                    to_print_in_json.append(current_opts_estimator)

                current_test_params = run_tabular_model(dataset, current_opts, 
                                                        n_train = opts_htru2['N_train'], n_test = opts_htru2['N_test'])
                                                        
                current_mse = mean_squared_error(current_test_params[1], current_test_params[0])

                results.loc[len(results)] = [current_mse, oblique_method, estim]

            iterator.set_description('[n_estim = %d]' % (estim), refresh = False)


    to_json(folder, filename + "--settings", to_print_in_json)
    make_boxplot_estimators(results, ["n_estimators", "mse", "method"], 
                        path = folder + folder_plots + filename + "--plot", save = True)





################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "tabular_boxplot_estimators_htru2/"
path_name_plots = "plots/" 

filename_results = "tabular_boxplot_estimators_htru2"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_tabular_boxplot_estimators_htru2(path_results, path_name_plots, filename_results)

