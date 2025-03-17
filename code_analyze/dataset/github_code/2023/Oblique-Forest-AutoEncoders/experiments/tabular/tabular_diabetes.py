################################################################################################### 
#                               Load modules
###################################################################################################
import sys
module_path = "./misc/"
sys.path.insert(0, module_path)

from datasets import load_dataset
from models import run_tabular_model
from utils import plot_results_tabular, makedir, to_json



################################################################################################### 
#                               Config
###################################################################################################
opts_diabetes = {
    'name': 'diabetes',
    'random_state': None,
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
    'oblique_method': 'HHCART',
    'verbose_model': False,
}


opts_hhcart = {
    'hhcart_method': 'proj',
    'hhcart_tau': 1e-4,
    'max_depth': 3,
    'min_samples': 2,
}

if opts_model["oblique_method"] == "HHCART":
    opts_estimator = opts_hhcart
else:
    raise ValueError("The oblique tree must be of type HHCART !")



################################################################################################### 
#                               Function
###################################################################################################
def run_tabular_diabetes(folder, folder_plots, filename):
    dataset = load_dataset(opts_diabetes)
    opts = [opts_model, opts_estimator, opts_diabetes]
    test_params = run_tabular_model(dataset, opts)

    to_json(folder, filename + "--settings", opts)
    plot_results_tabular(test_params, [2, 3, 4, 5], path = folder + folder_plots + filename + "--plot", save = True)



################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "tabular_diabetes/"
path_name_plots = "plots/" 

filename_results = "tabular_diabetes"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_tabular_diabetes(path_results, path_name_plots, filename_results)

