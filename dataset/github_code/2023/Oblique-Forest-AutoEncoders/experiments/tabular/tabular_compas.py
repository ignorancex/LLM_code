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
opts_compas = {
    'name': 'compas',
    'random_state': None,
    'train_path': 'D:/compas',
    'test_path': 'D:/compas',
    'train_filename': 'train_compas', 
    'test_filename': 'test_compas',
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
    'hhcart_method': 'eig',
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
def run_tabular_compas(folder, folder_plots, filename):
    dataset = load_dataset(opts_compas)
    opts = [opts_model, opts_estimator, opts_compas]
    test_params = run_tabular_model(dataset, opts)

    to_json(folder, filename + "--settings", opts)
    plot_results_tabular(test_params, [5, 6, 7, 8], limits = [(0, 25), (150, 200), (1100, 1200), (195, 315)], 
                                    path = folder + folder_plots + filename + "--plot", save = True)




################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "tabular_compas/"
path_name_plots = "plots/" 

filename_results = "tabular_compas"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_tabular_compas(path_results, path_name_plots, filename_results)

