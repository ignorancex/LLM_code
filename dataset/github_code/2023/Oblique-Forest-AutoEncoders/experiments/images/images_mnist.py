################################################################################################### 
#                               Load modules
###################################################################################################
import sys
module_path = "./misc/"
sys.path.insert(0, module_path)

from datasets import load_dataset
from models import run_image_model, get_stacked_image_results
from utils import plot_results_images, makedir, to_json



################################################################################################### 
#                               Config
###################################################################################################
opts_mnist = {
    'name': 'mnist',
    'random_state': None,
    'resize_val': None,
    'N_train': 100,
    'N_test': 10,
    'verbose': False,
}


opts_model = {
    'n_estimators': 300,
    'max_samples': 1.0,
    'max_features': 0.75,
    'bootstrap': True,
    'bootstrap_features': False,
    'oob_score': False,
    'warm_start': False,
    'n_jobs': -1,
    'bounds_constraints': [0, 255],
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
def run_images_mnist(folder, folder_plots, filename):
    dataset = load_dataset(opts_mnist)
    opts = [opts_model, opts_estimator, opts_mnist]
    test_params, image_params = run_image_model(dataset, opts)
    stacked_array = get_stacked_image_results(test_params, image_params)

    to_json(folder, filename + "--settings", opts)
    plot_results_images(stacked_array, image_params[2], path = folder + folder_plots + filename + "--plot", save = True)




################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "images_mnist/"
path_name_plots = "plots/" 

filename_results = "images_mnist"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_images_mnist(path_results, path_name_plots, filename_results)

