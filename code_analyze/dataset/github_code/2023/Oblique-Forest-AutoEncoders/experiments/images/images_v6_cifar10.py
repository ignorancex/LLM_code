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
opts_cifar10 = {
    'name': 'cifar10',
    'random_state': None,
    'train_path': None,
    'test_path': None,
    'train_filename': None, 
    'test_filename': None,
    'resize_val': None,
    'N_train': 800,
    'N_test': 10,
    'verbose': True,
}


opts_model = {
    'n_estimators': 1000,
    'max_samples': 0.5,
    'max_features': 0.25,
    'bootstrap': True,
    'bootstrap_features': False,
    'oob_score': False,
    'warm_start': False,
    'n_jobs': -1,
    'bounds_constraints': [0, 255],
    'oblique_method': 'RandCART',
    'verbose_model': False,
}


opts_randcart = {
    'max_depth': 3,
    'min_samples': 2,
    'compare_with_cart': False,
}

if opts_model["oblique_method"] == "RandCART":
    opts_estimator = opts_randcart
else:
    raise ValueError("The oblique tree must be of type RandCART !")



################################################################################################### 
#                               Function
###################################################################################################
def run_images_v6_cifar10(folder, folder_plots, filename):

    dataset = load_dataset(opts_cifar10)
    opts = [opts_model, opts_estimator, opts_cifar10]
    test_params, image_params = run_image_model(dataset, opts)
    stacked_array = get_stacked_image_results(test_params, image_params)

    to_json(folder, filename + "--settings", opts)
    plot_results_images(stacked_array, image_params[2], path = folder + folder_plots + filename + "--plot", save = True)
    



################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "images_v6_cifar10/"
path_name_plots = "plots/" 

filename_results = "images_v6_cifar10"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_images_v6_cifar10(path_results, path_name_plots, filename_results)

