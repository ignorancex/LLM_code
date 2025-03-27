################################################################################################### 
#                               Load modules
###################################################################################################
import sys
module_path = "./misc/"
sys.path.insert(0, module_path)

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from models import run_image_model, get_image_results
from utils import make_barplot_time, makedir, to_json, mean_ssim



################################################################################################### 
#                               Config
###################################################################################################
opts_mnist = {
    'name': 'mnist',
    'random_state': None,
    'resize_val': None,
    'N_train': 50,
    'N_test': 10,
    'verbose': True,
}


opts_model = {
    'n_estimators': 50,
    'max_samples': 0.75,
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
    'hhcart_tau': 1e-4,
    'max_depth': 3,
    'min_samples': 2,
}


opts_simulations = {
    'n_runs': 10,
    'all_hhcart_methods': ["eig", "svd", "fast_ica", "factor", "proj"],
}


if opts_model["oblique_method"] == "HHCART":
    opts_estimator = opts_hhcart
else:
    raise ValueError("The oblique tree must be of type HHCART !")



################################################################################################### 
#                               Function
###################################################################################################
def run_images_barplot_ssim_eig_methods(folder, folder_plots, filename):

    dataset = load_dataset(opts_mnist)
    results = pd.DataFrame(columns = ['ssim', 'method'])

    to_print_in_json = [opts_model, opts_estimator, opts_mnist]
    to_print_in_json.append({'n_runs': opts_simulations['n_runs']})


    for i in range(opts_simulations["n_runs"]):

        iterator = tqdm(opts_simulations["all_hhcart_methods"], total = len(opts_simulations["all_hhcart_methods"]), 
            desc = '[hhcart_method = %s] ' % ("eig"), 
            bar_format = "{desc} : {percentage:.2f}% | {bar} | [{n_fmt} / {total_fmt}]", position = 0, mininterval = 10)


        for hhcart_method in iterator:

            opts_estimator["hhcart_method"] = hhcart_method         
            current_opts = [opts_model, opts_estimator, opts_mnist]

            if i == opts_simulations["n_runs"] - 1:
                to_print_in_json.append({'hhcart_method': opts_estimator["hhcart_method"]})


            current_test_params, current_image_params = run_image_model(dataset, current_opts)
            results_reshaped, X_test_reshaped = get_image_results(current_test_params, current_image_params)

            current_ssim = mean_ssim(X_test_reshaped, results_reshaped)
            results.loc[len(results)] = [current_ssim, hhcart_method]

            iterator.set_description('[hhcart_method = %s]' % (hhcart_method), refresh = False)


    to_json(folder, filename + "--settings", to_print_in_json)
    make_barplot_time(results, ["ssim", "method"], path = folder + folder_plots + filename + "--plot", save = True)





################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "images_barplot_ssim_eig_methods/"
path_name_plots = "plots/" 

filename_results = "images_barplot_ssim_eig_methods"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_images_barplot_ssim_eig_methods(path_results, path_name_plots, filename_results)

