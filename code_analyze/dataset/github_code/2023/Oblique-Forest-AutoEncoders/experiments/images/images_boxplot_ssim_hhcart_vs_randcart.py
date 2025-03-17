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
from utils import make_boxplot_estimators, makedir, to_json, mean_ssim




################################################################################################### 
#                               Config
###################################################################################################
opts_chd2r = {
    'name': 'chd2r',
    'random_state': None,
    'train_path': 'D:/CHD2R/',
    'test_path': None,
    'train_filename': None, 
    'test_filename': None,
    'resize_val': 28,
    'test_size': None,
    'N_train': 50,
    'N_test': 10,
    'verbose': False,
}


opts_model = {
    'max_samples': 0.25,
    'max_features': 0.5,
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
    'hhcart_method': 'proj',
    'hhcart_tau': 1e-4,
    'max_depth': 30,
    'min_samples': 2,
}

opts_randcart = {
    'max_depth': 30,
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
def run_images_boxplot_ssim_hhcart_vs_randcart(folder, folder_plots, filename):

    dataset = load_dataset(opts_chd2r)
    results = pd.DataFrame(columns = ['ssim', 'method', 'n_estimators'])

    to_print_in_json = [opts_model, opts_chd2r]
    to_print_in_json.append({'n_runs': opts_simulations['n_runs']})


    for i in range(opts_simulations["n_runs"]):

        iterator = tqdm(opts_simulations["estimators"], total = len(opts_simulations["estimators"]), 
            desc = '[n_estim = %d] ' % (0), 
            bar_format = "{desc} : {percentage:.2f}% | {bar} | [{n_fmt} / {total_fmt}]", position = 0, mininterval = 10)


        for estim in iterator:
            for oblique_method in ["HHCART", "RandCART"]:

                opts_model["n_estimators"] = estim
                opts_model["oblique_method"] = oblique_method
                
                opts_estimator = opts_hhcart if oblique_method == "HHCART" else opts_randcart                

                current_opts = [opts_model, opts_estimator, opts_chd2r]

                if i == opts_simulations["n_runs"] - 1:
                    to_print_in_json.append({'n_estimators': opts_model["n_estimators"]})
                    to_print_in_json.append(opts_estimator)

                current_test_params, current_image_params = run_image_model(dataset, current_opts)
                results_reshaped, X_test_reshaped = get_image_results(current_test_params, current_image_params)

                current_ssim = mean_ssim(X_test_reshaped, results_reshaped)

                results.loc[len(results)] = [current_ssim, oblique_method, estim]

            iterator.set_description('[n_estim = %d]' % (estim), refresh = False)


    to_json(folder, filename + "--settings", to_print_in_json)
    make_boxplot_estimators(results, ["n_estimators", "ssim", "method"], 
                            path = folder + folder_plots + filename + "--plot", save = True)


    


################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "images_boxplot_ssim_hhcart_vs_randcart/"
path_name_plots = "plots/" 

filename_results = "images_boxplot_ssim_hhcart_vs_randcart"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_images_boxplot_ssim_hhcart_vs_randcart(path_results, path_name_plots, filename_results)

