################################################################################################### 
#                               Load modules
###################################################################################################
import sys
module_path = "./misc/"
sys.path.insert(0, module_path)

from tqdm import tqdm

from datasets import load_dataset
from models import train_image_model_with_time
from utils import basic_plot_results, makedir, to_json



################################################################################################### 
#                               Config
###################################################################################################
opts_oxford_flowers = {
    'name': 'oxford_flowers',
    'random_state': None,
    'train_path': 'D:/Oxford_Flowers/',
    'test_path': None,
    'train_filename': None, 
    'test_filename': None,
    'resize_val': 28,
    'test_size': None,
    'N_train': 50,
    'N_test': 10,
    'verbose': True,
}


opts_model = {
    'max_samples': 0.25,
    'max_features': 1.0,
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
    'max_depth': 10,
    'min_samples': 2,
}


opts_simulations = {
    'estimators': list(range(1, 100, 5))[1:],
}


if opts_model["oblique_method"] == "HHCART":
    opts_estimator = opts_hhcart
else:
    raise ValueError("The oblique tree must be of type HHCART !")



################################################################################################### 
#                               Function
###################################################################################################
def run_images_n_estimators_vs_fit_time(folder, folder_plots, filename):

    dataset = load_dataset(opts_oxford_flowers)
    list_time = []

    to_print_in_json = [opts_model, opts_estimator, opts_oxford_flowers]

    iterator = tqdm(opts_simulations["estimators"], total = len(opts_simulations["estimators"]), 
        desc = '[n_estim = %d] ' % (0), 
        bar_format = "{desc} : {percentage:.2f}% | {bar} | [{n_fmt} / {total_fmt}]", position = 0, mininterval = 10) 


    for estim in iterator:

        opts_model["n_estimators"] = estim
        opts = [opts_model, opts_estimator, opts_oxford_flowers]

        to_print_in_json.append({'n_estimators': opts_model["n_estimators"]})

        total_fit_time = train_image_model_with_time(dataset, opts)
        list_time.append(total_fit_time)

        iterator.set_description('[n_estim = %d]' % (estim), refresh = False)
        

    
    to_json(folder, filename + "--settings", to_print_in_json)
    basic_plot_results(opts_simulations["estimators"], list_time, ["n_estimators", "fit_time"], 
                        path = folder + folder_plots + filename + "--plot", save = True)


    


################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "images_n_estimators_vs_fit_time/"
path_name_plots = "plots/" 

filename_results = "images_n_estimators_vs_fit_time"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_images_n_estimators_vs_fit_time(path_results, path_name_plots, filename_results)




