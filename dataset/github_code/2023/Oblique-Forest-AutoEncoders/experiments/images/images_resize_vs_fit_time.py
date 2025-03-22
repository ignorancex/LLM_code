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
    'N_train': 50,
    'N_test': 10,
    'verbose': False,
}


opts_model = {
    'n_estimators': 30,
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
    'resize_values': [28, 32, 36, 42, 46, 50, 54, 60, 64],
}


if opts_model["oblique_method"] == "HHCART":
    opts_estimator = opts_hhcart
else:
    raise ValueError("The oblique tree must be of type HHCART !")



################################################################################################### 
#                               Function
###################################################################################################
def run_images_resize_vs_fit_time(folder, folder_plots, filename):    

    list_resize = opts_simulations["resize_values"]
    list_time = []

    to_print_in_json = [opts_model, opts_estimator]

    iterator = tqdm(list_resize, total = len(list_resize), 
        desc = '[resize_val = %d] ' % (0), 
        bar_format = "{desc} : {percentage:.2f}% | {bar} | [{n_fmt} / {total_fmt}]", position = 0, mininterval = 10) 



    for resize_val in iterator:

        opts_oxford_flowers['resize_val'] = resize_val

        dataset = load_dataset(opts_oxford_flowers)
        opts = [opts_model, opts_estimator, opts_oxford_flowers]

        total_fit_time = train_image_model_with_time(dataset, opts)
        list_time.append(total_fit_time)

        iterator.set_description('[resize_val = %d]' % (resize_val), refresh = False)
        to_print_in_json.append({'resize_val': opts_oxford_flowers['resize_val']})
        


    to_json(folder, filename + "--settings", to_print_in_json)
    basic_plot_results(list_resize, list_time, ["resize_values", "fit_time"], 
                            path = folder + folder_plots + filename + "--plot", save = True)



################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "images_resize_vs_fit_time/"
path_name_plots = "plots/" 

filename_results = "images_resize_vs_fit_time"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_images_resize_vs_fit_time(path_results, path_name_plots, filename_results)

