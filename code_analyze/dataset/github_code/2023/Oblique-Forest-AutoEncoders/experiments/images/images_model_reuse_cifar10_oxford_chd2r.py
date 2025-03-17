################################################################################################### 
#                               Load modules
###################################################################################################
import sys
module_path = "./misc/"
sys.path.insert(0, module_path)

from datasets import load_dataset
from models import train_image_model, decode_image_model, get_stacked_image_results
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
    'N_train': 1000,
    'N_test': 10,
    'verbose': True,
}



opts_chd2r = {
    'name': 'chd2r',
    'random_state': None,
    'train_path': 'D:/CHD2R/',
    'test_path': None,
    'train_filename': None, 
    'test_filename': None,
    'resize_val': 32,
    'N_train': 100,
    'N_test': 10,
    'verbose': True,
}


opts_oxford_flowers = {
    'name': 'oxford_flowers',
    'random_state': None,
    'train_path': 'D:/Oxford_Flowers/',
    'test_path': None,
    'train_filename': None, 
    'test_filename': None,
    'resize_val': 32,
    'N_train': 100,
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
    'oblique_method': 'HHCART',
    'verbose_model': False,
}


opts_hhcart = {
    'hhcart_method': 'proj',
    'hhcart_tau': 1e-4,
    'max_depth': 3,
    'min_samples': 2,
}


opts_randcart = {
    'max_depth': 3,
    'min_samples': 2,
    'compare_with_cart': False,
}

if opts_model["oblique_method"] == "RandCART":
    opts_estimator = opts_randcart
elif opts_model["oblique_method"] == "HHCART":
    opts_estimator = opts_hhcart
else:
    raise ValueError("The oblique tree must be of type HHCART or RandCART !")



################################################################################################### 
#                               Function
###################################################################################################
def run_images_model_reuse_cifar10_oxford_chd2r(folder, folder_plots, filename):
    to_print_in_json = [opts_model, opts_estimator, opts_cifar10, opts_chd2r, opts_oxford_flowers]


    dataset_cifar10 = load_dataset(opts_cifar10)
    full_opts_1 = [opts_model, opts_estimator, opts_cifar10]

    dataset_chd2r = load_dataset(opts_chd2r)
    full_opts_2 = [opts_model, opts_estimator, opts_chd2r]

    dataset_oxford = load_dataset(opts_oxford_flowers)
    full_opts_3 = [opts_model, opts_estimator, opts_oxford_flowers]


    channel_models = train_image_model(dataset_cifar10, full_opts_1)

    test_params_chd2r, image_params_chd2r = decode_image_model(channel_models, dataset_chd2r, full_opts_2)
    test_params_oxford, image_params_oxford = decode_image_model(channel_models, dataset_oxford, full_opts_3)

    stacked_array_oxford = get_stacked_image_results(test_params_oxford, image_params_oxford)
    stacked_array_chd2r = get_stacked_image_results(test_params_chd2r, image_params_chd2r)

    plot_results_images(stacked_array_oxford, image_params_oxford[2], path = folder + folder_plots + filename + "--[oxford]" + "--plot", save = True)
    plot_results_images(stacked_array_chd2r, image_params_chd2r[2], path = folder + folder_plots + filename + "--[chd2r]" + "--plot", save = True)
    
    to_json(folder, filename + "--settings", to_print_in_json)



################################################################################################### 
#                               Main
###################################################################################################
general_path = "./results/"
path_results = general_path + "images_model_reuse_cifar10_oxford_chd2r/"
path_name_plots = "plots/" 

filename_results = "images_model_reuse_cifar10_oxford_chd2r"

makedir(path_results, True)
makedir(path_results + path_name_plots, False)

run_images_model_reuse_cifar10_oxford_chd2r(path_results, path_name_plots, filename_results)

