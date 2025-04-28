import sys
import importlib
sys.path.insert(0, '../')

import mortalityForecast as mf

import multiprocessing

param_tuple = ({
    'country' : mf.data_loader.SWEDEN,
    'first_year_train' : 1952,
    'last_year_train' : 2011,
    'first_year_test' : 2012,
    'last_year_test' : 2021,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'model' : 'radial_basis',
    'num_basis' : 20,
    'tau' : 10,
    'latent_dim' : 4
    # 'model' : 'linear',
    # 'nn_layers' : 1,
    # 'latent_dim' : 5
},)


def train(param):

    exposure_all, deaths_all = mf.load_data(param, train = True, test = True)
    exposure_train, deaths_train = mf.load_data(param, train = True, test = False)
    exposure_test, deaths_test = mf.load_data(param, train = False, test = True)

    model = mf.Mortality(param)

    if param['model'] == 'linear':
        model_string = "%s_%s_%d_%d_%s_%d_%d_%d" %(param['country'].name, 'linear', param['first_year_train'], param['last_year_train'], param['sex'], param['max_age'], param['nn_layers'], param['latent_dim'])
        file_name = "../trainedModelsLinear/test_" + model_string
    elif param['model'] == 'radial_basis':
        model_string = "%s_%s_%d_%d_%s_%d_%d_%d" %(param['country'].name, 'radial_basis', param['first_year_train'], param['last_year_train'], param['sex'], param['max_age'], param['num_basis'], param['latent_dim'])
        file_name = "../trainedModelsRadial/test_" + model_string
    
    model = model.fit(exposure = exposure_train, 
                      deaths = deaths_train, 
                      num_steps = 5000, 
                      log_freq = 100, 
                      lr = 0.01,
                      checkpoint_freq = 100,
                      lr_decay = 1.0,
                      save_file = file_name)


if __name__ == '__main__':
    processes = []
    for param in param_tuple:
        process = multiprocessing.Process(target=train, 
                                        args=(param,))
        processes.append(process)
        process.start()

    for proc in processes:
        proc.join()