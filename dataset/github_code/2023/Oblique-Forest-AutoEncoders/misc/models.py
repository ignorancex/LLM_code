################################################################################################### 
#                               Load modules
###################################################################################################
import sys, time

module_path = "./modules/"
sys.path.insert(0, module_path)

import numpy as np

from segmentor import MSE, MeanSegmentor
from HHCART import HouseHolder_CART
from RandCART import Rand_CART
from UnsupervisedBaggingRegressor import BaggingRegressorEmbedding





################################################################################################### 
#                               Models
###################################################################################################
def define_model(opts):
    opts_model, opts_estimator, opts_dataset = opts

    oblique_method = opts_model["oblique_method"]
    assert isinstance(oblique_method, str)

    n_estimators = opts_model["n_estimators"]
    assert isinstance(n_estimators, (int, type(None)))
    
    max_samples = opts_model["max_samples"]
    assert isinstance(max_samples, (float, type(None)))

    max_features = opts_model["max_features"]
    assert isinstance(max_features, (float, type(None)))

    bootstrap = opts_model["bootstrap"]
    assert isinstance(bootstrap, (bool, type(None)))

    bootstrap_features = opts_model["bootstrap_features"]
    assert isinstance(bootstrap_features, (bool, type(None)))

    oob_score = opts_model["oob_score"]
    assert isinstance(oob_score, (bool, type(None)))

    warm_start = opts_model["warm_start"]
    assert isinstance(warm_start, (bool, type(None)))

    n_jobs = opts_model["n_jobs"]
    assert isinstance(n_jobs, (int, type(None)))

    bound_constraints = opts_model["bounds_constraints"]
    assert isinstance(bound_constraints, (list, type(None)))




    max_depth = opts_estimator["max_depth"]
    assert isinstance(max_depth, (int, type(None)))

    min_samples = opts_estimator["min_samples"]
    assert isinstance(min_samples, (int, type(None)))



    if opts_model["oblique_method"] == "RandCART":
        compare_with_cart = opts_estimator["compare_with_cart"]
        assert isinstance(compare_with_cart, (bool, type(None)))


    if opts_model["oblique_method"] == "HHCART":
        hhcart_method = opts_estimator["hhcart_method"]
        assert isinstance(hhcart_method, (str, type(None)))
    
        hhcart_tau = opts_estimator["hhcart_tau"]
        assert isinstance(hhcart_tau, (float, type(None), int))
    



    random_state = opts_dataset["random_state"]
    assert isinstance(random_state, (int, type(None)))

    verbose = opts_model["verbose_model"]
    assert isinstance(verbose, (bool, type(None), int))



    criterion = MSE()
    segmentor = MeanSegmentor()
    
    if oblique_method == 'HHCART':
        estimator = HouseHolder_CART(
                            impurity = criterion,
                            segmentor = segmentor,
                            method = hhcart_method,
                            tau = hhcart_tau,
                            max_depth = max_depth,
                            min_samples = min_samples,
        )

    elif oblique_method == 'RandCART':
        estimator = Rand_CART(
                            impurity = criterion,
                            segmentor = segmentor,
                            max_depth = max_depth,
                            min_samples = min_samples,
                            compare_with_cart = compare_with_cart
        )
    
    else:
        raise NotImplementedError("Other estimators are not available.")


    model = BaggingRegressorEmbedding(
                            base_estimator = estimator,
                            n_estimators = n_estimators,
                            max_samples = max_samples,
                            max_features = max_features,
                            bootstrap = bootstrap,
                            bootstrap_features = bootstrap_features,
                            oob_score = oob_score,
                            warm_start = warm_start,
                            n_jobs = n_jobs,
                            random_state = random_state,
                            verbose = verbose
        )

    return model





def run_tabular_model(dataset, opts, n_train = None, n_test = None):
    X_train, X_test = dataset["X_train"], dataset["X_test"]

    if n_train is not None:
        X_train = X_train[:n_train, :]

    if n_test is not None:
        X_test = X_test[:n_test, :]

    model = define_model(opts)
    model.fit(X_train)
    X_decoded = model.decode(X_test, opts[0]["bounds_constraints"])
    
    test_params = [X_decoded, X_test]
    return test_params
    



def run_tabular_model_with_time(dataset, opts, n_test = None):
    X_train, X_test = dataset["X_train"], dataset["X_test"]
    if n_test is not None:
        X_test = X_test[:n_test, :]
        
    model = define_model(opts)

    start_fit_time = time.time()
    model.fit(X_train)
    total_fit_time = time.time() - start_fit_time
    
    start_decode_time = time.time()
    X_decoded = model.decode(X_test, opts[0]["bounds_constraints"])
    total_decode_time = time.time() - start_decode_time

    test_params = [X_decoded, X_test, total_fit_time, total_decode_time]
    return test_params



def run_image_model(dataset, opts):
    X_train, X_test = dataset["X_train"], dataset["X_test"]

    width, height, channels = dataset["width"], dataset["height"], dataset["channels"]
    N_test = X_test.shape[0]

    if channels == 1:
        results = np.zeros((N_test, width * height))
    elif channels == 3:
        results = np.zeros((N_test, width * height, channels))


    for c in range(channels):
        
        if channels == 1:
            X_train_on_channels = X_train
            X_test_on_channels = X_test
        elif channels == 3:
            X_train_on_channels = X_train[:, :, c]
            X_test_on_channels = X_test[:, :, c]

        model = define_model(opts)
        model.fit(X_train_on_channels)        
        X_decoded = model.decode(X_test_on_channels, opts[0]["bounds_constraints"])
        
        if channels == 1:
            results = X_decoded
        elif channels == 3:
            results[:, :, c] = X_decoded

    image_params = [width, height, channels]
    test_params = [results, X_test]
    return test_params, image_params
        



def get_stacked_image_results(test_params, image_params):
    results, X_test = test_params
    test_images = np.copy(X_test)
    
    N_test = results.shape[0]
    width, height, channels = image_params
    
    if channels == 1:
        test_images = X_test.reshape(N_test, width, height)
        results = results.reshape(N_test, width, height)
    elif channels == 3:
        test_images = test_images.reshape(N_test, width, height, channels)
        results = results.reshape(N_test, width, height, channels)

    stacked_array_1 = np.hstack(([ test_images[i, :] for i in range(test_images.shape[0]) ]))
    stacked_array_2 = np.hstack(([ results[i, :] for i in range(results.shape[0]) ]))
    stacked_array = np.vstack((stacked_array_1, stacked_array_2))

    stacked_array = stacked_array.astype(np.uint8)
    return stacked_array



def get_image_results(test_params, image_params):
    results, X_test = test_params
    test_images = np.copy(X_test)
    
    N_test = results.shape[0]
    width, height, channels = image_params
    
    if channels == 1:
        test_images = X_test.reshape(N_test, width, height)
        results = results.reshape(N_test, width, height)
    elif channels == 3:
        test_images = test_images.reshape(N_test, width, height, channels)
        results = results.reshape(N_test, width, height, channels)

    return [results, test_images]




def train_image_model(dataset, opts):
    X_train = dataset["X_train"]

    channels = dataset["channels"]
    channel_models = []
    
    for c in range(channels):
        print('Channel: ', c, '\n')
        
        if channels == 1:
            X_train_on_channels = X_train
        elif channels == 3:
            X_train_on_channels = X_train[:, :, c]

        model = define_model(opts)
        model.fit(X_train_on_channels)        
        channel_models.append(model)
        
    return channel_models



def decode_image_model(channel_models, dataset, opts):
    X_test = dataset["X_test"]

    width, height, channels = dataset["width"], dataset["height"], dataset["channels"]
    N_test = X_test.shape[0]

    if channels == 1:
        results = np.zeros((N_test, width * height))
    elif channels == 3:
        results = np.zeros((N_test, width * height, channels))


    for c in range(channels):
        
        if channels == 1:
            X_test_on_channels = X_test
        elif channels == 3:
            X_test_on_channels = X_test[:, :, c]

        X_decoded = channel_models[c].decode(X_test_on_channels, opts[0]["bounds_constraints"])
        
        if channels == 1:
            results = X_decoded
        elif channels == 3:
            results[:, :, c] = X_decoded

    image_params = [width, height, channels]
    test_params = [results, X_test]
    return test_params, image_params
        




    
def train_image_model_with_time(dataset, opts):
    X_train = dataset["X_train"]
    channels = dataset["channels"]
    
    for c in range(channels):
        
        if channels == 1:
            X_train_on_channels = X_train
        elif channels == 3:
            X_train_on_channels = X_train[:, :, c]

        model = define_model(opts)

        start_fit_time = time.time()
        model.fit(X_train_on_channels)
        total_fit_time = time.time() - start_fit_time
            
    return total_fit_time