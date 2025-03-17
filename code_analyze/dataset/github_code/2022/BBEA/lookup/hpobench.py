import json
import os

import h5py
import numpy as np

try:
    import ConfigSpace
except ImportError as ie:
    raise ModuleNotFoundError("The module is missing to run HPO-benchmark: {}".format(ie))


class FCNetBenchmark(object):

    def __init__(self, path, dataset="./lookup/fcnet_protein_structure_data.hdf5", seed=None):

        cs = self.get_configuration_space()
        self.names = [h.name for h in cs.get_hyperparameters()]

        self.data = h5py.File(os.path.join(path, dataset), "r")

        self.X = []
        self.y = []
        self.c = []
        self.b = [] # budget
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y = []
        self.c = []
        self.b = []

        self.rng = np.random.RandomState(self.seed)

    def get_best_configuration(self):

        """
        Returns the best configuration in the dataset that achieves the lowest test performance.

        :return: Returns tuple with the best configuration, its final validation performance and its test performance
        """

        configs, te, ve = [], [], []
        for k in self.data.keys():
            configs.append(json.loads(k))
            te.append(np.mean(self.data[k]["final_test_error"]))
            ve.append(np.mean(self.data[k]["valid_mse"][:, -1]))

        b = np.argmin(te)

        return configs[b], ve[b], te[b]

    def objective_function(self, config, budget=100, **kwargs):

        assert 0 < budget <= 100  # check whether budget is in the correct bounds

        i = self.rng.randint(4)

        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        test = np.mean(self.data[k]["final_test_error"])
        valid = self.data[k]["valid_mse"][i]
        runtime = self.data[k]["runtime"][i]

        if budget != 100:
            test = valid

        time_per_epoch = runtime / 100  # divide by the maximum number of epochs

        rt = time_per_epoch * budget

        self.X.append(config)
        self.y.append(valid[budget - 1])
        self.c.append(rt)
        self.b.append(budget)

        return test, valid[budget - 1], rt

    def objective_function_learning_curve(self, config, budget=100):

        assert 0 < budget <= 100  # check whether budget is in the correct bounds

        index = self.rng.randint(4)

        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        test = np.mean(self.data[k]["final_test_error"])
        lc = [self.data[k]["valid_mse"][index][i] for i in range(budget)]
        runtime = self.data[k]["runtime"][index]

        if budget != 100:
            test = min(lc)

        time_per_epoch = runtime / 100 # divide by the maximum number of epochs

        rt = [time_per_epoch * (i + 1) for i in range(budget)]

        self.X.append(config)
        self.y.append(lc[-1])
        self.c.append(rt[-1])
        self.b.append(budget)

        return test, lc, rt

    def objective_function_deterministic(self, config, budget=100, index=0, **kwargs):

        assert 0 < budget <= 100  # check whether budget is in the correct bounds

        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        valid = self.data[k]["valid_mse"][index]
        runtime = self.data[k]["runtime"][index]

        time_per_epoch = runtime / 100 # divide by the maximum number of epochs

        rt = time_per_epoch * budget

        self.X.append(config)
        self.y.append(valid[budget - 1])
        self.c.append(rt)
        self.b.append(budget)

        return valid[budget - 1], rt

    def objective_function_test(self, config, **kwargs):

        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        test = np.mean(self.data[k]["final_test_error"])
        runtime = np.mean(self.data[k]["runtime"])

        return test, runtime

    def get_results(self, result_type='test', ignore_invalid_configs=False):
        
        print("{} configurations had been retrieved.".format(len(self.X)))
        inc, y_star_valid, y_star_test = self.get_best_configuration()
        #print("Best config: {}, {}, {}".format(inc, y_star_valid, y_star_test))
        #regret_validation = []
        #regret_test = []
        
        errors = []
        exec_times = []
        opt_times = []
        model_indices = []
        train_epochs = []

        inc_valid = np.inf
        inc_test = np.inf
        
        for i in range(len(self.X)):
            val_v = self.y[i]
            
            rt = self.c[i]
            test_v, _ = self.objective_function_test(self.X[i])
            #print("test error: {}, time: {}".format(test_v, rt))
            budget = self.b[i]
            train_epochs.append(budget)

            if inc_valid > val_v:
                inc_valid = val_v
                inc_test = test_v

            #regret_validation.append(float(inc_valid - y_star_valid))
            #regret_test.append(float(inc_test - y_star_test))

            exec_times.append(float(rt))
            if result_type == 'test':
                if budget == 100: 
                    errors.append(float(test_v)) 
                else:
                    errors.append(float(val_v))
            elif result_type == 'validation':
                errors.append(float(val_v))
            else:
                raise ValueError("Invalid report type: {}".format(result_type))
            opt_times.append(0) # FIXME:ignore surrogate modeling time
            model_indices.append(i)

        res = dict()
        #res['regret_validation'] = regret_validation
        #res['regret_test'] = regret_test
        res['opt_time'] = opt_times 
        res['exec_time'] = exec_times
        res['error'] = errors  
        res['model_idx'] = model_indices
        res['train_epoch'] = train_epochs

        return res

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_units_1", [16, 32, 64, 128, 256, 512]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_units_2", [16, 32, 64, 128, 256, 512]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_1", [0.0, 0.3, 0.6]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_2", [0.0, 0.3, 0.6]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
        cs.add_hyperparameter(
            ConfigSpace.OrdinalHyperparameter("init_lr", [5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("batch_size", [8, 16, 32, 64]))
        return cs


class FCNetSliceLocalizationBenchmark(FCNetBenchmark):

    def __init__(self, data_dir="./"):
        self.name = 'FCNetSliceLocalizationBenchmark'
        super(FCNetSliceLocalizationBenchmark, self).__init__(path=data_dir,
                                                              dataset="fcnet_slice_localization_data.hdf5")


class FCNetProteinStructureBenchmark(FCNetBenchmark):

    def __init__(self, data_dir="./"):
        self.name = 'FCNetProteinStructureBenchmark'
        super(FCNetProteinStructureBenchmark, self).__init__(path=data_dir, dataset="fcnet_protein_structure_data.hdf5")


class FCNetNavalPropulsionBenchmark(FCNetBenchmark):

    def __init__(self, data_dir="./"):
        self.name = 'FCNetNavalPropulsionBenchmark'
        super(FCNetNavalPropulsionBenchmark, self).__init__(path=data_dir, dataset="fcnet_naval_propulsion_data.hdf5")


class FCNetParkinsonsTelemonitoringBenchmark(FCNetBenchmark):

    def __init__(self, data_dir="./"):
        self.name = 'FCNetParkinsonsTelemonitoringBenchmark'
        super(FCNetParkinsonsTelemonitoringBenchmark, self).__init__(path=data_dir,
                                                                     dataset="fcnet_parkinsons_telemonitoring_data.hdf5")
