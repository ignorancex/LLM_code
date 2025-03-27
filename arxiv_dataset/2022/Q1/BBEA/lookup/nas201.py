
import numpy as np

from lookup.nas201bench.api import NAS201Bench

try:
    import ConfigSpace
except ImportError as ie:
    pass

class NAS201Benchmark(object):
    
    def __init__(self, name, multi_fidelity=True):

        self.name = name
        self.multi_fidelity = multi_fidelity
        self.dataset = NAS201Bench(name)
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.budgets = []
        self.max_epochs = 200

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.budgets = []

    def objective_function(self, config, budget=200):
        
        if self.multi_fidelity is False:
            assert budget == 200

        if type(config) != int:
            m_i = self.dataset.get_arch_index(config)
        else:
            m_i = config

        self.X.append(m_i)
        test_loss, val_loss, train_time, info = self.dataset.train(m_i, n_epochs=budget)

        self.y_test.append(test_loss)
        self.y_valid.append(val_loss)
        self.costs.append(train_time)
        self.budgets.append(budget)

        return test_loss, val_loss, train_time


    def get_results(self, result_type='test', ignore_invalid_configs=True):
        errors = []
        exec_times = []
        opt_times = []
        model_indices = []

        inc_valid = np.inf
        inc_test = np.inf
        print("{} configurations had been retrieved.".format(len(self.X)))
        
        for i in range(len(self.X)):

            if ignore_invalid_configs and self.costs[i] == 0:
                continue

            val_v = self.y_valid[i]
            test_v = self.y_test[i]
            if inc_valid > val_v:
                inc_valid = val_v
                inc_test = test_v

            if i in self.budgets and self.budgets[i] != self.max_epochs:
                test_v = val_v # if a configuration terminated early, no way to know final test performance

            rt = self.costs[i]
            exec_times.append(float(rt))
            if result_type == 'test':
                errors.append(float(test_v))
            elif result_type == 'validation':
                errors.append(float(val_v))
            else:
                raise ValueError("Invalid report type: {}".format(result_type))
            opt_times.append(0) # FIXME:ignore surrogate modeling time
            if type(self.X[i]) == int: 
                model_indices.append(self.X[i])
            else:
                model_indices.append(i)

        res = dict()
        res['opt_time'] = opt_times 
        res['exec_time'] = exec_times
        res['error'] = errors  
        res['model_idx'] = model_indices

        return res

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()
        search_range = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        for i in range(1, 4):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_range))
        #print("Config space: {}".format(cs))
        return cs    
