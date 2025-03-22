import os


import numpy as np
from lookup.nas101bench import api
from lookup.nas101bench.lib import graph_util

try:
    import ConfigSpace
except ImportError as ie:
    pass

# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3] # # all available operations
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix
TRAIN_EPOCHS = [4, 12, 36, 108]

# Immutable variables
MAX_EDGES = 9
VERTICES = 7


class NASCifar10(object):

    def __init__(self, name, data_dir, multi_fidelity=True):

        self.name = name
        self.multi_fidelity = multi_fidelity
        if self.multi_fidelity:
            self.dataset = api.NASBench(os.path.join(data_dir, 'nasbench_full.tfrecord'))
        else:
            self.dataset = api.NASBench(os.path.join(data_dir, 'nasbench_only108.tfrecord'))
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.budgets = []

        self.y_star_valid = 0.04944576819737756  # lowest mean validation error
        self.y_star_test = 0.056824247042338016  # lowest mean test error

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.budgets = []

    @staticmethod
    def objective_function(self, config):
        pass

    def record_invalid(self, config, valid, test, costs):
        self.X.append(config)
        self.y_valid.append(valid)
        self.y_test.append(test)
        self.costs.append(costs)

    def record_valid(self, config, data, model_spec):

        self.X.append(config)

        # compute mean test error for the final budget
        #_, metrics = self.dataset.get_metrics_from_spec(model_spec)
        #mean_test_error = 1 - np.mean([metrics[108][i]["final_test_accuracy"] for i in range(3)])
        #self.y_test.append(mean_test_error)
        test_error = 1 - data["test_accuracy"]
        self.y_test.append(test_error)

        # compute validation error for the chosen budget
        valid_error = 1 - data["validation_accuracy"]
        self.y_valid.append(valid_error)

        runtime = data["training_time"]
        self.costs.append(runtime)

    @staticmethod
    def get_configuration_space():
        pass

    def get_results(self, result_type='test', ignore_invalid_configs=True):

        #regret_validation = []
        #regret_test = []
        
        #rt = 0
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

            if i in self.budgets and self.budgets[i] != 108:
                test_v = val_v # if a configuration terminated early, no way to know final test performance

            #regret_validation.append(float(inc_valid - self.y_star_valid))
            #regret_test.append(float(inc_test - self.y_star_test))
            #rt += self.costs[i]
            #runtime.append(float(rt))
            rt = self.costs[i]
            exec_times.append(float(rt))
            if result_type == 'test':
                errors.append(float(test_v))
            elif result_type == 'validation':
                errors.append(float(val_v))
            else:
                raise ValueError("Invalid report type: {}".format(result_type))
            opt_times.append(0) # FIXME:ignore surrogate modeling time
            model_indices.append(i)

        res = dict()
        #res['regret_validation'] = regret_validation
        #res['regret_test'] = regret_test
        #res['runtime'] = runtime
        res['opt_time'] = opt_times 
        res['exec_time'] = exec_times
        res['error'] = errors  
        res['model_idx'] = model_indices

        return res


class NASCifar10A(NASCifar10):

    def __init__(self, data_dir, multi_fidelity=True):
        super(NASCifar10A, self).__init__('NASCifar10A', data_dir, multi_fidelity=multi_fidelity)

    def bin_array(self, num, m):
        """Convert a positive integer num into an m-bit bit vector"""
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8).tolist()

    def build(self, h):

        ops = [h['ops1'], h['ops2'], h['ops3'], h['ops4'], h['ops5']]
        links = [h['link0'], h['link1'], h['link2'], h['link3'], h['link4'], h['link5'], 0]
        
        MAX_LINKS = [63, 31, 15, 7, 3, 1, 0]
        
        n_iv = len(ops) # number of inner vertices (input, output excluded)
        ops.insert(0, INPUT)
        ops.append(OUTPUT)
        n_v = n_iv + 2
        edge_spots = int(n_v * (n_v - 1) / 2)
        #debug("node operation list: ")
        #debug("{}".format(ops))

        # validate links value
        for i in range(len(links)):
            if links[i] > MAX_LINKS[i]:
                raise ValueError("Invalid link value at {}: {} > {}".format(i, links[i], MAX_LINKS[i]))
                
        matrix = []
        for i in range(n_v):
            vec = self.bin_array(links[i], n_v)
            matrix.append(vec)

        return np.array(matrix)

    def objective_function(self, config, budget=108):
        
        if self.multi_fidelity is False:
            assert budget == 108
        
        matrix = self.build(config)

        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.record_invalid(config, 1, 1, 0)
            return 1, 1, 0

        labeling = [config["ops%d" % i] for i in range(1, 6)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
           data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.record_invalid(config, 1, 1, 0)
            return 1, 1, 0

        self.record_valid(config, data, model_spec)
        self.budgets.append(budget)

        return 1 - data["test_accuracy"], 1 - data["validation_accuracy"], data["training_time"]

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("ops1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("ops2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("ops3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("ops4", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("ops5", ops_choices))
        
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("link0", [i for i in range(64)]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("link1", [i for i in range(32)]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("link2", [i for i in range(16)]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("link3", [i for i in range(8)]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("link4", [i for i in range(4)]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("link5", [i for i in range(2)]))        
        
        return cs


class NASCifar10B(NASCifar10):

    def __init__(self, data_dir, multi_fidelity=True):
        super(NASCifar10B, self).__init__('NASCifar10B', data_dir, multi_fidelity=multi_fidelity)


    def objective_function(self, config, budget=108):
        if self.multi_fidelity is False:
            assert budget == 108

        bitlist = [0] * (VERTICES * (VERTICES - 1) // 2)
        for i in range(MAX_EDGES):
            bitlist[config["edge_%d" % i]] = 1
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit

        matrix = np.fromfunction(graph_util.gen_is_edge_fn(out),
                                 (VERTICES, VERTICES),
                                 dtype=np.int8)
        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.record_invalid(config, 1, 1, 0)
            return 1, 1, 0

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.record_invalid(config, 1, 1, 0)
            return 1, 1, 0

        self.record_valid(config, data, model_spec)
        self.budgets.append(budget)

        return 1 - data["test_accuracy"], 1 - data["validation_accuracy"], data["training_time"]

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        cat = [i for i in range((VERTICES * (VERTICES - 1)) // 2)]
        for i in range(MAX_EDGES):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, cat))
        return cs


class NASCifar10C(NASCifar10):

    def __init__(self, data_dir, multi_fidelity=True):
        super(NASCifar10C, self).__init__('NASCifar10C', data_dir, multi_fidelity=multi_fidelity)


    def objective_function(self, config, budget=108):
        if self.multi_fidelity is False:
            assert budget == 108

        edge_prob = []
        for i in range(VERTICES * (VERTICES - 1) // 2):
            edge_prob.append(config["edge_%d" % i])

        idx = np.argsort(edge_prob)[::-1][:config["num_edges"]]
        binay_encoding = np.zeros(len(edge_prob))
        binay_encoding[idx] = 1
        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(VERTICES * (VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = binay_encoding[i]

        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.record_invalid(config, 1, 1, 0)
            return 1, 1, 0

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.record_invalid(config, 1, 1, 0)
            return 1, 1, 0

        self.record_valid(config, data, model_spec)
        self.budgets.append(budget)

        return 1 - data["test_accuracy"], 1 - data["validation_accuracy"], data["training_time"]

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("num_edges", 0, MAX_EDGES))

        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("edge_%d" % i, 0, 1))
        return cs
