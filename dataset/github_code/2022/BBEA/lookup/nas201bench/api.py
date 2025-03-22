import os

from .nas_201_api  import NASBench201API as API
from .models       import CellStructure, get_search_spaces


def config2structure_func(max_nodes):
    def config2structure(config):
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = config[node_str]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)
    return config2structure


class NAS201Bench(object):

    def __init__(self, dataset,
                arch_nas_dataset='./lookup/NAS-Bench-201-v1_1-096897.pth'):
        self.max_nodes = 4
        self.max_epoches = 200

        if dataset == 'cifar10':
            dataset = 'cifar10-valid'
        self.dataset = dataset
        self.name = "nas-bench-201"
        if not os.path.isfile(arch_nas_dataset):
            raise ValueError("No lookup data available")
        
        self.api = API(arch_nas_dataset)
        self.search_space = get_search_spaces('cell', "nas-bench-201")
        self.convert_func = config2structure_func(self.max_nodes)

    def verify(self, cand):
        return True

    def get_search_space(self):
        return self.search_space

    def convert_structure(self, config):
        return self.convert_func(config)

    def get_arch_index(self, config):
        structure = self.convert_structure(config)
        return self.api.query_index_by_arch(structure)

    def get_eval_info(self, arch_index, n_epochs=None):
        if n_epochs != None and n_epochs > 0:
            i_epochs = n_epochs - 1
        else:
            i_epochs = n_epochs

        info = self.api.get_more_info(arch_index, self.dataset, i_epochs, 
                                     hp='200', is_random=True)
        return info

    def train(self, arch_index, n_epochs=None):
        info = self.get_eval_info(arch_index, n_epochs)
        cur_time = info['train-all-time']
        val_loss = 1.0
        test_loss = 1.0
        #print("#{} ({} epochs): {}".format(arch_index, n_epochs, info))

        if not 'train-accuracy' in info:
            raise ValueError("Invalid eval info #{}: {}".format(arch_index, info))

        if n_epochs == None or n_epochs == self.max_epoches:
            if 'valid-accuracy' in info:
                cur_acc = info['valid-accuracy'] # Note: 0 ~ 100
                val_loss = 1.0 - float(cur_acc / 100)
                cur_time += info['valid-per-time']

            if 'test-accuracy' in info:
                cur_acc = info['test-accuracy'] # Note: 0 ~ 100
                test_loss = 1.0 - float(cur_acc / 100)
                cur_time += info['test-per-time']

        elif 'valtest-accuracy' in info:
            cur_acc = info['valtest-accuracy'] # Note: 0 ~ 100
            test_loss = 1.0 - float(cur_acc / 100)
            val_loss = 1.0 - float(cur_acc / 100)
            cur_time += info['valtest-per-time']                         
        
        return test_loss, val_loss, cur_time, info  

    def query_by_arch(self, arch):
        return self.api.query_by_arch(arch, '200')
    

    