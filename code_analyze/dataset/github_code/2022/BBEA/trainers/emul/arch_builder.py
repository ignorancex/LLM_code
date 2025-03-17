from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from xoa.commons.logger import *

import lookup.nas101bench.api as api


# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3] # # all available operations
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix
TRAIN_EPOCHS = [4, 12, 36, 108]
MAX_EDGES = 9

MIN_VERTICES = 3
MAX_VERTICES = 7

# Immutable variables
NUM_VERTICES = 7 
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix - max: 21
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed

NAS_BENCH_DATA = None # singleton variable to avoid reloading dataset

class NAS101Builder():
    
    def __init__(self, data_path, config_type):
        
        if not os.path.isfile(data_path):
            error("Invalid NAS 101 dataset path: {}".format(data_path))
            raise ValueError("Dataset path not found: {}".format(data_path))
        
        if not 'NAS-Bench-101' in config_type:
            raise ValueError("Invalid configuration type: {}".format(config_type))

        self.config_type = config_type
        self.nas_bench = self.load_dataset(data_path)
        super(NAS101Builder, self).__init__()        
    
    def get_config_type(self):
        return self.config_type

    def verify(self, cand):
        if 'type2' in self.config_type:
            c = self.build_type2(cand)
        elif 'type1' in self.config_type:
            c = self.build_type1(cand)                    
        else:
            c = self.build(cand)
        
        if self.is_valid(c):
            return True
        else:
            return False

    def load_dataset(self, tfrecord):
        global NAS_BENCH_DATA
        try:
            if NAS_BENCH_DATA == None:
                NAS_BENCH_DATA = api.NASBench(tfrecord)
            return NAS_BENCH_DATA
        except Exception as ex:
            error("Dataset loading fail: {}".format(tfrecord))

    def bin_array(self, num, m):
        """Convert a positive integer num into an m-bit bit vector"""
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8).tolist()

    def is_valid(self, cell):
        return self.nas_bench.is_valid(cell)

    def query(self, cell, epochs):
        return self.nas_bench.query(cell, epochs=epochs)

    def get_metrics_from_spec(self, cell):
        return self.nas_bench.get_metrics_from_spec(cell)

    def build(self, h):
        if '-type2' in self.config_type:
            cell = self.build_type2(h)
        elif '-type1' in self.config_type:
            cell = self.build_type1(h)                
        elif self.config_type == 'NAS-Bench-101':
            cell = self.build_type0(h)
        else:
            raise TypeError("Not supported configuration space type: {}".format(self.config_type))
        return cell

    def build_type0(self, h):
        ops = [ h['ops1'], h['ops2'], h['ops3'], h['ops4'], h['ops5'] ]
        links = [int(h['link0']), int(h['link1']), int(h['link2']), 
                int(h['link3']), int(h['link4']), int(h['link5']), 0]
        
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
        matrix = np.array(matrix)
        #debug("{}".format(links))
        #debug("Adj. matrix:")
        #debug(matrix)

        return api.ModelSpec(matrix=matrix,   # output layer
                            # Operations at the vertices of the module, matches order of matrix.
                            ops=ops)

    def build_type1(self, h):        
        ops = [h['ops1'], h['ops2'], h['ops3'], h['ops4'], h['ops5']]
        n_iv = len(ops) # number of inner vertices (input, output excluded)
        ops.insert(0, INPUT)
        ops.append(OUTPUT)

        n_v = n_iv + 2
        selected = []
        max_edges = 9
        for i in range(max_edges):
            k = "edge_{}".format(i)
            v = h[k]
            selected.append(v)
        selected = list(set(selected)) # remove duplicated

        matrix = []
        slot_index = 0
        for i in range(n_v):
            vec = np.array([ 0 for j in range(n_v) ])
            for k in range(i+1, n_v):
                if slot_index in selected:        
                    vec[k] = 1
                slot_index = slot_index + 1

            matrix.append(vec)
        
        matrix = np.array(matrix)
        #debug("{}".format(links))
        #debug("Adj. matrix:")
        #debug(matrix)

        return api.ModelSpec(matrix=matrix,   # output layer
                            # Operations at the vertices of the module, matches order of matrix.
                            ops=ops)

    def build_type2(self, h):    
        ops = [h['ops1'], h['ops2'], h['ops3'], h['ops4'], h['ops5']]
        n_iv = len(ops) # number of inner vertices (input, output excluded)
        ops.insert(0, INPUT)
        ops.append(OUTPUT)

        n_v = n_iv + 2
        edge_spots = int(n_v * (n_v - 1) / 2)        
        num_edges = h['num_edges']
        edge_values = []
        for i in range(edge_spots):
            k = 'edge_{}'.format(i)
            v = h[k]
            edge_values.append(v)
        edge_values = np.array(edge_values)
        
        selected = [] # selected edges by higher order
        if num_edges > 0:
            selected = edge_values.argsort()[-num_edges:][::-1]
        #debug("Selected edge index: {}".format(selected))       

        matrix = []
        slot_index = 0
        for i in range(n_v):
            vec = np.array([ 0 for j in range(n_v) ])
            for k in range(i+1, n_v):
                if slot_index in selected:        
                    vec[k] = 1
                slot_index = slot_index + 1

            matrix.append(vec)
        
        matrix = np.array(matrix)
        #debug("{}".format(links))
        #debug("Adj. matrix:")
        #debug(matrix)

        return api.ModelSpec(matrix=matrix,   # output layer
                            # Operations at the vertices of the module, matches order of matrix.
                            ops=ops)
