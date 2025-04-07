import sys
import os
import pickle
import json
import numpy as np
import copy
import math
import torch
import dgl
import random

from dgl import DGLGraph
#from mergedgraph import MergedDGLGraph, unmerge

limit = 30000
allowed_delta = 250
path_saves = 'saves/'

# Relations are integers
RelTensor = torch.LongTensor
# Normalization factors are floats
NormTensor = torch.Tensor

USING_3D = True
class HumanGraph(DGLGraph):
    def __init__(self, data, alt, mode='train', debug=False):
        super(HumanGraph, self).__init__()
        self.labels = None
        self.features = None
        self.num_rels = -1
        self.mode = mode
        self.debug = debug

        self.set_n_initializer(dgl.init.zero_initializer)
        self.set_e_initializer(dgl.init.zero_initializer)

        if alt == '1':
            self.initializeWithAlternative1(data)
        else:
            print(f'Unknown network alternative {alt}')
            sys.exit(-1)

    @staticmethod
    def get_node_types_one_hot():
        return ['superbody', 'body', 'nose', 'left_ear', 'left_eye', 'left_shoulder', 'left_elbow', 'left_wrist',
                'left_hip', 'left_knee', 'left_ankle', 'right_ear', 'right_eye', 'right_shoulder', 'right_elbow',
                'right_wrist', 'right_hip', 'right_knee', 'right_ankle']

    @staticmethod
    def get_cam_types():
        return ['cam1', 'cam2', 'cam3']


    @staticmethod
    def get_body_parts():
        return {'e', 'ey', 'n', 's', 'el', 'w', 'h', 'k', 'a'}
        # e  = ear          r = right
        # s  = shoulder     l = left
        # el = elbow        b = body (global_node)
        # ey = eye
        # w  = wrist
        # h  = hip
        # k  = knee
        # a  = ankle
        # n  = nose

    @staticmethod
    def get_body_rels():
        return {'s_el', 'el_w', 's_h', 'h_k', 'k_a', 'n_e', 'n_s', 'n_ey'}

    @staticmethod
    def get_all_features():
        return HumanGraph.get_node_types_one_hot() + HumanGraph.get_cam_types() + HumanGraph.get_joint_metric_features() #+ HumanGraph.get_body_metric_features()

    @staticmethod
    def get_joint_metric_features():
        if USING_3D:
            return ['x_position', 'y_position', 'z_position', 'i_coordinate', 'j_coordinate', 'score']
        else:
            return ['i_coordinate', 'j_coordinate', 'score']

    @staticmethod
    def get_body_metric_features():
        return ['x', 'y', 'z', 'orientation_sin', 'orientation_cos'] #, 'timestamp']

    @staticmethod
    def get_rels():
        rels = set()
        body_parts = HumanGraph.get_body_parts()
        body_rels = HumanGraph.get_body_rels()
        # Add body relations
        for relations in body_rels:
            split = relations.split('_')
            if split[0] == 'n':
                rels.add(split[0] + '_' + 'r' + split[1])
                rels.add(split[0] + '_' + 'l' + split[1])
            else:
                rels.add('r' + split[0] + '_' + 'r' + split[1])
                rels.add('l' + split[0] + '_' + 'l' + split[1])
        # Add pair relations, relations with body (global node) and self relations
        for part in body_parts:
            if part == 'n':
                rels.add('b_n')
                rels.add('n_n')  # self-loop
            else:
                rels.add('r' + part + '_' + 'l' + part)
                rels.add('r' + part + '_' + 'r' + part)  # self-loops
                rels.add('l' + part + '_' + 'l' + part)  # self-loops
                rels.add('b' + '_' + 'r' + part)
                rels.add('b' + '_' + 'l' + part)
        # Adding inverses
        for e in list(rels):
            split = e.split('_')
            rels.add(split[1] + '_' + split[0])
        # Add global self relations
        rels.add('b_b')  # self-loop
        rels.add('sb2b')
        rels.add('b2sb')

        return sorted(list(rels))

    def initializeWithAlternative1(self, data):
        
        # We create a map to store the types of the nodes. We'll use it to compute edges' types
        id_by_type = dict()
        self.num_rels = len(HumanGraph.get_rels())

        # Feature dimensions
        body_metric_features = HumanGraph.get_body_metric_features()
        all_features = HumanGraph.get_all_features()
        feature_dimensions = len(all_features)

        # Compute the number of nodes
        # One for superbody (global node) + cameras*joints
        n_nodes = 1
        for cam in data['superbody']:
            n_nodes += 1 + len(cam['joints'])

        # print('Nodes {} body(1) joints({})'.format(n_nodes, len(data['joints'])))

        # Create the tensors
        self.features = np.zeros([n_nodes, feature_dimensions])
        self.labels = np.zeros([1, len(body_metric_features)])  # A 1x5 tensor
        # Generate the graph itself and fill tensor's data
        self.add_nodes(n_nodes)
        # print(minId, data['links'])
        self.labels[0][body_metric_features.index('x')] = data['superbody'][0]['ground_truth'][0]/4000.
        self.labels[0][body_metric_features.index('y')] = data['superbody'][0]['ground_truth'][1]/4000.
        self.labels[0][body_metric_features.index('z')] = data['superbody'][0]['ground_truth'][2]/4000.
        self.labels[0][body_metric_features.index('orientation_sin')] = 0.7*math.sin(data['superbody'][0]['ground_truth'][3])
        self.labels[0][body_metric_features.index('orientation_cos')] = 0.7*math.cos(data['superbody'][0]['ground_truth'][3])
#        self.labels[0][body_metric_features.index('timestamp')] = 0.

        self.features[0, all_features.index('superbody')] = 1.
        max_used_id = 1  # 0 for the superbody (global node)

        # Feature vectors and ids #
        # Create subgraph for each camera
        if self.debug:
            self.type_map_debug = dict()
            self.edges_debug = dict()
            counter = 0
        for cam in data['superbody']:
            camera_number = cam['cameraId']
            typeMap = dict()
            id_by_type.clear()
            if self.debug:
                counter += 1
                # print(cam['cameraId'])
                self.edges_debug[camera_number] = []

            typeMap[max_used_id] = 'b'
            id_by_type['b'] = max_used_id
            self.features[max_used_id, all_features.index('body')] = 1.
            camera_feature = HumanGraph.get_cam_types()[camera_number-1]
            self.features[max_used_id, all_features.index(camera_feature)] = 1.
            self.add_edge(0, max_used_id, {'rel_type': RelTensor([[HumanGraph.get_rels().index('sb2b')]]),
                                    'norm': NormTensor([[1.]])})
            if self.debug:
                self.edges_debug[camera_number].append(tuple([0, max_used_id]))
            self.add_edge(max_used_id, 0, {'rel_type': RelTensor([[HumanGraph.get_rels().index('b2sb')]]),
                                    'norm': NormTensor([[1.]])})
            if self.debug:
                self.edges_debug[camera_number].append(tuple([max_used_id, 0]))

            max_used_id += 1

            # Joints
            for joint, values in cam['joints'].items():
                if joint == 'nose':  # Special case because it hasn't got underscore
                    typeMap[max_used_id] = 'n'
                    id_by_type['n'] = max_used_id
                elif joint.split('_')[1] == 'elbow' or joint.split('_')[1] == 'eye':  # Special case because the abbreviation has 2 letters (el, ey)
                    typeMap[max_used_id] = joint[0] + joint.split('_')[1][0] + joint.split('_')[1][1]
                    id_by_type[typeMap[max_used_id]] = max_used_id
                else:
                    typeMap[max_used_id] = joint[0] + joint.split('_')[1][0]
                    id_by_type[typeMap[max_used_id]] = max_used_id
                self.features[max_used_id, all_features.index(joint)] = 1.
                if USING_3D:
                    self.features[max_used_id, all_features.index('x_position')] = values[0] / 4000.
                    self.features[max_used_id, all_features.index('y_position')] = values[1] / 4000.
                    self.features[max_used_id, all_features.index('z_position')] = values[2] / 4000.
                self.features[max_used_id, all_features.index('i_coordinate')] = (values[3]-320.) / 320.
                self.features[max_used_id, all_features.index('j_coordinate')] = (240.-values[4]) / 240.
                self.features[max_used_id, all_features.index('score')] = values[5]
                self.features[max_used_id, all_features.index(camera_feature)] = 1.
                max_used_id += 1

            # Edges #
            for relation in HumanGraph.get_rels():
                if relation in ['sb2b', 'b2sb']:
                    continue
                split = relation.split('_')
                node_type1 = split[0]
                node_type2 = split[1]
                if (node_type1 in id_by_type) and (node_type2 in id_by_type):
                    self.add_edge(id_by_type[node_type1], id_by_type[node_type2],
                                  {'rel_type': RelTensor([[HumanGraph.get_rels().index(relation)]]),
                                   'norm': NormTensor([[1.]])})
                    if self.debug:
                        self.edges_debug[camera_number].append(tuple([id_by_type[node_type1], id_by_type[node_type2]]))

            if self.debug:
                self.type_map_debug[camera_number] = copy.deepcopy(typeMap)
                if counter == 1:
                    self.type_map_debug[camera_number][0] = 'sb'



# ############________________________________________________________________________________________############## #

class CalibrationDataset(object):
    def __init__(self, path, mode, alt, init_line=-1, end_line=-1, verbose=True):
        super(CalibrationDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.alt = alt
        self.init_line = init_line
        self.end_line = end_line
        self.data = []
        self.num_rels = -1
        try:
            if self.mode != 'run':
                self.load_from_file()
            else:
                self._load(alt)
        except FileNotFoundError:
            self._load(alt)
        self._preprocess()
        if verbose:
            print('{} scenarios loaded.'.format(len(self.data)))

    def set_of_cams(self, l):
        return set([scenario.camera_identifier for scenario in l])

    def all_cameras(self, l):
        set_cams = self.set_of_cams(l)
        lencams = len(set_cams)
        return lencams >= 3

    def compute_delta(self, a, b):
        delta = self.scenarios[b]['timestamp'] - self.scenarios[a]['timestamp']
        assert(delta >= 0)
        return delta

    def _load(self, alt):
        if type(self.path) is str:
            file = open(self.path, "rb")
            self.scenarios = json.loads(file.read())['data_set']
            # For every measurement...
            for idx in range(len(self.scenarios)):
                # print('==========================================================')
                if idx % 100 == 0:
                    print(idx)
                if idx >= limit:
                    break
                self.data.append(HumanGraph(self.scenarios[idx], alt))
            delattr(self, 'scenarios')
        else:
            self.data.append(HumanGraph(self.path, alt))

        self.num_rels = self.data[0].num_rels
        self.graph = dgl.batch(self.data)
        self.features = torch.from_numpy(np.concatenate([element.features for element in self.data]))
        self.labels = torch.from_numpy(np.concatenate([element.labels for element in self.data]))
        if self.mode is not 'run':
            self.save_to_file()

    def get_dataset_name(self):
        return 'dataset_' + self.mode + '_s_' + str(limit) + '.pickle'

    def load_from_file(self):
        filename = self.get_dataset_name()

        with open(path_saves + filename, 'rb') as f:
            self.data = pickle.load(f)

        self.num_rels = self.data[0].num_rels
        self.graph = dgl.batch(self.data)
        self.features = torch.from_numpy(np.concatenate([element.features for element in self.data]))
        self.labels = torch.from_numpy(np.concatenate([element.labels for element in self.data]))

    def save_to_file(self):
        filename = self.get_dataset_name()
        os.makedirs(os.path.dirname(path_saves), exist_ok=True)

        with open(path_saves + filename, 'wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    def _preprocess(self):
        pass  # We don't really need to do anything, all pre-processing is done in the _load method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.data[item].features, self.data[item].labels
