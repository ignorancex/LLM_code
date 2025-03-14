"""
Utitlities for liver datasets
Modified from
https://github.com/microsoft/Recursive-Cascaded-Networks/tree/master/data_util/liver.py
"""


import numpy as np
import json
import os
import h5py
import _pickle as pickle
from .data import Split
from scipy.ndimage import zoom

def get_range(imgs):
    r = np.any(imgs.reshape(imgs.shape[0], -1), axis=-1).nonzero()
    return np.min(r), np.max(r)


class Hdf5Reader:
    def __init__(self, path):
        try:
            self.file = h5py.File(path, "r")
        except Exception:
            print('{} not found!'.format(path))
            self.file = None

    def __getitem__(self, key):
        data = {'id': key}
        if self.file is None:
            return data
        group = self.file[key]
        for k in group:
            data[k] = group[k]
        return data


class FileManager:
    def __init__(self, files):
        self.files = {}
        for k, v in files.items():
            self.files[k] = Hdf5Reader(v["path"])

    def __getitem__(self, key):
        p = key.find('/')
        if key[:p] in self.files:
            ret = self.files[key[:p]][key[p+1:]]
            ret['id'] = key.replace('/', '_')
            return ret
        elif '/' in self.files:
            ret = self.files['/'][key]
            ret['id'] = key.replace('/', '_')
            return ret
        else:
            raise KeyError('{} not found'.format(key))


class Dataset:
    def __init__(self, args, split_path, image_size=128, affine=False,
                    mask=False, paired=False, task=None,
                    batch_size=None, discriminator=False,
                    pretrained_flow_path='./Teacher_deformations'):
        pretrained_flow_path = os.path.abspath(pretrained_flow_path) + '/Liver'
        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])
        self.subset = {}
        self.fraction = image_size * 1.0 / 128
        self.image_size = image_size

        for k, v in config['subsets'].items():
            self.subset[k] = {}
            print(k)
            for entry in v:
                self.subset[k][entry] = self.files[entry]

        self.paired = paired

        def convert_int(key):
            try:
                return int(key)
            except ValueError as e:
                return key
        self.schemes = dict([(convert_int(k), v)
                             for k, v in config['schemes'].items()])

        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))

        self.task = task
        if self.task is None:
            self.task = config.get("task", "registration")
        if not isinstance(self.task, list):
            self.task = [self.task]
        self.discriminator_flag = discriminator
        if self.discriminator_flag:
            self.agg_flows = {}
            for f_name in sorted(os.listdir(pretrained_flow_path)):
                print("Loading " + f_name)
                with open(os.path.join(pretrained_flow_path, f_name), 'rb') as f:
                    flows = pickle.load(f)
                    self.agg_flows.update(flows)

            print("Loading the teacher deformations successfully")

        self.batch_size = batch_size

    def get_pairs_adj(self, data):
        pairs = []
        d1 = None
        for d2 in data:
            if d1 is None:
                d1 = d2
            else:
                pairs.append((d1, d2))
                pairs.append((d2, d1))
                d1 = None
        return pairs

    def get_pairs(self, data, ordered=True):
        pairs = []
        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                if i != j:
                    if ordered or i < j:
                        pairs.append((d1, d2))
        return pairs

    def generate_pairs(self, arr, loop=False):
        if self.paired:
            sets = self.get_pairs_adj(arr)
        else:
            sets = self.get_pairs(arr, ordered=True)

        while True:
            if loop:
                np.random.shuffle(sets)
            for d1, d2 in sets:
                yield (d1, d2)
            if not loop:
                break

    def generator(self, subset, batch_size=None, loop=False):
        if batch_size is None:
            batch_size = self.batch_size
        valid_mask = np.ones([6], dtype=np.bool)
        scheme = self.schemes[subset]
        num_data = 0
        if 'registration' in self.task:
            generators = [(self.generate_pairs(list(self.subset[k].values()), loop))
                          for k, fraction in scheme.items()]
            fractions = [int(np.round(fraction * batch_size))
                         for k, fraction in scheme.items()]

            if self.discriminator_flag:
                i = 0
                while True:
                    if i == 0:
                        ret = dict()
                        ret['voxel1'] = np.zeros(
                            (batch_size, self.image_size, self.image_size,
                                self.image_size, 1), dtype=np.float32)
                        ret['voxel2'] = np.zeros(
                            (batch_size, self.image_size, self.image_size,
                                self.image_size, 1), dtype=np.float32)
                        ret['seg1'] = np.zeros(
                            (batch_size, 128, 128, 128, 1), dtype=np.float32)
                        ret['seg2'] = np.zeros(
                            (batch_size, 128, 128, 128, 1), dtype=np.float32)
                        ret['point1'] = np.ones(
                            (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                        ret['point2'] = np.ones(
                            (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                        ret['id1'] = np.empty((batch_size), dtype='<U40')
                        ret['id2'] = np.empty((batch_size), dtype='<U40')
                        ret['agg_flow'] = np.zeros(
                            (batch_size, 128, 128, 128, 3), dtype=np.float32)
                    d1, d2 = next(generators[0])
                    id = d1['id'] + '-' + d2['id']
                    if id not in self.agg_flows.keys():
                        continue
                    ret['agg_flow'][i] = self.agg_flows[id]
                    ret['voxel1'][i, ..., 0], ret['voxel2'][i, ...,0] \
                            = d1['volume'], d2['volume']

                    if 'segmentation' in d1:
                        ret['seg1'][i, ..., 0] = d1['segmentation']
                    if 'segmentation' in d2:
                        ret['seg2'][i, ..., 0] = d2['segmentation']

                    if 'point' in d1:
                        ret['point1'][i] = d1['point'][...][valid_mask]
                    if 'point' in d2:
                        ret['point2'][i] = d2['point'][...][valid_mask]

                    ret['id1'][i] = d1['id']
                    ret['id2'][i] = d2['id']
                    i += 1

                    if i == batch_size:
                        i = 0
                        yield ret
            else:
                while True:
                    ret = dict()
                    ret['voxel1'] = np.zeros(
                        (batch_size, self.image_size, self.image_size,
                            self.image_size, 1), dtype=np.float32)
                    ret['voxel2'] = np.zeros(
                        (batch_size, self.image_size, self.image_size,
                            self.image_size, 1), dtype=np.float32)
                    ret['seg1'] = np.zeros(
                        (batch_size, self.image_size, self.image_size,
                            self.image_size, 1), dtype=np.float32)
                    ret['seg2'] = np.zeros(
                        (batch_size, self.image_size, self.image_size,
                            self.image_size, 1), dtype=np.float32)
                    ret['point1'] = np.ones(
                        (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                    ret['point2'] = np.ones(
                        (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                    ret['agg_flow'] = np.zeros(
                        (batch_size, 128, 128, 128, 3), dtype=np.float32)
                    ret['id1'] = np.empty((batch_size), dtype='<U40')
                    ret['id2'] = np.empty((batch_size), dtype='<U40')

                    i = 0
                    flag = True
                    nums = fractions
                    for gen, num in zip(generators, nums):
                        assert not self.paired or num % 2 == 0
                        for t in range(num):
                            try:
                                while True:
                                    d1, d2 = next(gen)
                                    break
                            except StopIteration:
                                flag = False
                                break

                            ret['voxel1'][i, ..., 0], ret['voxel2'][i, ...,0] \
                                    = d1['volume'], d2['volume']

                            if 'segmentation' in d1:
                                ret['seg1'][i, ..., 0] = d1['segmentation']
                            if 'segmentation' in d2:
                                ret['seg2'][i, ..., 0] = d2['segmentation']
                            if 'point' in d1:
                                ret['point1'][i] = d1['point'][...][valid_mask]
                            if 'point' in d2:
                                ret['point2'][i] = d2['point'][...][valid_mask]

                            ret['id1'][i] = d1['id']
                            ret['id2'][i] = d2['id']
                            i += 1
                    num_data += 1
                    if flag:
                        assert i == batch_size
                        yield ret
                    else:
                        yield ret
                        break
