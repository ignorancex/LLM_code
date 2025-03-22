from sys import prefix
import numpy as np
import torch
from mmcv.parallel import DataContainer


class SparseMask:
    def __init__(self, prefix=''):
        if prefix:
            prefix = f"{prefix}_"
        self.prefix = prefix
        self.suffix = "_sparse"

        self.receiver_idx = []
        self.sender_idx = []
        self.relation_idx = []

    def add(self, r_idx, s_idx, e_idx=None):
        if isinstance(r_idx, list):
            r_idx = np.array(r_idx)
        if isinstance(s_idx, list):
            s_idx = np.array(s_idx)

        assert r_idx.shape[0] == s_idx.shape[0]

        self.receiver_idx.append(r_idx)
        self.sender_idx.append(s_idx)
        if e_idx is not None:
            if isinstance(e_idx, list):
                e_idx = np.array(e_idx)
            self.relation_idx.append(e_idx)
    
    def get_sparse(self, row_size, col_size, separate=False, unique=True, with_value=True, dtype=np.float32):
        if separate:
            assert unique == False, f"Since we have edge index"
            r_idx, s_idx = self.get_rs_idx(unique=False)
            if len(self.relation_idx) <= 0:
                self.relation_idx.append(np.arange(r_idx.shape[0]))
            e_idx = np.concatenate(self.relation_idx, axis=0)
            r_mask = dict(
                    indices=np.stack([r_idx, e_idx], axis=0),
                    size=[np.array([row_size]), np.array([e_idx.shape[0]])],)
            s_mask = dict(
                    indices=np.stack([s_idx, e_idx], axis=0),
                    size=[np.array([col_size]), np.array([e_idx.shape[0]])],)
            if with_value:
                r_mask['values'] = np.ones(r_idx.shape[0]).astype(dtype)
                s_mask['values'] = np.ones(s_idx.shape[0]).astype(dtype)
            rst = {
                f"{self.prefix}r_mask{self.suffix}": r_mask,
                f"{self.prefix}s_mask{self.suffix}": s_mask,}
        else:
            r_idx, s_idx = self.get_rs_idx(unique=unique)
            rs_mask = dict(
                indices=np.stack([r_idx, s_idx], axis=0),
                size=[np.array([row_size]), np.array([col_size])],)
            if with_value:
                rs_mask['values'] = np.ones(r_idx.shape[0]).astype(dtype)
            rst = {
                f"{self.prefix}mask{self.suffix}": rs_mask}
        
        return rst
    
    def get_rs_idx(self, unique=True):
        r_idx = np.concatenate(self.receiver_idx, axis=0)
        s_idx = np.concatenate(self.sender_idx, axis=0)
        if unique:
            rs_pair = np.array(list(set(zip(r_idx, s_idx))))
            r_idx = rs_pair[:, 0]
            s_idx = rs_pair[:, 1]
        return r_idx, s_idx

    def get_rs_dict(self, unique=True):
        r_idx, s_idx = self.get_rs_idx(unique=unique)
        return {
            f"{self.prefix}edge": {
                'row': r_idx,
                'col': s_idx,
            }
        }

def np_tensor(data, float_type=np.float32, int_type=np.int64, to_container=True):
    if isinstance(data, dict):
        rst = {
            key: np_tensor(val, float_type=float_type, int_type=int_type, to_container=to_container)
            for key, val in data.items()
        }
    elif isinstance(data, list):
        rst = [np_tensor(val, float_type=float_type, int_type=int_type, to_container=to_container) for val in data]
    else:
        if data.dtype == np.float64:
            data = data.astype(float_type)
        elif data.dtype == np.int32:
            data = data.astype(int_type)
        rst = torch.from_numpy(data)
        if to_container:
            rst = DataContainer(rst, stack=False)
    return rst

def collate_sparse(batch_list, samples_per_gpu=1, num_head=1):
    # 2 * n
    indices = [i['indices'] for i in batch_list]
    data = [i['values'] for i in batch_list if 'values' in i.keys()]
    sizes = [i['size'] for i in batch_list]

    collated_sparse_data = dict(
        indices=indices,
        size=sizes,
    )
    if len(data) > 0:
        collated_sparse_data['values'] = data

    return collated_sparse_data

def stack_sparse(batch_list):
    batch_offset = [0] + [i['size'][0] for i in batch_list]
    batch_offset = np.cumsum(batch_offset)

    indices = []
    data = []
    sizes = [batch_offset[-1]] * 2
    for i in range(len(batch_list)):
        indices.append(batch_list[i]['indices'] + batch_offset[i])
        if 'values' in batch_list[i].keys():
            data.append(batch_list[i]['values'])
    
    indices = np.concatenate(indices, axis=-1)
    collated_sparse_data = dict(
        indices=indices,
        size=sizes,
    )
    if len(data) > 0:
        data = np.concatenate(data, axis=-1)
        collated_sparse_data['values'] = data
    return collated_sparse_data
