import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
from os.path import join as pjoin
import random
from torch.utils.data._utils.collate import default_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import remove_joints_augmentation, add_joint_augmentation
from model.conditioners import T5Conditioner


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

""" extract parents based on first frame """
def get_motion_parents(motion):
    joints_num = motion.shape[1]
    parents_map = np.sum(motion[0]**2, axis=2)
    parents = [-1]
    for j in range(1, joints_num):
        j_parent = np.where(parents_map[j] != 0)[0][0]
        parents.append(j_parent)
    return parents

""" create temporal mask template for window size"""
def create_temporal_mask_for_window(window, max_len):
    margin = window // 2
    mask = torch.zeros(max_len+1, max_len+1)
    mask[:, 0] = 1
    for i in range(max_len+1):
        mask[i, max(0, i - margin):min(max_len + 1, i + margin + 2)] = 1
    return mask

'''For use of training text motion matching model, and evaluations'''
class MotionDataset(data.Dataset):
    def __init__(self, opt, cond_dict, temporal_window, t5_name, balanced):
        print("in MotionDataset constructor")
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.cond_dict = cond_dict
        self.balanced = balanced
        data_dict = {}
        all_object_types = self.cond_dict.keys()
        new_name_list = []
        length_list = []
        self.t5_conditioner = T5Conditioner(name=t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')

        for object_type in all_object_types:
            parents = self.cond_dict[object_type]['parents']
            tpos_first_frame = self.cond_dict[object_type]['tpos_first_frame']
            joint_relations = self.cond_dict[object_type]['joint_relations']
            joints_graph_dist = self.cond_dict[object_type]['joints_graph_dist']
            offsets = self.cond_dict[object_type]['offsets']
            joints_names = self.cond_dict[object_type]['joints_names']
            joints_names_embs = self.encode_joints_names(joints_names).detach().cpu().numpy()
            kinematic_chains = self.cond_dict[object_type]['kinematic_chains']
            object_motions = [f for f in os.listdir(opt.motion_dir) if f.startswith(f'{object_type}_')]
            
            for name in object_motions:
                try:
                    motion = np.load(pjoin(opt.motion_dir, name))
                    data_dict[name] = {
                                        'motion': motion,
                                        'length': len(motion),
                                        'object_type': object_type,
                                        'parents': parents,
                                        'joints_graph_dist': joints_graph_dist,
                                        'joints_relations': joint_relations,
                                        'tpos_first_frame': tpos_first_frame,
                                        'offsets': offsets,
                                        'joints_names_embs': joints_names_embs,
                                        'kinematic_chains': kinematic_chains
                                       }
                                       
                    new_name_list.append(name)
                    length_list.append(len(motion))
                except:
                    pass
                
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.temporal_mask_template = create_temporal_mask_for_window(temporal_window, self.max_motion_length)
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def encode_joints_names(self, joints_names): # joints names should be padded with None to be of max_len 
        names_tokens = self.t5_conditioner.tokenize(joints_names)
        embs = self.t5_conditioner(names_tokens)
        return embs
    
    def inv_transform(self, x, y):
        mean = self.cond_dict[y['object_type']]['mean']
        std = self.cond_dict[y['object_type']]['std']
        return x * std + mean
    
    def augment(self, data):
        object_type = data['object_type']
        if object_type != "Dragon":
            aug_type = random.choice([0, 1, 2])
        else: 
            aug_type = random.choice([0, 1])
        mean = self.cond_dict[object_type]['mean']
        std = self.cond_dict[object_type]['std']
        if aug_type == 0: #no augmentation
            return data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains'], mean, std
        elif aug_type == 1: # remove_joints
            removal_rate = random.choice([0.1, 0.2, 0.3])
            return  remove_joints_augmentation(data, removal_rate, mean, std) 
        else: #add joint
            return add_joint_augmentation(data, mean, std)
        
    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        if self.balanced:
            idx = item #self.pointer + item (handled in weighted sampler)
        else:
            idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std = self.augment(data)
        "Z Normalization"
        # Normalize all coords but rotations 
        std += 1e-6 # for stability
        motion = (motion - mean[None, :]) / std[None, :]
        motion = np.nan_to_num(motion)
        ind = 0
        tpos_first_frame =  (tpos_first_frame - mean) / std
        tpos_first_frame = np.nan_to_num(tpos_first_frame)
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        elif m_length > self.max_motion_length:
            ind = random.randint(0, m_length - self.max_motion_length)
            motion = motion[ind: ind + self.max_motion_length]
            m_length = self.max_motion_length
            

        return motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, mean, std, self.opt.max_joints

class TruebonesSampler(WeightedRandomSampler):
    def __init__(self, data_source):
        num_samples = len(data_source)
        object_types = data_source.motion_dataset.cond_dict.keys()
        name_list = data_source.motion_dataset.name_list
        total_samples = len(name_list)
        weights = np.zeros(total_samples)
        object_share = 1.0/len(object_types)
        pointer = data_source.motion_dataset.pointer
        for object_type in object_types:
            object_indices = [i for i in range(num_samples) if i>=pointer and name_list[i].startswith(f'{object_type}_')]
            object_prob = object_share / len(object_indices)
            weights[object_indices] = object_prob
        super().__init__(num_samples=num_samples, weights=weights)
    
class Truebones(data.Dataset):
    def __init__(self, split="train", temporal_window=31, t5_name='t5-base', **kwargs):
        print("in TruebonesMixedJoints constructor")
        abs_base_path = f'.'
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(device)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.max_motion_length = min(opt.max_motion_length, kwargs['num_frames'])
        self.opt = opt
        self.balanced = kwargs['balanced']
        self.objects_subset = kwargs['objects_subset']
        print('Loading Truebones dataset')
        cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
        subset = opt.subsets_dict[self.objects_subset] 
        cond_dict = {k:cond_dict[k] for k in subset if k in cond_dict}
        print(f'Dataset subset {self.objects_subset} consists of {len(cond_dict.keys())} characters')
            
        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.motion_dataset = MotionDataset(self.opt, cond_dict, temporal_window, t5_name, self.balanced)
        assert len(self.motion_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.motion_dataset.__getitem__(item)

    def __len__(self):
        return self.motion_dataset.__len__()
