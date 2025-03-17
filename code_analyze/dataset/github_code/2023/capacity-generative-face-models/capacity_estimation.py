# ///////////// Copyright 2023 Vishnu Boddeti. All rights reserved. /////////////
# //
# //   Project     : Capacity of Generative Face Models
# //   File        : capacity_estimation.py
# //   Description : Capacity estimation and associated utility functions.
# //                 This includes both class conditional and unconditional generative models.
# //                 Capacity is estimated w.r.t demographic attributes like gender and age.
# //
# //   Created On: 08/20/2023
# //   Created By: Vishnu Boddeti <mailto:vishnu@msu.edu> and Gautam Sreekumar
# ////////////////////////////////////////////////////////////////////////////

import os
import pickle
import math

import numpy as np
import scipy.special as sp

from utils import load_features, get_parameters
from constants import *

__all__ = ['CapacityEstimator', 'get_cosine_bounds', 'ratio_hyperspherical_caps']

def get_cosine_bounds(X, quantile=0.05):
        cosine_dist = np.dot(X, X.transpose())
        min_val = np.min(cosine_dist, axis=1)
        
        mask = np.ones(cosine_dist.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        cosine_dist = cosine_dist * mask
        
        max_val = np.max(cosine_dist, axis=1)
        range_val = max_val - min_val

        value = np.quantile(min_val, quantile)
        total_angle = np.arccos(value) * 180 / np.pi

        return total_angle, min_val, max_val

def ratio_hyperspherical_caps(inter_class_angle, intra_class_angle, cos_delta, sin_delta, d):
    
    # compute cos(\theta) where \theta is the solid
    # angle corresponding to the inter-class hyper-spherical cap
    cos_theta = np.cos(inter_class_angle)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    cos_omega = cos_theta * cos_delta - sin_theta * sin_delta
    
    x = 1 - cos_omega * cos_omega
    a = (d - 1)/2
    b = 0.5

    numerator = sp.betainc(a, b, x)
    index = np.where(cos_omega < 0)
    numerator[index] = 0.5 + numerator[index]
    
    # compute cos(\phi) where \phi is the solid
    # angle corresponding to the intra-class hyper-spherical cap
    cos_phi = np.cos(intra_class_angle)
    sin_phi = np.sqrt(1 - cos_phi**2)
    cos_omega = cos_phi * cos_delta - sin_phi * sin_delta

    x = 1 - cos_omega * cos_omega
    a = (d - 1)/2
    b = 0.5

    denominator = sp.betainc(a, b, x)
    index = np.where(cos_omega < 0)
    denominator[index] = 0.5 + denominator[index]

    capacity = numerator / denominator
    capacity[capacity < 1] = 1
    
    return capacity

class CapacityEstimator(object):
    def __init__(self, args, gen_type='unconditional') -> None:
        
        self.args = args
        
        self.gen_type = gen_type
        self.dataset = args.dataset
        self.quantile = args.quantile
        self.face_model = args.face_model
        self.output_path = args.output_path
        self.ref_dataset = args.ref_dataset
        self.metadata_file = args.metadata_file
        self.max_samples = args.max_samples
        
        self.print_banner()
        
        intra_class_angle, _, _, threshold = get_parameters(self.face_model, self.ref_dataset)
        
        # set threshold values
        self.step = 0.01
        self.cos_delta = np.arange(min(threshold) - self.step * 2, max(threshold) + self.step * 2, self.step)
        self.sin_delta = np.sqrt(1 - self.cos_delta**2)
        
        self.X, self.df = load_features(
            self.dataset,
            self.face_model,
            metadata_file=self.metadata_file,
            root_path=None
        )
        
        self.d = self.X.shape[1]
        
        # saves memory and compute, comment this if you have enough memory
        if self.X.shape[0] > self.max_samples:
            self.X = self.X[:self.max_samples, :]
            self.df = self.df[:self.max_samples]
        
        if self.gen_type == 'conditional':
            # we get class statistics from the features
            self.class_stats = self._get_cos_value_from_X()
            # we calculate the intra-class angle from the features itself.
            min_cos_vals = np.array(self.class_stats['min_cos_vals'])
            intra_class_med_cos_val = np.cos(np.median(np.arccos(min_cos_vals)))
            self.intra_class_angle = math.acos(intra_class_med_cos_val) / 2
        elif self.gen_type == 'unconditional':
            # we calculate the intra-class angle from feature model and reference dataset.
            cos_value = math.cos(math.acos(intra_class_angle) / 2)
            self.intra_class_angle = np.arccos(cos_value)
        else:
            raise NotImplementedError
        
    def print_banner(self):
        print("---- Estimating capacity of "+ \
                    bcolors.DATASET + \
                    f"{self.dataset} " + \
                    bcolors.ENDC + \
                    "using " + \
                    bcolors.FACE_MODEL + \
                    f"{self.face_model} " + \
                    bcolors.ENDC + \
                    "features with " + \
                    bcolors.ENDC + \
                    bcolors.REF_DATASET + \
                    f"{self.ref_dataset}" + \
                    bcolors.ENDC + \
                    " as reference dataset ----")
        
            
    def get_capacity(self, write_to_pkl_file=False):
        # considering half the angle here
        total_angle, _1, _2 = get_cosine_bounds(self.X, quantile=self.quantile)
        inter_class_angle = total_angle / 2
        inter_class_angle = inter_class_angle * np.pi / 180

        capacity = ratio_hyperspherical_caps(
            inter_class_angle,
            self.intra_class_angle,
            self.cos_delta,
            self.sin_delta,
            self.d
        )
        data = {
            'cos_delta': self.cos_delta.tolist(),
            'capacity': capacity,
            'inter_class_angle': [inter_class_angle],
            'intra_class_angle': [self.intra_class_angle]
        }
        
        if self.gen_type == 'conditional':
            data['min_cos_vals'] = self.class_stats['min_cos_vals'].tolist()

        if write_to_pkl_file:
            temp = os.path.join(self.output_path, 'pkl_files', self.dataset)
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            fname = os.path.join(temp,f'{self.face_model}_{self.ref_dataset}_capacity.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(data, f)
    
    def get_capacity_gender(self, write_to_pkl_file=False):
        if self.gen_type == 'conditional':
            ids = self.df['ids']
            ids = np.array([int(_) for _ in ids])
            male_class_ids = np.asarray(self.class_stats['gender'] == 0).nonzero()[0]
            female_class_ids = np.asarray(self.class_stats['gender'] == 1).nonzero()[0]
        
        # compute capacity for gender (male)
        if self.gen_type == 'conditional':
            X_temp = np.zeros((0, self.X.shape[1]))
            for i in male_class_ids:
                where_idx = np.asarray(ids == i).nonzero()
                X_temp = np.vstack((X_temp, self.X[where_idx]))
        elif self.gen_type == 'unconditional':
            X_temp = self.X[self.df['gender'][:self.X.shape[0]] == 0]
        else:
            raise NotImplementedError
        
        total_angle, _1, _2 = get_cosine_bounds(X_temp, quantile=self.quantile)
        inter_class_angle_male = (total_angle / 2) * np.pi / 180
        
        if self.gen_type == 'conditional':
            intra_class_med_cos_val = np.cos(np.median(np.arccos(self.class_stats['min_cos_vals'][male_class_ids])))
            intra_class_angle_male = math.acos(intra_class_med_cos_val) / 2
        else:
            intra_class_angle_male = self.intra_class_angle
        
        capacity_male = ratio_hyperspherical_caps(
            inter_class_angle_male,
            intra_class_angle_male,
            self.cos_delta,
            self.sin_delta,
            self.d
        )

        # compute capacity for gender (female)
        if self.gen_type == 'conditional':
            X_temp = np.zeros((0, self.X.shape[1]))
            for i in female_class_ids:
                where_idx = np.asarray(ids == i).nonzero()
                X_temp = np.vstack((X_temp, self.X[where_idx]))
        elif self.gen_type == 'unconditional':
            X_temp = self.X[self.df['gender'][:self.X.shape[0]] == 1]
        else:
            raise NotImplementedError

        total_angle, _1, _2 = get_cosine_bounds(X_temp, quantile=self.quantile)
        inter_class_angle_female = (total_angle / 2) * np.pi / 180
        if self.gen_type == 'conditional':
            intra_class_med_cos_val = np.cos(np.median(np.arccos(self.class_stats['min_cos_vals'][female_class_ids])))
            intra_class_angle_female = math.acos(intra_class_med_cos_val) / 2
        else:
            intra_class_angle_female = self.intra_class_angle
        capacity_female = ratio_hyperspherical_caps(
            inter_class_angle_female,
            intra_class_angle_female,
            self.cos_delta,
            self.sin_delta,
            self.d
        )

        gender = ['male']*capacity_male.shape[0] + ['female']*capacity_female.shape[0]
        capacity = capacity_male.tolist() + capacity_female.tolist()
        data_gender = {
            'cos_delta': 2 * self.cos_delta.tolist(),
            'capacity':capacity,
            'gender':gender,
            'inter_class_angle_male': [inter_class_angle_male],
            'inter_class_angle_female': [inter_class_angle_female],
            'intra_class_angle_male': [intra_class_angle_male],
            'intra_class_angle_female': [intra_class_angle_female],
        }

        if write_to_pkl_file:
            temp = os.path.join(self.output_path, 'pkl_files', self.dataset)
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            fname = os.path.join(temp, f'{self.face_model}_{self.ref_dataset}_capacity_gender.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(data_gender, f)


    def get_capacity_age(self, write_to_pkl_file=False): 
        if self.gen_type == 'unconditional':
            pass
        else:
            raise NotImplementedError
        
        age_max = int(self.df['age'].max() / 10 + 1)  * 10
        ages = np.arange(0, age_max, 10)

        age_list = []
        capacity_list = []
        inter_class_angle_list = []
        intra_class_angle_list = []

        for i in range(1, len(ages)):
            ind1 = self.df['age'] <= ages[i]
            ind2 = self.df['age'] >= ages[i-1]
            ind = ind1 & ind2
            ind = ind[:self.X.shape[0]]

            X_age = self.X[ind]
            total_angle, _1, _2 = get_cosine_bounds(X_age, quantile=self.quantile)
            inter_class_angle = (total_angle / 2) * np.pi / 180

            tmp_age = [str(ages[i])] * len(self.cos_delta.tolist())
            tmp_capacity = ratio_hyperspherical_caps(
                inter_class_angle,
                self.intra_class_angle,
                self.cos_delta,
                self.sin_delta,
                self.d
            ).tolist()
            
            inter_class_angle_list.append(inter_class_angle)
            intra_class_angle_list.append(self.intra_class_angle)
            
            age_list = age_list + tmp_age
            capacity_list = capacity_list + tmp_capacity

        data_age = {
            'cos_delta': (len(ages) - 1) * self.cos_delta.tolist(),
            'capacity': capacity_list,
            'age': age_list,
            'inter_class_angle': inter_class_angle_list,
            'intra_class_angle': intra_class_angle_list
        }

        if write_to_pkl_file:
            temp = os.path.join(self.output_path, 'pkl_files', self.dataset)
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            fname = os.path.join(temp, f'{self.face_model}_{self.ref_dataset}_capacity_age.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(data_age, f)

    def _get_cos_value_from_X(self):
        ids = self.df['ids']
        ids = np.array([int(_) for _ in ids])
        unique_ids = np.unique(ids)

        class_stats = {}
        class_stats['age'] = []
        class_stats['gender'] = []
        class_stats['min_cos_vals'] = []
        
        for i in unique_ids:
            where_idx = ids == i
            age = self.df['age'][where_idx]
            gender = self.df['gender'][where_idx]
            age = np.mean(age)
            gender = np.mean(gender)
            
            if gender > 0.5:
                gender = 1
            else:
                gender = 0
            
            feat_ = self.X[where_idx]
            cosine_dist = np.dot(feat_, feat_.transpose())
            min_val = np.min(cosine_dist, axis=1)
            
            class_stats['age'].append(age)
            class_stats['gender'].append(gender)
            class_stats['min_cos_vals'].append(np.min(min_val))
            
        class_stats['age'] = np.array(class_stats['age'])
        class_stats['gender'] = np.array(class_stats['gender'])
        class_stats['min_cos_vals'] = np.array(class_stats['min_cos_vals'])
        
        return class_stats