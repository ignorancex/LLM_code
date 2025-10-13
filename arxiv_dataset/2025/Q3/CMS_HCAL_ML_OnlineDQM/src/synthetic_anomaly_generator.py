import torch
import argparse
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import utilities as util

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
data_path = os.path.abspath(os.path.dirname(current_path)+"/data")
sys.path.append(data_path)

hcal_digimap_3d_meta_template = util.load_json(rf"{data_path}\hcal_digimap_3d_meta_template.json")


class SyntheticDQMChannelAnomalyGenerator():

    def __init__(self, subdetector_name, **kwargs):
        self.subdetector_name = subdetector_name
        self.mask_keep = kwargs.get("mask_keep", None)

        if self.subdetector_name == "he":
            self.anml_dim_range = {"ieta": [16, 29], "iphi": [1, 72], "depth": [1, 7]}
        elif self.subdetector_name == "hb":
            self.anml_dim_range = {"ieta": [1, 16], "iphi": [1, 72], "depth": [1, 4]}

        ieta_valid_max_size = self.anml_dim_range["ieta"][1] - self.anml_dim_range["ieta"][0] + 1
        depth_valid_max_size = self.anml_dim_range["depth"][1] - self.anml_dim_range["depth"][0] + 1
        self.anml_spatial_dim = kwargs.get("anml_spatial_dim", {"ieta": [1, ieta_valid_max_size+1], "iphi": [1, 5], "depth": [1, depth_valid_max_size+1]}) # spatial size of anomaly, >0 and <max+1 length of subdetector. one rbx sector covers 4 iphi's 

        print(self.anml_dim_range)
        print(self.anml_spatial_dim)
        
        # setting checking
        for axis, v in self.anml_spatial_dim.items():
            valid_max_size = self.anml_dim_range[axis][1] - self.anml_dim_range[axis][0] + 1 + 1
            assert valid_max_size > 0
            assert v[0] < v[1], f"the min and max dim of spatial anomaly along the {axis} is not valid."
            assert v[0] > 0, f"the min of spatial anomaly along the {axis} is not valid."
            assert v[1] <= valid_max_size, f"the min of spatial anomaly along the {axis} is not valid."

    def get_valid_ls_for_timewindowing(self, data_np_nd, memorysize):
        data_np = data_np_nd.copy()
        # exclude nan and zero ls from the test set
        zero_nan_ls = np.where(np.nansum(data_np, axis=tuple(np.arange(1, data_np.ndim))) == 0)[0]
        
        # to avoid missing appear in history ls in sliding time window
        zero_nan_ls_list = []
        for i in zero_nan_ls:
            zero_nan_ls_list.extend(list(np.arange(i, i+memorysize))) 
        print("missing ls: {}, with time window memorysize: {}".format(len(set(zero_nan_ls)), len(set(zero_nan_ls_list))))
        
        ls_index_reset = np.arange(data_np.shape[0])
        ls_index_sel = sorted(list(set(ls_index_reset).difference(set(zero_nan_ls_list))))
        return ls_index_sel

    def synthetic_anomaly_generator_base(self, data_np_nd, **kwargs):
        '''
        data_np_nd: [lsxieetaxiphixdepthxfeature]
        feature:1
        '''
        # anml_type = kwargs.get("anml_type", "dqm_regional")
        # anml_dim_range = kwargs.get("anml_dim_range", {"ieta": [16, 29], "iphi": [1, 72], "depth": [1, 7]})
        
        anml_gen_dict = kwargs.get("anml_gen_dict", None)
        update_strength = kwargs.get("update_strength", True)
        skip_init_ls = kwargs.get("skip_init_ls", 5) # most of the time, initial ls are a kind of noisy
        anml_strength = kwargs.get("anml_strength", 0)  # rd, anomaly = rd*channel_value
        target_ls_size = kwargs.get("target_ls_size", 1)
        is_ls_adjust = kwargs.get("is_ls_adjust", False)
        anml_strength_apply = kwargs.get("anml_strength_apply", "relative")
        anml_value_max_thr = kwargs.get("anml_value_max_thr", None)

        seed = kwargs.get("seed", 100)
        print(data_np_nd.shape)
        num_ls = data_np_nd.shape[0]
        if anml_gen_dict is None: # initial
            np.random.seed(seed)
            anml_gen_dict = {}
            ls_rand = self.synthetic_anomaly_ls_generator(
                num_ls, skip_init_ls=skip_init_ls, target_ls_size=target_ls_size, is_ls_adjust=is_ls_adjust)

            data_np_nd_with_anml = data_np_nd[ls_rand]

            data_np_nd_anml_label = np.zeros(data_np_nd_with_anml.shape)
            anml_gen_dict["ls_rand"] = ls_rand
            anml_gen_dict["seed"] = seed
            anml_gen_dict["loc"] = {}
            for i, ls in tqdm(enumerate(ls_rand)):
                anml_range_idx = self.synthetic_anomaly_spatial_location_generator(i, **kwargs)
                data_np_nd_with_anml[i], data_np_nd_anml_label[i], anml_strength_assigned = self.synthetic_anomaly_apply(
                    data_np_nd_with_anml[i], anml_range_idx[1:], anml_strength, use_random=True, inplace=True, anml_strength_apply=anml_strength_apply, anml_value_max_thr=anml_value_max_thr)
                anml_gen_dict["loc"][i] = {
                    "ls": ls, "anml_range_idx": anml_range_idx, "anml_strength": anml_strength_assigned, "anml_value_max_thr": anml_value_max_thr}
                
        else: # load from previous generator
            data_np_nd_with_anml = data_np_nd
            data_np_nd_anml_label = np.zeros(data_np_nd_with_anml.shape)

            for i, anml_gen in tqdm(anml_gen_dict["loc"].items()):
                if not update_strength:
                    anml_strength = anml_gen["anml_strength"]
                anml_value_max_thr = anml_gen["anml_value_max_thr"]
                data_np_nd_with_anml[i], data_np_nd_anml_label[i], anml_strength_assigned = self.synthetic_anomaly_apply(
                    data_np_nd_with_anml[i], anml_gen["anml_range_idx"][1:], anml_strength, use_random=False, inplace=True, anml_strength_apply=anml_strength_apply, anml_value_max_thr=anml_value_max_thr)

        data_np_nd_with_anml = torch.Tensor(data_np_nd_with_anml)
        data_np_nd_anml_label = torch.BoolTensor(data_np_nd_anml_label)

        return data_np_nd_with_anml, data_np_nd_anml_label, anml_gen_dict

    def synthetic_anomaly_ls_generator(self, source_num_ls, skip_init_ls=5, target_ls_size=1, is_ls_adjust=False):
        if is_ls_adjust:
            start_ls = np.random.randint(
                skip_init_ls, source_num_ls - target_ls_size, 1)
            ls_rand = np.arange(start_ls, start_ls + target_ls_size)
        else:
            ls_rand = np.random.choice(
                np.arange(skip_init_ls, source_num_ls), target_ls_size)
        return ls_rand

    def synthetic_anomaly_spatial_location_generator(self, ls, **kwargs):
        '''
        data_np_nd: [lsxieetaxiphixdepthxfeature]
        feature:1
        '''
        anml_dim_range = kwargs.get("anml_dim_range", self.anml_dim_range)
        mask_keep = kwargs.get("mask_keep",  self.mask_keep)
        anml_spatial_dim = kwargs.get("anml_spatial_dim", self.anml_spatial_dim)

        out_of_mask = True
        while(out_of_mask):
            anml_range_idx = []
            for i, (key, value) in enumerate(anml_dim_range.items()):
                edge_idx = np.random.randint(value[0], value[1]+1, 2)
                dim_range = np.abs(np.diff(edge_idx)) 

                while ((dim_range < anml_spatial_dim[key][0]) or (dim_range > anml_spatial_dim[key][1])):
                    edge_idx = np.random.randint(value[0], value[1]+1, 2)
                    dim_range = np.abs(np.diff(edge_idx)) 
                
                if key == "ieta":
                    edge_idx = np.random.choice([-1, 1], 1)*edge_idx
        
                edge_idx = util.segidx_arrayidx_mapper(
                    edge_idx, axis=key, istoarray=True)
        
                anml_range_idx.append(slice(edge_idx.min(), edge_idx.max() + 1))
            
            if mask_keep is not None:
                out_of_mask = mask_keep[tuple(anml_range_idx)].sum() == 0
            else:
                out_of_mask = False
            
            anml_range_idx = (ls, ) + tuple(anml_range_idx) + (0, )
            
        return anml_range_idx

    def synthetic_anomaly_apply(self, data_np_nd_ls, anml_range_idx, anml_strength, use_random=False, inplace=False, anml_strength_apply="relative", anml_value_max_thr=None):
        ls = anml_range_idx[0]
        if not inplace:
            data_np_nd_ls_with_anml = util.copy.deepcopy(data_np_nd_ls)
        else:
            data_np_nd_ls_with_anml = data_np_nd_ls
        data_np_nd_ls_anml_label = np.zeros(data_np_nd_ls_with_anml.shape)

        zero_mask = data_np_nd_ls_with_anml == 0
        nan_mask = np.isnan(data_np_nd_ls_with_anml)
        
        if not isinstance(anml_strength, (list, np.ndarray)):
            anml_strength = [anml_strength]
            
        if use_random:
            anml_strength_assigned = np.array(np.random.choice(
                anml_strength, 1))  # assumes a uniform distribution
        else:
            anml_strength_assigned = anml_strength
        
        
        isequal_withaml_strength_mask = data_np_nd_ls_with_anml == anml_strength_assigned
                
        #print("anml_strength_assigned:", anml_strength_assigned)
        if anml_strength_apply == "relative":
            if anml_value_max_thr is not None:
                isamlstrength_below_thr_idx = anml_strength_assigned[0]*data_np_nd_ls_with_anml[anml_range_idx] <= anml_value_max_thr
                data_np_nd_ls_with_anml[anml_range_idx][isamlstrength_below_thr_idx] = anml_strength_assigned[0]*data_np_nd_ls_with_anml[anml_range_idx][isamlstrength_below_thr_idx]
                data_np_nd_ls_anml_label[anml_range_idx][isamlstrength_below_thr_idx] = 1 if anml_strength_assigned[0] != 1 else 0
            else:
                data_np_nd_ls_with_anml[anml_range_idx] = anml_strength_assigned[0]*data_np_nd_ls_with_anml[anml_range_idx]
                data_np_nd_ls_anml_label[anml_range_idx] = 1 if anml_strength_assigned[0] != 1 else 0
            
        else:
            # absolute error
            isequal_withaml_strength_mask = data_np_nd_ls_with_anml[anml_range_idx] == anml_strength_assigned[0]
            
            data_np_nd_ls_with_anml[anml_range_idx] = anml_strength_assigned[0]
            data_np_nd_ls_anml_label[anml_range_idx] = 1
            
            # data_np_nd_ls_with_anml[anml_range_idx][isequal_withaml_strength_mask] = anml_strength_assigned
            data_np_nd_ls_anml_label[anml_range_idx][isequal_withaml_strength_mask] = 0
            
        data_np_nd_ls_with_anml[nan_mask] = np.nan
        data_np_nd_ls_with_anml[zero_mask] = 0
        
        data_np_nd_ls_anml_label[nan_mask | zero_mask] = 0
        
        return data_np_nd_ls_with_anml, data_np_nd_ls_anml_label, anml_strength_assigned

    def add_history_timewindow_to_synthetic_anomaly(self, data_np_nd: np.ndarray, data_np_nd_with_anml: torch.Tensor, data_np_nd_anml_label: torch.Tensor, anml_gen_dict: dict, memorysize: int):
        # add history ls to the LS with anomaly
        data_np_nd_with_anml_t = torch.cat(
            [data_np_nd_with_anml.unsqueeze(1)]*memorysize, axis=1)
        data_np_nd_with_anml_t.shape
        data_np_nd_anml_label_t = torch.cat(
            [data_np_nd_anml_label.unsqueeze(1)]*memorysize, axis=1)
        data_np_nd_anml_label_t.shape

        # anml_gen_dict
        for i, ls in tqdm(enumerate(anml_gen_dict["ls_rand"])):
            data_np_nd_with_anml_t[i, :-1] = torch.Tensor(data_np_nd[ls-memorysize+1:ls])
            data_np_nd_anml_label_t[i, :-1] = False

        return data_np_nd_with_anml_t, data_np_nd_anml_label_t

    def synthetic_anomaly_generator_timewindow(self, data_np_nd: np.ndarray, data_np_nd_with_anml: torch.Tensor, data_np_nd_anml_label: torch.Tensor, anml_gen_dict: dict, memorysize: int, **kwargs):
        # add same anomaly to all history ls
        data_np_nd_with_anml_tw = torch.cat(
            [data_np_nd_with_anml.unsqueeze(1)]*memorysize, axis=1).detach().numpy()
        data_np_nd_with_anml_tw.shape
        
        data_np_nd_anml_label_tw = torch.cat(
            [data_np_nd_anml_label.unsqueeze(1)]*memorysize, axis=1).detach().numpy()
        data_np_nd_anml_label_tw.shape
        
        # data_np_nd_anml_label_t
        # anml_gen_dict
        for i, ls in tqdm(enumerate(anml_gen_dict["ls_rand"])):
            data_np_nd_with_anml_tw[i, :-1] = data_np_nd[ls-memorysize+1:ls]

        for tw in tqdm(range(memorysize)):
            data_np_nd_with_anml_tw[:, tw], data_np_nd_anml_label_tw[:, tw], _ = self.synthetic_anomaly_generator_base(data_np_nd_with_anml_tw[:, tw],
                                                                                                        anml_gen_dict=anml_gen_dict, **kwargs
                                                                                                        )
                                                                                                    
        data_np_nd_with_anml_tw = torch.Tensor(data_np_nd_with_anml_tw)
        data_np_nd_anml_label_tw = torch.BoolTensor(data_np_nd_anml_label_tw)

        return data_np_nd_with_anml_tw, data_np_nd_anml_label_tw

    def generate_anml_spatial_temporal(self, test_np_filename: str, anml_gen_size: int, anml_strength: list, memorysize: int, **kwargs):
        '''
        test_np_filename = "{}_test_dataset_m1_n18_ls_range_100_-10_norm_depth_agg".format(dataset)
        '''
        skip_init_ls = kwargs.get("skip_init_ls", None)
        issave = kwargs.get("issave", False)
        dataset = kwargs.get("dataset", "")
        anml_tag = kwargs.get("anml_tag", "")
        anml_strength_apply = kwargs.get("anml_strength_apply", "relative") 
        
        data_np_nd_test = util.load_npdata("{}/{}/{}.npy".format(data_path, dataset, test_np_filename))
        data_np_nd_test.shape
        
        ls_index_sel = self.get_valid_ls_for_timewindowing(data_np_nd_test, memorysize)
        
        if skip_init_ls is None:
            skip_init_ls = memorysize

        # generate the base anomaly labels (non-temporal): generate random LS, random spatial location (boxes) and sizes.
        data_np_nd_with_anml, data_np_nd_anml_label, anml_gen_dict = self.synthetic_anomaly_generator_base(data_np_nd_test[ls_index_sel], 
                                                                                            anml_strength=anml_strength[0],
                                                                                            target_ls_size=anml_gen_size,
                                                                                            **kwargs
                                                                                            )
        print(data_np_nd_with_anml.shape, data_np_nd_anml_label.shape)
        ls_rand = anml_gen_dict["ls_rand"]
        # print(anml_gen_dict)

        # below preserves the generated anomaly config from the base to generate anomaly using time-windowing and multiple anomaly strength setting

        # ls used for anomaly generation, will be used for normal label
        data_np_nd_with_healthy_tw, data_np_nd_healthy_label_tw = self.synthetic_anomaly_generator_timewindow(
                                                                data_np_nd_test[ls_index_sel],
                                                                # data_np_nd_with_anml,
                                                                torch.Tensor(data_np_nd_test[np.array(ls_index_sel)[ls_rand]]), 
                                                                0*data_np_nd_anml_label, 
                                                                anml_gen_dict, memorysize, 
                                                                update_strength=True, 
                                                                anml_strength=1, anml_strength_apply="relative"
                                                                )
                                                                

        # anomaly on the last ls in a time window
        data_np_nd_with_anml_t, data_np_nd_anml_label_t = self.add_history_timewindow_to_synthetic_anomaly(data_np_nd_test[ls_index_sel], 
                                                            data_np_nd_with_anml, 
                                                            data_np_nd_anml_label, anml_gen_dict, memorysize
                                                            )
        
        print(data_np_nd_with_anml_t.shape, data_np_nd_anml_label_t.shape)
        
        # anomaly on all ls in a time window
        data_np_nd_with_anml_tw, data_np_nd_anml_label_tw = self.synthetic_anomaly_generator_timewindow(
                                                            data_np_nd_test[ls_index_sel],
                                                            torch.Tensor(data_np_nd_test[np.array(ls_index_sel)[ls_rand]]), 
                                                            data_np_nd_anml_label, 
                                                            anml_gen_dict, memorysize, 
                                                            update_strength=False, anml_strength_apply=anml_strength_apply
                                                            )
        print(data_np_nd_with_anml_tw.shape, data_np_nd_anml_label_tw.shape)

        # anomaly on all ls in a time window with several decay factor Rd
        data_np_nd_with_anml_tw_multiple = [data_np_nd_with_anml_tw]
        data_np_nd_anml_label_tw_multiple = [data_np_nd_anml_label_tw]
        for anml_multiply_factor in anml_strength[1:]:
            #make sure data_np_nd_with_anml resets to healthy
            data_np_nd_with_anml_tw_, data_np_nd_anml_label_tw_ = self.synthetic_anomaly_generator_timewindow(
                                                                util.copy.deepcopy(data_np_nd_test[ls_index_sel]),
                                                                # data_np_nd_with_anml,    
                                                                torch.Tensor(data_np_nd_test[np.array(ls_index_sel)[ls_rand]]), 
                                                                util.copy.deepcopy(data_np_nd_anml_label), 
                                                                util.copy.deepcopy(anml_gen_dict), memorysize, 
                                                                update_strength=True, 
                                                                anml_strength=anml_multiply_factor, anml_strength_apply=anml_strength_apply
                                                                )
            print(data_np_nd_with_anml_tw_.shape)
            data_np_nd_with_anml_tw_multiple.append(util.copy.deepcopy(data_np_nd_with_anml_tw_))
            data_np_nd_anml_label_tw_multiple.append(util.copy.deepcopy(data_np_nd_anml_label_tw_))

        data_np_nd_with_anml_tw_multiple = torch.cat(data_np_nd_with_anml_tw_multiple, axis=0)
        data_np_nd_anml_label_tw_multiple = torch.cat(data_np_nd_anml_label_tw_multiple, axis=0)
        print(data_np_nd_with_anml_tw_multiple.shape, data_np_nd_anml_label_tw_multiple.shape)
                                                        
        if issave:
            kwargs.update({ "test_np_filename": test_np_filename, "anml_gen_size": anml_gen_size, "anml_strength": anml_strength, "memorysize": memorysize})
            util.save_json("{}/{}/{}_anml_kwargs_{}_{}.json".format(data_path, dataset, test_np_filename, anml_tag, anml_gen_size), kwargs)
            util.save_pickle("{}/{}/{}_anml_gen_dict_{}_{}.pkl".format(data_path, dataset, test_np_filename, anml_tag, anml_gen_size), anml_gen_dict)
            
            util.save_npdata("{}/{}/{}_data_np_nd_with_healthy_tw_{}_{}.npy".format(data_path, dataset, test_np_filename, anml_tag, anml_gen_size), data_np_nd_with_healthy_tw)
            util.save_npdata("{}/{}/{}_data_np_nd_healthy_label_tw_{}_{}.npy".format(data_path, dataset, test_np_filename, anml_tag, anml_gen_size), data_np_nd_healthy_label_tw)
            
            util.save_npdata("{}/{}/{}_data_np_nd_with_anml_{}_{}.npy".format(data_path, dataset, test_np_filename, anml_tag, anml_gen_size), data_np_nd_with_anml)
            util.save_npdata("{}/{}/{}_data_np_nd_anml_label_{}_{}.npy".format(data_path, dataset, test_np_filename, anml_tag, anml_gen_size), data_np_nd_anml_label)

            util.save_npdata("{}/{}/{}_data_np_nd_with_anml_tw_multiple_{}_{}.npy".format(data_path, dataset, test_np_filename, anml_tag, anml_gen_size), data_np_nd_with_anml_tw_multiple)
            util.save_npdata("{}/{}/{}_data_np_nd_anml_label_tw_multiple_{}_{}.npy".format(data_path, dataset, test_np_filename, anml_tag, anml_gen_size), data_np_nd_anml_label_tw_multiple)
            
        else:
            return anml_gen_dict, data_np_nd_with_anml, data_np_nd_anml_label, data_np_nd_with_anml_t, data_np_nd_anml_label_t, data_np_nd_with_anml_tw_multiple, data_np_nd_anml_label_tw_multiple

class SyntheticDQMChannelAnomalyLoader():
    def __init__(self, test_np_filename, **kwargs):
        self.test_np_filename = test_np_filename
        self.dataset = kwargs.get("dataset", "")
        self.anml_gen_size = kwargs.get("anml_gen_size", "")

    def load_health_ls_used_for_amnl_gen(self, **kwargs):
        anml_tag = kwargs.get("anml_tag", "")
        
        data_np_nd_with_healthy_tw = util.load_npdata("{}/{}/{}_data_np_nd_with_healthy_tw_{}_{}.npy".format(data_path, self.dataset, self.test_np_filename, anml_tag, self.anml_gen_size))
        data_np_nd_healthy_label_tw = util.load_npdata("{}/{}/{}_data_np_nd_healthy_label_tw_{}_{}.npy".format(data_path, self.dataset, self.test_np_filename, anml_tag, self.anml_gen_size))
        print(data_np_nd_with_healthy_tw.shape, data_np_nd_healthy_label_tw.shape)
        return torch.Tensor(data_np_nd_with_healthy_tw), torch.BoolTensor(data_np_nd_healthy_label_tw)
        
    def load_generated_anml(self, **kwargs):
        anml_tag = kwargs.get("anml_tag", "")
        
        data_np_nd_with_anml = util.load_npdata("{}/{}/{}_data_np_nd_with_anml_{}_{}.npy".format(data_path, self.dataset, self.test_np_filename, anml_tag, self.anml_gen_size))
        data_np_nd_anml_label = util.load_npdata("{}/{}/{}_data_np_nd_anml_label_{}_{}.npy".format(data_path, self.dataset, self.test_np_filename, anml_tag, self.anml_gen_size))
        print(data_np_nd_with_anml.shape, data_np_nd_anml_label.shape)
        return torch.Tensor(data_np_nd_with_anml), torch.BoolTensor(data_np_nd_anml_label)
        
    def load_anml_gen_dict(self, **kwargs):
        anml_tag = kwargs.get("anml_tag", "")
        anml_gen_dict = util.load_pickle("{}/{}/{}_anml_gen_dict_{}_{}.pkl".format(data_path, self.dataset, self.test_np_filename, anml_tag, self.anml_gen_size))
        anml_gen_config = util.load_json("{}/{}/{}_anml_kwargs_{}_{}.json".format(data_path, self.dataset, self.test_np_filename, anml_tag, self.anml_gen_size))
        return anml_gen_dict, anml_gen_config


def dqm_synthetic_anomaly_generator(**kwargs):
    """
    input dataset size: [ls X t X ieta X iphi X depth X feature]
    """
    print(kwargs)
    subdetector_name = kwargs.pop("subdetector_name", "he")
    mask = kwargs.get("mask", None)
    dataset = kwargs.get("dataset", None)
    test_np_filename = kwargs.get("test_np_filename", None) # model_test_data_ts:  [ls X ieta X iphi X depth X feature]
    memorysize = kwargs.get("memorysize", 5) # timewindow size for temporal models
    anml_gen_size = kwargs.get("anml_gen_size", 1000) # number of randomly selected LS
    anml_strength = kwargs.get("anml_strength", [2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]) # list of anomaly strength
    seed = kwargs.get("seed", 101)
    issave = kwargs.get("issave", True)
    anml_tag = kwargs.get("anml_tag", "hot_degard_dead_channel")

    objAnmlGen = SyntheticDQMChannelAnomalyGenerator(subdetector_name=subdetector_name, mask_keep=~mask, **kwargs)

    objAnmlGen.generate_anml_spatial_temporal(
                                    test_np_filename, 
                                    dataset=dataset,
                                    anml_tag=anml_tag,
                                    memorysize=memorysize, 
                                    # mask_keep=~mask, 
                                    skip_init_ls=memorysize,
                                    anml_strength=anml_strength,
                                    anml_gen_size=anml_gen_size,
                                    seed=seed,
                                    issave=issave
                                    )


if __name__ == '__main__':
    """Main entry function."""

    parser = argparse.ArgumentParser(description="dqm synthetic anomaly generator")
    
    parser.add_argument('-s', '--subdetector_name', type=str,  
                        default="",
                        help='data source selection')
    parser.add_argument('-d', '--datatype', type=str, choices=['iid', 'ts'], default="ts",
                        help='data type selection')
    parser.add_argument('-m', '--memorysize', type=int, default=5,
                        help='model temporal memory size')
    parser.add_argument('-ds', '--dataset', type=str, default="HEHB",
                        help='data source selection')
    parser.add_argument('-dp', '--data_path', type=str, default=data_path,
                        help='data_path')
    parser.add_argument('-gl', '--anml_gen_size', type=int, default=1000,
                    help='anml_gen_size')   
    parser.add_argument('-gsd', '--seed', type=int, default=101,
                    help='seed')   
    parser.add_argument('-df', '--test_np_filename', type=str, default=None,
                        help='test_np_filename')           
    parser.add_argument('-tg', '--anml_tag', type=str, default="anomaly_channel",
                        help='anml_tag')   
    parser.add_argument('--save', action='store_true',
                        help='save', default=False) 

    args = parser.parse_args()
    print(args)

    try:
        subdetector_mask = util.load_npdata(rf"{args.data_path}\HCAL_CONFIG\{args.subdetector_name}_segmentation_config_mask.npy").astype(bool)
        print(f"hcal segmentation map is loaded: {subdetector_mask.shape}")
    except Exception as ex:
        print(ex)
        subdetector_mask = None

    anml_strength = [2.0, 0.8, 0.6, 0.4, 0.2, 0.0] # noisy-hot channel, degarded channels, dead channel

    args = vars(args)
    dqm_synthetic_anomaly_generator(mask=subdetector_mask, anml_strength=anml_strength, **args)


'''
!python synthetic_anomly_generator.py -s he -ds HEHB -m 5 -df model_test_data_ts -gl 1000 -gsd 101 -tg anomaly_channel
'''
