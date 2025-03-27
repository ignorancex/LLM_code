import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
import scipy.sparse as sp
from tqdm import tqdm
import lib.utils as utils
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                          GlobalStorage, NodeStorage)
from torch_geometric.typing import EdgeType, NodeType, OptTensor

class IndexedData(Data):
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
            edge_attr: OptTensor = None, y: OptTensor = None,
            pos: OptTensor = None, id=None , **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if pos is not None:
            self.pos = pos
        if id is not None:
            self.id = id

        for key, value in kwargs.items():
            setattr(self, key, value)


def encode_onehot(labels):
    label_list = labels.tolost()
    #while isinstance(label_list, list):

    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

class ParseData(object):

    def __init__(self, dataset_path,args,suffix='_springs5',mode="interp"):
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.suffix_all = args.suffix_all
        self.baseline = args.baseline
        self.mode = mode
        self.random_seed = args.random_seed
        self.args = args
        self.total_step = args.total_ode_step
        self.cutting_edge = args.cutting_edge
        self.num_pre = args.extrap_num

        self.max_loc = None
        self.min_loc = None
        self.max_vel = None
        self.min_vel = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


    def load_data_joint_(self,args,sample_percent,batch_size,data_type="train", customize_cut_num = None, gem_grad_clip=False, task_id=None,mask_select=False):
        self.args = args
        self.batch_size,self.sample_percent = batch_size,sample_percent
        cut_num = 20000 if data_type == "train" else 5000
        self.observed_extrap_percent = args.extrap_percent_train if data_type=='train' else args.extrap_percent_test
        split_data_func = self.split_data #if args.sampledata_load else self.split_data_no_sample

        if customize_cut_num is not None:
            cut_num = customize_cut_num

        # Loading Data
        # gem_clip_grad is used in GEM to load data and computes the clip on the gradients
        # load multiple task data
        end_tid = task_id if gem_grad_clip else len(self.suffix_all)  # +1
        locs, vels, edgess, timess, edgess_NRI = [], [], [], [], []
        loc_observed_all,vel_observed_all,times_observed_all,edges_all, series_list_de_all = [], [], [], [], []
        for s in self.suffix_all[0:end_tid]:
            interp_extrap_func = self.interp_extrap_VCell if s[0:3] in ['VCe'] else self.interp_extrap
            loc = np.load(self.dataset_path + '/loc_' + data_type + s + '.npy', allow_pickle=True)[:cut_num]
            locs.append(np.load(self.dataset_path + '/loc_' + data_type + s + '.npy', allow_pickle=True)[:cut_num])
            vel = None if s[0:3] in ['_nt', 'NRW', 'VCe'] else np.load(self.dataset_path + '/vel_' + data_type + s + '.npy',
                                  allow_pickle=True)[:cut_num]
            vels.append(vel)

            self.num_graph = loc.shape[0]
            self.num_atoms = loc.shape[1]
            self.feature = loc[0][0][0].shape[0] if s[0:3] in ['_nt', 'NRW', 'VCe'] else \
                loc[0][0][0].shape[0] + vel[0][0][0].shape[0]

            edges = np.load(self.dataset_path + '/edges_' + data_type + s + '.npy',allow_pickle=True)[:cut_num]
            [edgess.append(e) for e in edges]

            # Graph Dataloader --USING NRI
            edges = np.reshape(edges, [-1, self.num_atoms ** 2])
            edges = np.array((edges + 1) / 2, dtype=np.int64) if not args.keep_original_edge_types else encode_onehot(
                edges)  # binarize, only values larger than 1 can have positive values
            edges = torch.LongTensor(edges)  # ndarray
            # Exclude self edges
            off_diag_idx = np.ravel_multi_index(
                np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
                [self.num_atoms, self.num_atoms])
            edges = edges[:, off_diag_idx]

            [edgess_NRI.append(e) for e in edges]

            times = np.load(self.dataset_path + '/times_' + data_type + s + '.npy', allow_pickle=True)[:cut_num]
            timess.append(np.load(self.dataset_path + '/times_' + data_type + s + '.npy', allow_pickle=True)[:cut_num])

            if s[0:8] in ["_springs", "_charged"]:
                # Normalize features to [-1, 1], across test and train dataset
                if self.max_loc == None:
                    loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
                    vel, self.max_vel, self.min_vel = self.normalize_features(vel, self.num_atoms)
                else:
                    loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
                    vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1

            elif s[0:4] == '_ntu':
                self.timelength = 300
            elif s[0:3] == 'NRW':
                # Normalize features to [-1, 1], across test and train dataset
                if self.max_loc == None:
                    loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
                else:
                    loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
            elif s[0:3] == 'VCe':
                # Normalize features to [-1, 1], across test and train dataset
                loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms, normalize_mode=args.normalizeVCellfeat)  # [num_sims,num_atoms, (timestamps,2)]

            # split data w.r.t interp and extrap, also normalize times
            if self.mode == "in":
                loc_en, vel_en, times_en = interp_extrap_func(loc, vel, times, self.mode, data_type)
                loc_de, vel_de, times_de = loc_en, vel_en, times_en
            elif self.mode == "ex":
                loc_en, vel_en, times_en, loc_de, vel_de, times_de = interp_extrap_func(loc, vel, times, self.mode,
                                                                                        data_type,
                                                                                        mask_select=mask_select)
            # Encoder dataloader
            series_list_observed, loc_observed, vel_observed, times_observed = split_data_func(loc_en, vel_en,
                                                                                               times_en)  # full observation in series..., sample partial observations into the other

            # Decoder Dataloader
            if self.mode == "in":
                series_list_de = series_list_observed
            elif self.mode == "ex":
                # self.decoder concats all balls of all trajs together
                series_list_de = self.decoder_data(loc_de, vel_de, times_de)  # uniform the lengths and padding
            series_list_de_all += series_list_de
            [loc_observed_all.append(l) for l in loc_observed]
            if vel_observed is None:
                vel_observed_all=None
            else:
                [vel_observed_all.append(l) for l in vel_observed]
            [times_observed_all.append(l) for l in times_observed]
            #vel_observed_all+=vel_observed
            #times_observed_all+=times_observed


        if self.mode == "in":
            time_begin = 0
        else:
            time_begin = 1
        loc_observed_all = np.asarray(loc_observed_all)
        vel_observed_all = np.asarray(vel_observed_all) if vel_observed_all is not None else None
        times_observed_all = np.asarray(times_observed_all)
        edges_all = np.asarray(edgess)
        edgess_NRI = np.asarray(edgess_NRI)
        encoder_data_loader, graph_data_loader = self.transfer_data(loc_observed_all, vel_observed_all, edges_all,
                                                                    times_observed_all, time_begin=time_begin) # transfer the data into data of each trajectory, so that can used in dataloader

        # Graph Dataloader --USING NRI
        #graph_data_loader = Loader(edgess_NRI, batch_size=self.batch_size)
        graph_data_loader = DataLoader(edgess_NRI, batch_size=self.batch_size)

        # Decoder Dataloader
        decoder_data_loader = Loader(series_list_de_all, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]

        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch

    def load_data_joint(self,args,sample_percent,batch_size,data_type="train", customize_cut_num = None,task_id=None,mask_select=False):
        encoder_data_loaders, decoder_data_loaders, graph_data_loaders, num_batchs = [],[],[],[]
        for suffix in self.suffix_all:
            encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch = self.load_data_single_task(args,sample_percent,batch_size, suffix, data_type=data_type, customize_cut_num = customize_cut_num)
            encoder_data_loaders.append(encoder_data_loader)
            decoder_data_loaders.append(decoder_data_loader)
            graph_data_loaders.append(graph_data_loader)
            num_batchs.append(num_batch)
        return encoder_data_loaders, decoder_data_loaders, graph_data_loaders, num_batchs

    def load_data_bc(self,args,sample_percent,batch_size,data_type="train", customize_cut_num = None,task_id=None,mask_select=False):
        encoder_data_loaders, decoder_data_loaders, graph_data_loaders, num_batchs = [],[],[],[]
        for i,suffix in enumerate(self.suffix_all):
            if args.method == 'er':
                customize_cut_num = args.er_args_buffered_ids[i]
            encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch = self.load_data_single_task(args,sample_percent,batch_size, suffix, data_type=data_type, customize_cut_num = customize_cut_num)
            encoder_data_loaders.append(encoder_data_loader)
            decoder_data_loaders.append(decoder_data_loader)
            graph_data_loaders.append(graph_data_loader)
            num_batchs.append(num_batch)
        return encoder_data_loaders, decoder_data_loaders, graph_data_loaders, num_batchs


    def load_data_single_task(self,args,sample_percent,batch_size, suffix, data_type="train", customize_cut_num = None,mask_select=False):
        self.args = args
        self.batch_size,self.sample_percent = batch_size,sample_percent
        cut_num = 20000 if data_type == "train" else 5000
        self.observed_extrap_percent = args.extrap_percent_train if data_type=='train' else args.extrap_percent_test
        interp_extrap_func = self.interp_extrap_VCell if suffix[0:3] in ['VCe'] else self.interp_extrap
        split_data_func = self.split_data #if args.sampledata_load else self.split_data_no_sample

        if customize_cut_num is not None:
            selected_ids = list(range(customize_cut_num)) if isinstance(customize_cut_num, int) else customize_cut_num

        # Loading Data
        # only load current task data
        loc = np.load(self.dataset_path + '/loc_' + data_type + suffix + '.npy', allow_pickle=True)[selected_ids]  # n_trajs * n_balls * [traj_length,2]
        vel = None if suffix[0:3] in ['_nt', 'NRW', 'VCe'] else np.load(
                self.dataset_path + '/vel_' + data_type + suffix + '.npy', allow_pickle=True)[selected_ids]  # n_trajs * n_balls * [traj_length,2]
        edges = np.load(self.dataset_path + '/edges_' + data_type + suffix + '.npy', allow_pickle=True)[selected_ids]  # n_trajs * n_balls * n_balls
        times = np.load(self.dataset_path + '/times_' + data_type + suffix + '.npy', allow_pickle=True)[selected_ids]  # n_trajs * n_balls * [traj_length], int64

        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature =loc[0][0][0].shape[0] if suffix[0:3] in ['_nt','NRW','VCe'] else loc[0][0][0].shape[0] + vel[0][0][0].shape[0]

        if suffix[0:8] in ["_springs", "_charged"]:
            # Normalize features to [-1, 1], across test and train dataset
            loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
            vel, self.max_vel, self.min_vel = self.normalize_features(vel, self.num_atoms)
        elif suffix[0:4]=='_ntu':
            self.timelength = 300
        elif suffix[0:3]=='NRW':
            # Normalize features to [-1, 1], across test and train dataset
            loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
        elif suffix[0:3]=='VCe':
            # Normalize features to [-1, 1], across test and train dataset
            loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms, normalize_mode=args.normalizeVCellfeat)  # [num_sims,num_atoms, (timestamps,2)]

        # split data w.r.t interp and extrap, also normalize times
        if self.mode=="in":
            loc_en,vel_en,times_en = interp_extrap_func(loc,vel,times,self.mode,data_type)
            loc_de,vel_de,times_de = loc_en,vel_en,times_en
        elif self.mode == "ex":
            loc_en,vel_en,times_en,loc_de,vel_de,times_de = interp_extrap_func(loc,vel,times,self.mode,data_type,mask_select=mask_select)

        #Encoder dataloader
        series_list_observed, loc_observed, vel_observed, times_observed = split_data_func(loc_en, vel_en, times_en) # full observation in series..., sample partial observations into the other
        if self.mode == "in":
            time_begin = 0
        else:
            time_begin = 1
        encoder_data_loader, graph_data_loader = self.transfer_data(loc_observed, vel_observed, edges,
                                                                    times_observed, time_begin=time_begin) # transfer the data into data of each trajectory, so that can used in dataloader


        # Graph Dataloader --USING NRI
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64) if not args.keep_original_edge_types else encode_onehot(edges)# binarize, only values larger than 1 can have positive values
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])

        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size)

        # Decoder Dataloader
        if self.mode=="in":
            series_list_de = series_list_observed
        elif self.mode == "ex":
            # self.decoder concats all balls of all trajs together
            series_list_de = self.decoder_data(loc_de,vel_de,times_de) # uniform the lengths and padding
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]

        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch


    def load_data(self,args,sample_percent,batch_size,data_type="train", customize_cut_num = None, gem_grad_clip=False, task_id=None,mask_select=False):
        self.args = args
        self.batch_size,self.sample_percent = batch_size,sample_percent
        cut_num = 20000 if data_type == "train" else 5000
        self.observed_extrap_percent = args.extrap_percent_train if data_type=='train' else args.extrap_percent_test
        interp_extrap_func = self.interp_extrap_VCell if self.suffix[0:3] in ['VCe'] else self.interp_extrap
        split_data_func = self.split_data #if args.sampledata_load else self.split_data_no_sample

        if customize_cut_num is not None:
            cut_num = customize_cut_num

        # Loading Data
        # gem_clip_grad is used in GEM to load data and computes the clip on the gradients
        if (self.baseline in ['er','joint'] and data_type=='train') or (gem_grad_clip and data_type=='train'):
            # load multiple task data
            end_tid = task_id if gem_grad_clip else len(self.suffix_all) #+1
            loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix_all[0] + '.npy', allow_pickle=True)[:cut_num]
            vel = None if self.suffix[0:3] in ['_nt','NRW','VCe'] else np.load(self.dataset_path + '/vel_' + data_type + self.suffix_all[0] + '.npy', allow_pickle=True)[
                  :cut_num] # vel is useless for ntu
            edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix_all[0] + '.npy', allow_pickle=True)[
                    :cut_num]
            times = np.load(self.dataset_path + '/times_' + data_type + self.suffix_all[0] + '.npy', allow_pickle=True)[
                    :cut_num]
            for s in self.suffix_all[1:end_tid]:
                loc = np.concatenate((loc, np.load(self.dataset_path + '/loc_' + data_type + s + '.npy',
                                                   allow_pickle=True)[:cut_num]), 0)
                vel = None if self.suffix[0:3] in ['_nt','NRW','VCe'] else np.concatenate((vel, np.load(self.dataset_path + '/vel_' + data_type + s + '.npy',
                                                   allow_pickle=True)[:cut_num]), 0)
                edges = np.concatenate((edges, np.load(self.dataset_path + '/edges_' + data_type + s + '.npy',
                                                       allow_pickle=True)[:cut_num]), 0)
                times = np.concatenate((times, np.load(self.dataset_path + '/times_' + data_type + s + '.npy',
                                                       allow_pickle=True)[:cut_num]), 0)
        else:
            # only load current task data
            loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num] # n_trajs * n_balls * [traj_length,2]
            vel = None if self.suffix[0:3] in ['_nt','NRW','VCe'] else np.load(self.dataset_path + '/vel_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num] # n_trajs * n_balls * [traj_length,2]
            edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # n_trajs * n_balls * n_balls
            times = np.load(self.dataset_path + '/times_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # n_trajs * n_balls * [traj_length], int64

        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature =loc[0][0][0].shape[0] if self.suffix[0:3] in ['_nt','NRW','VCe'] else loc[0][0][0].shape[0] + vel[0][0][0].shape[0]

        if self.suffix[0:8] in ["_springs", "_charged"]:
            # Normalize features to [-1, 1], across test and train dataset
            if self.max_loc == None:
                loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
                vel, self.max_vel, self.min_vel = self.normalize_features(vel, self.num_atoms)
            else:
                loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
                vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1

        elif self.suffix[0:4]=='_ntu':
            self.timelength = 300
        elif self.suffix[0:3]=='NRW':
            # Normalize features to [-1, 1], across test and train dataset
            if self.max_loc == None:
                loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
            else:
                loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
        elif self.suffix[0:3]=='VCe':
            # Normalize features to [-1, 1], across test and train dataset
            if self.max_loc == None:
                loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms, normalize_mode=args.normalizeVCellfeat)  # [num_sims,num_atoms, (timestamps,2)]
            else:
                if args.normalizeVCellfeat:
                    loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1

        # split data w.r.t interp and extrap, also normalize times
        if self.mode=="in":
            loc_en,vel_en,times_en = interp_extrap_func(loc,vel,times,self.mode,data_type)
            loc_de,vel_de,times_de = loc_en,vel_en,times_en
        elif self.mode == "ex":
            loc_en,vel_en,times_en,loc_de,vel_de,times_de = interp_extrap_func(loc,vel,times,self.mode,data_type,mask_select=mask_select)

        #Encoder dataloader
        series_list_observed, loc_observed, vel_observed, times_observed = split_data_func(loc_en, vel_en, times_en) # full observation in series..., sample partial observations into the other
        if self.mode == "in":
            time_begin = 0
        else:
            time_begin = 1
        encoder_data_loader, graph_data_loader = self.transfer_data(loc_observed, vel_observed, edges,
                                                                    times_observed, time_begin=time_begin) # transfer the data into data of each trajectory, so that can used in dataloader


        # Graph Dataloader --USING NRI
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64) if not args.keep_original_edge_types else encode_onehot(edges)# binarize, only values larger than 1 can have positive values
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])

        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size)

        # Decoder Dataloader
        if self.mode=="in":
            series_list_de = series_list_observed
        elif self.mode == "ex":
            # self.decoder concats all balls of all trajs together
            series_list_de = self.decoder_data(loc_de,vel_de,times_de) # uniform the lengths and padding
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]

        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch

    def load_data_lode(self,args,sample_percent,batch_size,data_type="train", customize_cut_num = None, gem_grad_clip=False, task_id=None,mask_select=False):
        # specialized for Latent-ODE
        self.args = args
        self.batch_size,self.sample_percent = batch_size,sample_percent
        cut_num = 20000 if data_type == "train" else 5000
        self.observed_extrap_percent = args.extrap_percent_train if data_type=='train' else args.extrap_percent_test
        interp_extrap_func = self.interp_extrap_VCell if self.suffix[0:3] in ['VCe'] else self.interp_extrap
        split_data_func = self.split_data #if args.sampledata_load else self.split_data_no_sample

        if customize_cut_num is not None:
            cut_num = customize_cut_num

        # Loading Data
        # gem_clip_grad is used in GEM to load data and computes the clip on the gradients
        if (self.baseline in ['er','joint'] and data_type=='train') or (gem_grad_clip and data_type=='train'):
            # load multiple task data
            end_tid = task_id if gem_grad_clip else len(self.suffix_all) #+1
            loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix_all[0] + '.npy', allow_pickle=True)[:cut_num]
            vel = None if self.suffix[0:3] in ['_nt','NRW','VCe'] else np.load(self.dataset_path + '/vel_' + data_type + self.suffix_all[0] + '.npy', allow_pickle=True)[
                  :cut_num] # vel is useless for ntu
            edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix_all[0] + '.npy', allow_pickle=True)[
                    :cut_num]
            times = np.load(self.dataset_path + '/times_' + data_type + self.suffix_all[0] + '.npy', allow_pickle=True)[
                    :cut_num]
            for s in self.suffix_all[1:end_tid]:
                loc = np.concatenate((loc, np.load(self.dataset_path + '/loc_' + data_type + s + '.npy',
                                                   allow_pickle=True)[:cut_num]), 0)
                vel = None if self.suffix[0:3] in ['_nt','NRW','VCe'] else np.concatenate((vel, np.load(self.dataset_path + '/vel_' + data_type + s + '.npy',
                                                   allow_pickle=True)[:cut_num]), 0)
                edges = np.concatenate((edges, np.load(self.dataset_path + '/edges_' + data_type + s + '.npy',
                                                       allow_pickle=True)[:cut_num]), 0)
                times = np.concatenate((times, np.load(self.dataset_path + '/times_' + data_type + s + '.npy',
                                                       allow_pickle=True)[:cut_num]), 0)
        else:
            # only load current task data
            loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num] # n_trajs * n_balls * [traj_length,2]
            vel = None if self.suffix[0:3] in ['_nt','NRW','VCe'] else np.load(self.dataset_path + '/vel_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num] # n_trajs * n_balls * [traj_length,2]
            edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # n_trajs * n_balls * n_balls
            times = np.load(self.dataset_path + '/times_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # n_trajs * n_balls * [traj_length], int64

        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature =loc[0][0][0].shape[0] if self.suffix[0:3] in ['_nt','NRW','VCe'] else loc[0][0][0].shape[0] + vel[0][0][0].shape[0]

        if self.suffix[0:8] in ["_springs", "_charged"]:
            # Normalize features to [-1, 1], across test and train dataset
            if self.max_loc == None:
                loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
                vel, self.max_vel, self.min_vel = self.normalize_features(vel, self.num_atoms)
            else:
                loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
                vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1

        elif self.suffix[0:4]=='_ntu':
            self.timelength = 300
        elif self.suffix[0:3]=='NRW':
            # Normalize features to [-1, 1], across test and train dataset
            if self.max_loc == None:
                loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
            else:
                loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
        elif self.suffix[0:3]=='VCe':
            # Normalize features to [-1, 1], across test and train dataset
            if self.max_loc == None:
                loc, self.max_loc, self.min_loc = self.normalize_features(loc, self.num_atoms, normalize_mode=args.normalizeVCellfeat)  # [num_sims,num_atoms, (timestamps,2)]
            else:
                if args.normalizeVCellfeat:
                    loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1

        # split data w.r.t interp and extrap, also normalize times
        loc = loc.reshape(self.num_graph*self.num_atoms,-1)
        vel = None if vel is None else vel.reshape(self.num_graph*self.num_atoms,-1)
        if self.mode=="in":
            loc_en,vel_en,times_en = interp_extrap_func(loc,vel,times,self.mode,data_type)
            loc_de,vel_de,times_de = loc_en,vel_en,times_en
        elif self.mode == "ex":
            loc_en,vel_en,times_en,loc_de,vel_de,times_de = interp_extrap_func(loc,vel,times,self.mode,data_type,mask_select=mask_select)

        #Encoder dataloader
        series_list_observed, loc_observed, vel_observed, times_observed = split_data_func(loc_en, vel_en, times_en) # full observation in series..., sample partial observations into the other
        if self.mode == "in":
            time_begin = 0
        else:
            time_begin = 1
        encoder_data_loader, graph_data_loader = self.transfer_data(loc_observed, vel_observed, edges,
                                                                    times_observed, time_begin=time_begin) # transfer the data into data of each trajectory, so that can used in dataloader


        # Graph Dataloader --USING NRI
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64) if not args.keep_original_edge_types else encode_onehot(edges)# binarize, only values larger than 1 can have positive values
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])

        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size)

        # Decoder Dataloader
        if self.mode=="in":
            series_list_de = series_list_observed
        elif self.mode == "ex":
            # self.decoder concats all balls of all trajs together
            series_list_de = self.decoder_data(loc_de,vel_de,times_de) # uniform the lengths and padding
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]

        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch


    def interp_extrap(self,loc,vel,times,mode,data_type,mask_select=False):
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel) if vel is not None else None # no vel in motion data
        times_observed = np.ones_like(times)
        if mode =="in":
            if data_type== "test":
                # remove the data for testing in extrapolation setting.
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre] if self.args.system=='physics' else loc[i][j]
                        if vel is not None:
                            vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre] if self.args.system=='physics' else times[i][j]
                return loc_observed,vel_observed,times_observed/self.total_step
            else:
                return loc,vel,times/self.total_step

        elif mode == "ex":# split into 2 parts and normalize t seperately
            loc_extrap = np.ones_like(loc)
            vel_extrap = np.ones_like(vel)
            times_extrap = np.ones_like(times)

            norm_timelength = self.total_step if self.args.system == 'physics' else self.total_step / 2

            if data_type == "test":
                if mask_select:
                    # for mask selection, only select half of the first part of the data
                    for i in range(self.num_graph):
                        for j in range(self.num_atoms):
                            times_current = times[i][j]
                            split_id = self.total_step//2 if self.args.system == 'physics' else times_current.shape[0] // 4
                            end_id = self.total_step if self.args.system == 'physics' else times_current.shape[0] // 2
                            times_current_mask = np.where(times_current < split_id, times_current, 0)
                            num_observe_current = np.argmax(times_current_mask) + 1
                            loc_observed[i][j] = loc[i][j][:num_observe_current]
                            if vel is not None:
                                vel_observed[i][j] = vel[i][j][:num_observe_current]
                            times_observed[i][j] = times[i][j][:num_observe_current]

                            loc_extrap[i][j] = loc[i][j][num_observe_current:end_id]
                            if vel is not None:
                                vel_extrap[i][j] = vel[i][j][num_observe_current:end_id]
                            times_extrap[i][j] = times[i][j][num_observe_current:end_id]-split_id
                    times_observed = times_observed / norm_timelength
                    times_extrap = times_extrap / norm_timelength
                else:
                    for i in range(self.num_graph):
                        for j in range(self.num_atoms):
                            split_id = self.num_pre if self.args.system == 'physics' else times[i][j].shape[0] // 2
                            offset = norm_timelength if self.args.system == 'physics' else split_id
                            loc_observed[i][j] = loc[i][j][:-split_id]
                            if vel is not None:
                                vel_observed[i][j] = vel[i][j][:-split_id]
                            times_observed[i][j] = times[i][j][:-split_id]

                            loc_extrap[i][j] = loc[i][j][-split_id:]
                            if vel is not None:
                                vel_extrap[i][j] = vel[i][j][-split_id:]
                            times_extrap[i][j] = times[i][j][-split_id:]- offset #to make extrap and observation all start from 0
                    times_observed = times_observed / norm_timelength
                    times_extrap = times_extrap/ norm_timelength  # 40 extrap steps also sampled from 60 steps, during generation
            else:
                #norm_timelength = self.total_step if self.args.system == 'physics' else self.total_step / 2
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        times_current = times[i][j]
                        split_id = self.total_step//2 if self.args.system == 'physics' else times_current.shape[0] // 2
                        times_current_mask = np.where(times_current<split_id,times_current,0)
                        num_observe_current = np.argmax(times_current_mask)+1

                        loc_observed[i][j] = loc[i][j][:num_observe_current]
                        if vel is not None:
                            vel_observed[i][j] = vel[i][j][:num_observe_current]
                        times_observed[i][j] = times[i][j][:num_observe_current]

                        loc_extrap[i][j] = loc[i][j][num_observe_current:]
                        if vel is not None:
                            vel_extrap[i][j] = vel[i][j][num_observe_current:]
                        times_extrap[i][j] = times[i][j][num_observe_current:] - split_id # for the normalization below

                times_observed = times_observed/norm_timelength
                #times_extrap = (times_extrap - split_id) / norm_timelength
                times_extrap = times_extrap / norm_timelength

            return loc_observed,vel_observed,times_observed,loc_extrap,vel_extrap,times_extrap
    def interp_extrap_VCell(self,loc,vel,times,mode,data_type,mask_select=False):
        loc_observed = np.ones_like(loc)
        vel_observed = None # no vel in vcell data
        times_observed = np.ones_like(times)
        if mode == "ex":# split into 2 parts and normalize t seperately
            loc_extrap = np.ones_like(loc)
            vel_extrap = np.ones_like(vel)
            times_extrap = np.ones_like(times)

            norm_timelength = 0.0
            for i in range(self.num_graph):
                for j in range(self.num_atoms):
                    #observation_length = int(round(times[i][j].shape[0]*self.observed_extrap_percent))
                    #times_current = times[i][j][0:observation_length]
                    norm_timelength = max(norm_timelength,times[i][j].max())
            norm_timelength = norm_timelength if self.args.normalizeVCelltime else 1.

            if data_type == "test":
                if mask_select:
                    # for mask selection, only select half of the first part of the data
                    for i in range(self.num_graph):
                        for j in range(self.num_atoms):
                            times_current = times[i][j]
                            split_id = int(times_current.shape[0] * self.observed_extrap_percent * self.observed_extrap_percent)
                            end_id = int(times_current.shape[0] * self.observed_extrap_percent)
                            times_id = np.arange(times_current.shape[0])
                            times_current_mask = np.where(times_id < split_id, times_current, 0)
                            num_observe_current = np.argmax(times_current_mask) + 1
                            loc_observed[i][j] = loc[i][j][:num_observe_current]
                            times_observed[i][j] = times[i][j][:num_observe_current]

                            loc_extrap[i][j] = loc[i][j][num_observe_current:end_id]
                            offset = times_current[split_id] if self.args.offset_extrap else 0.
                            times_extrap[i][j] = times[i][j][num_observe_current:end_id]-offset
                    times_observed = times_observed / norm_timelength
                    times_extrap = times_extrap / norm_timelength
                else:
                    for i in range(self.num_graph):
                        for j in range(self.num_atoms):
                            split_id = int(times[i][j].shape[0] * self.observed_extrap_percent)
                            offset = times[i][j][split_id] if self.args.offset_extrap else 0.
                            loc_observed[i][j] = loc[i][j][:split_id]

                            times_observed[i][j] = times[i][j][:split_id]

                            loc_extrap[i][j] = loc[i][j][split_id:]
                            times_extrap[i][j] = times[i][j][split_id:]- offset
                    times_observed = times_observed / norm_timelength
                    times_extrap = times_extrap/ norm_timelength  # 40 extrap steps also sampled from 60 steps
            else:
                #norm_timelength = self.total_step if self.args.system == 'physics' else self.total_step / 2
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        times_current = times[i][j]
                        split_id = int(times_current.shape[0] * self.observed_extrap_percent)
                        times_id = np.arange(times_current.shape[0])
                        times_current_mask = np.where(times_id<split_id,times_current,0)
                        num_observe_current = np.argmax(times_current_mask)+1 # should == split_id

                        loc_observed[i][j] = loc[i][j][:num_observe_current]
                        times_observed[i][j] = times[i][j][:num_observe_current]

                        loc_extrap[i][j] = loc[i][j][num_observe_current:]
                        offset = times[i][j][split_id] if self.args.offset_extrap else 0.
                        times_extrap[i][j] = times[i][j][num_observe_current:] - offset # for the normalization below

                times_observed = times_observed/norm_timelength
                #times_extrap = (times_extrap - split_id) / norm_timelength
                times_extrap = times_extrap / norm_timelength

            return loc_observed,vel_observed,times_observed,loc_extrap,vel_extrap,times_extrap

    def split_data(self,loc,vel,times):
        # generate partial observations by sampling
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel) if vel is not None else None
        times_observed = np.ones_like(times)

        # split encoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j][1:])  # [2500] num_train * num_ball
                if vel is not None:
                    vel_list.append(vel[i][j][1:])
                times_list.append(times[i][j][1:])

        series_list = []
        odernn_list = []
        for i, loc_series in enumerate(loc_list):
            # for encoder data
            graph_index = i // self.num_atoms
            atom_index = i % self.num_atoms
            length = len(loc_series)
            preserved_idx = sorted(
                np.random.choice(np.arange(length), int(length * self.sample_percent), replace=False))
            loc_observed[graph_index][atom_index] = loc_series[preserved_idx]
            if vel is not None:
                vel_observed[graph_index][atom_index] = vel_list[i][preserved_idx]
            times_observed[graph_index][atom_index] = times_list[i][preserved_idx]
            '''
            # for odernn encoder
            feature_observe = np.zeros((self.timelength, self.feature))  # [T,D]
            times_observe = -1 * np.ones(self.timelength)  # maximum #[T], padding -1
            mask_observe = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_observe[:len(times_list[i][preserved_idx])] = times_list[i][preserved_idx]
            feature_observe[:len(times_list[i][preserved_idx])] = np.concatenate(
                (loc_series[preserved_idx], vel_list[i][preserved_idx]), axis=1) if vel is not None else loc_series[preserved_idx]
            mask_observe[:len(times_list[i][preserved_idx])] = 1

            tt_observe = torch.FloatTensor(times_observe)
            vals_observe = torch.FloatTensor(feature_observe)
            masks_observe = torch.FloatTensor(mask_observe)

            odernn_list.append((tt_observe, vals_observe, masks_observe))
            '''
            # for decoder data, padding and mask
            # initialize
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D], self.timelength is set in self.normalize()
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed
            # fill in with values
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1) if vel is not None else loc_series
            times_predict[:len(times_list[i])] = times_list[i]
            mask_predict[:len(times_list[i])] = 1

            vals = torch.FloatTensor(feature_predict)
            tt = torch.FloatTensor(times_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list, loc_observed, vel_observed, times_observed

    def split_data_no_sample(self,loc,vel,times):
        # generate partial observations by sampling
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel) if vel is not None else None
        times_observed = np.ones_like(times)

        # split encoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j][1:])  # [2500] num_train * num_ball
                if vel is not None:
                    vel_list.append(vel[i][j][1:])
                times_list.append(times[i][j][1:])

        series_list = []
        odernn_list = []
        for i, loc_series in enumerate(loc_list):
            # for encoder data
            graph_index = i // self.num_atoms
            atom_index = i % self.num_atoms
            loc_observed[graph_index][atom_index] = loc_series
            if vel is not None:
                vel_observed[graph_index][atom_index] = vel_list[i]
            times_observed[graph_index][atom_index] = times_list[i]
            '''
            # for odernn encoder
            feature_observe = np.zeros((self.timelength, self.feature))  # [T,D]
            times_observe = -1 * np.ones(self.timelength)  # maximum #[T], padding -1
            mask_observe = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_observe[:len(times_list[i][preserved_idx])] = times_list[i][preserved_idx]
            feature_observe[:len(times_list[i][preserved_idx])] = np.concatenate(
                (loc_series[preserved_idx], vel_list[i][preserved_idx]), axis=1) if vel is not None else loc_series[preserved_idx]
            mask_observe[:len(times_list[i][preserved_idx])] = 1

            tt_observe = torch.FloatTensor(times_observe)
            vals_observe = torch.FloatTensor(feature_observe)
            masks_observe = torch.FloatTensor(mask_observe)

            odernn_list.append((tt_observe, vals_observe, masks_observe))
            '''
            # for decoder data, padding and mask
            # initialize
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D], self.timelength is set in self.normalize()
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed
            # fill in with values
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1) if vel is not None else loc_series
            times_predict[:len(times_list[i])] = times_list[i]
            mask_predict[:len(times_list[i])] = 1

            vals = torch.FloatTensor(feature_predict)
            tt = torch.FloatTensor(times_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list, loc_observed, vel_observed, times_observed

    def decoder_data(self, loc, vel, times):
        # split decoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j])  # [2500] num_train * num_ball
                if self.suffix[0:8] in ["_springs", "_charged"]:
                    vel_list.append(vel[i][j])
                times_list.append(times[i][j])

        series_list = []
        for i, loc_series in enumerate(loc_list):
            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1) if self.suffix[0:8] in ["_springs", "_charged"] else loc_series
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list

    def transfer_data(self,loc, vel, edges, times, time_begin=0):
        data_list = []
        graph_list = []
        edge_size_list = []

        #for i in tqdm(range(self.num_graph)):
        for i in range(loc.shape[0]):
            data_per_graph, edge_data, edge_size = self.transfer_one_graph(loc[i], vel[i], edges[i], times[i],
                                                                           time_begin=time_begin) if vel is not None else self.transfer_one_graph(loc[i], None, edges[i], times[i],
                                                                           time_begin=time_begin, id=f'{i}')
            data_list.append(data_per_graph)
            graph_list.append(edge_data)
            edge_size_list.append(edge_size)

        #print("average number of edges per graph is %.4f" % np.mean(np.asarray(edge_size_list)))
        data_loader = DataLoader(data_list, batch_size=self.batch_size)
        graph_loader = DataLoader(graph_list, batch_size=self.batch_size)

        return data_loader, graph_loader

    def transfer_one_graph(self,loc, vel, edge, time, time_begin=0, mask=True, forward=False, id=None):
        # Creating x : [N,D]
        # Creating edge_index
        # Creating edge_attr
        # Creating edge_type
        # Creating y: [N], value= num_steps
        # Creeating pos 【N】
        # forward: t0=0;  otherwise: t0=tN/2

        # compute cutting window size:
        if self.cutting_edge:
            if self.suffix[0:8] in ["_springs", "_charged"] or self.suffix[0:3]=='NRW':
                max_gap = (self.total_step - 40*self.sample_percent) /self.total_step #args.total_ode_step 60 sample-percent-train/特殊的他0.6

            elif self.suffix[0:3] in ['VCe']:
                max_gap = (self.total_step - 30 * self.sample_percent) / self.total_step if self.args.VCell_window_size is None else self.args.VCell_window_size

            else:
                max_gap = (self.total_step - 30 * self.sample_percent) / self.total_step
        else:
            max_gap = 100

        if self.mode=="in":
            forward= False
        elif self.mode=="ex":
            forward=True

        num_atoms = loc.shape[0]
        y = np.zeros(num_atoms) # num of observed snapshots for each ball
        x_ = list()
        x_pos_ = list()
        x_pos_int_ = list() #denote the positions with interger instead of decimal times

        # Creating x, y, x_pos
        for i, loc_ball in enumerate(loc):
            vel_ball = vel[i] if self.suffix[0:8] in ["_springs", "_charged"] else None
            time_ball = time[i]

            # Creating y
            y[i] = len(time_ball) # num of observed snapshots for each ball

            # Creating x (concat loc and vel) and x_pos
            xi = np.concatenate((loc_ball, vel_ball),1) if self.suffix[0:8] in ["_springs", "_charged"] else loc_ball
            x_.append(xi)
            x_pos_i = time_ball - time_begin
            x_pos_.append(x_pos_i)
            x_pos_i_int = np.arange(0,y[i]) #denote the positions with interger instead of decimal times
            x_pos_int_.append(x_pos_i_int)


            '''
            # Creating x and x_pos, by traverse each ball's sequence

            node_time = {t:time_ball[t] for t in range(loc_ball.shape[0])}
            for t in range(loc_ball.shape[0]):
                xt_feature = np.concatenate((loc_ball[t], vel_ball[t])) if self.suffix[0:8] in ["_springs", "_charged"] else loc_ball[t]
                x.append(xt_feature)

                x_pos.append(time_ball[t] - time_begin)
                node_time[node_number] = time_ball[t]

                if i not in ball_nodes.keys():
                    ball_nodes[i] = [node_number]
                else:
                    ball_nodes[i].append(node_number)

                node_number += 1


            aa=0
            '''

        x = np.concatenate(x_,0) # concat trajs and velos of all balls along temporal dimension
        x_pos = np.concatenate(x_pos_,0)
        x_pos_int = np.concatenate(x_pos_int_,0)

        # Adding self-loop
        edge_with_self_loop = edge + np.eye(num_atoms, dtype=int)

        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for i in range(len(x_pos))], axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for i in range(len(x_pos))], axis=0) # pairwise temporal distance between each snapshot of each node and every other snapshot of every other node
        edge_time_matrix_int = np.concatenate([np.asarray(x_pos_int).reshape(-1, 1) for i in range(len(x_pos_int))], axis=1) - np.concatenate(
            [np.asarray(x_pos_int).reshape(1, -1) for i in range(len(x_pos_int))], axis=0) # pairwise temporal distance between each snapshot of each node and every other snapshot of every other node
        edge_exist_matrix = np.zeros((len(x_pos), len(x_pos))) # spatial temporal adj

        # connect two observations if their corresponding nodes are connected
        for i in range(num_atoms):
            for j in range(num_atoms):
                if edge_with_self_loop[i][j] != 0: # exact edge values are ignored
                    sender_index_start = int(np.sum(y[:i])) #start of sender ball i
                    sender_index_end = int(sender_index_start + y[i]) # end of sender ball i
                    receiver_index_start = int(np.sum(y[:j]))
                    receiver_index_end = int(receiver_index_start + y[j])
                    if i == j:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = 1
                    else:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = -1

        # remove edges outside a fixed window size
        if self.args.system == 'VCell' and max_gap>1.:
            temporal_connect_matrix = edge_time_matrix_int
        else:
            temporal_connect_matrix = edge_time_matrix
        if mask == None:
            edge_time_matrix = np.where(abs(temporal_connect_matrix)<=max_gap,edge_time_matrix,-2)
            edge_matrix = (edge_time_matrix + 2) * abs(edge_exist_matrix)  # padding 2 to avoid equal time been seen as not exists.
        elif forward == True:  # sender nodes are those whose time is larger. t0 = 0
            edge_time_matrix = np.where((edge_time_matrix >= 0) & (abs(temporal_connect_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)
        elif forward == False:  # sender nodes are those whose time is smaller. t0 = tN/2
            edge_time_matrix = np.where((edge_time_matrix <= 0) & (abs(temporal_connect_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)

        _, edge_attr_same = self.convert_sparse(edge_exist_matrix * edge_matrix)
        edge_is_same = np.where(edge_attr_same > 0, 1, 0).tolist() # whether edge is self-loop

        edge_index, edge_attr = self.convert_sparse(edge_matrix) # indices and attr of entries in the sparsified matrix
        edge_attr = edge_attr - 2
        edge_index_original, _ = self.convert_sparse(edge) # edge value is ignored

        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        edge_is_same = torch.FloatTensor(np.asarray(edge_is_same))

        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)

        graph_index_original = torch.LongTensor(edge_index_original)
        edge_data = Data(x = torch.ones(num_atoms),edge_index = graph_index_original,y=id)

        #graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same)
        graph_data = IndexedData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same, id=id)
        edge_size = edge_index.shape[1]

        return graph_data,edge_data,edge_size

    def transfer_one_graph_lode(self,loc, vel, edge, time, time_begin=0, mask=True, forward=False, id=None):
        # Creating x : [N,D]
        # Creating edge_index
        # Creating edge_attr
        # Creating edge_type
        # Creating y: [N], value= num_steps
        # Creeating pos 【N】
        # forward: t0=0;  otherwise: t0=tN/2

        # compute cutting window size:
        if self.cutting_edge:
            if self.suffix[0:8] in ["_springs", "_charged"] or self.suffix[0:3]=='NRW':
                max_gap = (self.total_step - 40*self.sample_percent) /self.total_step #args.total_ode_step 60 sample-percent-train/特殊的他0.6

            elif self.suffix[0:3] in ['VCe']:
                max_gap = (self.total_step - 30 * self.sample_percent) / self.total_step if self.args.VCell_window_size is None else self.args.VCell_window_size

            else:
                max_gap = (self.total_step - 30 * self.sample_percent) / self.total_step
        else:
            max_gap = 100

        if self.mode=="in":
            forward= False
        elif self.mode=="ex":
            forward=True

        num_atoms = loc.shape[0]
        y = np.zeros(num_atoms) # num of observed snapshots for each ball
        x_ = list()
        x_pos_ = list()
        x_pos_int_ = list() #denote the positions with interger instead of decimal times

        # Creating x, y, x_pos
        for i, loc_ball in enumerate(loc):
            vel_ball = vel[i] if self.suffix[0:8] in ["_springs", "_charged"] else None
            time_ball = time[i]

            # Creating y
            y[i] = len(time_ball) # num of observed snapshots for each ball

            # Creating x (concat loc and vel) and x_pos
            xi = np.concatenate((loc_ball, vel_ball),1) if self.suffix[0:8] in ["_springs", "_charged"] else loc_ball
            x_.append(xi)
            x_pos_i = time_ball - time_begin
            x_pos_.append(x_pos_i)
            x_pos_i_int = np.arange(0,y[i]) #denote the positions with interger instead of decimal times
            x_pos_int_.append(x_pos_i_int)


            '''
            # Creating x and x_pos, by traverse each ball's sequence

            node_time = {t:time_ball[t] for t in range(loc_ball.shape[0])}
            for t in range(loc_ball.shape[0]):
                xt_feature = np.concatenate((loc_ball[t], vel_ball[t])) if self.suffix[0:8] in ["_springs", "_charged"] else loc_ball[t]
                x.append(xt_feature)

                x_pos.append(time_ball[t] - time_begin)
                node_time[node_number] = time_ball[t]

                if i not in ball_nodes.keys():
                    ball_nodes[i] = [node_number]
                else:
                    ball_nodes[i].append(node_number)

                node_number += 1


            aa=0
            '''

        x = np.concatenate(x_,0) # concat trajs and velos of all balls along temporal dimension
        x_pos = np.concatenate(x_pos_,0)
        x_pos_int = np.concatenate(x_pos_int_,0)

        # Adding self-loop
        edge_with_self_loop = edge + np.eye(num_atoms, dtype=int)

        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for i in range(len(x_pos))], axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for i in range(len(x_pos))], axis=0) # pairwise temporal distance between each snapshot of each node and every other snapshot of every other node
        edge_time_matrix_int = np.concatenate([np.asarray(x_pos_int).reshape(-1, 1) for i in range(len(x_pos_int))], axis=1) - np.concatenate(
            [np.asarray(x_pos_int).reshape(1, -1) for i in range(len(x_pos_int))], axis=0) # pairwise temporal distance between each snapshot of each node and every other snapshot of every other node
        edge_exist_matrix = np.zeros((len(x_pos), len(x_pos))) # spatial temporal adj

        # connect two observations if their corresponding nodes are connected
        for i in range(num_atoms):
            for j in range(num_atoms):
                if edge_with_self_loop[i][j] != 0: # exact edge values are ignored
                    sender_index_start = int(np.sum(y[:i])) #start of sender ball i
                    sender_index_end = int(sender_index_start + y[i]) # end of sender ball i
                    receiver_index_start = int(np.sum(y[:j]))
                    receiver_index_end = int(receiver_index_start + y[j])
                    if i == j:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = 1
                    else:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = -1

        # remove edges outside a fixed window size
        if self.args.system == 'VCell' and max_gap>1.:
            temporal_connect_matrix = edge_time_matrix_int
        else:
            temporal_connect_matrix = edge_time_matrix
        if mask == None:
            edge_time_matrix = np.where(abs(temporal_connect_matrix)<=max_gap,edge_time_matrix,-2)
            edge_matrix = (edge_time_matrix + 2) * abs(edge_exist_matrix)  # padding 2 to avoid equal time been seen as not exists.
        elif forward == True:  # sender nodes are those whose time is larger. t0 = 0
            edge_time_matrix = np.where((edge_time_matrix >= 0) & (abs(temporal_connect_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)
        elif forward == False:  # sender nodes are those whose time is smaller. t0 = tN/2
            edge_time_matrix = np.where((edge_time_matrix <= 0) & (abs(temporal_connect_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)

        _, edge_attr_same = self.convert_sparse(edge_exist_matrix * edge_matrix)
        edge_is_same = np.where(edge_attr_same > 0, 1, 0).tolist() # whether edge is self-loop

        edge_index, edge_attr = self.convert_sparse(edge_matrix) # indices and attr of entries in the sparsified matrix
        edge_attr = edge_attr - 2
        edge_index_original, _ = self.convert_sparse(edge) # edge value is ignored

        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        edge_is_same = torch.FloatTensor(np.asarray(edge_is_same))

        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)

        graph_index_original = torch.LongTensor(edge_index_original)
        edge_data = Data(x = torch.ones(num_atoms),edge_index = graph_index_original,y=id)

        #graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same)
        graph_data = IndexedData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same, id=id)
        edge_size = edge_index.shape[1]

        return graph_data,edge_data,edge_size


    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise. Since in human dataset, it join the data of four tags (belt, chest, ankles) into a single time series
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True,
                                                    return_inverse=True) #【including 0 ]
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])


        for b, ( tt, vals, mask) in enumerate(batch):

            indices = inverse_indices[offset:offset + len(tt)]

            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask

        # get rid of the padding timepoint
        combined_tt = combined_tt[1:]
        combined_vals = combined_vals[:,1:,:]
        combined_mask = combined_mask[:,1:,:]

        combined_tt = combined_tt.float()


        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "mask": combined_mask,
            }
        return data_dict

    def normalize_features(self,inputs, num_balls, normalize_mode='individual'): # individual or universal normalization or no normalization
        '''

        :param inputs: [num-train, num-ball,(timestamps,2)]
        :return:
        '''
        value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]  # [2500] num_train * num_ball
        self.timelength = max(value_list_length)
        max_values = [torch.max(torch.tensor(balls[i])).item() for i in range(num_balls) for balls in inputs]
        min_values = [torch.min(torch.tensor(balls[i])).item() for i in range(num_balls) for balls in inputs]
        value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs] # i is outer loop
        value_padding = pad_sequence(value_list,batch_first=True,padding_value = 0)
        max_value = torch.max(value_padding).item()
        min_value = torch.min(value_padding).item()

        # Normalize to [-1, 1]
        if normalize_mode=='individual':
            for i in range(len(inputs)):
                for j in range(len(inputs[0])):
                    max_value_, min_value_ = torch.max(torch.tensor(inputs[i][j])).item(),torch.min(torch.tensor(inputs[i][j])).item()
                    scale = max_value_ - min_value_ if (max_value_ - min_value_)!=0 else 1
                    #print('max',max_value_,'min',min_value_)
                    inputs[i][j] = (inputs[i][j] - min_value_) * 2 / scale - 1
        elif normalize_mode=='universal':
            inputs = (inputs - min_value) * 2 / (max_value - min_value) - 1
        elif normalize_mode is None:
            pass
        else:
            raise ValueError('illegal normalization mode')
        return inputs,max_value,min_value

    def convert_sparse(self,graph):
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr

'''
import numpy as np
data = np.load('/store/NTU-RGB-D/xview/train_data.npy')
labels = pickle.load(open('/store/NTU-RGB-D/xview/train_label.pkl','rb'))
names = labels[0]
d = {i:{j: {k:[] for k in range(1,41,1)} for j in range(1,4,1)} for i in range(1,61,1)}
for n in names:
    action = int(n.split('.')[0].split('A')[1])
    camera = int(n.split('P')[0].split('C')[1])
    person = int(n.split('P')[1].split('R')[0])
    d[action][camera][person].append(n)

d = {}
for a in range(60):
    dc = {}
    for c in range(3):
'''
