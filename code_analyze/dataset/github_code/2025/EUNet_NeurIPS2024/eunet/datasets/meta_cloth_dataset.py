import numpy as np
import os
import tqdm
import mmcv

from .builder import DATASETS
from eunet.datasets.utils import MetaClothReader
from eunet.datasets.utils import diff_pos, writeH5, readH5, np_tensor
from eunet.datasets.utils.mesh import get_unneighbor_edges
from .base_dataset import BaseDataset
from .utils.metacloth import ATTR_RANGE


@DATASETS.register_module()
class MetaClothDynamicDataset(BaseDataset):

    def __init__(self, env_cfg, phase, **kwargs):
        super(MetaClothDynamicDataset, self).__init__(env_cfg=env_cfg, phase=phase, **kwargs)

    def _init_reader(self):
        self.data_reader = MetaClothReader(self.env_cfg.clothenv_base, phase=self.phase)

    def filter_data(self, seq_list=[]):
        data_list = []
        for seq in seq_list:
            for i in self.data_list:
                split_seq = i[0].split('/')
                if int(split_seq[-1]) == int(seq):
                    data_list.append(i)
        self.data_list = data_list

    def preprocess(self, sheet_path, **kwargs):
        # sheet format: seq\tnum
        sample_seq = self.data_reader.seq_list

        data_list = []
        for seq in sample_seq:
            sample_info = self.data_reader.read_info(seq)
            num_frames = 120 # Specifically for newly dataset
            data_list.append(f"{seq}\t0\t{num_frames}")
        with open(sheet_path, 'w') as f:
            f.write("\n".join(data_list))
        return True
    
    ## Dynamic data process *********************************************************************************************
    def load_raw_dynamic(self, seq_num, frame_idx, **kwargs):
        pos = self.data_reader.read_garment_vertices(seq_num, 'Plane', frame=frame_idx)
        offset = np.array([0, pos.shape[0]])
        names = ['Plane']
        g_types = self.data_reader.garment_type['Plane'].reshape(1, 1)
        raw_gravity = self.data_reader.read_gravity(seq_num)
        gravity = np.concatenate([np.zeros((1, 3)), raw_gravity.reshape(1, -1)], axis=-1)

        return g_types, names, offset, pos, gravity
    
    def process_raw_dynamic(self, g_types, names, offset, pos, gravity, **kwargs):
        # Noise Augmentation
        add_noise = self.generate_noise(pos.shape[0], pos.shape[1], self.noise_range)
        noise_list = [add_noise]

        # Concat needed info
        state_list = []
        state_list.append(pos)
        state_list = np.concatenate(state_list, axis=-1)
        add_noise = np.concatenate(noise_list, axis=-1)

        # Parse data
        garment_offset = offset
        garment_data = state_list[garment_offset[0]:garment_offset[-1]]
        garment_noise = add_noise[garment_offset[0]:garment_offset[-1]]
        trans_data = np.zeros((1, 9))
        garment_states = dict(state=garment_data, noise=garment_noise)
        # h_state is no use here, placeholder only
        forces_states = dict(h_state=garment_data, gravity=gravity, trans=trans_data)

        return g_types, names, offset, dict(garment=garment_states, forces=forces_states)
    
    def generate_noise(self, num_verts, num_dim, noise_range):
        if noise_range is not None:
            n_std = np.random.choice(noise_range, 1)
            assert n_std > 0
            add_noise = np.random.normal(loc=0.0, scale=n_std, size=(num_verts, num_dim))
        else:
            add_noise = np.zeros((num_verts, num_dim))
        return add_noise
    
     ## Static data process *********************************************************************************************
    def load_raw_static(self, seq_num, **kwargs):
        mesh_name, mesh_offset, mesh_face_offset, mesh_faces, mesh_templates = [], [], [], None, None
        mesh_name = ['Plane']

        # Load T-pose template
        seq_info = self.data_reader.read_info(seq_num)
        g_meta = seq_info['cloth']
        F, T = self.data_reader.read_garment_topology(seq_num, garment=g_meta['name'])
        mesh_templates = T
        mesh_faces = F
        mesh_offset = np.array([0, T.shape[0]])
        mesh_face_offset = np.array([0, len(F)])
        assert mesh_templates.shape[0] == mesh_offset[-1]
        return mesh_name, mesh_offset, mesh_face_offset, mesh_faces, mesh_templates
    
    def process_raw_static(self, seq_num, mesh_name, mesh_offset, mesh_face_offset, mesh_faces, mesh_templates, **kwargs):
        # Parse data
        garment_offset = mesh_offset
        garment_faces_offset = mesh_face_offset
        garment_name = mesh_name[:len(garment_offset)-1]
        garment_templates = mesh_templates[garment_offset[0]:garment_offset[-1]]
        garment_faces = mesh_faces[garment_faces_offset[0]:garment_faces_offset[-1]] -garment_offset[0]
        # Meta info
        indices = garment_offset
        indices_weight = np.concatenate([
            np.ones(garment_offset[i+1]-garment_offset[i]) / (garment_offset[i+1]-garment_offset[i])
            for i in range(len(garment_offset)-1)],
            axis=0)

        # Mask pin nodes
        garment_num = garment_offset[-1] - garment_offset[0]
        vert_mask = np.ones((garment_num, 1))
        pin_idx = self.data_reader.read_pin_verts()
        for idx in pin_idx:
            vert_mask[idx] = 0.0

        # Attribute mask
        attr_list = []
        garment_mass = []
        seq_info = self.data_reader.read_info(seq_num)
        for g_idx, g_name in enumerate(garment_name):
            attr = self.data_reader.read_garment_attributes(seq_num, g_name, info=seq_info)
            mass = np.full((garment_offset[g_idx+1]-garment_offset[g_idx], 1), attr[2])
            garment_mass.append(mass)
            attr[2] = 0 # TODO: erase the mass
            # Check duplicated attr
            attr_list.append(attr)
            attr_exist = len(attr_list)-1
            assert len(attr_list) == 1

        attr_list = np.stack(attr_list, axis=0)
        garment_mass = np.concatenate(garment_mass, axis=0)

        static_data = dict(
            vert_mask=vert_mask,
            mass=garment_mass,
            indices=indices,
            indices_weight=indices_weight,
            faces=garment_faces,
            h_faces=garment_faces, # No use but to align with the pipeline
            attr=attr_list,
            templates=garment_templates,
        )
            
        f_connect_list, f_connect_edge_list = [], []
        face2edge_list = []
        edge_offset = 0
        for g_idx, g_name in enumerate(garment_name):
            f_connect, f_connect_edge, face2edge_connectivity = self.data_reader.read_garment_polygon_params(seq_num, g_name)
            # Move the offset for further concat
            f_offset = garment_faces_offset[g_idx]
            v_offset = garment_offset[g_idx]
            f_connect += f_offset
            f_connect_edge += v_offset
            face2edge_connectivity += edge_offset

            edge_offset += f_connect_edge.shape[0]
            f_connect_list.append(f_connect)
            f_connect_edge_list.append(f_connect_edge)
            face2edge_list.append(face2edge_connectivity)
        f_connect_list = np.concatenate(f_connect_list, axis=0)
        f_connect_edge_list = np.concatenate(f_connect_edge_list, axis=0)
        face2edge_list = np.concatenate(face2edge_list, axis=0)
        static_data.update(dict(
            f_connectivity=f_connect_list,
            f_connectivity_edges=f_connect_edge_list,
            face2edge_connectivity=face2edge_list))
        
        # Load randomed vert mask for noise
        assert len(garment_name) == 1
        distance_matrix = self.data_reader.read_garment_floyd(seq_num, garment_name[0])
        distance_matrix = np.array(distance_matrix)
        center_id = np.random.randint(2, garment_num) # avoid the pinned verts
        filtered_verts, first_neighbor_list = get_unneighbor_edges(distance_matrix, center_id, first_neighbor=True)
        hop_mask = np.zeros_like(vert_mask)
        hop_mask[filtered_verts] = 1.0
        first_neighbor_mask = np.zeros_like(vert_mask)
        first_neighbor_mask[first_neighbor_list] = 1.0
        static_data.update(dict(hop_mask=hop_mask))
        static_data.update(dict(first_neighbor_mask=first_neighbor_mask))
        
        return static_data, mesh_name, mesh_offset, mesh_face_offset, mesh_faces, mesh_templates

    def load_candidate_frames(self, seq_num, frame_idx):
        g_types, names, offset = None, None, None
        input_states_list = []

        start_frame = frame_idx-self.env_cfg.history
        # extra 1 is to align with other pipeline, in practice only the first 3 frames are used, the last frame (4th) is no use for this
        for i in range(1+self.env_cfg.step+self.env_cfg.history):
            cur_frame = start_frame+i
            ## Behavior of input_states: 1. stack with history
            g_types, names, offset, input_states = self.pack_dynamic(seq_num, cur_frame)
            input_states_list.append(input_states)
        
        return g_types, names, offset, input_states_list

    def pack_dynamic(self, seq_num, frame_idx, **kwargs):
        g_types, names, offset, pos, gravity = self.load_raw_dynamic(seq_num, frame_idx)
        g_types, names, offset, input_states = self.process_raw_dynamic(g_types, names, offset, pos, gravity, **kwargs)
        return g_types, names, offset, input_states

    def wrap_frames(self, input_states_list, g_noncat_key=[]):
        history_frame = self.env_cfg.history
        input_data_list = []
        gt_list = []

        # Parse input_frame: list of dict -> dict of list
        garment_keys = input_states_list[0]['garment'].keys()
        forces_keys = input_states_list[0]['forces'].keys()

        for i in range(history_frame, history_frame+1+self.env_cfg.step-1):
            # Collect gt
            gt_list.append(dict(
                vertices=input_states_list[i+1]['garment']['state'].copy(),
                trans=input_states_list[i+1]['forces']['trans'].copy()
            ))

            # Collapse garment
            garment_data = {
                key: [input_states_list[j]['garment'][key] for j in range(i, i-history_frame-1, -1)]
                for key in garment_keys if key not in g_noncat_key
            }
            ## Concat key
            garment_data = {
                key: np.concatenate(val, axis=-1)
                for key, val in garment_data.items() if key not in g_noncat_key
            }
            ## Add noncat key
            garment_data.update({
                key: input_states_list[i]['garment'][key]
                for key in g_noncat_key})
            
            # Collapse forces
            ## Concat and filter no use key
            forces_data = {
                key: np.concatenate([input_states_list[j+1]['forces'][key] for j in range(i, i-history_frame-1, -1)], axis=-1)
                for key in forces_keys
            }
            input_data = dict()
            input_data.update(garment_data)
            input_data.update(forces_data)
            input_data_list.append(input_data)

        input_data_list = np_tensor(input_data_list)
        gt_list = np_tensor(gt_list)
        return input_data_list, gt_list
    
    def pack_static(self, seq_num, **kwargs):
        mesh_name, mesh_offset, mesh_face_offset, mesh_faces, mesh_templates = self.load_raw_static(seq_num, **kwargs)
        static_data, _mesh_name, _mesh_offset, _mesh_face_offset, _mesh_faces, _mesh_templates = self.process_raw_static(seq_num, mesh_name, mesh_offset, mesh_face_offset, mesh_faces, mesh_templates, **kwargs)
        static_data = np_tensor(static_data)
        return static_data
    
    def __getitem__(self, idx):
        seq_num, frame_idx = self.data_list[idx]

        g_types, names, offset, input_states_list = self.load_candidate_frames(seq_num, frame_idx)
        g_noncat_key = ['noise']
        input_data_list, gt_list = self.wrap_frames(input_states_list, g_noncat_key=g_noncat_key)

        static_data = self.pack_static(seq_num)

        # Manually adjust the g_types according to the params
        mass = static_data['mass'].data[0]
        attr = static_data['attr'].data[0]
        eps = 1e-3
        if (attr[0] * ATTR_RANGE['tension'][1] - 5).abs() < eps  and (attr[1] * ATTR_RANGE['bending'][1] - 0.5).abs() < eps:
            g_types[0, 0] = 0
        elif (attr[0] * ATTR_RANGE['tension'][1] - 80).abs() < eps and (attr[1] * ATTR_RANGE['bending'][1] - 150.0).abs() < eps:
            g_types[0, 0] = 1
        elif (attr[0] * ATTR_RANGE['tension'][1] - 40).abs() < eps and (attr[1] * ATTR_RANGE['bending'][1] - 10.0).abs() < eps:
            g_types[0, 0] = 2
        elif (attr[0] * ATTR_RANGE['tension'][1] - 15).abs() < eps and (attr[1] * ATTR_RANGE['bending'][1] - 0.5).abs() < eps:
            g_types[0, 0] = 3
        else:
            assert False
            g_types[0, 0] = 4
        static_data['indices_type'] = np_tensor(g_types)

        inputs = dict(
            dynamic=input_data_list,
            static=static_data,
        )

        gt_label = gt_list
        seq_dir, seq_int = seq_num.split('/')
        seq_int = int(seq_int)
        meta=dict(sequence=seq_int, frame=frame_idx+1)

        input_data = dict(
            inputs=inputs,
            gt_label=gt_label,
            meta=meta,)
        return input_data