import os
import numpy as np
import torch
import scipy.io as sio

from .builder import DATASETS
from .base_dataset import BaseDataset
from eunet.datasets.utils import np_tensor
from .utils.hood_common import triangles_to_edges
from .utils.phys import get_vertex_mass
from eunet.datasets.utils import readPKL

from .utils.metacloth import ATTR_RANGE
from .utils import Cloth3DReader


@DATASETS.register_module()
class Cloth3DDynamicDataset(BaseDataset):

    def __init__(self, env_cfg, phase, **kwargs):
        super(Cloth3DDynamicDataset, self).__init__(env_cfg=env_cfg, phase=phase, **kwargs)
        
    def _init_reader(self):
        self.data_reader = Cloth3DReader(self.env_cfg.clothenv_base, self.env_cfg.meta_path, phase=self.phase)

    def preprocess(self, sheet_path, **kwargs):
        # sheet format: seq\tnum
        sample_seq = self.data_reader.seq_list

        data_list = []
        for seq in sample_seq:
            sample_info = sio.loadmat(os.path.join(self.data_reader.data_dir, seq, "info.mat"), struct_as_record=False, squeeze_me=True)
            num_frames = sample_info['trans'].shape[1] if len(sample_info['trans'].shape) > 1 else 1
            data_list.append(f"{seq}\t0\t{num_frames}")
        with open(sheet_path, 'w') as f:
            f.write("\n".join(data_list))
        return True
    
    def prepare_rollout(self, current_garment, current_human, batch_data, **kwargs):
        if current_garment is not None:
            batch_data['inputs']['dynamic'][0]['state'] = [np_tensor(current_garment)]
        return batch_data

    ## Dynamic data process *********************************************************************************************
    def load_raw_dynamic(self, seq_num, frame_idx, **kwargs):
        sample_info = self.data_reader.read_info(seq_num)
        g_types = []
        names = []
        for g_name in sample_info['outfit'].keys():
            g_tp = self.data_reader.read_garment_type(g_name)
            if g_tp is not None:
                g_types.append(g_tp)
                names.append(g_name)
        assert len(g_types) == 1
        g_types = np.stack(g_types, axis=0)
        names.append('human')
        garment_pos = self.data_reader.read_garment_vertices(seq_num, names[0], frame=frame_idx, absolute=True)
        human_pos = self.data_reader.read_human(seq_num, frame=frame_idx, absolute=True)[0]
        offset = np.cumsum([0, garment_pos.shape[0], human_pos.shape[0]])
        pos = np.concatenate([garment_pos, human_pos], axis=0)
        gravity = np.array([0, 0, 0, 0, 0, -9.81]).reshape(1, -1) / ((self.env_cfg.clothenv_base.fps * self.env_cfg.clothenv_base.dt) ** 2)
        return g_types, names, offset, pos, gravity
    
    def process_raw_dynamic(self, g_types, names, offset, pos, gravity, **kwargs):
        garment_offset = offset[0:-1]
        human_offset = offset[-2:]

        state_list = pos
        garment_data = state_list[garment_offset[0]:garment_offset[-1]]
        human_data = state_list[human_offset[0]:human_offset[1]]
        garment_states = dict(state=garment_data)
        forces_states = dict(h_state=human_data, gravity=gravity)

        return g_types, names, offset, dict(garment=garment_states, forces=forces_states)

    def pack_dynamic(self, seq_num, frame_idx, **kwargs):
        g_types, names, offset, pos, gravity = self.load_raw_dynamic(seq_num, frame_idx)
        g_types, names, offset, input_states = self.process_raw_dynamic(g_types, names, offset, pos, gravity, **kwargs)
        return g_types, names, offset, input_states
    
    ## Static data process *********************************************************************************************
    def load_raw_static(self, seq_num, **kwargs):
        mesh_face_offset = [0]
        mesh_faces = []

        human_V, human_F = self.data_reader.read_human(seq_num, frame=0, absolute=True)
        mesh_templates = []
        seq_info = self.data_reader.read_info(seq_num)
        for g_idx, g_name in enumerate(seq_info['outfit'].keys()):
            g_tp = self.data_reader.read_garment_type(g_name)
            if g_tp is None:
                continue
            F, T = self.data_reader.read_garment_topology(seq_num, garment=g_name)
            mesh_templates.append(T)
            mesh_faces.append(F)
            mesh_face_offset.append(F.shape[0])
        mesh_faces.append(human_F+T.shape[0])
        mesh_face_offset.append(human_F.shape[0])
        mesh_templates.append(np.zeros((human_V.shape[0], mesh_templates[0].shape[1])))
        
        mesh_face_offset = np.cumsum(mesh_face_offset)
        mesh_faces = np.concatenate(mesh_faces, axis=0)
        mesh_templates = np.concatenate(mesh_templates, axis=0)

        return mesh_face_offset, mesh_faces, mesh_templates
    
    def process_raw_static(self, seq_num, mesh_name, mesh_offset, mesh_face_offset, mesh_faces, mesh_templates, **kwargs):
        garment_offset = mesh_offset[0:-1]
        garment_faces_offset = mesh_face_offset[0:-1]
        garment_name = mesh_name[:len(garment_offset)-1]
        garment_templates = mesh_templates[garment_offset[0]:garment_offset[-1]]
        human_offset = mesh_offset[-2:]
        human_faces_offset = mesh_face_offset[-2:]
        garment_faces = mesh_faces[garment_faces_offset[0]:garment_faces_offset[-1]] -garment_offset[0]
        garment_faces = garment_faces.astype(np.int64)
        garment_edges = triangles_to_edges(torch.tensor(garment_faces).unsqueeze(0)).detach().cpu().numpy()
        # Meta info
        indices = garment_offset
        recover_scalar = 1.0
        indices_weight = np.concatenate([
            np.ones(garment_offset[i+1]-garment_offset[i]) * recover_scalar / (garment_offset[i+1]-garment_offset[i])
            for i in range(len(garment_offset)-1)],
            axis=0)
        
        ## Get human faces
        human_faces = mesh_faces[human_faces_offset[0]:mesh_face_offset[-1]] - human_offset[0]
        human_faces = human_faces.astype(np.int64)

        attr_list = []
        garment_mass = []
        vert_mask_list = []
        seq_info = self.data_reader.read_info(seq_num)
        for g_idx, g_name in enumerate(garment_name):
            attr = self.data_reader.read_garment_attributes_metacloth(seq_num, g_name, info=seq_info)
            mass_density = attr[2] * 484 / (4.8**2)
            template = mesh_templates[mesh_offset[g_idx]:mesh_offset[g_idx+1]]
            faces = mesh_faces[mesh_face_offset[g_idx]:mesh_face_offset[g_idx+1]]
            faces = faces - np.min(faces)
            mass = get_vertex_mass(template, faces, mass_density).reshape(-1, 1)
            pin_vert = self.data_reader.read_garment_pinverts(seq_num, g_name)
            pin_mask = np.ones((garment_offset[g_idx+1]-garment_offset[g_idx], 1))
            if pin_vert is not None and self.phase != 'train':
                for p_v in pin_vert:
                    pin_mask[p_v, 0] = 0
            vert_mask_list.append(np.array(pin_mask))
            garment_mass.append(mass)
            attr[2] = 0
            # Check duplicated attr
            attr_exist = -1
            if attr_exist < 0:
                attr_list.append(attr)
                attr_exist = len(attr_list)-1
            # To adapt to hood pipeline
            lame_mu = np.array([attr[0]*ATTR_RANGE['tension'][1]], dtype=np.float32)
            lame_mu_input = np.array([attr[0]], dtype=np.float32)
            lame_lambda = lame_mu
            lame_lambda_input = lame_mu_input
            bending_coeff = np.array([attr[1]*ATTR_RANGE['bending'][1]], dtype=np.float32)
            bending_coeff_input = np.array([attr[1]], dtype=np.float32)
        vert_mask = np.concatenate(vert_mask_list, axis=0)
        assert len(attr_list) == 1
        attr_list = np.stack(attr_list, axis=0)
        garment_mass = np.concatenate(garment_mass, axis=0)

        static_fix_data = dict()
        static_data = dict(
            edges=garment_edges,
            vert_mask=vert_mask,
            mass=garment_mass,
            indices=indices,
            indices_weight=indices_weight,
            faces=garment_faces,
            attr=attr_list,
            h_faces=human_faces,
            templates=garment_templates,
            # To adapt to hood. CURRENTLY ONLY SUPPORT ONE GARMENT
            lame_mu_raw=lame_mu,
            lame_lambda_raw=lame_lambda,
            bending_coef_raw=bending_coeff,
            lame_mu=lame_mu_input,
            lame_lambda=lame_lambda_input,
            bending_coef=bending_coeff_input,
        )
        garment_num = garment_offset[-1] - garment_offset[0]
        human_num = human_offset[-1] - human_offset[0]

        f_connect_list, f_connect_edge_list = [], []
        f_area_list = []
        face2edge_list = []
        edge_offset = 0
        for g_idx, g_name in enumerate(garment_name):
            f_connect, f_connect_edge, face2edge_connectivity, f_area = self.data_reader.read_garment_polygon_params(seq_num, g_name, padding=True)
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
            f_area_list.append(f_area)

        f_connect_list = np.concatenate(f_connect_list, axis=0)
        f_connect_edge_list = np.concatenate(f_connect_edge_list, axis=0)
        face2edge_list = np.concatenate(face2edge_list, axis=0)
        static_data.update(dict(
            f_connectivity=f_connect_list,
            f_connectivity_edges=f_connect_edge_list,
            face2edge_connectivity=face2edge_list))

        f_area_list = np.concatenate(f_area_list, axis=0)
        static_data.update(dict(f_area=f_area_list))

        ## To adjust to hood inputs with node type
        ## 0 normal, 1 obstacle, 3 pinned
        vertex_type = (1-vert_mask)*3
        vertex_level = np.zeros_like(vertex_type)
        h_vertex_type = np.ones((human_num, 1))
        h_vertex_level = np.zeros_like(h_vertex_type)
        static_data.update(dict(
            vertex_type=np.concatenate([vertex_type, h_vertex_type], axis=0).astype(np.int64),
            vertex_level=np.concatenate([vertex_level, h_vertex_level], axis=0).astype(np.int64)))
        # Load human mask to mask out hands
        filter_dict = readPKL(self.env_cfg.clothenv_base.smpl_segm)
        filter_idx = ['rightHand', 'rightHandIndex1', 'leftHand', 'leftHandIndex1']
        filter_mask = np.ones((human_num, 1))
        for f_idx in filter_idx:
            masked_idx_list = filter_dict[f_idx]
            for m_idx in masked_idx_list:
                filter_mask[m_idx, 0] = 0
        static_fix_data.update(dict(human_vert_mask=filter_mask))
        return static_data, static_fix_data
    
    def postprocess_static(self, static_data, static_fix_data):
        static_data = np_tensor(static_data)
        static_data.update(static_fix_data)

        return static_data

    def pack_static(self, seq_num, mesh_name, mesh_offset, **kwargs):
        mesh_face_offset, mesh_faces, mesh_templates = self.load_raw_static(seq_num, **kwargs)
        static_data, static_fix_data = self.process_raw_static(seq_num, mesh_name, mesh_offset, mesh_face_offset, mesh_faces, mesh_templates, **kwargs)
        static_data = self.postprocess_static(static_data, static_fix_data)
        return static_data 
    
    def load_candidate_frames(self, seq_num, frame_idx):
        g_types, names, offset = None, None, None
        input_states_list = []

        start_frame = frame_idx-self.env_cfg.history
        # 1 for the ground truth
        for i in range(1+self.env_cfg.step+self.env_cfg.history):
            cur_frame = start_frame+i
            ## Behavior of input_states: 1. stack with history
            g_types, names, offset, input_states = self.pack_dynamic(seq_num, cur_frame)
            input_states_list.append(input_states)
        
        return g_types, names, offset, input_states_list
    
    def wrap_frames(self, g_types, names, offset, input_states_list, g_noncat_key=[]):
        history_frame = self.env_cfg.history
        input_data_list = []
        gt_list = []

        garment_keys = input_states_list[0]['garment'].keys()
        forces_keys = input_states_list[0]['forces'].keys()

        for i in range(history_frame, history_frame+1+self.env_cfg.step-1):
            # Collect gt
            gt_list.append(dict(
                vertices=input_states_list[i+1]['garment']['state'].copy(),
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

        # Convert to Datacontainer
        input_data_list = np_tensor(input_data_list)
        gt_list = np_tensor(gt_list)
        return input_data_list, gt_list

    def __getitem__(self, idx):
        seq_num, frame_idx = self.data_list[idx]
        
        g_types, names, offset, input_states_list = self.load_candidate_frames(seq_num, frame_idx)
        input_data_list, gt_list = self.wrap_frames(g_types, names, offset, input_states_list)

        static_data = self.pack_static(seq_num, names, offset)
        assert len(g_types) == 1
        static_data['indices_type'] = np_tensor(g_types)

        inputs = dict(
            dynamic=input_data_list,
            static=static_data,
        )

        gt_label = gt_list
        # The frame means predict frames, in dynamic model this need plus 1
        meta=dict(sequence=int(seq_num), frame=frame_idx+1)

        input_data = dict(
            inputs=inputs,
            gt_label=gt_label,
            meta=meta,)
        return input_data