import os
import torch
import numpy as np

from .builder import DATASETS
from eunet.datasets.utils.sparse_data import SparseMask
from eunet.datasets.utils import np_tensor
from .base_dataset import BaseDataset

from .utils.hood import load_garments_dict, make_garment_smpl_dict, make_obstacle_dict
import smplx
from smplx import SMPL
from .utils.hood_common import pickle_load, separate_arms, NodeType, triangles_to_edges
from .utils.hood_garment_smpl import GarmentSMPL
from .utils.hood_coarse import make_coarse_edges
from .utils.phys import get_vertex_mass, get_face_areas, make_Dm_inv, get_face_connectivity_combined
from mmcv import Config
from typing import Dict, Tuple, List
import pickle
import pandas as pd
from .utils.metacloth import DEFAULT_CONFIG_SET
from .utils.metacloth import ATTR_RANGE as META_ATTR_RANGE
from eunet.datasets.utils import readPKL


class VertexBuilder:
    """
    Helper class to build garment and body vertices from a sequence of SMPL poses.
    """

    def __init__(self, mcfg):
        self.mcfg = mcfg

    @staticmethod
    def build(sequence_dict: dict, f_make, idx_start: int, idx_end: int = None, garment_name: str = None) -> np.ndarray:
        """
        Build vertices from a sequence of SMPL poses using the given `f_make` function.
        :param sequence_dict: a dictionary of SMPL parameters
        :param f_make: a function that takes SMPL parameters and returns vertices
        :param idx_start: first frame index
        :param idx_end: last frame index
        :param garment_name: name of the garment (None for body)
        :return: [Nx3] mesh vertices
        """

        betas = sequence_dict['betas']
        if len(betas.shape) == 2 and betas.shape[0] != 1:
            betas = betas[idx_start: idx_end]

        verts = f_make(sequence_dict['body_pose'][idx_start: idx_end],
                       sequence_dict['global_orient'][idx_start: idx_end],
                       sequence_dict['transl'][idx_start: idx_end],
                       betas, garment_name=garment_name)

        return verts

    def pad_lookup(self, lookup: np.ndarray) -> np.ndarray:
        """
        Pad the lookup sequence to the required number of steps.
        """
        n_lookup = lookup.shape[0]
        n_topad = self.mcfg.lookup_steps - n_lookup

        if n_topad == 0:
            return lookup

        padlist = [lookup] + [lookup[-1:]] * n_topad
        lookup = np.concatenate(padlist, axis=0)
        return lookup

    def pos2tensor(self, pos: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array of vertices to a tensor and permute the axes into [VxNx3] (torch geometric format)
        """
        pos = torch.tensor(pos).permute(1, 0, 2)
        if not self.mcfg.wholeseq and pos.shape[1] == 1:
            pos = pos[:, 0]
        return pos

    def add_verts(self, sequence_dict: dict, idx: int, f_make, object_key: str, history: int,
                  **kwargs) -> Tuple:
        """
        Builds the vertices from the given SMPL pose sequence and adds them to the HeteroData sample.
        :param sample: HetereoData object
        :param sequence_dict: sequence of SMPL parameters
        :param idx: frame index (not used if self.mcfg.wholeseq is True)
        :param f_make: function that takes SMPL parameters and returns vertices
        :param object_key: name of the object to build vertices for ('cloth' or 'obstacle')
        :return: updated HeteroData object
        """

        N_steps = sequence_dict['body_pose'].shape[0]
        pos_dict = {}

        # Build the vertices for the whole sequence
        if self.mcfg.wholeseq:
            all_vertices = VertexBuilder.build(sequence_dict, f_make, 0, None,
                                               **kwargs)

        # Build the vertices for several frames starting from `idx`
        else:
            n_lookup = 1
            if self.mcfg.lookup_steps > 0:
                # 1: for next pos
                # history: for prev history input
                # 1: to calculate the init velocity
                n_lookup = min(self.mcfg.lookup_steps, N_steps - idx - 1 - history - 1)
            all_vertices = VertexBuilder.build(sequence_dict, f_make, idx, idx + 1 + history + 1 + n_lookup,
                                               **kwargs)

        # Deal with history input here
        pos = all_vertices[1:]
        vel = all_vertices[1:] - all_vertices[:-1]
        return pos, vel


class NoiseMaker:
    """
    Helper class to add noise to the garment vertices.
    """

    def __init__(self, mcfg: Config):
        self.mcfg = mcfg

    def add_noise(self, pos, vel, vertex_type) -> Tuple:
        """
        Add gaussian noise with std == self.mcfg.noise_scale to `pos` and `prev_pos`
        tensors in `sample['cloth']`
        :param sample: HeteroData
        :return: sample: HeteroData
        """
        if self.mcfg.noise_scale == 0:
            return pos, vel

        world_pos = pos[1]
        if len(vertex_type.shape) == 1:
            vertex_type = vertex_type[..., None]

        noise = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)
        noise_prev = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)

        mask = vertex_type == NodeType.NORMAL
        if len(mask.shape) == 2 and len(noise.shape) == 3:
            mask = mask.unsqueeze(-1)
        noise = noise * mask

        pos[1] += noise
        pos[0] += noise_prev
        vel[1] = pos[1] - pos[0]
        vel[0] += noise_prev
        return pos, vel
    
    def add_external(self, pos, vertex_type, history) -> Tuple:
        """
        Add gaussian noise with std == self.mcfg.noise_scale to `pos` and `prev_pos`
        tensors in `sample['cloth']`
        :param sample: HeteroData
        :return: sample: HeteroData
        """
        n_frame, n_verts, n_dim = pos.shape
        external = np.zeros_like(pos)
        if self.mcfg.external == 0:
            return external

        vtype_mask = np.tile(vertex_type == NodeType.NORMAL, (n_frame, 1, 1))
        r_acc = (np.random.rand(n_frame, 1, n_dim) - 0.5) * 2 * self.mcfg.external
        r_acc[:, :, -1] += self.mcfg.external
        r_acc = np.tile(r_acc, (1,n_verts,1))
        r_acc = r_acc * vtype_mask

        random_mask = np.random.rand(n_frame, n_verts, 1) > 0.5
        r_acc = r_acc * random_mask
        r_acc[:history+1] *= 0
        return r_acc
    
class GarmentBuilder:
    """
    Class to build the garment meshes from SMPL parameters.
    """

    def __init__(self, mcfg: Config, garments_dict: dict, garment_smpl_model_dict: Dict[str, GarmentSMPL]):
        """
        :param mcfg: config
        :param garments_dict: dictionary with data for all garments
        :param garment_smpl_model_dict: dictionary with SMPL models for all garments
        """
        self.mcfg = mcfg
        self.garments_dict = garments_dict
        self.garment_smpl_model_dict = garment_smpl_model_dict

        self.vertex_builder = VertexBuilder(mcfg)
        self.noise_maker = NoiseMaker(mcfg)

    def make_cloth_verts(self, body_pose: np.ndarray, global_orient: np.ndarray, transl: np.ndarray, betas: np.ndarray,
                         garment_name: str) -> np.ndarray:
        """
        Make vertices of a garment `garment_name` in a given pose

        :param body_pose: SMPL pose parameters [Nx69] OR [69]
        :param global_orient: SMPL global_orient [Nx3] OR [3]
        :param transl: SMPL translation [Nx3] OR [3]
        :param betas: SMPL betas [Nx10] OR [10]
        :param garment_name: name of the garment in `self.garment_smpl_model_dict`

        :return: vertices [NxVx3]
        """
        body_pose = torch.FloatTensor(body_pose)
        global_orient = torch.FloatTensor(global_orient)
        transl = torch.FloatTensor(transl)
        betas = torch.FloatTensor(betas)

        garment_smpl_model = self.garment_smpl_model_dict[garment_name]

        if len(body_pose.shape) == 1:
            body_pose = body_pose.unsqueeze(0)
            global_orient = global_orient.unsqueeze(0)
            transl = transl.unsqueeze(0)
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)

        wholeseq = self.mcfg.wholeseq or body_pose.shape[0] > 1
        full_pose = torch.cat([global_orient, body_pose], dim=1)

        if wholeseq and betas.shape[0] == 1:
            betas = betas.repeat(body_pose.shape[0], 1)

        with torch.no_grad():
            vertices = garment_smpl_model.make_vertices(betas=betas, full_pose=full_pose, transl=transl).numpy()

        if not wholeseq:
            vertices = vertices[0]

        return vertices

    def add_vertex_type(self, garment_pos, garment_name: str) -> np.array:
        """
        Add `vertex_type` tensor to `sample['cloth']`

        utils.common.NodeType.NORMAL (0) for normal vertices
        utils.common.NodeType.HANDLE (3) for pinned vertices

        if `self.mcfg.pinned_verts` == True, take `vertex_type` from `self.garments_dict`
        else: fill all with utils.common.NodeType.NORMAL (0)

        :param sample: HeteroData sample
        :param garment_name: name of the garment in `self.garments_dict`

        :return: sample['cloth'].vertex_type: torch.LongTensor [Vx1]
        """
        garment_dict = self.garments_dict[garment_name]

        if self.mcfg.pinned_verts:
            vertex_type = garment_dict['node_type'].astype(np.int64)
        else:
            V = garment_pos.shape[0]
            vertex_type = np.zeros((V, 1)).astype(np.int64)

        return vertex_type

    def resize_restpos(self, restpos: np.array) -> np.array:
        """
        Randomly resize resting geometry of a garment
        with scale from `self.mcfg.restpos_scale_min` to `self.mcfg.restpos_scale_max`

        :param restpos: Vx3
        :return: resized restpos: Vx3
        """
        if self.mcfg.restpos_scale_min == self.mcfg.restpos_scale_max == 1.:
            return restpos

        scale = np.random.rand()
        scale *= (self.mcfg.restpos_scale_max - self.mcfg.restpos_scale_min)
        scale += self.mcfg.restpos_scale_min

        mean = restpos.mean(axis=0, keepdims=True)
        restpos -= mean
        restpos *= scale
        restpos += mean

        return restpos

    def make_shaped_restpos(self, sequence_dict: dict, garment_name: str) -> np.ndarray:
        """
        Create resting pose geometry for a garment in SMPL zero pose with given SMPL betas

        :param sequence_dict: dict with
            sequence_dict['body_pose'] np.array SMPL body pose [Nx69]
            sequence_dict['global_orient'] np.array SMPL global_orient [Nx3]
            sequence_dict['transl'] np.array SMPL translation [Nx3]
            sequence_dict['betas'] np.array SMPL betas [10]
        :param garment_name: name of the garment in `self.garment_smpl_model_dict`
        :return: zeroposed garment with given shape [Vx3]
        """
        body_pose = np.zeros_like(sequence_dict['body_pose'][:1])
        global_orient = np.zeros_like(sequence_dict['global_orient'][:1])
        transl = np.zeros_like(sequence_dict['transl'][:1])
        verts = self.make_cloth_verts(body_pose,
                                      global_orient,
                                      transl,
                                      sequence_dict['betas'], garment_name=garment_name)
        return verts

    def add_restpos(self, sequence_dict: dict, garment_name: str) -> np.array:
        """
        Add resting pose geometry to `sample['cloth']`

        :param sample: HeteroData
        :param sequence_dict: dict with SMPL parameters
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return: sample['cloth'].rest_pos: torch.FloatTensor [Vx3]
        """
        garment_dict = self.garments_dict[garment_name]
        if self.mcfg.use_betas_for_restpos:
            rest_pos = self.make_shaped_restpos(sequence_dict, garment_name)[0]
        else:
            rest_pos = self.resize_restpos(garment_dict['rest_pos'])

        return rest_pos

    def add_faces_and_edges(self, garment_name: str) -> Tuple:
        """
        Add garment faces to `sample['cloth']`
        Add bi-directional edges to `sample['cloth', 'mesh_edge', 'cloth']`

        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            sample['cloth'].faces_batch: torch.LongTensor [3xF]
            ample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]

            src, dst)
        """

        garment_dict = self.garments_dict[garment_name]

        faces = garment_dict['faces']
        # src to dst
        edges = triangles_to_edges(torch.tensor(faces).unsqueeze(0)).detach().cpu().numpy()
        return faces, edges

    def make_vertex_level(self, num_verts: int, coarse_edges_dict: Dict[int, np.array]) -> np.array:
        """
        Add `vertex_level` labels to `sample['cloth']`
        for each garment vertex, `vertex_level` is the number of the deepest level the vertex is in
        starting from `0` for the most shallow level

        :param sample: HeteroData
        :param coarse_edges_dict: dictionary with list of edges for each coarse level
        :return: sample['cloth'].vertex_level: torch.LongTensor [Vx1]
        """
        N = num_verts
        vertex_level = np.zeros((N, 1)).astype(np.int64)
        for i in range(self.mcfg.n_coarse_levels):
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            nodes_unique = np.unique(edges_coarse.reshape(-1))
            vertex_level[nodes_unique] = i + 1
        return vertex_level

    def add_coarse(self, garment_name: str, num_verts: int) -> Tuple:
        """
        Add coarse edges to `sample` as `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`.
        where `i` is the number of the coarse level (starting from `0`)

        :param sample: HeteroData
        :param garment_name:
        :return: sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]
        """
        if self.mcfg.n_coarse_levels == 0:
            return [], np.zeros((num_verts, 1)).astype(np.int64)

        garment_dict = self.garments_dict[garment_name]
        faces = garment_dict['faces']

        center_nodes = garment_dict['center']
        center = np.random.choice(center_nodes)
        if 'coarse_edges' not in garment_dict:
            garment_dict['coarse_edges'] = dict()

        if center in garment_dict['coarse_edges']:
            coarse_edges_dict = garment_dict['coarse_edges'][center]
        else:
            coarse_edges_dict = make_coarse_edges(faces, center, n_levels=self.mcfg.n_coarse_levels)
            garment_dict['coarse_edges'][center] = coarse_edges_dict

        # for each level `i` add edges to sample as  `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`
        coarse_edge_list = []
        for i in range(self.mcfg.n_coarse_levels):
            key = f'coarse_edge{i}'
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            edges_coarse = np.concatenate([edges_coarse, edges_coarse[:, [1, 0]]], axis=0)
            # coarse_edges = torch.tensor(edges_coarse.T)
            coarse_edges = edges_coarse.T
            coarse_edge_list.append(coarse_edges)

        # add `vertex_level` labels to sample
        vertex_level = self.make_vertex_level(num_verts, coarse_edges_dict)

        return coarse_edge_list, vertex_level

    def add_button_edges(self, garment_name: str) -> np.array:
        """
        Add set of node pairs that should serve as buttons (needed for unzipping/unbuttoning demonstration)
        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return: if `button_edges` are on,
            sample['cloth'].button_edges_batch: torch.LongTensor [2xBE]
        """

        # if button_edges flag is off, do nothing
        if not hasattr(self.mcfg, 'button_edges') or not self.mcfg.button_edges:
            return None

        garment_dict = self.garments_dict[garment_name]

        # if there are no buttons for the given garment, do nothing
        if 'button_edges' not in garment_dict:
            return None

        button_edges = garment_dict['button_edges']

        return button_edges

    def build(self, sequence_dict: dict, idx: int, garment_name: str, history: int) -> Dict:
        """
        Add all data for the garment to the sample

        :param sample: HeteroData
        :param sequence_dict: dictionary with SMPL parameters
        :param idx: starting index in a sequence (not used if  `self.mcfg.wholeseq`)
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            if self.mcfg.wholeseq:
                sample['cloth'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
                sample['cloth'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
                sample['cloth'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame
            else:
                sample['cloth'].prev_pos torch.FloatTensor [Vx3]: vertex positions at the previous frame
                sample['cloth'].pos torch.FloatTensor [Vx3]: vertex positions at the current frame
                sample['cloth'].target_pos torch.FloatTensor [Vx3]: vertex positions at the next frame
                sample['cloth'].lookup torch.FloatTensor [VxLx3] (L == self.mcfg.lookup_steps): vertex positions at several future frames

            sample['cloth'].rest_pos torch.FloatTensor [Vx3]: vertex positions in the canonical pose
            sample['cloth'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['cloth'].vertex_type torch.LongTensor [Vx1]: vertex type (0 - regular, 3 - pinned)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

            sample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]: mesh edges

            for each coarse level `i` in [0, self.mcfg.n_coarse_levels]:
                sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]: coarse edges at level `i`

            if self.mcfg.button edges and the garment has buttons:
                sample['cloth'].button_edges_batch: torch.LongTensor [2xBE]: button edges

        """
        pos, vel = self.vertex_builder.add_verts(sequence_dict, idx, self.make_cloth_verts, "cloth",
                                               garment_name=garment_name, history=history)

        vertex_type = self.add_vertex_type(pos, garment_name)
        pos, vel = self.noise_maker.add_noise(pos, vel, vertex_type)
        random_external = self.noise_maker.add_external(pos, vertex_type, history)
        template = self.add_restpos(sequence_dict, garment_name)
        faces, edges = self.add_faces_and_edges(garment_name)
        hie_edges, vertex_level = self.add_coarse(garment_name, pos.shape[-2])
        button_edges = self.add_button_edges(garment_name)

        return pos, vel, vertex_type, template, faces, edges, hie_edges, vertex_level, button_edges, random_external


class BodyBuilder:
    """
    Class for building body meshed from SMPL parameters
    """

    def __init__(self, mcfg: Config, smpl_model: SMPL, obstacle_dict: dict):
        """
        :param mcfg: Config
        :param smpl_model:
        :param obstacle_dict: auxiliary data for the obstacle
                obstacle_dict['vertex_type']: vertex type (1 - regular obstacle node, 2 - hand node (omitted during inference to avoid body self-penetrations))
        """
        self.smpl_model = smpl_model
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg
        self.vertex_builder = VertexBuilder(mcfg)

    def make_smpl_vertices(self, body_pose: np.ndarray, global_orient: np.ndarray, transl: np.ndarray,
                           betas: np.ndarray, **kwargs) -> np.ndarray:
        """
        Create body vertices from SMPL parameters (used in VertexBuilder.add_verts)

        :param body_pose: SMPL pose parameters [Nx69] OR [69]
        :param global_orient: SMPL global_orient [Nx3] OR [3]
        :param transl: SMPL translation [Nx3] OR [3]
        :param betas: SMPL betas [Nx10] OR [10]

        :return: vertices [NxVx3]
        """
        body_pose = torch.FloatTensor(body_pose)
        global_orient = torch.FloatTensor(global_orient)
        transl = torch.FloatTensor(transl)
        betas = torch.FloatTensor(betas)
        if len(body_pose.shape) == 1:
            body_pose = body_pose.unsqueeze(0)
            global_orient = global_orient.unsqueeze(0)
            transl = transl.unsqueeze(0)
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)
        wholeseq = self.mcfg.wholeseq or body_pose.shape[0] > 1

        with torch.no_grad():
            smpl_output = self.smpl_model(betas=betas, body_pose=body_pose, transl=transl, global_orient=global_orient)
        vertices = smpl_output.vertices.numpy().astype(np.float32)

        if not wholeseq:
            vertices = vertices[0]

        return vertices

    def add_vertex_type(self, num_verts) -> np.array:
        """
        Add vertex type field to the obstacle object in the sample
        """
        N = num_verts
        if 'vertex_type' in self.obstacle_dict:
            vertex_type = self.obstacle_dict['vertex_type']
        else:
            vertex_type = np.ones((N, 1)).astype(np.int64)
        return vertex_type

    def add_faces(self) -> np.array:
        """
        Add body faces to the obstacle object in the sample
        """
        faces = self.smpl_model.faces.astype(np.int64)
        return faces

    def add_vertex_level(self, num_verts) -> np.array:
        """
        Add vertex level field to the obstacle object in the sample (always 0 for the body)
        """
        N = num_verts
        vertex_level = np.zeros((N, 1))
        return vertex_level

    def build(self, sequence_dict: dict, idx: int, history: int) -> Tuple:
        """
        Add all data for the body (obstacle) to the sample
        :param sample: HeteroData object to add data to
        :param sequence_dict: dict with SMPL parameters
        :param idx: index of the current frame in the sequence
        
        :return:
            if self.mcfg.wholeseq:
                sample['obstacle'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
                sample['obstacle'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
                sample['obstacle'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame
            else:
                sample['obstacle'].prev_pos torch.FloatTensor [Vx3]: vertex positions at the previous frame
                sample['obstacle'].pos torch.FloatTensor [Vx3]: vertex positions at the current frame
                sample['obstacle'].target_pos torch.FloatTensor [Vx3]: vertex positions at the next frame
                sample['obstacle'].lookup torch.FloatTensor [VxLx3] (L == self.mcfg.lookup_steps): vertex positions at several future frames

            sample['obstacle'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['obstacle'].vertex_type torch.LongTensor [Vx1]: vertex type (1 - regular obstacle, 2 - omitted)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)


        
        """

        pos, vel = self.vertex_builder.add_verts(sequence_dict, idx, self.make_smpl_vertices, "obstacle", history=history)
        num_verts = pos.shape[-2]
        vertex_type = self.add_vertex_type(num_verts)
        faces = self.add_faces()
        vertex_level = self.add_vertex_level(num_verts)
        return pos, vel, vertex_type, faces, vertex_level

class SequenceLoader:
    def __init__(self, mcfg, data_path, betas_table=None):
        self.mcfg = mcfg
        self.data_path = data_path
        self.betas_table = betas_table

    def process_sequence(self, sequence: dict) -> dict:
        """
        Apply transformations to the SMPL sequence
        :param sequence: dict with SMPL parameters
        :return: processed dict with SMPL parameters
        """
        #
        # from SNUG, eliminates hand-body penetrations
        if self.mcfg.separate_arms:
            body_pose = sequence['body_pose']
            global_orient = sequence['global_orient']
            full_pos = np.concatenate([global_orient, body_pose], axis=1)
            full_pos = separate_arms(full_pos)
            sequence['global_orient'] = full_pos[:, :3]
            sequence['body_pose'] = full_pos[:, 3:]

        # sample random SMPLX beta parameters
        if self.mcfg.random_betas:
            betas = sequence['betas']
            random_betas = np.random.rand(*betas.shape)
            random_betas = random_betas * self.mcfg.betas_scale * 2
            random_betas -= self.mcfg.betas_scale
            sequence['betas'] = random_betas

        # zero-out hand pose (eliminates unrealistic hand poses)
        sequence['body_pose'][:, -6:] *= 0

        # zero-out all SMPL beta parameters
        if self.mcfg.zero_betas:
            sequence['betas'] *= 0

        return sequence

    def load_sequence(self, fname: str, betas_id: int=None) -> dict:
        """
        Load sequence of SMPL parameters from disc
        and process it

        :param fname: file name of the sequence
        :param betas_id: index of the beta parameters in self.betas_table
                        (used only in validation to generate sequences for metrics calculation
        :return: dict with SMPL parameters:
            sequence['body_pose'] np.array [Nx69]
            sequence['global_orient'] np.array [Nx3]
            sequence['transl'] np.array [Nx3]
            sequence['betas'] np.array [10]
        """
        filepath = os.path.join(self.data_path, fname + '.pkl')
        with open(filepath, 'rb') as f:
            sequence = pickle.load(f)

        assert betas_id is None or self.betas_table is not None, "betas_id should be specified only in validation mode with valid betas_table"

        if self.betas_table is not None:
            sequence['betas'] = self.betas_table[betas_id]

        sequence = self.process_sequence(sequence)

        return sequence
    
class Loader:
    """
    Class for building HeteroData objects containing all data for a single sample
    """

    def __init__(self, mcfg: Config, garments_dict: dict, smpl_model: SMPL,
                 garment_smpl_model_dict: Dict[str, GarmentSMPL], obstacle_dict: dict, train_mode: bool, betas_table=None):
        '''
            attr_option:
                None: No need attr like hood
                random: use for training
                fixed: use for testing
        '''
        self.sequence_loader = SequenceLoader(mcfg, mcfg.data_root, betas_table=betas_table)
        self.garment_builder = GarmentBuilder(mcfg, garments_dict, garment_smpl_model_dict)
        self.body_builder = BodyBuilder(mcfg, smpl_model, obstacle_dict)

        self.data_path = mcfg.data_root
        self.mcfg = mcfg
        # yzx -> xyz
        self.rot_correct_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).reshape(1, 3, 3)
        self.attr_option = mcfg.get('attr_option', None)
        self.train_mode = train_mode

    def _rotate_vector(self, vec):
        n_frame, n_row, n_dim = vec.shape
        assert n_dim % 3 == 0
        vec = vec.reshape(-1, 3, 1)
        new_vec = np.matmul(self.rot_correct_matrix, vec)
        new_vec = new_vec.reshape(n_frame, n_row, n_dim)
        return new_vec

    def load_sample(self, fname: str, idx: int, garment_name: str, betas_id: int, history: int = 1):
        """
        Build HeteroData object for a single sample
        :param fname: name of the pose sequence relative to self.data_path
        :param idx: index of the frame to load (not used if self.mcfg.wholeseq == True)
        :param garment_name: name of the garment to load
        :param betas_id: index of the beta parameters in self.betas_table (only used to generate validation sequences when comparing to snug/ssch)
        :return: HelteroData object (see BodyBuilder.build and GarmentBuilder.build for details)
        """
        # Body info
        sequence = self.sequence_loader.load_sequence(fname, betas_id=betas_id)
        # Edges are directed: src, dst
        pos, vel, vertex_type, template, faces, edges, hie_edges, vertex_level, button_edges, random_external = self.garment_builder.build(sequence, idx, garment_name, history=history)
        h_pos, h_vel, h_vertex_type, h_faces, h_vertex_level = self.body_builder.build(sequence, idx, history=history)
        # Load human mask to mask out hands
        filter_dict = readPKL(self.mcfg.smpl_segm)
        filter_idx = ['rightHand', 'rightHandIndex1', 'leftHand', 'leftHandIndex1']
        filter_mask = np.ones((h_pos.shape[1], 1))
        for f_idx in filter_idx:
            masked_idx_list = filter_dict[f_idx]
            for m_idx in masked_idx_list:
                filter_mask[m_idx, 0] = 0

        vel = vel / self.mcfg.dt
        h_vel = h_vel / self.mcfg.dt
        gravity = np.array([0, 0, 0, 0, -9.81, 0]).reshape(1, -1) / ((self.mcfg.fps * self.mcfg.dt) ** 2)

        # Rotate the arrays from yzx to xyz
        pos = self._rotate_vector(pos)
        vel = self._rotate_vector(vel)
        template = self._rotate_vector(template[None, :])[0]
        h_pos = self._rotate_vector(h_pos)
        h_vel = self._rotate_vector(h_vel)
        gravity = self._rotate_vector(gravity[None, :])[0]

        # Format it
        ## Dynamic
        g_types = [np.array([1])]
        names = [garment_name, 'human']
        offset = np.cumsum([0, pos.shape[1], h_pos.shape[1]])
        assert pos.shape[0] == h_pos.shape[0]
        input_states_list = []
        for f_idx in range(pos.shape[0]):
            g_data = np.concatenate([pos[f_idx], vel[f_idx]], axis=-1)
            g_noise = np.zeros_like(pos[f_idx])
            h_data = np.concatenate([h_pos[f_idx], h_vel[f_idx]], axis=-1)
            g_dict = dict(state=g_data, noise=g_noise, external=random_external[f_idx])
            h_dict = dict(h_state=h_data, trans=np.zeros((1,9)), gravity=gravity)
            input_states_list.append(dict(garment=g_dict, forces=h_dict))
        
        dynamic_list = [g_types, names, offset, input_states_list]
        
        # Static
        attr = None # This is placeholder for material model to calculate energy
        f_connectivity, f_connectivity_edges, face2edge_connectivity = get_face_connectivity_combined(faces, padding=True)

        # According to the attribute option
        ## Update related attr to fit material model with inputs
        prior_attr_range = META_ATTR_RANGE
        assert self.attr_option is not None
        if self.attr_option == 'random':
            attr_raw = np.random.choice(DEFAULT_CONFIG_SET, 1)[0]
        elif self.attr_option.isdigit():
            # For inference. This is not to compare with the ground truth yet.
            attr_idx = int(self.attr_option)
            assert attr_idx < len(DEFAULT_CONFIG_SET)
            attr_raw = DEFAULT_CONFIG_SET[attr_idx]
        else:
            # Set a default one
            attr_raw = DEFAULT_CONFIG_SET[0]
        # For EUNet input
        attr = np.array([
            attr_raw['tension_stiffness'] / prior_attr_range['tension'][1],
            attr_raw['bending_stiffness'] / prior_attr_range['bending'][1],
            0, # mass
            attr_raw.get('tension_damping', 0) / prior_attr_range['tension_damping'][1],
            attr_raw.get('bending_damping', 0) / prior_attr_range['bending_damping'][1],
        ]).reshape(1, -1)
        # To match the input format of StVK, but not exactly the same meaning
        lame_mu = torch.from_numpy(np.array([attr_raw['tension_stiffness']], dtype=np.float32))
        lame_mu_input = torch.from_numpy(np.array([attr_raw['tension_stiffness'] / prior_attr_range['tension'][1]], dtype=np.float32))
        lame_lambda = lame_mu
        lame_lambda_input = lame_mu_input
        bending_coeff = torch.from_numpy(np.array([attr_raw['bending_stiffness']], dtype=np.float32))
        bending_coeff_input = torch.from_numpy(np.array([attr_raw['bending_stiffness'] / prior_attr_range['bending'][1]], dtype=np.float32))
        # Training on cloth made of 484 vertices with size 4.8
        # The garments have denser vertices
        mass_density = attr_raw['mass'] * 484 / (4.8**2)
        if self.train_mode:
            mass_range = np.random.rand() * (4-0.25) + 0.25
        else:
            mass_range = 1.0
        mass_density *= mass_range
        mass = get_vertex_mass(template, faces, mass_density).reshape(-1, 1)

        if self.mcfg.get("mass_scalar", None) is not None:
            mass *= self.mcfg.mass_scalar

        face_offset = np.cumsum([0, faces.shape[0], h_faces.shape[0]])
        static_list = self.pack_static(
            offset, face_offset, names, template,
            np.concatenate([faces, h_faces+offset[-2]], axis=0),
            edges, mass,
            lame_mu.detach().cpu().numpy(), lame_lambda.detach().cpu().numpy(), bending_coeff.detach().cpu().numpy(),
            lame_mu_input.detach().cpu().numpy(), lame_lambda_input.detach().cpu().numpy(), bending_coeff_input.detach().cpu().numpy(),
            np.concatenate([vertex_type, h_vertex_type], axis=0), np.concatenate([vertex_level, h_vertex_level], axis=0),
            f_connectivity, f_connectivity_edges, face2edge_connectivity,
            hie_edges, attr=attr, human_vert_mask=filter_mask)
        
        return dynamic_list, static_list
    
    def pack_static(self,
                    mesh_offset, mesh_face_offset, mesh_name, garment_templates, mesh_faces,
                    g_edges, garment_mass,
                    lame_mu_raw, lame_lambda_raw, bending_coef_raw,
                    lame_mu, lame_lambda, bending_coef,
                    vertex_type, vertex_level,
                    f_connectivity, f_connectivity_edges, face2edge_connectivity,
                    hie_edges_list, attr=None, human_vert_mask=None):
        # Parse data
        garment_offset = mesh_offset[0:-1]
        garment_faces_offset = mesh_face_offset[0:-1]
        garment_name = mesh_name[:len(garment_offset)-1]
        human_offset = mesh_offset[-2:]
        human_faces_offset = mesh_face_offset[-2:]
        garment_faces = mesh_faces[garment_faces_offset[0]:garment_faces_offset[-1]] -garment_offset[0]
        # Meta info
        indices = garment_offset
        indices_weight = np.concatenate([
            np.ones(garment_offset[i+1]-garment_offset[i]) / (garment_offset[i+1]-garment_offset[i])
            for i in range(len(garment_offset)-1)],
            axis=0)
        
        ## Get human faces
        human_faces = mesh_faces[human_faces_offset[0]:mesh_face_offset[-1]] - human_offset[0]
        human_faces = human_faces.astype(np.int64)

        garment_num = garment_offset[-1] - garment_offset[0]
        vert_mask = np.ones((garment_num, 1)) # Placeholder

        static_fix_data = dict()
        static_data = dict(
            vert_mask=vert_mask,
            vertex_type=vertex_type,
            vertex_level=vertex_level,
            mass=garment_mass,
            indices=indices,
            indices_weight=indices_weight,
            faces=garment_faces,
            edges=g_edges, # Used to generate graph
            # indices_type=
            h_faces=human_faces,
            templates=garment_templates,
        )
        static_sparse_data = dict()
        garment_num = garment_offset[-1] - garment_offset[0]

        # Hierachy
        if self.mcfg.get("hie", False):
            for i in range(len(hie_edges_list)):
                hie_mask = SparseMask(prefix=f'bistride{i}')
                # src, dst
                cur_hie = hie_edges_list[i]
                # Need dst, src
                hie_mask.add(cur_hie[1], cur_hie[0])
                static_sparse_data.update(hie_mask.get_sparse(garment_num, garment_num, with_value=False, unique=False))
        Dm_inv = make_Dm_inv(torch.from_numpy(garment_templates), torch.from_numpy(garment_faces)).detach().cpu().numpy()
        f_area = get_face_areas(garment_templates, garment_faces).reshape(-1, 1)
        static_data.update(dict(
            Dm_inv=Dm_inv,
            f_area=f_area,
            lame_mu_raw=lame_mu_raw,
            lame_lambda_raw=lame_lambda_raw,
            bending_coef_raw=bending_coef_raw,
            lame_mu=lame_mu,
            lame_lambda=lame_lambda,
            bending_coef=bending_coef,
            f_connectivity=f_connectivity,
            f_connectivity_edges=f_connectivity_edges,
            face2edge_connectivity=face2edge_connectivity,))
        
        # According to the attribute option
        # To fit material related processing and material energy calculation
        if attr is not None:
            static_data['attr'] = attr
        attr_mask = SparseMask(prefix='attr')
        r_p = np.arange(garment_offset[0], garment_offset[-1])
        s_p = np.zeros(r_p.shape[0]).astype(int)
        attr_mask.add(np.array(r_p), np.array(s_p))
        static_sparse_data.update(attr_mask.get_sparse(garment_num, 1, with_value=False))
        if human_vert_mask is not None:
            static_fix_data.update(dict(human_vert_mask=human_vert_mask))
        return static_data, static_fix_data, static_sparse_data, mesh_name, mesh_offset, mesh_face_offset, mesh_faces
    
    def wrap_frames(self, g_types, names, offset, input_states_list, g_noncat_key=[], history=1, step=5):
        history_frame = history
        input_data_list = []
        gt_list = []

        # Parse input_frame: list of dict -> dict of list
        garment_keys = input_states_list[0]['garment'].keys()
        forces_keys = input_states_list[0]['forces'].keys()

        for i in range(history_frame, history_frame+1+step-1):
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

        # Convert to Datacontainer
        input_data_list = np_tensor(input_data_list)
        gt_list = np_tensor(gt_list)
        return input_data_list, gt_list

    def postprocess_static(self, static_data, static_fix_data, static_sparse_data):
        static_data = np_tensor(static_data)
        static_sparse_data = np_tensor(static_sparse_data, to_container=False)
        ## Append fix data
        static_data.update(static_fix_data)
        static_data.update(static_sparse_data)

        return static_data

    
@DATASETS.register_module()
class HoodDataset(BaseDataset):
    def __init__(self, env_cfg, phase, val_seq=10, **kwargs):
        """
        Dataset class for building training and validation samples
        :param loader: Loader object
        :param datasplit: pandas DataFrame with the following columns:
            id: sequence name relative to loader.data_path
            length: number of frames in the sequence
            garment: name of the garment
        :param wholeseq: if True, load the whole sequence, otherwise load a single frame
        """
        self.env_cfg = env_cfg
        self.phase = phase
        self.rollout = env_cfg.get('rollout', False)

        self.history = env_cfg.get("history", 0)
        self.step = env_cfg.get("step", 1)
        self.init_frame = env_cfg.get("init_frame", 1)
        self.omit_frame = env_cfg.get("omit_frame", 1)

        # Init loader
        self.loader = self._init_loader()
        # Init datasplit
        self.datasplit = self._init_split()
        if self.phase == 'val':
            self.datasplit = self.datasplit.iloc[:val_seq]
        self.wholeseq = env_cfg.get("wholeseq", False)

        self._set_len()

    def _init_reader(self):
        return
    
    def _init_loader(self):
        mcfg = self.env_cfg.base
        garment_dict_path = os.path.join(mcfg.aux_data, mcfg.garment_dict_file)
        garments_dict = load_garments_dict(garment_dict_path)

        smpl_model_path = os.path.join(mcfg.aux_data, mcfg.smpl_model)
        smpl_model = smplx.SMPL(smpl_model_path)

        garment_smpl_model_dict = make_garment_smpl_dict(garments_dict, smpl_model)
        obstacle_dict = make_obstacle_dict(mcfg)

        if mcfg.get('single_sequence_file', None) is None:
            mcfg.data_root = os.path.join(mcfg.data_root, mcfg.orig_data_root)
            pass

        if mcfg.get('betas_file', None) is not None:
            betas_table = pickle_load(os.path.join(mcfg.aux_data, mcfg.betas_file))['betas']
        else:
            betas_table = None

        loader = Loader(mcfg, garments_dict,
                        smpl_model, garment_smpl_model_dict, obstacle_dict=obstacle_dict, betas_table=betas_table, train_mode=self.phase=='train')
        return loader
    
    def _init_split(self):
        mcfg = self.env_cfg.base
        if mcfg.get("single_sequence_file", None) is not None:
            datasplit = pd.DataFrame()
            datasplit['id'] = [mcfg.single_sequence_file]
            datasplit['garment'] = [mcfg.single_sequence_garment]
        else:
            split_path = os.path.join(mcfg.aux_data, mcfg.split_path)
            datasplit = pd.read_csv(split_path, dtype='str')
        return datasplit

    def _set_len(self):
        if self.wholeseq:
            self._len = self.datasplit.shape[0]
        else:
            all_lens = self.datasplit.length.tolist()
            # Pad history with 0
            self.all_lens = [int(x) - max(7, self.init_frame+self.omit_frame+self.history) for x in all_lens]
            self._len = sum(self.all_lens)

    def filter_data(self, seq_list=[]):
        self.datasplit = self.datasplit.iloc[seq_list]
        self._set_len()

    def _find_idx(self, index: int) -> Tuple[str, int, str]:
        """
        Takes a global index and returns the sequence name, frame index and garment name for it
        """
        fi = 0
        while self.all_lens[fi] <= index:
            index -= self.all_lens[fi]
            fi += 1
        return self.datasplit.iloc[fi].id, index, self.datasplit.iloc[fi].garment, self.datasplit.iloc[fi].name

    def __getitem__(self, item: int) -> Dict:
        """
        Load a sample given a global index
        """

        betas_id = None
        if self.wholeseq:
            fname = self.datasplit.id[item]
            garment_name = self.datasplit.garment[item]
            idx = 0

            if 'betas_id' in self.datasplit:
                betas_id = int(self.datasplit.betas_id[item])
        else:
            fname, idx, garment_name, file_idx = self._find_idx(item)

        dynamic_list, static_list = self.loader.load_sample(fname, idx, garment_name, betas_id=betas_id, history=self.history)
        input_data_list, gt_list = self.loader.wrap_frames(dynamic_list[0], dynamic_list[1], dynamic_list[2], dynamic_list[3], g_noncat_key=['external'], history=self.history, step=self.step)
        static_data = self.loader.postprocess_static(static_list[0], static_list[1], static_list[2])
        static_data['indices_type'] = np_tensor(np.array(dynamic_list[0]))

        inputs = dict(
            dynamic=input_data_list,
            static=static_data,
        )

        gt_label = gt_list
        meta=dict(sequence=int(file_idx), frame=idx+1)

        input_data = dict(
            inputs=inputs,
            gt_label=gt_label,
            meta=meta,)
        return input_data
    
    def __len__(self) -> int:
        return self._len
    
    def prepare_rollout(self, current_garment, current_human, batch_data, **kwargs):
        if current_garment is not None:
            batch_data['inputs']['dynamic'][0]['state'] = [np_tensor(current_garment)]
        return batch_data