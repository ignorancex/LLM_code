import os
from itertools import chain
from mmcv import Config
import numpy as np
from torch import zeros_like
from eunet.datasets.utils import readJSON, readPKL, readOBJ, writePKL, quads2tris
from eunet.datasets.smpl import SMPLModel
from .cloth3d_util import loadInfo, zRotMatrix, proj, readPC2Frame
from pickle import UnpicklingError
from eunet.datasets.utils.mesh import laplacianDict
import shutil
from .metacloth import ATTR_RANGE
from .phys import get_face_connectivity_combined, get_face_areas
from .hood_coarse import make_coarse_edges
from .hood_common import random_between_log


FABRIC_DAMPING = {
	# Only for testing
	'cotton': [15, 0.5, 0.3, 5, 0],
	'denim': [40, 10, 1.0, 25, 0],
	'leather': [80, 150, 0.4, 25, 0],
	'rubber': [15, 25, 3, 25, 0],
	'silk': [5, 0.5, 0.15, 0, 0],
	
}

GARMENT_TYPE = [
	'Tshirt', 
	'Top',
	'Jumpsuit',
	'Dress',
]


class Cloth3DReader:
	def __init__(self, clothenv_cfg, meta_path, phase='train'):
		self.cfg = clothenv_cfg
		self.phase = phase
		self.data_dir = clothenv_cfg.root_dir
		self.generated_dir = clothenv_cfg.generated_dir
		with open(meta_path, 'r') as f:
			meta = f.readlines()
		self.seq_list = [i.split('\t')[0] for i in meta]
			
		self.smpl = {
			'f': SMPLModel(os.path.join(self.cfg.smpl_dir, 'model_f.pkl')),
			'm': SMPLModel(os.path.join(self.cfg.smpl_dir, 'model_m.pkl'))
		}

		self.garment_type = dict()
		one_hot = np.eye(len(GARMENT_TYPE))
		for g_name, g_code in zip(GARMENT_TYPE, one_hot):
			self.garment_type[g_name] = g_code
		
	""" 
	Read sample info 
	Input:
	- sample: name of the sample e.g.:'01_01_s0'
	"""
	def read_info(self, sample):
		info_path = os.path.join(self.data_dir, sample, 'info')
		if not os.path.exists(info_path+'.mat'):
			new_info_path = os.path.join(self.data_dir, sample, 'info')
			shutil.copy(new_info_path+'.mat', info_path+'.mat')
			info_path = new_info_path

		return loadInfo(info_path)
	
	def parse_garment_names(self, sample=None, info=None):
		if info is None:
			info = self.read_info(sample)
		g_names = list(info['outfit'].keys())
		return g_names
		
	""" Human data """
	"""
	Read SMPL parameters for the specified sample and frame
	Inputs:
	- sample: name of the sample
	- frame: frame number
	"""
	def read_smpl_params(self, sample, frame):
		# Read sample data
		info = self.read_info(sample)
		# SMPL parameters
		gender = 'm' if info['gender'] else 'f'
		if len(info['poses'].shape) == 1: frame = None
		pose = info['poses'][:, frame].reshape(self.smpl[gender].pose_shape)
		shape = info['shape']
		trans = info['trans'][:, frame].reshape(self.smpl[gender].trans_shape)
		return gender, pose, shape, trans
	
	"""
	Computes human mesh for the specified sample and frame
	Inputs:
	- sample: name of the sample
	- frame: frame number
	Outputs:
	- V: human mesh vertices
	- F: mesh faces
	"""
	def read_human(self, sample, frame=None, absolute=False, rot_z=False):
		# Read sample data
		info = self.read_info(sample)
		# SMPL parameters
		gender, pose, shape, trans = self.read_smpl_params(sample, frame)
		# Compute SMPL
		_, V, _ = self.smpl[gender].set_params(pose=pose, beta=shape, trans=trans if absolute else None, with_body=True)
		J = self.smpl[gender].J
		# V -= J[0:1] # This may not need, since our smpl model already minus this; thus J[0:1] == 0
		# Apply rotation on z-axis
		F = self.smpl[gender].faces.copy()
		if rot_z:
			zRot = zRotMatrix(info['zrot'])
			return zRot.dot(V.T).T, F
		else:
			return V, F

	def read_human_joints(self, sample, frame, with_body=False):
		# compute
		# SMPL parameters
		gender, pose, shape, trans = self.read_smpl_params(sample, frame)
		# Compute SMPL
		G, V, _ = self.smpl[gender].set_params(pose=pose, beta=shape, trans=trans, with_body=with_body)
		J = self.smpl[gender].J.copy()
		weights = self.smpl[gender].weights.copy()
		return J, G, V, weights
	
	""" Garment data """
	"""
	Reads garment vertices location for the specified sample, garment and frame
	Inputs:
	- sample: name of the sample
	- garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
	- frame: frame number
	- absolute: True for absolute vertex locations, False for locations relative to SMPL root joint	
	Outputs:
	- V: 3D vertex locations for the specified sample, garment and frame
	"""
	def read_garment_vertices(self, sample, garment, frame=None, absolute=False, rot_z=False):
		# Read garment vertices (relative to root joint)
		pc16_path = os.path.join(self.data_dir, sample, garment + '.pc16')
		V = readPC2Frame(pc16_path, frame, True)
		# Read sample data
		info = self.read_info(sample)		
		if absolute:
			# Transform to absolute
			if len(info['trans'].shape) == 1: frame = None
			V += info['trans'][:,frame].reshape((1,3))
		# Apply rotation on z-axis
		if rot_z:
			zRot = zRotMatrix(info['zrot'])
			return zRot.dot(V.T).T
		else:
			return V
	
	def read_garment_template(self, sample, garment, absolute=False, rot_z=False):
		# Read garment vertices (relative to root joint)
		obj_path = os.path.join(self.data_dir, sample, garment + '.obj')
		V = readOBJ(obj_path)[0]
		# Read sample data
		info = self.read_info(sample)		
		if absolute:
			# Transform to absolute
			if len(info['trans'].shape) == 1: frame = None
			V += info['trans'][:,frame].reshape((1,3))
		# Apply rotation on z-axis
		if rot_z:
			return V
		else:
			zRot = zRotMatrix(-info['zrot'])
			return zRot.dot(V.T).T
			

	"""
	Reads garment faces for the specified sample and garment
	Inputs:
	- sample: name of the sample
	- garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
	Outputs:
	- F: mesh faces	
	"""
	def read_garment_topology(self, sample, garment, triangle=True):
		# Read OBJ file
		obj_path = os.path.join(self.data_dir, sample, garment + '.obj')
		assert os.path.exists(obj_path)
		T, F, _, _ = readOBJ(obj_path)
		if triangle:
			F = quads2tris(F)
		return F, T

	def read_garment_pinverts(self, sample, garment):
		path = os.path.join(self.data_dir, sample, f'{garment}_pin_vert.json')

		if not os.path.exists(path):
			return None
		
		with open(path, 'r') as f:
			pin_dict = readJSON(path)
		return pin_dict['pin_vert']
	
	"""	
	Reads garment UV map for the specified sample and garment
	Inputs:
	- sample: name of the sample
	- garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
	Outputs:
	- Vt: UV map vertices
	- Ft: UV map faces		
	"""
	def read_garment_UVMap(self, sample, garment):
		# Read OBJ file
		obj_path = os.path.join(self.data_dir, sample, garment + '.obj')
		return readOBJ(obj_path)[2:]

	def read_camera(self, sample):
		# Read sample data
		info = self.read_info(sample)
		# Camera location
		# TODO: clean meta
		camLoc = info['scene']['camera']
		# Camera projection matrix
		# TODO: camera 
		return proj(camLoc)
	
	def read_garment_polygon_params(self, sample, garment, padding=False):
		# Only for test
		padding_suffix = '_npad' if not padding else '_pad'
		phys_dir = os.path.join(self.generated_dir, sample)
		os.makedirs(phys_dir, exist_ok=True)
		phys_path = os.path.join(phys_dir, f"{garment}_polygon{padding_suffix}.pkl")
		try:
			data = readPKL(phys_path)
		except (UnpicklingError, FileNotFoundError, EOFError):
			F, T = self.read_garment_topology(sample, garment)
			f_area = get_face_areas(T, F).reshape(-1, 1)
			f_connectivity, f_connectivity_edges, face2edge_connectivity = get_face_connectivity_combined(F, padding=padding)
			data = {
				'f_area': f_area,
				'f_connectivity': f_connectivity,
				'f_connectivity_edges': f_connectivity_edges,
				'face2edge_connectivity': face2edge_connectivity}
			writePKL(phys_path, data)

		return data['f_area'], data['f_connectivity'], data['f_connectivity_edges'], data['face2edge_connectivity']
	
	def read_garment_attributes_metacloth(self, sample, garment, info=None):
		# Read garment vertices (relative to root joint)
		if info is None:
			info = self.read_info(sample)
		fabric = np.float32(FABRIC_DAMPING[info['outfit'][garment]['fabric']])
		attr = np.array([
			fabric[0] / ATTR_RANGE['tension'][1],
			fabric[1] / ATTR_RANGE['bending'][1],
			fabric[2],
			# Previously got 2 slots for self friction and friction with human
			# Now add damping attr
			fabric[3] / ATTR_RANGE['tension_damping'][1],
			fabric[4] / ATTR_RANGE['bending_damping'][1]])
		return attr
	
	def read_garment_type(self, garment):
		if garment not in self.garment_type.keys():
			return None
		return self.garment_type[garment]
	
	def read_garment_laplacianDict(self, sample, garment, faces=None):
		assert faces is not None, "Need to control the scale effects"
		if faces is None:
			faces, _ = self.read_garment_topology(sample, garment)
		lap_path = os.path.join(self.cfg.generated_dir, sample, f'{garment}_lap_dict.pkl')
		try:
			data = readPKL(lap_path)
		except (UnpicklingError, FileNotFoundError, EOFError):
			laplacian_mask_dict, cross_mask_dict, edge_neighbor_dict = laplacianDict(faces, with_neighbor=False)
			data = {
				'lap_mask': laplacian_mask_dict,
				'cross_mask': cross_mask_dict,
				'edge_neighbor': edge_neighbor_dict}
			writePKL(lap_path, data)
		
		return data['lap_mask'], data['cross_mask'], data['edge_neighbor']
	
	def read_garment_var_params(self, sample, garment):
		lame_mu, lame_mu_normed = random_between_log(23600.0, 23600.0, shape=[1], return_norm=True)
		lame_lambda, lame_lambda_normed = random_between_log(44400.0, 44400.0, shape=[1], return_norm=True)
		return lame_mu.detach().cpu().numpy(), lame_mu_normed.detach().cpu().numpy(), lame_lambda.detach().cpu().numpy(), lame_lambda_normed.detach().cpu().numpy()
	
	def read_coarse_edges(self, sample, garment, num_verts, n_coarse_levels, faces=None):
		"""
		Add coarse edges to `sample` as `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`.
		where `i` is the number of the coarse level (starting from `0`)

		:param sample: HeteroData
		:param garment_name:
		:return: sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]
		"""
		if n_coarse_levels == 0:
			return None
		
		hie_path = os.path.join(self.generated_dir, sample, f"hie_{garment}_{str(n_coarse_levels)}.pkl")
		try:
			data = readPKL(hie_path)
		except (UnpicklingError, FileNotFoundError, EOFError):
			if faces is None:
				faces, _ = self.read_garment_topology(sample, garment)
			# Randomly choose center of the mesh
			# center of a graph is a node with minimal eccentricity (distance to the farthest node)
			center_nodes = np.random.choice(num_verts)
			center = center_nodes

			coarse_edges_dict = make_coarse_edges(faces, center, n_levels=n_coarse_levels)
			# for each level `i` add edges to sample as  `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`
			coarse_edge_list = []
			for i in range(n_coarse_levels):
				key = f'coarse_edge{i}'
				edges_coarse = coarse_edges_dict[i].astype(np.int64)
				edges_coarse = np.concatenate([edges_coarse, edges_coarse[:, [1, 0]]], axis=0)
				# coarse_edges = torch.tensor(edges_coarse.T)
				coarse_edges = edges_coarse.T
				coarse_edge_list.append(coarse_edges)
			data = dict(coarse_edges=coarse_edge_list, center=center)
			writePKL(hie_path, data)
		
		return data['coarse_edges']
		
# TESTING
if __name__ == '__main__':
	sample = '135_02_s8'
	frame = 0
	garment = 'Tshirt'
	
	config_path = '/home/ydshao/VirtualenvProjects/MMGarment/configs/_base_/datasets/clothenv.py'
	phase = ''
	cfg = Config.fromfile(config_path)
	
	reader = ClothEnvReader(cfg.clothenv_base, phase=phase)
	F = reader.read_garment_topology(sample, garment)
	Vt, Ft = reader.read_garment_UVMap(sample, garment)