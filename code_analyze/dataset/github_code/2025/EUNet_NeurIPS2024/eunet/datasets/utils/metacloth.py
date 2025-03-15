from itertools import chain
import os
import numpy as np
from pickle import UnpicklingError
from eunet.datasets.utils import readJSON, readPKL, readOBJ, writePKL
from eunet.datasets.utils.mesh import floyd_map

from .phys import get_face_connectivity_combined


META_FN = 'meta.json'

GARMENT_TYPE = [
    'Plane',
]

TYPE_NAMES = [
    'silk', 'leather', 'denim', 'cotton']

ATTR_RANGE = dict(
    tension=(15, 300),
    bending=(15, 150),
    mass=(0.2, 3.0),
    tension_damping=(0, 30),
    bending_damping=(0, 30),
)

DEFAULT_CONFIG_SET = [
    dict( # Silk
        tension_stiffness=5.,
        compression_stiffness=5.,
        shear_stiffness=5.,
        bending_stiffness=0.5,
        mass=0.15,
        tension_damping=0.0,
        compression_damping=0.0,
        shear_damping=0.0,
        bending_damping=0.0,),
    dict( # Leather
        tension_stiffness=80.,
        compression_stiffness=80.,
        shear_stiffness=80.,
        bending_stiffness=150.,
        mass=0.4,
        tension_damping=25.0,
        compression_damping=25.0,
        shear_damping=25.0,
        bending_damping=0.0,),
    dict( # Denim
        tension_stiffness=40.,
        compression_stiffness=40.,
        shear_stiffness=40.,
        bending_stiffness=10.,
        mass=1.0,
        tension_damping=25.0,
        compression_damping=25.0,
        shear_damping=25.0,
        bending_damping=0.0,),
    dict( # Cotton
        tension_stiffness=15.,
        compression_stiffness=15.,
        shear_stiffness=15.,
        bending_stiffness=0.5,
        mass=0.3,
        tension_damping=5.0,
        compression_damping=5.0,
        shear_damping=5.0,
        bending_damping=0.0,),
]

class MetaClothReader:
    def __init__(self, env_cfg, phase='train'):
        self.cfg = env_cfg
        self.phase = phase
        self.data_dir = env_cfg.root_dir
        self.generated_dir = env_cfg.generated_dir
        self.seq_list = []
        split_meta = readJSON(env_cfg.split_meta)
        if isinstance(phase, list):
            phase_meta = list(chain(*[split_meta[i] for i in phase]))
        else:
            phase_meta = split_meta[phase]
        for seq_num in phase_meta:
            self.seq_list.append(seq_num)

        self.garment_type = dict()
        one_hot = np.eye(len(GARMENT_TYPE))
        for g_name, g_code in zip(GARMENT_TYPE, one_hot):
            self.garment_type[g_name] = g_code
        
        self.template_meta = readJSON(os.path.join(env_cfg.garment_dir, env_cfg.meta_name))
    
    """ 
	Read sample info 
	Input:
	- sample: name of the sample e.g.:'01_01_s0'
	"""
    def read_info(self, sample):
        info_path = os.path.join(self.data_dir, sample, META_FN)
        infos = readJSON(info_path)
        # Add one hot coding
        infos['cloth']['type'] = self.garment_type[infos['cloth']['name']]
        return infos
    
    def read_pin_verts(self, sample=None, garment=None):
        pin_idx = np.array(self.template_meta['pin_verts'])
        return pin_idx
	
    def read_garment_vertices(self, sample, garment, frame=None):
		# Read garment vertices (relative to root joint)
        garment_path = os.path.join(self.data_dir, sample, garment + '.pkl')
        if not os.path.exists(garment_path):
            assert False
            return None
        garment_seq = readPKL(garment_path)
        V = garment_seq['vertices']
        if frame is not None:
            V = garment_seq['vertices'][frame]
        return V
    
    def read_gravity(self, sample, info=None):
        if info is None:
            info = self.read_info(sample)
        g = self.cfg.get('gravity_override', None)
        if g is None:
            g = info['gravity']
        gravity = np.array(g) / ((self.cfg.fps*self.cfg.dt)**2)
        return gravity

    def read_garment_topology(self, sample, garment, info=None):
		# Read OBJ file
        template_path = os.path.join(self.cfg.garment_dir, self.cfg.mesh_name)
        if info is None:
            info = self.read_info(sample)
        template_scale = info['cloth']['scale']
        T, F, _, _ = readOBJ(template_path)
        T *= template_scale
        F = np.array(F)
        return F, T

    def read_garment_attributes(self, sample, garment, info=None):
		# Read garment vertices (relative to root joint)
        if info is None:
            info = self.read_info(sample)

        g_meta = info['cloth']
        assert g_meta['name'] == garment
        attr = np.array([
            g_meta['cloth']['tension_stiffness'] / ATTR_RANGE['tension'][1],
            g_meta['cloth']['bending_stiffness'] / ATTR_RANGE['bending'][1],
            g_meta['cloth']['mass'],
            g_meta['cloth'].get('tension_damping', 0) / ATTR_RANGE['tension_damping'][1],
            g_meta['cloth'].get('bending_damping', 0) / ATTR_RANGE['bending_damping'][1]])
        return attr
    
    def read_garment_obj(self, sample, garment, info=None):
        if info is None:
            info = self.read_info(sample)
        g_cfg = None
        for i in range(len(info['simulate']['garment'])):
            cur_cfg = info['simulate']['garment'][i]
            if cur_cfg['name'] == garment:
                g_cfg = cur_cfg
                break
        assert g_cfg is not None
        obj_path = os.path.join(self.cfg.garment_dir, g_cfg['mesh_path'])
        V, F, Vt, Ft = readOBJ(obj_path)
        return V, F
    
    def read_garment_vertices_topology(self, sample, garment, frame):
		# Read garment vertices (relative to root joint)
        garment_path = os.path.join(self.data_dir, sample, garment + '.pkl')
        garment_seq = readPKL(garment_path)
        V = garment_seq['vertices'][frame]
        F = garment_seq['faces']
        F, T = self.read_garment_topology(sample, garment)
        return V, F, T
    
    def read_garment_polygon_params(self, sample, garment, padding=True):
        padding_suffix = '_npad' if not padding else '_pad'
        phys_dir = os.path.join(self.cfg.generated_dir, sample)
        os.makedirs(phys_dir, exist_ok=True)
        phys_path = os.path.join(phys_dir, f"{garment}_{self.cfg.mesh_name.replace('.obj', '')}_polygon{padding_suffix}.pkl")
        try:
            data = readPKL(phys_path)
        except (UnpicklingError, FileNotFoundError, EOFError):
            F, T = self.read_garment_topology(sample, garment)
            f_connectivity, f_connectivity_edges, face2edge_connectivity = get_face_connectivity_combined(F, padding=padding)
            data = {
                'f_connectivity': f_connectivity,
                'f_connectivity_edges': f_connectivity_edges,
                'face2edge_connectivity': face2edge_connectivity}
            writePKL(phys_path, data)

        return data['f_connectivity'], data['f_connectivity_edges'], data['face2edge_connectivity']
    
    def read_garment_floyd(self, sample, garment):
        floydpath = os.path.join(self.cfg.garment_dir, f"{garment}_{self.cfg.mesh_name.replace('.obj', '')}_floyd_dict.pkl")
        try:
            data = readPKL(floydpath)
        except (UnpicklingError, FileNotFoundError, EOFError):
            faces, _ = self.read_garment_topology(sample, garment)
            floyd_dist = floyd_map(faces)
            data = {
                'floyd_dist': floyd_dist.astype(np.int64).tolist()}
            writePKL(floydpath, data)
        
        return data['floyd_dist']