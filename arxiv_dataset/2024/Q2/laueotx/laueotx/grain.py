import numpy as np, os
from laueotx.utils import logging as utils_logging
from laueotx import crystalography, laue_math
from laueotx import rodrigues_space
LOGGER = utils_logging.get_logger(__file__)
from laueotx.config import ALGEBRA_ENGINE, PRECISION

if ALGEBRA_ENGINE == 'tensorflow':
    import tensorflow as tf

# from memory_profiler import profile

def load_HKL_data(file_hkl):

    if not os.path.isabs(file_hkl):

        package_root = os.path.dirname(os.path.abspath(__file__))
        file_hkl = os.path.join(package_root, '../hkl_data/', file_hkl)

    assert os.path.isfile(file_hkl), f'file {file_hkl} not found'
    
    dtype = [('hkl', 'a16'),
             ('d_hkl', 'f8'),
             ('F2', 'f8'),
             ('M', 'i4')]
    data_hkl_load =np.genfromtxt(file_hkl, skip_header=1, dtype=dtype)

    hkl_plane = np.zeros([len(data_hkl_load), 3], dtype=PRECISION)
    for i in range(len(data_hkl_load)):
        hkl_plane[i,:] = np.fromstring(str(data_hkl_load['hkl'][i], 'utf-8').strip('[').strip(']'), sep=',')

    dtype = [('hkl', 'f4', 3),
             ('d_hkl', 'f4'),
             ('F2', 'f4'),
             ('M', 'i4'),
             ('hkl_norm', 'f4')]

    data_hkl = np.empty(len(data_hkl_load), dtype=dtype)
    data_hkl['hkl'] = hkl_plane
    data_hkl['d_hkl'] = data_hkl_load['d_hkl']
    data_hkl['F2'] = data_hkl_load['F2']
    data_hkl['M'] = data_hkl_load['M']
    data_hkl['hkl_norm'] = np.linalg.norm(hkl_plane, axis=1)

    return data_hkl




class Grain():

    def __init__(self, material, file_hkl, beamline, rs=None, x=None, x_tol=[[2.5,2.5,2.5]], tetragonal=False, rotation_type='rp'):
        """
        Initialize a grain with given material and hkl specs. 
        :param material: string name of the material (see crystalography.py)
        :param file_hkl: file name of the hkl specifications of the material
        :param beamline: Beamline object (see beamline.py)
        :param r: rotation matrix, or tensor of rotations
        :param x: position, or tensor of positions
        :param tetragonal: if to use tatragonal geometry
        """

        self.material = material
        self.beamline = beamline
        self.tetragonal = tetragonal
        self.material_const = crystalography.crystalography_data[self.material]
        self.x_tol = np.array(x_tol)
        self.rotation_type = rotation_type

        # TODO: add eqn references
        self.B = laue_math.get_B_matrix(a=self.material_const['a'],
                                        b=self.material_const['b'],
                                        c=self.material_const['c'],
                                        alpha=np.radians(self.material_const['alpha']),
                                        beta=np.radians(self.material_const['beta']),
                                        gamma=np.radians(self.material_const['gamma']))

        self.Ba = self.B * 2. * np.pi

        self.hkl = load_HKL_data(file_hkl)
        self.hkl_planes = self.hkl['hkl']
        self.n_planes = len(self.hkl)

        # TODO: add eqn references
        self.v = np.dot(self.Ba, self.hkl_planes.transpose())

        # TODO: add eqn references
        self.inv_v = 1./(self.v[0,:]**2 + self.v[1,:]**2 + self.v[2,:]**2)

        # here encode limits
        if self.tetragonal:
            self.lims = np.array([1., 1., np.sqrt(2)-1.], dtype=PRECISION)

        else:
            if self.rotation_type == 'rp':
                self.lims = np.full(3, fill_value=np.sqrt(2)-1., dtype=PRECISION)
            elif self.rotation_type == 'mrp':
                self.lims = np.full(3, fill_value=0.18568445, dtype=PRECISION)
            # self.lims = np.ones(3, dtype=PRECISION)

        if rs is None:
            self.rs = self.r = self.ubr = self.lbr = None
        else:
            self.set_orientation(rs)

        if x is None:    
            self.x = self.x_Omega = None
        else:
            self.set_position(x)

    def set_orientation(self, rs):

        self.rs = np.array(rs, dtype=PRECISION)
        self.lbrs, self.ubrs = rodrigues_space.generate_constraints(self.rs)
        self.orientation_constraints()

    def set_position(self, x):

        self.x = np.atleast_2d(x).astype(PRECISION)
        # self.x_Omega0 = np.einsum('ojk,ik->oj', self.beamline.O, self.x)
        # self.x_Omega0 = np.einsum('ojk,ik->oj', self.beamline.O, self.x[[0]])
        self.x_Omega = np.einsum('oij,...j->...oi', self.beamline.O, self.x)
        pass

    def get_n_batches(self, batch_size):

        return int(np.ceil(len(self.rs)/batch_size))


    def orientation_constraints(self, permute=False):
        """
        TODO: add math reference
        No idea if the name of this fuction is suitable
        """

        self.lbr = self.lbrs * np.expand_dims(self.lims, axis=0)
        self.ubr = self.ubrs * np.expand_dims(self.lims, axis=0)
        self.r   = self.rs   * np.expand_dims(self.lims, axis=0)

    def get_laue_diffraction_rays_sample(self, batch=np.s_[:None]):

        assert self.r is not None, "set rotation first by calling grain.set_orientation(rs)"

        L, select_bragg = laue_math.laue_continuous_source(r=self.r[batch],
                                                           v=self.v,
                                                           inv_v=self.inv_v,
                                                           om=self.beamline.omegas_sincos,
                                                           lim_lambda=[self.beamline.lambda_min, self.beamline.lambda_max], 
                                                           rotation_type=self.rotation_type)

        
        return L, select_bragg

    def get_laue_diffraction_rays_lab(self, L=None, select_bragg=None, batch=np.s_[:None]):

        if L is None and select_bragg is None:
            L, select_bragg = self.get_laue_diffraction_rays_sample(batch=batch)


        if ALGEBRA_ENGINE == 'numpy':
        
            L_lab = np.einsum('oij,...opj->...opi', self.beamline.O, L)
        
        elif ALGEBRA_ENGINE == 'tensorflow':
        
            with tf.device('gpu'):
                
                L_lab = tf.einsum('oij,...opj->...opi', self.beamline.O, L)

        return L_lab, select_bragg

    # @profile
    def generate_diffraction_spots(self, L=None, select_bragg=None, batch=np.s_[:None]):

        assert self.x_Omega is not None, "set position first by calling grain.set_position(xs)"

        # get the rays in lab coords
        L_lab, select_bragg = self.get_laue_diffraction_rays_lab(L=L, select_bragg=select_bragg, batch=batch)

        # get the positions of spots
        spot_positions, in_detector = self.beamline.get_spots_positions(L_lab, self.x_Omega[batch], select_bragg)

        return spot_positions, select_bragg, in_detector

    def select_and_split_spots_by_detector(self, spot_positions, select_bragg, in_detector):

        list_spot_split = []
        for i in range(self.beamline.detectors.n_detectors):

            sp_ = spot_positions[:,:,i,:]
            select = select_bragg & in_detector[:,:,i]
            list_spot_split.append(sp_[select])

        return list_spot_split
