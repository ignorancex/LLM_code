import numpy as np
from laueotx.utils import logging as utils_logging
from laueotx import laue_math
from laueotx.config import ALGEBRA_ENGINE, PRECISION
LOGGER = utils_logging.get_logger(__file__)

if ALGEBRA_ENGINE == 'tensorflow':
    import tensorflow as tf

class Detector():


    def __init__(self, side_length, position, tilt, detector_id=0, hole_diameter=20, tol_position=10, tol_tilt=10, num_pixels=4000, detector_type='forward'):
        """
        Constructor.
        :param tol_tilt:
        :param tol_position:
        """

        self.detector_type = detector_type
        self.num_pixels = num_pixels
        self.detector_id = detector_id
        self.tol_tilt = tol_tilt # deg
        self.tol_position = tol_position # 
        if self.detector_type == 'forward':
            self.orientation = 1
        elif self.detector_type == 'backward':
            self.orientation = -1

        self.normal = np.array([-1*self.orientation, 0, 0], dtype=PRECISION) #Vector normal to the detector plane (100 is backdiffraction -100 is forward diffraction)
        self.tilt = np.array(tilt) #detector tilts on the x y and z axis guess [rad]
        self.tilt_tol_arr = np.tile(np.radians(self.tol_tilt), reps=[1,3])
        self.side_length = side_length #side of a square deetector [mm]
        self.hole_diameter = hole_diameter #diameter of the hole [mm]
        self.position = np.array(position, dtype=PRECISION) #detector position with respect the rotation axis [mm]. Negative is backdiffraction.
        self.lo_position_tol = self.position - self.tol_position
        self.hi_position_tol = self.position + self.tol_position

        self.set_rotation_matrix()

    def set_rotation_matrix(self):

        self.rotation_matrix = laue_math.rotation_matrix(self.tilt)
        self.inverse_rotation_matrix = np.linalg.inv(self.rotation_matrix)
        self.rotated_normal = np.dot(self.rotation_matrix, self.normal)

    # @profile
    def plane_line_intersect(self, L_lab, x0, select_bragg=None):

        # TODO: change varialbe names to something more informative
        w = x0 - self.position
        N = -np.sum(np.expand_dims(self.rotated_normal, axis=0) * w, axis=1, keepdims=True)
        D = np.sum(np.expand_dims(self.rotated_normal, axis=[0,1]) * L_lab, axis=2)
        sI = N/D
        I = np.expand_dims(x0, axis=1) + np.expand_dims(sI, axis=-1) * L_lab
        return I

    def get_spots_positions(self, L_lab, x0, select_bragg=None):
        """
        Get spot positions on the detector
        :param L_lab: unit rays in lab refrence frame
        :param x0: 3D grain positions
        :param select_bragg: flag indicating whether the spot meets the Bragg's condition
        :return spot_pos_rot: positions of spots on the detector in 3D coordinates
        """

        I = self.plane_line_intersect(L_lab, x0, select_bragg=select_bragg)
        Ip = I - np.expand_dims(self.position, axis=[0,1])
        spot_pos = np.einsum('opj,ij->opi', Ip, self.inverse_rotation_matrix)
        # spot_pos[0,select_bragg[0],:]
        scattering_direction = np.sign(np.sum(L_lab * self.normal, axis=-1)) == self.orientation 
        # scattering_direction[0,select_bragg[0]]
        spot_x, spot_y = spot_pos[...,1], spot_pos[...,2]
        in_detector_x = np.abs(spot_x) < self.side_length/2 
        in_detector_y = np.abs(spot_y) < self.side_length/2 
        not_in_hole = np.sqrt(spot_x**2 + spot_y**2) > self.hole_diameter/2

        in_detector = scattering_direction & in_detector_x & in_detector_y & not_in_hole

        spot_pos_rot = np.einsum('opj,ij->opi', spot_pos, self.rotation_matrix) + np.expand_dims(self.position, axis=[0,1])

        return spot_pos_rot

    def rotate(self, tilt_new):

        self.tilt = tilt_new.astype(PRECISION)
        self.set_rotation_matrix()

    def shift(self, position_new):

        self.position = position_new.astype(PRECISION)

    def rotate_and_shift(self, tilt_new, position_new):
        
        self.rotate(tilt_new)
        self.shift(position_new)


class MultipleDetectors():

    def __init__(self, detectors):

        self.detectors = detectors
        self.n_detectors = len(self.detectors)
        for d in self.detectors:
            LOGGER.debug(f'created detector: {d.detector_type}')

        self.detector_type = f'multi_{self.n_detectors}'
        self.tol_tilt = np.array([d.tol_tilt for d in self.detectors])
        self.tol_position = np.array([d.tol_position for d in self.detectors])
        self.orientation = np.array([d.orientation for d in self.detectors])
        self.normal = np.array([d.normal for d in self.detectors])
        self.tilt_tol_arr = np.array([d.tilt_tol_arr for d in self.detectors])[:,0,:]
        self.side_length = np.array([d.side_length for d in self.detectors])
        self.hole_diameter = np.array([d.hole_diameter for d in self.detectors])
        self.lo_position_tol = np.array([d.lo_position_tol for d in self.detectors])
        self.hi_position_tol = np.array([d.hi_position_tol for d in self.detectors])
        self.detectors_dict = {d.detector_type: d for d in self.detectors}

        self.update()

    def update(self):

        self.tilt = np.array([d.tilt for d in self.detectors])
        self.position = np.array([d.position for d in self.detectors])
        self.rotation_matrix = np.array([d.rotation_matrix for d in self.detectors])
        self.inverse_rotation_matrix = np.array([d.inverse_rotation_matrix for d in self.detectors])
        self.rotated_normal = np.array([d.rotated_normal for d in self.detectors])

    # @profile
    def plane_line_intersect(self, L_lab, x0):
        # TODO: change varialbe names to something more informative
        # "Practical Geometry Algorithms" Sunday Page 61 Line-Plane Intersection

        if ALGEBRA_ENGINE == 'tensorflow':

            with tf.device('gpu'):

                w = tf.expand_dims(x0, axis=-2) - tf.expand_dims(self.position, axis=0)
                N = -tf.reduce_sum(tf.expand_dims(self.rotated_normal, axis=0) * w, axis=-1)
                D = tf.reduce_sum(self.rotated_normal * tf.expand_dims(L_lab, axis=-2), axis=-1)
                sI = tf.expand_dims(N, axis=-2)/D
                I = tf.expand_dims(tf.expand_dims(x0, axis=-2), axis=-2) + tf.expand_dims(sI, axis=-1) * tf.expand_dims(L_lab, axis=-2)


        elif ALGEBRA_ENGINE == 'numpy':

            # w0 = np.expand_dims(x0[0], axis=1) - np.expand_dims(self.position, axis=0)
            delta_x =  np.expand_dims(self.position, axis=0) - np.expand_dims(x0, axis=-2)
            # N0 = -np.sum(np.expand_dims(self.rotated_normal, axis=0) * w0, axis=2)
            N = np.sum(np.expand_dims(self.rotated_normal, axis=0) * delta_x, axis=-1)
            # N = -np.sum(np.expand_dims(self.rotated_normal, axis=0) * w, axis=2)

            # D0 = np.sum(np.expand_dims(self.rotated_normal, axis=[0,1]) * np.expand_dims(L_lab[0], axis=2), axis=3)
            D = np.sum(self.rotated_normal * np.expand_dims(L_lab, axis=-2), axis=-1)
            sI = np.expand_dims(N, axis=-2)/D
            # sI0 = np.expand_dims(N0, axis=-2)/D0
            # sI0 = np.expand_dims(N, axis=1)/D[0]
            # I0 = np.expand_dims(x0[0], axis=[1,2]) + np.expand_dims(sI0, axis=3) * np.expand_dims(L_lab[0], axis=2)
            I = np.expand_dims(x0, axis=[-2,-3]) + np.expand_dims(sI, axis=-1) * np.expand_dims(L_lab, axis=-2)

        return I

    # @profile
    def get_spots_positions(self, L_lab, x0, select_bragg=None, reference_frame='beamline'):
        """
        Get spot positions on the detector
        :param L_lab: unit rays in lab refrence frame
        :param x0: 3D grain positions
        :param select_bragg: flag indicating whether the spot meets the Bragg's condition
        :param reference_frame: reference frame for the output [beamline, detector]
        :return spot_pos_out: positions of spots on the detector in 3D coordinates
        :return in_detector: flag indicating whether the spot is inside the detector area
        """

        # get spot positions
        spot_pos = self.plane_line_intersect(L_lab, x0)

        # get position constraints

        if ALGEBRA_ENGINE == 'tensorflow':
            
            with tf.device('gpu'):
            
                spot_pos_detector = tf.einsum('...opdj,dij->...opdi', spot_pos-self.position, self.inverse_rotation_matrix) 
                scattering_direction = tf.math.sign(tf.reduce_sum(tf.expand_dims(L_lab, axis=-2) * self.normal, axis=-1)) == -1
                spot_x, spot_y = spot_pos_detector[...,1], spot_pos_detector[...,2]
                in_detector_x = tf.math.abs(spot_x) < self.side_length/2 
                in_detector_y = tf.math.abs(spot_y) < self.side_length/2 
                outside_hole = tf.math.sqrt(spot_x**2 + spot_y**2) > self.hole_diameter/2
                in_detector = scattering_direction & in_detector_x & in_detector_y & outside_hole        

        elif ALGEBRA_ENGINE == 'numpy':

            spot_pos_detector = np.einsum('...opdj,dij->...opdi', spot_pos-self.position, self.inverse_rotation_matrix) 
            scattering_direction = np.sign(np.sum(np.expand_dims(L_lab, axis=-2) * self.normal, axis=-1)) == -1
            spot_x, spot_y = spot_pos_detector[...,1], spot_pos_detector[...,2]
            in_detector_x = np.abs(spot_x) < self.side_length/2 
            in_detector_y = np.abs(spot_y) < self.side_length/2 
            outside_hole = np.sqrt(spot_x**2 + spot_y**2) > self.hole_diameter/2
            in_detector = scattering_direction & in_detector_x & in_detector_y & outside_hole
            pass

        if reference_frame=='beamline':

            spot_pos_out = spot_pos
        
        elif reference_frame=='detector':

            spot_pos_out = spot_pos_detector

        return spot_pos_out, in_detector



    def rotate_and_shift(self, list_tilt_new, list_position_new):

        assert self.n_detectors == len(list_position_new)
        assert self.n_detectors == len(list_tilt_new)

        for i, (det, tilt, pos) in enumerate(zip(self.detectors, list_tilt_new, list_position_new)): 
            det.rotate_and_shift(tilt, pos)

        self.update()



    def __getitem__(self, key):

        return self.detectors_dict[key]

def select_detector_and_angle(spot_pos, id_angle, id_detector, select_bragg, in_detector):

    select = np.expand_dims(select_bragg, axis=-1) & in_detector
    select_ = select[id_angle, :, id_detector] 
    spot_pos_ = spot_pos[id_angle, :, id_detector, :]
    return spot_pos_[select_]

def select_angle(spot_pos, id_angle, select_bragg):

    select_ = select_bragg[id_angle, : ]
    spot_pos_ = spot_pos[id_angle, ...]
    return spot_pos_[select_]
