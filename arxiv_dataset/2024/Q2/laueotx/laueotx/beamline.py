import numpy as np
from laueotx.detector import Detector, MultipleDetectors
from laueotx.utils import logging as utils_logging
from laueotx.config import ALGEBRA_ENGINE, PRECISION

if ALGEBRA_ENGINE == 'tensorflow':
    import tensorflow as tf

LOGGER = utils_logging.get_logger(__file__)

class Beamline():
    """
    Beamline parameters 
    %x is the beam direction towards the sample, y is left towards the sample,
    %z is down. 0,0,0 is the rotation axis position if there is no misalignment
    """

    def __init__(self, omegas, detectors, lambda_lim):
        """
        Constructor.
        :param omegas: array of rotation angles
        """

        self.omegas = omegas
        self.omegas_rad = np.radians(self.omegas)
        self.omegas_sincos = np.vstack([np.cos(self.omegas_rad), -np.sin(self.omegas_rad), np.zeros_like(self.omegas_rad)]).transpose()
        self.n_omegas = len(self.omegas) # list of omegas sampled
        self.lambda_min=lambda_lim[0]; # minimum incident wavelength
        self.lambda_max=lambda_lim[1];  # maximum incident wavelength
        self.O = np.zeros((self.n_omegas, 3, 3), dtype=PRECISION); # Initialize O
        self.invO = np.zeros((self.n_omegas, 3, 3), dtype=PRECISION); # Initialize O

        self.detectors = MultipleDetectors(detectors)

        for i in range(self.n_omegas):

            # eqn ??? in https://arxiv.org/abs/1902.03200
            self.O[i,0,:] = np.array([np.cos(self.omegas_rad[i]), -np.sin(self.omegas_rad[i]), 0.])
            self.O[i,1,:] = np.array([np.sin(self.omegas_rad[i]),  np.cos(self.omegas_rad[i]), 0.])
            self.O[i,2,:] = np.array([0., 0., 1.])
            self.invO[i,:,:] = np.linalg.inv(self.O[i,:,:])

        LOGGER.debug(f'created beamline: wavelengths [{self.lambda_min:2.2e}, {self.lambda_max:2.2e}]')

    def __str__(self):

        s = ''
        s += f'n_omegas={self.n_omegas}\n'
        s += f'lambda_min={self.lambda_min}\n'
        s += f'lambda_max={self.lambda_max}\n'
        return s


    # @profile
    def get_spots_positions(self, L_lab, x0, select_bragg=None):

        spot_pos, in_detector = self.detectors.get_spots_positions(L_lab, x0, select_bragg)

        # import ipdb; ipdb.set_trace(); 
        # pass

        if ALGEBRA_ENGINE == 'tensorflow':
           
            with tf.device('gpu'):
           
                spot_pos_inv = tf.einsum('...opdj,oij->...opdi', spot_pos, self.invO)

        elif ALGEBRA_ENGINE == 'numpy':
            
            spot_pos_inv = np.einsum('...opdj,oij->...opdi', spot_pos, self.invO)
                    

        # if self.detectors.position[0][0]==-170:

        #     from fastlaue3dnd.detector import select_detector_and_angle, select_angle
        #     spot_pos_inv__0_0 = select_detector_and_angle(spot_pos_inv, 0, 0, select_bragg, in_detector)
        #     spot_pos_inv__0_1 = select_detector_and_angle(spot_pos_inv, 0, 1, select_bragg, in_detector)

        #     spot_pos_inv__1_0 = select_detector_and_angle(spot_pos_inv, 1, 0, select_bragg, in_detector)
        #     spot_pos_inv__1_1 = select_detector_and_angle(spot_pos_inv, 1, 1, select_bragg, in_detector)

        #     import ipdb; ipdb.set_trace(); pass

        return spot_pos_inv, in_detector