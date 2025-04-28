import numpy as np, warnings
from laueotx.utils import logging as utils_logging
from laueotx.utils import arrays as utils_arrays
from laueotx.utils import io as utils_io
from laueotx import config
LOGGER = utils_logging.get_logger(__file__)

if config.ALGEBRA_ENGINE == 'tensorflow':
    import tensorflow as tf


dtype_peaks = [('x', np.float32),
               ('y', np.float32),
               ('omega', np.float32),
               ('id', np.int64),
               ('id_angle', np.int64),
               ('id_detector', np.int64),
               ('id_global', np.int64)]

def arr_to_rec(arr, dtype):

    rec = np.empty(len(arr), dtype=dtype)
    for i, p in enumerate(rec.dtype.names):
        rec[p] = arr[:,i]
    return rec

def get_peaks_rec(mat):



    if mat['Peaks'].shape[1] == 4:
        mat['Peaks'] = np.concatenate([mat['Peaks'], np.zeros((len(mat['Peaks']), 3))], axis=1)

    # convert to rec
    peaks_rec = arr_to_rec(mat['Peaks'], dtype_peaks)

    # get ids
    peaks_rec['id_global'] = np.arange(len(peaks_rec))
    peaks_rec['id'] = np.arange(len(peaks_rec))
    peaks_rec['id_angle'] = -1
    peaks_rec['id_detector'] = -1

    return peaks_rec


def get_peaks_from_input_mat(mat, detector_id):
    """
    Here implement routines to read data from matlab arrays
    """

    if type(mat) is np.ndarray:
        if mat.shape[-1] == 4:
            peaks = get_peaks_rec(mat)
        else:
            raise Exception(f'detected peaks array should be Nx4, is {peaks.shape}')

    elif type(mat) is dict:

        peaks = get_peaks_rec(mat)

    else:
        raise Exception('wrong peaks data input format')

    peaks['id_detector'] = detector_id

    return peaks


def get_peaks_from_input_h5(objects_all, detector_id):

    select = objects_all['i_det'] == detector_id
    objects = utils_arrays.rewrite(objects_all[select])
    
    peaks_rec = np.empty(len(objects), dtype=dtype_peaks)
    peaks_rec['x'] = objects['x']
    peaks_rec['y'] = objects['y']
    peaks_rec['omega'] = objects['ang_deg']
    peaks_rec['id'] = np.arange(len(objects))
    peaks_rec['id_angle'] = objects['i_ang']
    peaks_rec['id_detector'] = objects['i_det']
    peaks_rec['id_global'] = np.arange(len(objects_all))[select]

    return peaks_rec

class PeaksData():

    def __init__(self,
                 detector_type,
                 beamline,
                 filepath=None,
                 peaks_rec=None,
                 apply_coord_shift=False,
                 flip_x_coord=False, 
                 omegas_use=None):
        """
        PeaksDataset constructur
        :param detector_type: type of the detector type (backward, forward, etc)
        :param beamline: Beamline object from beamline.py
        :param filepath: path to a .mat filename containing peaks data
        :param peaks_rec: peaks recarray with peaks data (overrides filename)
        :param apply_coord_shift: if to apply coordinate shift (False for simulatinos)
        :param flip_x_coord: if to apply flip to x coord (False for simulations)
        """

        assert (filepath is not None) or (peaks_rec is not None), 'specify either peaks_rec or a valid .mat filepath'

        self.detector_type = detector_type
        self.beamline = beamline
        self.detector = self.beamline.detectors[self.detector_type]
        self.omegas_use = omegas_use

        if filepath is not None:
            
            if filepath.endswith('.mat'):

                import scipy.io
                input_mat = scipy.io.loadmat(filepath)
                self.peaks = get_peaks_from_input_mat(input_mat, self.detector.detector_id)

            elif filepath.endswith('.h5'):

                objects_all = utils_io.read_arrays(filepath)['objects']
                self.peaks = get_peaks_from_input_h5(objects_all, self.detector.detector_id)

        if self.omegas_use is not None:
            self.filter_omegas()
                
        if apply_coord_shift:
            self.shift_coords()

        if flip_x_coord:
            self.flip_x()

        if peaks_rec is not None:
            self.peaks = peaks_rec

        self.n_peaks = len(self.peaks)

        self.split_angles()

        self.to_arr()

        self.invO_flat = self.beamline.invO[self.peaks['id_angle']]

    def filter_omegas(self):

        select = np.in1d(self.peaks['omega'], np.array(self.omegas_use))
        LOGGER.info(f"using {len(self.omegas_use)}/{len(np.unique(self.peaks['omega']))} angles, keeping {np.count_nonzero(select)}/{len(select)} spots")

        self.peaks = self.peaks[select]
        omegas, id_angle = np.unique(self.peaks['omega'], return_inverse=True)
        self.peaks['id_angle'] = id_angle

    def shift_coords(self):

        # shift from pixels to milimeters
        # shift = lambda x: x*400/4000-200
        shift = lambda x: x/self.detector.num_pixels*self.detector.side_length - self.detector.side_length/2

        for c in ['x', 'y']:
            self.peaks[c] = shift(self.peaks[c])

    def flip_x(self):

        self.peaks['x'] *= -1

    def split_angles(self):

        uid_angle = np.unique(self.peaks['id_angle'])
        self.peaks_angles = np.empty(len(uid_angle), dtype=object)
        for i, u in enumerate(uid_angle):
            select = self.peaks['id_angle'] == u
            self.peaks_angles[i] = self.peaks[select]

        LOGGER.debug(f'detector {self.detector}: split peaks data into {len(self.peaks_angles)} angles')

    def to_arr(self):

        self.peaks_arr = np.concatenate([np.zeros((len(self.peaks),1)),
                                         np.expand_dims(self.peaks['x'], axis=1),
                                         np.expand_dims(self.peaks['y'], axis=1)], axis=1)



    def rotate(self):


        if ALGEBRA_ENGINE == 'tensorflow':

            # this two lines can be merged into a single op
            peaks_rot = np.dot(self.peaks_arr, self.detector.rotation_matrix.transpose()) + self.detector.position
            self.peaks_rot = np.einsum('pij,pj->pi', self.invO_flat, peaks_rot)



        elif ALGEBRA_ENGINE == 'numpy':

            # this two lines can be merged into a single op
            peaks_rot = np.dot(self.peaks_arr, self.detector.rotation_matrix.transpose()) + self.detector.position
            self.peaks_rot = np.einsum('pij,pj->pi', self.invO_flat, peaks_rot)

    def __getitem__(self, item):

        peaks = self.peaks[item].copy()
        pd = PeaksData(peaks_rec=peaks,
                       detector_type=self.detector_type, 
                       beamline=self.beamline,
                       apply_coord_shift=False,
                       flip_x_coord=False)

        return pd





class MultidetectorPeaksData():

    def __init__(self, list_peaks_data, beamline):

        self.list_peaks_data = list_peaks_data
        self.beamline = beamline
        self.peaks = np.concatenate([pd.peaks for pd in self.list_peaks_data])
        self.peaks_arr = np.concatenate([pd.peaks_arr for pd in self.list_peaks_data])
        self.id_detector = np.concatenate([np.full(len(pd.peaks_arr), i, dtype=np.int8) for i, pd in enumerate(self.list_peaks_data)])
        self.n_peaks_split = [l.n_peaks for l in self.list_peaks_data]
        self.n_peaks = len(self.peaks)
        self.peaks['id_global'] = np.arange(len(self.peaks), dtype=int)

        n_angles_per_detector = np.array([len(np.unique(pd.peaks['omega'])) for pd in self.list_peaks_data])
        assert np.all(n_angles_per_detector==n_angles_per_detector[0]), 'number of rotation angles is different for each detector, perhaps some frames have no peaks?'

    def __str__(self):

        s = f'n_peaks={self.n_peaks}\n'

        for i, n in enumerate(self.n_peaks_split):
            s += f'detector={i} n_peaks={n}\n'
        
        return s


    def rotate(self):

        peaks_rot = []
        # this should be vectorised
        for i, pd in enumerate(self.list_peaks_data):
            pd.rotate()
            peaks_rot.append(pd.peaks_rot)

        self.peaks_rot = np.concatenate(peaks_rot)

    def to_lab_coords(self):

        n_dim = 3

        _, det_index = np.unique(self.peaks['id_detector'], return_inverse=True)

        # detector params
        det_pos = np.array(self.beamline.detectors.position)
        det_tilt = np.array(self.beamline.detectors.tilt)

        # detector coord system
        peaks_det_x = self.peaks['x'] 
        peaks_det_y = self.peaks['y']

        # convert to lab system

        # get the extended coords in the detector
        peaks_pos = np.zeros((len(peaks_det_x), n_dim))
        peaks_pos[:,1] = peaks_det_x
        peaks_pos[:,2] = peaks_det_y

        # rotate according to the tilts
        from laueotx.laue_math_tensorised import batch_rotation_matrix_compat as batch_rotation_matrix
        dR = np.array(batch_rotation_matrix(det_tilt))
        peaks_det_R = dR[det_index]
        peaks_pos = tf.einsum('bij, bj -> bi', peaks_det_R, peaks_pos)

        # shift by the detector position wrt to the lab system
        peaks_det_pos = det_pos[det_index]
        peaks_pos += peaks_det_pos

        LOGGER.info(f"converted {len(self.peaks)} peaks to lab coords, n_det={len(np.unique(self.peaks['id_detector']))} n_ang={len(np.unique(self.peaks['id_angle']))}")


        return peaks_pos

    def __getitem__(self, list_select):

        from collections.abc import Iterable
        if isinstance(list_select, Iterable):

            assert len(list_select) == len(self.list_peaks_data)
            list_peaks_data = [pd[select] for pd, select in zip(self.list_peaks_data, list_select)]

        else:

            list_peaks_data = [pd[list_select] for pd in self.list_peaks_data]
    
        mpd = MultidetectorPeaksData(list_peaks_data, beamline=self.beamline)
        return mpd

        






            
