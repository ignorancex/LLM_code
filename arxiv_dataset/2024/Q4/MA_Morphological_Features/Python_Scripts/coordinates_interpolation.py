# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:54:50 2024

@author: Henry
"""

import os
from tqdm import tqdm
import json
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

class CineCoordInterpolator:
    def __init__(self, root_dir, coord_json_path, standardised_phase=30):
        """
        Initialize the CineInterpolator class.
        
        :param root_dir: The root directory where the data and output files are stored.
        :param standardised_phase: The number of phases to standardize across all patients (default is 30).
        """
        self.root_dir = root_dir
        self.coord_json_path = coord_json_path
        self.standardised_phase = standardised_phase
        self.ref_standardised_phase = np.arange(0, standardised_phase, 1)
        
        with open(self.coord_json_path) as f:
            self.Dict = json.load(f)

    def extrap1d(self, interpolator):
        """
        Extrapolate a 1D interpolation function to handle values outside the input range.
        
        :param interpolator: A scipy interpolation object (e.g., interp1d or CubicSpline).
        :return: A function that performs the extrapolation.
        """
        xs = interpolator.x
        ys = interpolator.y

        def pointwise(x):
            if x < xs[0]:
                return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
            elif x > xs[-1]:
                return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            else:
                return interpolator(x)

        def ufunclike(xs):
            return np.array(list(map(pointwise, np.array(xs))))

        return ufunclike

    def process_patient_data(self):
        """
        Process the patient data to interpolate and standardize the number of phases.
        """
        for patient in tqdm(self.Dict, unit='cases', desc='Patient', total=len(self.Dict)):
            self.Dict[patient]['ellipse_xx'] = {}  # x-coordinates for all LAX views
            self.Dict[patient]['ellipse_yy'] = {}  # y-coordinates for all LAX views
            self.Dict[patient]['ellipse_zz'] = {}  # z-coordinates for all LAX views

            for folder in self.Dict[patient]:
                if '_landmarks' in folder:
                    dict_idx1, dict_idx2 = self._get_view_indices(folder)
                    xyz_dict = self._extract_xyz_points(self.Dict[patient][folder])

                    if len(xyz_dict['phase']) == self.standardised_phase:
                        self._populate_ellipse_dict(patient, xyz_dict, dict_idx1, dict_idx2)
                    else:
                        self._interpolate_and_resample(xyz_dict, dict_idx1, dict_idx2, patient)

    def _get_view_indices(self, folder):
        """
        Get the indices for the views based on the folder name.
        
        :param folder: The folder name indicating the view.
        :return: A tuple of indices corresponding to the view.
        """
        if '2ch' in folder:
            return 0, 1
        elif 'lvot' in folder:
            return 2, 3
        elif '4ch' in folder:
            return 4, 5
        return None, None

    def _extract_xyz_points(self, folder_data):
        """
        Extract the x, y, z coordinates and phases from the folder data.
        
        :param folder_data: The data corresponding to a specific folder (view).
        :return: A dictionary containing the phase and coordinates.
        """
        xyz_dict = {
            'phase': [int(idx) for idx in folder_data],
            'point_x1': [folder_data[str(idx)][0][0] for idx in folder_data],
            'point_x2': [folder_data[str(idx)][1][0] for idx in folder_data],
            'point_y1': [folder_data[str(idx)][0][1] for idx in folder_data],
            'point_y2': [folder_data[str(idx)][1][1] for idx in folder_data],
            'point_z1': [folder_data[str(idx)][0][2] for idx in folder_data],
            'point_z2': [folder_data[str(idx)][1][2] for idx in folder_data]
        }
        return xyz_dict

    def _populate_ellipse_dict(self, patient, xyz_dict, dict_idx1, dict_idx2):
        """
        Populate the ellipse dictionaries for x, y, and z coordinates.
        
        :param patient: The patient ID or name.
        :param xyz_dict: The dictionary containing phase and coordinate data.
        :param dict_idx1: Index for the first point in the ellipse dictionary.
        :param dict_idx2: Index for the second point in the ellipse dictionary.
        """
        for idx in xyz_dict['phase']:
            self._initialize_ellipse_dict(patient, idx)

            self.Dict[patient]['ellipse_xx'][str(idx)][dict_idx1] = xyz_dict['point_x1'][idx]
            self.Dict[patient]['ellipse_xx'][str(idx)][dict_idx2] = xyz_dict['point_x2'][idx]
            self.Dict[patient]['ellipse_yy'][str(idx)][dict_idx1] = xyz_dict['point_y1'][idx]
            self.Dict[patient]['ellipse_yy'][str(idx)][dict_idx2] = xyz_dict['point_y2'][idx]
            self.Dict[patient]['ellipse_zz'][str(idx)][dict_idx1] = xyz_dict['point_z1'][idx]
            self.Dict[patient]['ellipse_zz'][str(idx)][dict_idx2] = xyz_dict['point_z2'][idx]

    def _initialize_ellipse_dict(self, patient, idx):
        """
        Initialize the ellipse dictionaries for a given phase if not already initialized.
        
        :param patient: The patient ID or name.
        :param idx: The phase index to initialize.
        """
        if str(idx) not in self.Dict[patient]['ellipse_xx']:
            self.Dict[patient]['ellipse_xx'][str(idx)] = [0] * 6
        if str(idx) not in self.Dict[patient]['ellipse_yy']:
            self.Dict[patient]['ellipse_yy'][str(idx)] = [0] * 6
        if str(idx) not in self.Dict[patient]['ellipse_zz']:
            self.Dict[patient]['ellipse_zz'][str(idx)] = [0] * 6

    def _interpolate_and_resample(self, xyz_dict, dict_idx1, dict_idx2, patient):
        """
        Interpolate and resample the coordinates to match the standardised phase count.
        
        :param xyz_dict: The dictionary containing phase and coordinate data.
        :param dict_idx1: Index for the first point in the ellipse dictionary.
        :param dict_idx2: Index for the second point in the ellipse dictionary.
        :param patient: The patient ID or name.
        """
        if xyz_dict['phase'][0] != 0:
            self._extrapolate_missing_phase_0(xyz_dict)

        if len(xyz_dict['phase']) != self.standardised_phase:
            self._resample_to_standardised_phase(xyz_dict)

        self._populate_ellipse_dict(patient, xyz_dict, dict_idx1, dict_idx2)

    def _extrapolate_missing_phase_0(self, xyz_dict):
        """
        Extrapolate missing phase 0 data if it's absent.
        
        :param xyz_dict: The dictionary containing phase and coordinate data.
        """
        for point_key in xyz_dict:
            if point_key != 'phase':
                f_i = interp1d(xyz_dict['phase'], xyz_dict[point_key], kind='cubic')
                f_x = self.extrap1d(f_i)

                extrapolated = list(f_x([0]))
                xyz_dict[point_key] = extrapolated + xyz_dict[point_key]

        xyz_dict['phase'] = [0] + xyz_dict['phase']

    def _resample_to_standardised_phase(self, xyz_dict):
        """
        Resample the data to match the standardised number of phases.
        
        :param xyz_dict: The dictionary containing phase and coordinate data.
        """
        resampled_standardised_phase = np.arange(0, len(xyz_dict['phase']), len(xyz_dict['phase'])/self.standardised_phase)
        for point_key in xyz_dict:
            if point_key != 'phase':
                spl = CubicSpline(xyz_dict['phase'], xyz_dict[point_key])
                resampled = spl(resampled_standardised_phase)
                xyz_dict[point_key] = [i for i in resampled]

        xyz_dict['phase'] = [int(p) for p in self.ref_standardised_phase]

    def save_interpolated_coordinates(self):
        """
        Save the interpolated world coordinates to a JSON file.
        
        :param output_filename: The name of the output file to save the data.
        """
        with open(os.path.join(self.root_dir, 'Cine_World_Coordinates_interpolated.json'), 'wt') as f:
            json.dump(self.Dict, f)
            
        return os.path.join(self.root_dir, 'Cine_World_Coordinates_interpolated.json')