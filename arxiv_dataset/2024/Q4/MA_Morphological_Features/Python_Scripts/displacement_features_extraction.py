# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:39:16 2024

Extract Features From Mitral Annulus and its best-fitted ellipse as data frame

Same as v3, but added Patient RYJ to the csv as identifier

@author: Henry
"""

import os
import math
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

class CineMitralAnnulusFeatureExtractor:
    def __init__(self, root_dir, labels_file, ellipse_features_dict_path):
        """
        Initialize the feature extractor class.

        :param root_dir: Root directory where the data is stored and output will be saved.
        :param labels_file: Path to the JSON file containing patient labels.
        :param ellipse_features_file: Path to the JSON file containing ellipse features.
        """
        self.root_dir = root_dir
        self.labels_dict = self._load_json(labels_file)
        self.ellipse_features_dict_path = self._load_json(ellipse_features_dict_path)
        self.label_range = list(np.arange(0, 30, 5))
        self.ellipse_features_all_phases = [
            'ellipse_area', 'ellipse_perimeter', 'ellipse_b', 
            'ellipse_a', 'ellipse_ba_ratio', 'ellipse_eccentricity'
        ]
        self.ellipse_features_exclude_phase0 = [
            'ellipse_normal_vec1', 'ellipse_normal_vec2', 
            'ellipse_normal_vec3', 'ellipse_theta', 'ellipse_tilt'
        ]

    def _load_json(self, filepath):
        """
        Load a JSON file.

        :param filepath: Path to the JSON file.
        :return: Loaded JSON data as a dictionary.
        """
        with open(filepath) as f:
            return json.load(f)

    def displacement_per_axis(self, patient_key, axis):
        """
        Calculate displacement of landmarks along a specific axis.

        :param patient_key: Key to identify the patient.
        :param axis: Axis along which to calculate displacement ('x', 'y', 'z').
        :return: Displacements and their corresponding names.
        """
        key = f'MA_points_{axis}x' if axis == 'x' else f'MA_points_{axis}y' if axis == 'y' else f'MA_points_{axis}z'
        displacements, displacements_names = [], []
        
        for idx in range(len(self.label_range) - 1):
            for point in range(6):  # 6 landmark points per phase (2 in each view)
                landmark_point_n1 = self.ellipse_features_dict[patient_key][key][str(self.label_range[idx])][point]
                landmark_point_n2 = self.ellipse_features_dict[patient_key][key][str(self.label_range[idx + 1])][point]
                displacement = landmark_point_n2 - landmark_point_n1
                displacement_name = f"{axis}_diff_{self.label_range[idx + 1]}_{self.label_range[idx]}_{point}"
                displacements.append(displacement)
                displacements_names.append(displacement_name)

        return displacements, displacements_names

    def ma_height(self, patient_key):
        """
        Calculate the height of the mitral annulus.

        :param patient_key: Key to identify the patient.
        :return: Heights and their corresponding names.
        """
        heights, heights_names = [], []
        for phase in self.label_range:
            distances = self.ellipse_features_dict[patient_key]['distances_MApoints_plane'][str(phase)]
            above_plane_indices = self.ellipse_features_dict[patient_key]['above_plane_MApoints_idx'][str(phase)]
            below_plane_indices = self.ellipse_features_dict[patient_key]['below_plane_MApoints_idx'][str(phase)]
            
            max_above = max([abs(distances[a_idx]) for a_idx in above_plane_indices])
            max_below = max([abs(distances[b_idx]) for b_idx in below_plane_indices])
            max_height = round(max_above + max_below, 2)
            height_name = f'height_{phase}'
            heights.append(max_height)
            heights_names.append(height_name)
        
        return heights, heights_names

    def extract_features(self):
        """
        Extract features for all patients and save the results to a CSV file.
        """
        data = []
        for patient_key in tqdm(self.ellipse_features_dict, unit='cases', desc='Patient', total=len(self.ellipse_features_dict)):
            MR_status = self.labels_dict[patient_key]['MR']
            NoVD_status = self.labels_dict[patient_key]['NoVD']
            
            if MR_status == 1 or NoVD_status == 1:
                f1_values, f1_names = self._extract_absolute_features(patient_key)
                f2_values, f2_names = self._extract_relative_features(patient_key)
                displacement_group_values, displacement_group_names = self._extract_displacement_features(patient_key)
                heights_values, heights_names = self.ma_height(patient_key)
                
                patient_class = 0 if NoVD_status == 1 else 1
                data_per_patient = [patient_key] + f1_values + f2_values + displacement_group_values + heights_values + [patient_class]
                data.append(data_per_patient)

        variable_list = ['Assession_Number'] + f1_names + f2_names + displacement_group_names + heights_names + ['class']
        filename = self._save_to_csv(data, variable_list)
        
        return filename

    def _extract_absolute_features(self, patient_key):
        """
        Extract absolute features for all phases.

        :param patient_key: Key to identify the patient.
        :return: Extracted feature values and their names.
        """
        f1_values, f1_names = [], []
        for f1 in self.ellipse_features_all_phases:
            for p in self.label_range:
                f1_value = self.ellipse_features_dict[patient_key][f1][str(p)]
                f1_name = f"{f1}_phase{p}"
                f1_values.append(f1_value)
                f1_names.append(f1_name)
        return f1_values, f1_names

    def _extract_relative_features(self, patient_key):
        """
        Extract relative features for all phases except phase 0.

        :param patient_key: Key to identify the patient.
        :return: Extracted feature values and their names.
        """
        f2_values, f2_names = [], []
        for f2 in self.ellipse_features_exclude_phase0:
            for p in self.label_range[1:]:  # exclude phase 0
                f2_value = self.ellipse_features_dict[patient_key][f2][str(p)]
                f2_name = f"{f2}_phase{p}"
                f2_values.append(f2_value)
                f2_names.append(f2_name)
        return f2_values, f2_names

    def _extract_displacement_features(self, patient_key):
        """
        Extract displacement features for mitral annulus landmark points.

        :param patient_key: Key to identify the patient.
        :return: Displacement values and their names.
        """
        displacements_x, displacements_names_x = self.displacement_per_axis(patient_key, 'x')
        displacements_y, displacements_names_y = self.displacement_per_axis(patient_key, 'y')
        displacements_z, displacements_names_z = self.displacement_per_axis(patient_key, 'z')

        displacements_magnitudes = []
        displacements_magnitudes_names = []
        for idx, (x, y, z) in enumerate(zip(displacements_x, displacements_y, displacements_z)):
            magnitude = round(math.sqrt(x**2 + y**2 + z**2), 2)
            name_temp = displacements_names_x[idx].split('diff')[-1]
            displacement_name = f'mag_diff{name_temp}'
            displacements_magnitudes.append(magnitude)
            displacements_magnitudes_names.append(displacement_name)

        displacement_group_values = displacements_x + displacements_y + displacements_z + displacements_magnitudes
        displacement_group_names = displacements_names_x + displacements_names_y + displacements_names_z + displacements_magnitudes_names
        return displacement_group_values, displacement_group_names

    def _save_to_csv(self, data, variable_list):
        """
        Save the extracted features to a CSV file.

        :param data: Data to save.
        :param variable_list: List of variable names (columns).
        """
        df = pd.DataFrame(data, columns=variable_list)
        df.to_csv(os.path.join(self.root_dir, 'Cine_MitralAnnulus_Features_FULL.csv'), index=False)
        
        return os.path.join(self.root_dir, 'Cine_MitralAnnulus_Features_FULL.csv')