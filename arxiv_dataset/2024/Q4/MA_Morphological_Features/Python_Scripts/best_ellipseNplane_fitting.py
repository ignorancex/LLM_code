# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:56:24 2024

In MA points dict, each ellipse list was saved in the following order:
    1. Mitral Anterior (2ch)
    2. Mitral Posterior (2ch)
    3. Mitral Septal (3ch)
    4. Mitral Freewall (3ch)
    5. Mitral Septal (4ch)
    6. Mitral Freewall (4ch)
    
This script replaces valve_landmarks_best_fit_ellipse.py

1. 3D ellipse point is found using the 6 labelled landmark points
2. Ellipse at phase 0 is rotated and translated, with the rotation matrix used
   for the rotational and translational for subsequent phases
   a. Rotation: Semi-major axis aligns with the x-axis
   b. Translation: Centroid is shifted to (0, 0, 0)
3. All the ellipse features are saved as .json file

@author: Henry
"""

import os
import json
import numpy as np
from tqdm import tqdm
from skimage.measure import EllipseModel
import matplotlib.pyplot as plt

class MitralAnnulusEllipseFitter:
    def __init__(self, root_dir, interpolated_json_path, QC=False):
        """
        Initialize the MitralAnnulusEllipseFitter class.

        :param root_dir: Root directory where the data is stored and output will be saved.
        :param QC: Boolean flag for enabling quality control visualizations (default is False).
        """
        self.root_dir = root_dir
        self.interpolated_json_path = interpolated_json_path
        self.QC = QC
        
        with open(self.interpolated_json_path) as f:
            self.MA_points_dict = json.load(f)
        self.feature_keys = [
            'ellipse_points', 'ellipse_area', 'ellipse_perimeter', 'ellipse_b',
            'ellipse_a', 'ellipse_ba_ratio', 'ellipse_theta', 'ellipse_eccentricity',
            'ellipse_normal_vec1', 'ellipse_normal_vec2', 'ellipse_normal_vec3',
            'ellipse_tilt', 'distances_MApoints_plane', 'above_plane_MApoints_idx',
            'below_plane_MApoints_idx', 'MA_points', 'MA_points_xx',
            'MA_points_yy', 'MA_points_zz', 'ellipse_xc', 'ellipse_yc'
        ]
        self.Dict = {patient_key: {key: {} for key in self.feature_keys} for patient_key in self.MA_points_dict}
        self.ref_centroid = None
        self.ref_rot_matrix = None

    def rodrigues_rot(self, P, n0, n1):
        """
        Apply Rodrigues' rotation formula to rotate points P from vector n0 to n1.

        :param P: Array of points to rotate.
        :param n0: Initial vector.
        :param n1: Target vector.
        :return: Rotated points.
        """
        if P.ndim == 1:
            P = P[np.newaxis, :]

        n0 = n0 / np.linalg.norm(n0)
        n1 = n1 / np.linalg.norm(n1)
        k = np.cross(n0, n1)
        k = k / np.linalg.norm(k)
        theta = np.arccos(np.dot(n0, n1))

        P_rot = np.zeros((len(P), 3))
        for i in range(len(P)):
            P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))

        return P_rot

    def fit_an_ellipse(self, P):
        """
        Fit an ellipse to a set of 3D points using Singular Value Decomposition (SVD).

        :param P: 3D points to fit the ellipse to.
        :return: Fitted ellipse model and the ellipse points in 3D space.
        """
        P_mean = P.mean(axis=0)
        P_centered = P - P_mean

        U, s, V = np.linalg.svd(P_centered, full_matrices=False)
        normal = V[2, :]

        P_xy = self.rodrigues_rot(P_centered, normal, [0, 0, 1])

        ell = EllipseModel()
        ell.estimate(P_xy[:, :2])

        n = 100
        xy = ell.predict_xy(np.linspace(0, 2 * np.pi, n))

        points = np.array([[xy[i, 0], xy[i, 1], 0] for i in range(len(xy))])
        ellipse_points_3d = self.rodrigues_rot(points, [0, 0, 1], normal) + P_mean

        return ell, ellipse_points_3d

    def rotate_to_x_axis(self, points):
        """
        Rotate a set of points so that the first principal component aligns with the x-axis.

        :param points: Points to rotate.
        :return: Rotated points, centroid, and rotation matrix.
        """
        centroid = np.mean(points, axis=0)
        translated_points = points - centroid
        _, _, vh = np.linalg.svd(np.cov(translated_points.T))
        rotation_matrix = vh.T
        rotated_points = np.dot(translated_points, rotation_matrix)

        return rotated_points, centroid, rotation_matrix

    def rotate_points(self, points, centroid, rotation_matrix):
        """
        Rotate a set of points based on a centroid and rotation matrix.

        :param points: Points to rotate.
        :param centroid: Centroid for translation.
        :param rotation_matrix: Rotation matrix to apply.
        :return: Rotated points.
        """
        translated_points = points - centroid
        rotated_points = np.dot(translated_points, rotation_matrix)
        return rotated_points

    def distance_to_plane(self, points, plane_normal, plane_point):
        """
        Compute the distance from each point to a specified plane.

        :param points: Points to measure distance from.
        :param plane_normal: Normal vector of the plane.
        :param plane_point: A point on the plane.
        :return: Distances from each point to the plane.
        """
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        distances = np.dot(points - plane_point, plane_normal)
        return distances

    def best_fit_plane_svd(self, points):
        """
        Find the best-fit plane for a set of points using SVD.

        :param points: Points to fit the plane to.
        :return: Normal vector of the plane and the centroid of the points.
        """
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        plane_normal = vh[-1, :]

        return plane_normal, centroid

    def find_points_above_below_plane(self, distances):
        """
        Identify points above and below a specified plane.

        :param distances: Distances of points from the plane.
        :return: Indices of points above and below the plane.
        """
        above_plane_indices = np.where(distances > 0)[0]
        below_plane_indices = np.where(distances < 0)[0]
        return above_plane_indices, below_plane_indices

    def tilt_angle(self, normal_vector):
        """
        Calculate the tilt angle of a plane relative to the z-axis.

        :param normal_vector: Normal vector of the plane.
        :return: Tilt angle in radians.
        """
        angle = np.arccos(np.dot(normal_vector, np.array([0, 0, 1])) / np.linalg.norm(normal_vector))
        return round(angle, 2)

    def ellipse_parameters(self, model):
        """
        Calculate ellipse parameters such as area, perimeter, and eccentricity.

        :param model: Fitted ellipse model.
        :return: Tuple containing ellipse parameters.
        """
        xc, yc, a, b, theta = model.params

        def correct_ellipse_model_params(a: float, b: float, theta: float) -> tuple:
        if a < b:
            if theta < np.pi / 2:
                a_b_swapped = True
                return theta + np.pi / 2, a_b_swapped
            else:
                a_b_swapped = True
                return theta - np.pi / 2, a_b_swapped
        else:
            if theta < 0:
                a_b_swapped = False
                return np.pi + theta, a_b_swapped
            else:
                a_b_swapped = False
                return theta, a_b_swapped
        
        theta, a_b_swapped = correct_ellipse_model_params(a, b, theta)
        
        if a_b_swapped:
            a = copy.deepcopy(b)
            b = copy.deepcopy(a)
            
        area = np.pi * a * b
        h = ((a - b) ** 2) / ((a + b) ** 2)
        perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        eccentricity = np.sqrt(1 - (b ** 2 / a ** 2))
        ba_ratio = b / a

        return round(a, 2), round(b, 2), round(ba_ratio, 2), round(theta, 2), round(area, 2), round(perimeter, 2), round(eccentricity, 2)

    def process_patient_data(self):
        """
        Process the mitral annulus data for each patient and phase, fitting ellipses and extracting features.
        """
        count = 0
        for patient_key in tqdm(self.MA_points_dict, unit='cases', desc='Patient', total=len(self.MA_points_dict)):
            for phase_key in self.MA_points_dict[patient_key]['ellipse_xx']:
                MA_points_3D = self._get_ma_points_3d(patient_key, phase_key)
                ellipse_model, ellipse_points = self.fit_an_ellipse(MA_points_3D)

                if phase_key == '0':
                    rotated_ellipse_points, self.ref_centroid, self.ref_rot_matrix = self.rotate_to_x_axis(ellipse_points)
                else:
                    rotated_ellipse_points = self.rotate_points(ellipse_points, self.ref_centroid, self.ref_rot_matrix)

                plane_normal, plane_centroid = self.best_fit_plane_svd(rotated_ellipse_points)
                rotated_MA_points = self.rotate_points(MA_points_3D, self.ref_centroid, self.ref_rot_matrix)
                distances = self.distance_to_plane(rotated_MA_points, plane_normal, plane_centroid)
                above_plane_indices, below_plane_indices = self.find_points_above_below_plane(distances)
                tilt = self.tilt_angle(plane_normal)

                if phase_key == '0':
                    tilt_subtract = 3.14 if tilt == 3.14 else 0
                tilt -= tilt_subtract

                a, b, ba_ratio, theta, area, perimeter, eccentricity = self.ellipse_parameters(ellipse_model)
                xc = int(np.mean([max(rotated_ellipse_points[:, 0]), min(rotated_ellipse_points[:, 0])]))
                yc = int(np.mean([max(rotated_ellipse_points[:, 1]), min(rotated_ellipse_points[:, 1])]))

                if phase_key == '0':
                    ref_theta = theta
                theta -= ref_theta

                self._store_features(patient_key, phase_key, rotated_ellipse_points, area, perimeter, b, a, ba_ratio, theta, tilt, eccentricity, plane_normal, distances, above_plane_indices, below_plane_indices, rotated_MA_points, xc, yc)

                if self.QC and count < 30:
                    self._quality_control_visualization(ellipse_points, rotated_ellipse_points, MA_points_3D, rotated_MA_points, plane_normal, plane_centroid, self.ref_centroid)

                count += 1

    def _get_ma_points_3d(self, patient_key, phase_key):
        """
        Retrieve 3D mitral annulus points for a specific patient and phase.

        :param patient_key: The patient identifier.
        :param phase_key: The phase identifier.
        :return: Array of 3D points.
        """
        MA_points_x = self.MA_points_dict[patient_key]['ellipse_xx'][phase_key]
        MA_points_y = self.MA_points_dict[patient_key]['ellipse_yy'][phase_key]
        MA_points_z = self.MA_points_dict[patient_key]['ellipse_zz'][phase_key]

        MA_points_3D = np.array([[MA_points_x[i], MA_points_y[i], MA_points_z[i]] for i in range(len(MA_points_x))])
        return MA_points_3D

    def _store_features(self, patient_key, phase_key, ellipse_points, area, perimeter, b, a, ba_ratio, theta, tilt, eccentricity, plane_normal, distances, above_plane_indices, below_plane_indices, MA_points, xc, yc):
        """
        Store extracted features in the dictionary.

        :param patient_key: The patient identifier.
        :param phase_key: The phase identifier.
        :param ellipse_points: Points defining the fitted ellipse.
        :param area: Ellipse area.
        :param perimeter: Ellipse perimeter.
        :param b: Semi-minor axis length.
        :param a: Semi-major axis length.
        :param ba_ratio: Ratio of semi-minor to semi-major axis.
        :param theta: Ellipse rotation angle.
        :param tilt: Tilt angle of the plane.
        :param eccentricity: Ellipse eccentricity.
        :param plane_normal: Normal vector of the fitted plane.
        :param distances: Distances of points from the plane.
        :param above_plane_indices: Indices of points above the plane.
        :param below_plane_indices: Indices of points below the plane.
        :param MA_points: Rotated mitral annulus points.
        :param xc: X-coordinate of the ellipse center.
        :param yc: Y-coordinate of the ellipse center.
        """
        self.Dict[patient_key]['ellipse_points'][phase_key] = ellipse_points.tolist()
        self.Dict[patient_key]['ellipse_area'][phase_key] = area
        self.Dict[patient_key]['ellipse_perimeter'][phase_key] = perimeter
        self.Dict[patient_key]['ellipse_b'][phase_key] = b
        self.Dict[patient_key]['ellipse_a'][phase_key] = a
        self.Dict[patient_key]['ellipse_ba_ratio'][phase_key] = ba_ratio
        self.Dict[patient_key]['ellipse_theta'][phase_key] = theta
        self.Dict[patient_key]['ellipse_tilt'][phase_key] = tilt
        self.Dict[patient_key]['ellipse_eccentricity'][phase_key] = eccentricity
        self.Dict[patient_key]['ellipse_normal_vec1'][phase_key] = plane_normal[0]
        self.Dict[patient_key]['ellipse_normal_vec2'][phase_key] = plane_normal[1]
        self.Dict[patient_key]['ellipse_normal_vec3'][phase_key] = plane_normal[2]
        self.Dict[patient_key]['distances_MApoints_plane'][phase_key] = distances.tolist()
        self.Dict[patient_key]['above_plane_MApoints_idx'][phase_key] = above_plane_indices.tolist()
        self.Dict[patient_key]['below_plane_MApoints_idx'][phase_key] = below_plane_indices.tolist()
        self.Dict[patient_key]['MA_points'][phase_key] = MA_points.tolist()
        self.Dict[patient_key]['MA_points_xx'][phase_key] = MA_points[:, 0].tolist()
        self.Dict[patient_key]['MA_points_yy'][phase_key] = MA_points[:, 1].tolist()
        self.Dict[patient_key]['MA_points_zz'][phase_key] = MA_points[:, 2].tolist()
        self.Dict[patient_key]['ellipse_xc'][phase_key] = xc
        self.Dict[patient_key]['ellipse_yc'][phase_key] = yc

    def _quality_control_visualization(self, ellipse_points, rotated_ellipse_points, MA_points_3D, rotated_MA_points, plane_normal, plane_centroid, centroid):
        """
        Visualize quality control plots for the ellipses and mitral annulus points.

        :param ellipse_points: Original ellipse points.
        :param rotated_ellipse_points: Rotated ellipse points.
        :param MA_points_3D: Original mitral annulus points.
        :param rotated_MA_points: Rotated mitral annulus points.
        :param plane_normal: Normal vector of the fitted plane.
        :param plane_centroid: Centroid of the fitted plane.
        :param centroid: Centroid of the original points.
        """
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')

        ax.scatter(ellipse_points[:, 0], ellipse_points[:, 1], ellipse_points[:, 2], color='r', label='Original Ellipse')
        ax.scatter(rotated_ellipse_points[:, 0], rotated_ellipse_points[:, 1], rotated_ellipse_points[:, 2], color='g', label='Rotated Ellipse')
        ax.scatter(MA_points_3D[:, 0], MA_points_3D[:, 1], MA_points_3D[:, 2], color='r', label='Original Points')
        ax.scatter(rotated_MA_points[:, 0], rotated_MA_points[:, 1], rotated_MA_points[:, 2], color='g', label='Rotated Points')

        xx, yy = np.meshgrid(np.linspace(min(rotated_ellipse_points[:, 0]), max(rotated_ellipse_points[:, 0]), 10), np.linspace(min(rotated_ellipse_points[:, 1]), max(rotated_ellipse_points[:, 1]), 10))
        zz = (-plane_normal[0] * (xx - plane_centroid[0]) - plane_normal[1] * (yy - plane_centroid[1]) + plane_normal[2] * plane_centroid[2]) / plane_normal[2]
        ax.plot_surface(xx, yy, zz, color='y', alpha=0.5)

        ax.legend()
        plt.show()

        fig, ax = plt.subplots(1, 3, figsize=(15, 11), dpi=150)
        ax[0].plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r--', lw=2, label='Original ellipse')
        ax[1].plot(ellipse_points[:, 0], ellipse_points[:, 2], 'r--', lw=2, label='Original ellipse')
        ax[2].plot(ellipse_points[:, 1], ellipse_points[:, 2], 'r--', lw=2, label='Original ellipse')

        ax[0].scatter(MA_points_3D[:, 0], MA_points_3D[:, 1], color='r', label='Original Points')
        ax[1].scatter(MA_points_3D[:, 0], MA_points_3D[:, 2], color='r', label='Original Points')
        ax[2].scatter(MA_points_3D[:, 1], MA_points_3D[:, 2], color='r', label='Original Points')

        ax[0].plot(rotated_ellipse_points[:, 0], rotated_ellipse_points[:, 1], 'g--', lw=2, label='Rotated ellipse')
        ax[1].plot(rotated_ellipse_points[:, 0], rotated_ellipse_points[:, 2], 'g--', lw=2, label='Rotated ellipse')
        ax[2].plot(rotated_ellipse_points[:, 1], rotated_ellipse_points[:, 2], 'g--', lw=2, label='Rotated ellipse')

        ax[0].scatter(rotated_MA_points[:, 0], rotated_MA_points[:, 1], color='g', label='Rotated Points')
        ax[1].scatter(rotated_MA_points[:, 0], rotated_MA_points[:, 2], color='g', label='Rotated Points')
        ax[2].scatter(rotated_MA_points[:, 1], rotated_MA_points[:, 2], color='g', label='Rotated Points')

        ax[0].scatter(centroid[0], centroid[1], color='b', label='Centroid')
        ax[1].scatter(centroid[0], centroid[2], color='b', label='Centroid')
        ax[2].scatter(centroid[1], centroid[2], color='b', label='Centroid')

        ax[2].legend()
        plt.show()

    def save_features(self):
        """
        Save the extracted features to a JSON file.
        """
        with open(os.path.join(self.root_dir, 'Cine_MitralAnnulus_Ellipse_Features.json'), 'wt') as f:
            json.dump(self.Dict, f)
            
        return os.path.join(self.root_dir, 'Cine_MitralAnnulus_Ellipse_Features.json')
