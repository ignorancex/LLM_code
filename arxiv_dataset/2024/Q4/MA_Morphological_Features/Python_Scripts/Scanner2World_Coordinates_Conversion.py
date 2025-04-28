# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:04:28 2022

Convert image coordinates to world coordinates

* NOTES: Write a function to check if all LAX views are fully labelled, if not,
output a text file with counts, patients, and reasons
    - 2-/4-ch: Likely image quality issues
    - 3-ch: likely coronal view and/or image quality issues

@author: Henry
"""

import os
import SimpleITK as sitk
import glob
import re
from tqdm import tqdm
import json
import pydicom as dcm

class CineWorldCoordinates:
    def __init__(self, root_dir, cine_images_dir):
        """
        Initialize the CineWorldCoordinates class.
        
        :param root_dir: The root directory where the data and output files are stored.
        :param cine_images_dir: The directory containing the CINE images.
        """
        self.root_dir = root_dir
        self.cine_images_dir = cine_images_dir
        self.labelled_list = self._load_labelled_list()  # Load the list of labeled patients
        self.coordinate_notes, self.Dict = self._load_coordinate_data()  # Load coordinate data if available

    def _load_labelled_list(self):
        """
        Load the list of patients with labeled valve landmarks from a text file.
        
        :return: A list of labeled patients.
        """
        labelled_list = []
        with open(os.path.join(self.root_dir, 'labelled_landmark_files_dicom.txt')) as f:
            labelled_list.insert(0, f.read())
        return labelled_list

    def _load_coordinate_data(self):
        """
        Load the coordinate conversion information from JSON and text files.
        
        :return: A tuple containing coordinate notes and a dictionary of world coordinates.
        """
        try:
            with open(os.path.join(self.root_dir, 'Data_CMR_Cine_dicom_coordinatestatus_notes.txt'), "r") as f:
                coordinate_notes = [f.read()]
            with open(os.path.join(self.root_dir, 'Cine_World_Coordinates.json'), 'r') as f:
                Dict = json.load(f)
        except FileNotFoundError:
            coordinate_notes = ['']
            Dict = {}
        return coordinate_notes, Dict

    def _atoi(self, text):
        """
        Convert text to integer if possible, otherwise return the text.
        
        :param text: The input text.
        :return: Integer if the text is a digit, otherwise the text itself.
        """
        return int(text) if text.isdigit() else text

    def _natural_keys(self, text):
        """
        Split text into a list of integers and non-integers for natural sorting.
        
        :param text: The input text to split.
        :return: A list of split text elements.
        """
        return [self._atoi(c) for c in re.split('(\d+)', text)]

    def _transform_point(self, from_image, view):
        """
        Convert image coordinates to world coordinates based on the view.
        
        :param from_image: The SimpleITK image from which to extract points.
        :param view: The view type ('2ch', 'lvot', or '4ch').
        :return: A list of physical points corresponding to valve landmarks.
        """
        view_mapping = {'2ch': (10, 15), 'lvot': (20, 25), '4ch': (30, 35)}
        p1_mag, p2_mag = view_mapping.get(view, (None, None))

        if p1_mag is None or p2_mag is None:
            return []

        file = []
        p1, p2 = None, None

        # Iterate over all pixels in the image to find the landmarks
        for x in range(from_image.GetSize()[0]):
            for y in range(from_image.GetSize()[1]):
                for z in range(from_image.GetSize()[2]):
                    pixel_value = int(from_image[x, y, z])
                    if pixel_value == p1_mag:
                        p1 = from_image.TransformIndexToPhysicalPoint((x, y, z))
                    elif pixel_value == p2_mag:
                        p2 = from_image.TransformIndexToPhysicalPoint((x, y, z))
        
        if p1 and p2:
            file = [p1, p2]
        
        return file

    def process_patients(self):
        """
        Process all patients in the CINE images directory to find and convert coordinates.
        """
        for patient in tqdm(os.listdir(self.cine_images_dir), unit='cases', desc='Patient', total=len(os.listdir(self.cine_images_dir))):
            if patient in self.labelled_list[0]:  # Check if the patient is in the labelled list
                self._process_patient(patient)

    def _process_patient(self, patient):
        """
        Process a single patient, checking for missing landmarks and converting coordinates.
        
        :param patient: The patient ID or name to process.
        """
        folders = os.listdir(os.path.join(self.cine_images_dir, patient))

        if not self._has_all_landmarks(folders):
            self._log_missing_landmarks(patient, folders)
        elif patient not in self.Dict:
            self.Dict[patient] = {}
            for folder in folders:
                if '_landmarks' in folder:
                    view = self._get_view_from_folder(folder)
                    files = glob.glob(os.path.join(self.cine_images_dir, patient, folder, '*.dcm'))
                    files.sort(key=self._natural_keys)
                    self._process_folder(patient, folder, files, view)

    def _has_all_landmarks(self, folders):
        """
        Check if all required CINE LAX views are present in the patient's folder.
        
        :param folders: List of folder names for the patient.
        :return: True if all landmarks are present, False otherwise.
        """
        return all(view in folders for view in ['cine_2ch_landmarks', 'cine_4ch_landmarks', 'cine_lvot_landmarks'])

    def _log_missing_landmarks(self, patient, folders):
        """
        Log missing landmarks for a patient and save the notes.
        
        :param patient: The patient ID or name.
        :param folders: List of folder names for the patient.
        """
        cine_2ch_exist = ' has' if 'cine_2ch_landmarks' in folders else ' has no'
        cine_4ch_exist = ' has' if 'cine_4ch_landmarks' in folders else ' has no'
        cine_3ch_exist = ' has' if 'cine_lvot_landmarks' in folders else ' has no'
        
        note = f"{patient}{cine_2ch_exist} cine_2ch_landmarks,{cine_4ch_exist} cine_4ch_landmarks, and{cine_3ch_exist} cine_3ch_landmarks."
        
        if note not in self.coordinate_notes:
            self.coordinate_notes.insert(0, note)
            self._save_notes()

    def _get_view_from_folder(self, folder):
        """
        Determine the view type based on the folder name.
        
        :param folder: Folder name containing landmark data.
        :return: The view type ('2ch', 'lvot', or '4ch').
        """
        if '2ch' in folder:
            return '2ch'
        elif 'lvot' in folder:
            return 'lvot'
        elif '4ch' in folder:
            return '4ch'
        return None

    def _process_folder(self, patient, folder, files, view):
        """
        Process a folder containing DICOM files to transform coordinates to world coordinates.
        
        :param patient: The patient ID or name.
        :param folder: Folder name containing the DICOM files.
        :param files: List of DICOM file paths to process.
        :param view: The view type ('2ch', 'lvot', or '4ch').
        """
        self.Dict[patient][folder] = {}
        for file in files:
            cine_img = sitk.ReadImage(file)
            cine_physical_points = self._transform_point(cine_img, view)
            
            dicom_file = dcm.read_file(file)
            instance_number = dicom_file.InstanceNumber
            idx = int(instance_number) - 1
            
            self.Dict[patient][folder][str(idx)] = cine_physical_points

    def _save_notes(self):
        """
        Save the coordinate status notes to a text file.
        """
        with open(os.path.join(self.root_dir, 'Data_CMR_Cine_dicom_coordinatestatus_notes.txt'), "w") as f:
            for line in self.coordinate_notes:
                f.write(f"{line}\n")

    def save_world_coordinates(self):
        """
        Save the dictionary of world coordinates to a JSON file.
        """
        with open(os.path.join(self.root_dir, 'Cine_World_Coordinates.json'), 'wt') as f:
            json.dump(self.Dict, f)
            
        return os.path.join(self.root_dir, 'Cine_World_Coordinates.json')