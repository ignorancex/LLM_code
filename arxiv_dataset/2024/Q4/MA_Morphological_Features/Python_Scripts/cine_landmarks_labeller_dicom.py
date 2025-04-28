# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 22:58:40 2023

Label landmarks on Cine 2-, 3-, and 4-ch MRIs for Valve Landmarks Model

If Cine is scanned in the wrong orientation, the whole sequence can be skipped 
by labelling a random coordinate if a timeout is set

Labelling instruction:
    - the landmarks to label and orders are instructed in the pop-up windows
    - left click to label, right click to remove, middle click to stop labelling
    - Remove timeout by setting the value to -ve
    - Programme will automatically quit if the target number of coordinates is not reached
    
Patients without all Cine LAX views (2-, 3-, 4-CH) are excluded

WARNING: The labelled landmarks slices are stored in ascending number based on range(N),
         not the actual Instance Number, but they are already sorted in the correct order.
         This is sorted in the Coordinates Conversion script, in which the converted
         coordinates are stored in the corresponding Instance Number.

@author: Henry
"""

import os
import imageio.v3 as iio
import pydicom as dcm
import numpy as np
import glob
import SimpleITK as sitk
from operator import itemgetter
import json
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import argparse

class CineLabeller:
    def __init__(self, root_dir = None, CINE_images_dir = None, labels_dict_path = None, 
                 timeout = None, N_phases2skip = None, target_pathology = None):
        """
        Initializes the CineLabeller class with the necessary parameters.
        
        :param CINE_images_dir: Directory containing CINE images.
        :param labels_dict_path: Path to the JSON file containing patient labels.
        :param timeout: Timeout for user input during interactive labeling.
        :param N: Interval for labeling every Nth slice.
        :param target_pathology: Target pathology to filter patients.
        """
        self.root_dir = root_dir
        self.CINE_images_dir = CINE_images_dir
        self.timeout = timeout
        self.N = N_phases2skip
        self.target_pathology = target_pathology

        # Load labels dictionary
        try:
            with open(labels_dict_path, 'r') as f:
                self.Labels_Dict = json.load(f)
        except:
            self.Labels_Dict = None

        # Load example images for reference
        self.example_2ch = iio.imread(os.path.join(root_dir, '2CH_example.png'))
        self.example_3ch = iio.imread(os.path.join(root_dir, '3CH_example.png'))
        self.example_4ch = iio.imread(os.path.join(root_dir, '4CH_example.png'))

        # Load labelled lists and notes
        self.labelled_list, self.label_notes = self._load_existing_labels()

    def _load_existing_labels(self):
        """
        Loads existing labeled images and notes from files.
        
        :return: Tuple containing the list of labeled images and notes.
        """
        try:
            with open(os.path.join(self.root_dir, 'labelled_landmark_files_dicom.txt', "r")) as f:
                labelled_list = f.readlines()
            with open(os.path.join(self.root_dir, 'Data_CMR_Cine_dicom_viewstatus_notes.txt', "r")) as f:
                label_notes = f.readlines()
        except FileNotFoundError:
            # If files are not found, initialize empty lists
            labelled_list = []
            label_notes = []
        return labelled_list, label_notes

    def correct_slice_ordering(self, dicom_file_lists, time_point):
        """
        Corrects the order of DICOM slices based on their position in the patient.
        
        :param dicom_file_lists: List of DICOM file paths for a patient.
        :param time_point: Specific time point to process.
        :return: List of sorted DICOM files by slice position.
        """
        image_position_patient_tag = '0020|0013'
        dicom_indexes = []

        # Iterate over each DICOM file to extract its position and index it
        for slice_index, dicom_file in enumerate(dicom_file_lists):
            reader = sitk.ImageFileReader()
            reader.SetFileName(dicom_file[time_point])
            reader.ReadImageInformation()

            # Extract the image position from metadata and store it with its index
            image_position_patient = int(reader.GetMetaData(image_position_patient_tag))
            dicom_indexes.append({
                'position': image_position_patient,
                'index': slice_index,
                'folder_order': time_point
            })

        # Sort DICOM files based on their positions in ascending order
        dicom_sorted_indexes = sorted(dicom_indexes, key=itemgetter('position'))
        dicom_sorted_files = [dicom_file_lists[key_value['index']][time_point] for key_value in dicom_sorted_indexes]

        return dicom_sorted_files

    def load_series_time(self, image):
        """
        Loads the series time from a DICOM image.
        
        :param image: Path to the DICOM image.
        :return: Series time as a string.
        """
        return self._load_dicom_metadata(image, '0008|0031')

    def load_acquisition_time(self, image):
        """
        Loads the acquisition time from a DICOM image.
        
        :param image: Path to the DICOM image.
        :return: Acquisition time as a string.
        """
        return self._load_dicom_metadata(image, '0008|0032')

    def _load_dicom_metadata(self, image, tag):
        """
        Helper function to load specific metadata from a DICOM file.
        
        :param image: Path to the DICOM image.
        :param tag: DICOM tag to retrieve metadata.
        :return: Metadata value corresponding to the specified tag.
        """
        reader = sitk.ImageFileReader()
        reader.SetFileName(image)
        reader.ReadImageInformation()
        return reader.GetMetaData(tag)

    def load_patient_data(self, patient_folders):
        """
        Loads and sorts patient data based on acquisition or series time.
        
        :param patient_folders: Dictionary of patient folder paths.
        :return: Dictionary containing sorted patient data and same-timepoint folders.
        """
        patient_data = {}
        sorted_folders_same_timepoint = {}

        # Iterate over each patient folder
        for key, folder_list in patient_folders.items():
            dicom_file_lists = [folder for folder in folder_list if len(folder) < 500]
            min_total_files = min(len(folder) for folder in dicom_file_lists)

            dicom_order_list = []
            files_list = []

            # Process each time point to sort files
            for time_point in range(min_total_files):
                file_list = self.correct_slice_ordering(dicom_file_lists, time_point)
                dicom_order_list.extend(file_list)
                files_list.append(file_list)

            # Determine reference time and method for sorting
            time_ref, method = self.get_time_reference(files_list)

            # Sort and load the image series based on the determined reference time
            sorted_files_list = sorted(files_list, key=lambda fl: dicom_order_list[files_list.index(fl)]['position'])
            sitk_temporal_images = []
            sorted_files_list_same_timepoint = []

            for sorted_file_list in sorted_files_list[:min_total_files]:
                time_temp = self.load_acquisition_time(sorted_file_list[0]) if method == 'acq' else self.load_series_time(sorted_file_list[0])

                if time_temp == time_ref:
                    reader = sitk.ImageSeriesReader()
                    reader.SetFileNames(sorted_file_list)
                    sitk_temporal_images.append(reader.Execute())
                    sorted_files_list_same_timepoint.append(sorted_file_list)

            patient_data[key] = sitk_temporal_images
            sorted_folders_same_timepoint[key] = sorted_files_list_same_timepoint

        return sorted_folders_same_timepoint

    def get_time_reference(self, files_list):
        """
        Determines the reference time (acquisition or series) for sorting DICOM files.
        
        :param files_list: List of DICOM file paths.
        :return: Tuple of reference time and method ('acq' or 'series').
        """
        acq_time_dict = {}
        series_time_dict = {}

        # Build dictionaries for acquisition and series times
        for files in files_list:
            acq_time_temp = self.load_acquisition_time(files[0])
            series_time_temp = self.load_series_time(files[0])

            acq_time_dict[acq_time_temp] = acq_time_dict.get(acq_time_temp, 0) + 1
            series_time_dict[series_time_temp] = series_time_dict.get(series_time_temp, 0) + 1

        # Determine if acquisition time should be used based on the frequency of occurrence
        if max(acq_time_dict.values()) > 1:
            acq_time_dict_diff = {k: abs(30 - v) for k, v in acq_time_dict.items()}
            time_ref = min(acq_time_dict_diff, key=acq_time_dict_diff.get)
            return time_ref, 'acq'

        # Otherwise, use series time as reference
        series_time_dict_diff = {k: abs(30 - v) for k, v in series_time_dict.items()}
        time_ref = min(series_time_dict_diff, key=series_time_dict_diff.get)
        return time_ref, 'series'

    def convert_and_save(self, dicom_file, arr, save_path):
        """
        Converts and saves a numpy array as a DICOM file.
        
        :param dicom_file: DICOM file object to be saved.
        :param arr: Numpy array representing the image data.
        :param save_path: Path where the DICOM file will be saved.
        """
        arr = arr.astype('uint16')
        dicom_file.PixelData = arr.tobytes()
        dicom_file.save_as(save_path)

    def label_coord(self, file_path, timeout, idx, message, n):
        """
        Handles the interactive labeling of DICOM images.
        
        :param file_path: Path to the DICOM file to be labeled.
        :param timeout: Timeout for user input during labeling.
        :param idx: Index of the current image slice.
        :param message: Message to be displayed for labeling instructions.
        :param n: Number of points to be labeled.
        :return: Tuple containing DICOM file, image, label array, and coordinates list.
        """
        dicom_file = dcm.read_file(file_path)
        image = dicom_file.pixel_array

        # Determine the reference image based on the sequence type
        ref_image = self.example_2ch if '2ch' in file_path else self.example_3ch if 'lvot' in file_path else self.example_4ch

        # Display the reference and current image for interactive labeling
        plt.subplot(1, 2, 1)
        plt.imshow(ref_image)
        plt.title('Reference Image for Labelling')
        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap='bone')
        plt.title(f'Cardiac Phase {idx}\n{message}')
        plt.grid(False)
        coord_list = plt.ginput(n, timeout=timeout)
        plt.show()

        return dicom_file, image, np.zeros(image.shape + (1,)), coord_list

    def get_equidistant_points(self, p1, p2, parts):
        """
        Generates equidistant points between two points.
        
        :param p1: Starting point (x, y).
        :param p2: Ending point (x, y).
        :param parts: Number of parts to divide the line segment into.
        :return: List of equidistant points between p1 and p2.
        """
        return list(zip(np.linspace(p1[0], p2[0], parts+1),
                        np.linspace(p1[1], p2[1], parts+1)))

    def process_patients(self):
        """
        Main method to process all patients in the directory based on the target pathology.
        """
        patients_in_dir = os.listdir(self.CINE_images_dir)

        # Process each patient in the directory
        for c, patient in enumerate(patients_in_dir):
            if self.Labels_Dict.get(patient, {}).get(self.target_pathology) == 1 or self.Labels_Dict == None or self.target_pathology == None:
                print(f"{c}. Processing: {patient}")
                sequences_list = os.listdir(os.path.join(self.CINE_images_dir, patient))

                if not self._has_all_lax_views(sequences_list):
                    self._handle_missing_views(patient, sequences_list)
                else:
                    self._process_sequences(patient, sequences_list)

    def _has_all_lax_views(self, sequences_list):
        """
        Checks if all required LAX views are present in the sequences list.
        
        :param sequences_list: List of sequences for a patient.
        :return: True if all views are present, False otherwise.
        """
        return all(view in sequences_list for view in ['cine_2ch', 'cine_4ch', 'cine_lvot'])

    def _handle_missing_views(self, patient, sequences_list):
        """
        Handles cases where a patient is missing required LAX views.
        
        :param patient: Patient ID or name.
        :param sequences_list: List of sequences available for the patient.
        """
        print(f'NOTICE: {patient} does not have all LAX views, thus excluded!')

        # Determine which views are missing
        cine_2ch_exist = ' has' if 'cine_2ch' in sequences_list else ' has no'
        cine_4ch_exist = ' has' if 'cine_4ch' in sequences_list else ' has no'
        cine_3ch_exist = ' has' if 'cine_lvot' in sequences_list else ' has no'

        note = f"{patient}{cine_2ch_exist} cine_2ch,{cine_4ch_exist} cine_4ch, and{cine_3ch_exist} cine_3ch."

        # Add a note about the missing views
        if note not in self.label_notes:
            self.label_notes.append(note)
            self._save_notes()

    def _save_notes(self):
        """
        Saves the current notes to a file.
        """
        with open(os.path.join(self.root_dir, 'Data_CMR_Cine_dicom_viewstatus_notes.txt', "w")) as f:
            f.write("\n".join(self.label_notes))

    def _process_sequences(self, patient, sequences_list):
        """
        Processes sequences for a given patient, handling labeling and interpolation.
        
        :param patient: Patient ID or name.
        :param sequences_list: List of sequences available for the patient.
        """
        patient_folders = {seq: [os.path.join(self.CINE_images_dir, patient, seq)] for seq in sequences_list if 'landmarks' not in seq}
        sequence_folder = self.load_patient_data(patient_folders)

        # Process each sequence for the patient
        for seq, sub_folder_files in sequence_folder.items():
            counter = 0
            N_range = range(0, len(sub_folder_files), self.N)
            coord_list_previous = []

            for idx, file_path in enumerate(sub_folder_files):
                file_path = file_path[0]
                if file_path not in self.labelled_list:
                    proceed, interpolate, end = self._determine_proceed(counter, idx, len(sub_folder_files), N_range)

                    if proceed:
                        dicom_file, image, label, coord_list = self.label_coord(file_path, self.timeout, idx, self._get_message(seq), self._get_n(seq))
                        coord_list_previous.append(coord_list)
                        counter += 1

                        self._save_labelled_image(dicom_file, image, label, file_path, idx)

                    if interpolate:
                        self._interpolate_labels(coord_list_previous, sub_folder_files, dicom_file, idx)

            plt.close()

    def _determine_proceed(self, counter, idx, num_files, N_range):
        """
        Determines whether to proceed with labeling, interpolation, or end the process.
        
        :param counter: Counter for the number of labeled images.
        :param idx: Current index of the image slice.
        :param num_files: Total number of files in the sequence.
        :param N_range: Range of indices for Nth slices to label.
        :return: Tuple indicating whether to proceed, interpolate, and end the process.
        """
        if counter == 0 or idx == num_files - 1 or idx in N_range:
            return True, idx in N_range or idx == num_files - 1, idx == num_files - 1
        return False, False, False

    def _get_message(self, seq):
        """
        Returns the labeling instructions based on the sequence type.
        
        :param seq: Sequence name (e.g., cine_2ch, cine_lvot, cine_4ch).
        :return: Instruction message for labeling.
        """
        if '2ch' in seq:
            return '1. Mitral Anterior; 2. Mitral Posterior'
        elif 'lvot' in seq:
            return '3. Aortic Septal; 4. Aortic Freewall; 7. Mitral Septal; 8. Mitral Freewall'
        elif '4ch' in seq:
            return '9. Tricuspid Septal; 10. Tricuspid Freewall; 5. Mitral Septal; 6. Mitral Freewall'

    def _get_n(self, seq):
        """
        Returns the number of points to label based on the sequence type.
        
        :param seq: Sequence name (e.g., cine_2ch, cine_lvot, cine_4ch).
        :return: Number of points to label.
        """
        if '2ch' in seq or 'lvot' in seq or '4ch' in seq:
            return 4

    def _save_labelled_image(self, dicom_file, image, label, file_path, idx):
        """
        Saves the labeled image and updates the list of labeled files.
        
        :param dicom_file: DICOM file object to be saved.
        :param image: Image data as a numpy array.
        :param label: Label data as a numpy array.
        :param file_path: Original file path for saving the labeled image.
        :param idx: Index of the current image slice.
        """
        saveout_dir = os.path.join(os.path.dirname(file_path), '_landmarks')
        os.makedirs(saveout_dir, exist_ok=True)

        # Save the labeled image and its corresponding label map
        self.convert_and_save(dicom_file, image, os.path.join(saveout_dir, f'slice{idx}.dcm'))
        self.convert_and_save(dicom_file, label, os.path.join(saveout_dir, f'slice{idx}_label.dcm'))

        self.labelled_list.append(file_path)
        self._save_labelled_list()

    def _save_labelled_list(self):
        """
        Saves the updated list of labeled files to a file.
        """
        with open(os.path.join(self.root_dir, 'labelled_landmark_files_dicom.txt', "w")) as f:
            f.write("\n".join(self.labelled_list))

    def _interpolate_labels(self, coord_list_previous, sub_folder_files, dicom_file, idx):
        """
        Interpolates labels between slices and saves the interpolated labels.
        
        :param coord_list_previous: List of coordinates from previous labeling.
        :param sub_folder_files: List of files in the current sequence.
        :param dicom_file: DICOM file object to be saved.
        :param idx: Index of the current image slice.
        """
        N_step_list = list(range(1, self.N, 1))
        N_step_list.reverse()

        for i, idx in enumerate(N_step_list):
            file_path = sub_folder_files[idx][0]
            image = dicom_file.pixel_array

            label = np.zeros(image.shape + (1,))
            interpolated_coord_list = [self.get_equidistant_points(coord_list_previous[-2][j], coord_list_previous[-1][j], self.N)[i + 1]
                                       for j in range(len(coord_list_previous[-1]))]

            self._assign_landmark_labels(file_path, interpolated_coord_list, label)
            self._save_labelled_image(dicom_file, image, label, file_path, idx)

    def _assign_landmark_labels(self, file_path, coord_list, label):
        """
        Assigns labels to landmarks based on the sequence type.
        
        :param file_path: Path to the DICOM file being labeled.
        :param coord_list: List of coordinates for the landmarks.
        :param label: Numpy array representing the label data.
        """
        if '2ch' in file_path:
            label[int(coord_list[0][1]), int(coord_list[0][0]), 0] = 10
            label[int(coord_list[1][1]), int(coord_list[1][0]), 0] = 15
        elif 'lvot' in file_path:
            label[int(coord_list[0][1]), int(coord_list[0][0]), 0] = 100
            label[int(coord_list[1][1]), int(coord_list[1][0]), 0] = 150
            label[int(coord_list[2][1]), int(coord_list[2][0]), 0] = 20
            label[int(coord_list[3][1]), int(coord_list[3][0]), 0] = 25
        elif '4ch' in file_path:
            label[int(coord_list[0][1]), int(coord_list[0][0]), 0] = 200
            label[int(coord_list[1][1]), int(coord_list[1][0]), 0] = 250
            label[int(coord_list[2][1]), int(coord_list[2][0]), 0] = 30
            label[int(coord_list[3][1]), int(coord_list[3][0]), 0] = 35