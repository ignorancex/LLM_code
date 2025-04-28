# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:48:19 2024

MA Morphological Features Pipeline

0. Cine MRI Labelling

1. Scanner to World Coordinates Conversion

2. Coordinates Interpolation

3. Find Best-fit Ellipse and Plane

4. Extract Morphological Features

5. Outlier Detections

6. Feature Selection (mRMR)

7. Feature Analysis (LDA & RF Classification)

@author: Henry
"""

import os
import argparse
from cine_landmarks_labeller_dicom import CineLabeller
from Scanner2World_Coordinates_Conversion import CineWorldCoordinates
from coordinates_interpolation import CineCoordInterpolator
from best_ellipseNplane_fitting import MitralAnnulusEllipseFitter
from displacement_features_extraction import CineMitralAnnulusFeatureExtractor
from outlier_detection import OutlierDetection
from feature_selection_mrmr import MRMRFeatureSelector
from Feature_Analysis import CineFeatureAnalysis

def main():
    # Create the main parser
    parser = argparse.ArgumentParser(description="Select which operation to run.")
    subparsers = parser.add_subparsers(dest='class_name', help='Choose which operation to run')
    
    # Subparser for CineLabeller
    parser_labeller = subparsers.add_parser('cine_labeller', help='Run CineLabeller')
    parser_labeller.add_argument("--root_dir", required = False, default = os.getcwd(), type = str,
                             help = "Root Directory to read-in or save-out files")
    parser_labeller.add_argument("--dcm_dir", required = True, type = str,
                             help = "Dicom Images Directory")
    parser_labeller.add_argument("--labels_path", required = False, default = None, type = str,
                             help = "Patients Pathology Label Path (with extension .json)")
    parser_labeller.add_argument("--timeout", required = False, default = 50, type = int,
                             help = "Idle time to exit programme")
    parser_labeller.add_argument("--n", required = False, default = 5, type = int,
                             help = "Number of phases to skip labelling")
    parser_labeller.add_argument("--pathology", required = False, default = None, type = str,
                             help = "Specific pathology to label")
    
    # Subparser for CinePipline
    parser_pipeline = subparsers.add_parser('cine_pipeline', help='Run CinePipeline')
    parser_pipeline.add_argument("--mode", required = False, default = 0, type = int,
                             help = "Run entire pipeline or individual process, see index for more information")
    parser_pipeline.add_argument("--root_dir", required = False, default = os.getcwd(), type = str,
                             help = "Root Directory to read-in or save-out files")
    parser_pipeline.add_argument("--dcm_dir", required = True, type = str,
                             help = "Dicom Images Directory")
    parser_pipeline.add_argument("--labels_path", required = False, default = None, type = str,
                             help = "Patients Pathology Label Path (with extension .json)")

    # Parse the arguments
    args = parser.parse_args()
    
    # Execute based on the selected class
    if args.class_name == 'cine_labeller':
        print('Opening Cine Labeller......')
        cine_labeller = CineLabeller(
                root_dir = args.root_dir,
                CINE_images_dir=args.dcm_dir,
                labels_dict_path=args.labels_path,
                timeout=args.timeout,
                N_phases2skip=args.n,
                pathology=args.pathology
            )
        cine_labeller.process_patients()
        
    elif args.class_name == 'cine_pipeline':
        mode = args.mode
        
        if mode == 0 or mode == 1:
            # 1. Convert Image coordinates to common World coordinates        
            print('Converting to World Coordinates......')
            cine_world_coordinates = CineWorldCoordinates(
                root_dir = args.root_dir,
                cine_images_dir=args.dcm_dir)
            cine_world_coordinates.process_patients()
            cine_world_coordinates_path = cine_world_coordinates.save_world_coordinates() 
        
        if mode == 0 or mode == 2:
            # 2. World coordinates interpolation (to find the coordinates of the skipped phases)
            print('Interpolating the Skipped World Coordinates......')
            cine_coordinates_interpolator = CineCoordInterpolator(
                root_dir = args.root_dir,
                coord_json_path = cine_world_coordinates_path
                )
            cine_coordinates_interpolator.process_patient_data()
            cine_world_coordinates_interpolated_path = cine_coordinates_interpolator.save_interpolated_coordinates()
        
        if mode == 0 or mode == 3:
            # 3. Find Best-fit Ellipse and Plane and the corresponding features from the labelled coordinates
            print('Finding Best-Fit Ellipse and Plane to Extract MA Features......')
            ellipse_plane_fitter = MitralAnnulusEllipseFitter(
                root_dir = args.root_dir,
                interpolated_json_path = cine_world_coordinates_interpolated_path
                )
            ellipse_plane_fitter.process_patient_data()
            ellipse_features_json_path = ellipse_plane_fitter.save_features()
        
        if mode == 0 or mode == 4:
            # 4. Extract displacement features
            print('Extracting MA Displacement Features......')
            displacement_features_extraction = CineMitralAnnulusFeatureExtractor(
                root_dir = args.root_dir,
                labels_file = args.labels_path,
                ellipse_features_dict_path = ellipse_features_json_path
                )
            full_features_json_path = displacement_features_extraction.extract_features()
        
        if mode == 0 or mode == 5:
            # 5. Outlier Detection
            print('Removing Individual Outlier Features......')
            feature_outlier_detection = OutlierDetection(
                root_dir = args.root_dir,
                features_csv = full_features_json_path
                )
            df_final = feature_outlier_detection.process_features()
            full_features_OR_csv_path = feature_outlier_detection.save_processed_data(df_final)
        
        if mode == 0 or mode == 6:
            # 6. Feature Selection
            print('Selecting the Top-K Features Using MRMR......')
            feature_selection = MRMRFeatureSelector(
                root_dir = args.root_dir,
                features_csv = full_features_OR_csv_path,
                K = 10
                )
            selected_features_csv_path = feature_selection.run()
        
        if mode == 0 or mode == 7:
            # 7. Feature Analysis (LDA & RF)
            print('Features Analysis - LDA and RF......')
            feature_analysis = CineFeatureAnalysis(
                root_dir = args.root_dir,
                features_csv = selected_features_csv_path
                )
            feature_analysis.run_analysis()
        
        
if __name__ == "__main__":
    main()
