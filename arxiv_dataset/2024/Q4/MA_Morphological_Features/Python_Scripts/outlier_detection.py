# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:08:10 2024

Remove outlier due to labelling error

The dataset is split into 2 groups for outlier detection:
    1. No MR
    2. MR

@author: Henry
"""

import os
import pandas as pd
import numpy as np

class OutlierDetection:
    def __init__(self, root_dir, features_csv):
        """
        Initialize the feature processor class.

        :param root_dir: Root directory where data is stored and output will be saved.
        :param features_csv: Path to the CSV file containing the features.
        """
        self.root_dir = root_dir
        self.features_csv = features_csv
        self.df = self._load_data()

    def _load_data(self):
        """
        Load the CSV data into a pandas DataFrame and drop unnecessary columns.

        :return: DataFrame with loaded data.
        """
        df = pd.read_csv(self.features_csv)
        return df.drop(columns=['Unnamed: 0'])

    def split_data_by_class(self):
        """
        Split the data into NoVD (class 0) and MR (class 1) groups.

        :return: Tuple of DataFrames (NoVD, MR), and their corresponding Assession Numbers.
        """
        NoVD_df = self.df[self.df['class'] == 0].reset_index(drop=True)
        NoVD_Assession_df = NoVD_df.pop('Assession_Number')
        NoVD_df = NoVD_df.drop(columns=['class'])
        
        MR_df = self.df[self.df['class'] == 1].reset_index(drop=True)
        MR_Assession_df = MR_df.pop('Assession_Number')
        MR_df = MR_df.drop(columns=['class'])

        return NoVD_df, MR_df, NoVD_Assession_df, MR_Assession_df

    def replace_outliers(self, df_to_update, feature_headers):
        """
        Replace outliers in the DataFrame using the IQR method with the mean of non-outlier values.

        :param df_to_update: DataFrame to update with outliers replaced.
        :param feature_headers: List of headers (features) to check for outliers.
        :return: Updated DataFrame with outliers replaced.
        """
        for header in feature_headers:
            header_values = df_to_update[header].values
            
            q1 = np.percentile(header_values, 25)
            q3 = np.percentile(header_values, 75)
            iqr = q3 - q1
            
            lower_threshold = q1 - (1.5 * iqr)
            upper_threshold = q3 + (1.5 * iqr)
            
            outlier_idx = np.where((header_values < lower_threshold) | (header_values > upper_threshold))[0]
            
            if len(outlier_idx) > 0:
                non_outliers = np.delete(header_values, outlier_idx)
                mean = np.mean(non_outliers)
                df_to_update.loc[outlier_idx, header] = mean
                
        return df_to_update

    def process_features(self):
        """
        Process the features by replacing outliers and combining the data back together.

        :return: Final DataFrame with processed features and class labels.
        """
        NoVD_df, MR_df, NoVD_Assession_df, MR_Assession_df = self.split_data_by_class()
        feature_headers = NoVD_df.columns.values.tolist()
        
        NoVD_df_updated = self.replace_outliers(NoVD_df, feature_headers)
        MR_df_updated = self.replace_outliers(MR_df, feature_headers)
        
        # Combine updated data and class labels
        y_df = pd.DataFrame([0] * len(NoVD_df_updated) + [1] * len(MR_df_updated), columns=['class'])
        df_updated = pd.concat([NoVD_df_updated, MR_df_updated], ignore_index=True)
        assession_df = pd.concat([NoVD_Assession_df, MR_Assession_df], ignore_index=True)
        
        df_final = pd.concat([assession_df, df_updated, y_df], axis=1)
        df_final.columns = ['Assession_Number'] + feature_headers + ['class']

        return df_final

    def save_processed_data(self, df_final):
        """
        Save the processed DataFrame to a CSV file.

        :param df_final: DataFrame containing the processed data.
        :param output_filename: Name of the output file.
        """
        output_path = os.path.join(self.root_dir, 'Cine_MitralAnnulus_Features_FULL_OR.csv')
        df_final.to_csv(output_path, index=False)
        
        return output_path