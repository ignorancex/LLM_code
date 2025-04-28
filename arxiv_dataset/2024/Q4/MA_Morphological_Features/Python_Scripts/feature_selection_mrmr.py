# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:44:23 2024

Select Top K Features using minimum Redundancy Maximum Relevance (mRMR)

Works when sample size is small but with high number of variables

Only difference with v3 is the normal vector components are converted to absolute values

@author: Henry
"""

import os
import pandas as pd
from mrmr import mrmr_classif

class MRMRFeatureSelector:
    def __init__(self, root_dir, features_csv, K=10):
        """
        Initialize the feature selector class.

        :param root_dir: Root directory where data is stored and output will be saved.
        :param features_csv: Path to the CSV file containing the features.
        :param K: Number of top features to select using mRMR.
        """
        self.root_dir = root_dir
        self.features_csv = features_csv
        self.K = K
        self.df = self._load_data()

    def _load_data(self):
        """
        Load the CSV data into a pandas DataFrame and drop unnecessary columns.

        :return: DataFrame with loaded data.
        """
        return pd.read_csv(self.features_csv)

    def preprocess_data(self):
        """
        Preprocess the data by extracting target, assession number, and applying absolute values 
        to specific feature columns.

        :return: Preprocessed DataFrame, target column, and assession number.
        """
        target = self.df['class']
        assession = self.df['Assession_Number']
        df_features = self.df.drop(columns=['Unnamed: 0', 'Assession_Number', 'class'])
        headers = df_features.columns.values.tolist()

        # Apply absolute value to the selected feature headers
        target_headers = ['normal_vec', 'plane_translation']
        for header in headers:
            for t_header in target_headers:
                if t_header in header:
                    df_features[header] = df_features[header].abs()

        return df_features, target, assession

    def select_top_k_features(self, df_features, target):
        """
        Select the top K features using the mRMR (Minimum Redundancy Maximum Relevance) algorithm.

        :param df_features: DataFrame containing the features.
        :param target: Series containing the target labels.
        :return: DataFrame with the top K features and their relevance scores.
        """
        TopK_features, relevance, _ = mrmr_classif(
            X=df_features, y=target, K=self.K, 
            relevance='f', redundancy='c', return_scores=True
        )

        TopK_features_df = df_features[TopK_features]
        TopK_features_relevance_df = relevance[TopK_features]

        return TopK_features_df, TopK_features_relevance_df

    def save_selected_features(self, TopK_features_df, assession, target, TopK_features):
        """
        Save the selected top K features, along with the assession number and target, to a CSV file.

        :param TopK_features_df: DataFrame containing the top K features.
        :param assession: Series containing the assession numbers.
        :param target: Series containing the target labels.
        :param TopK_features: List of top K feature names.
        """
        df_final = pd.concat([assession, TopK_features_df, target], axis=1)
        columns_names = ['Assession_Number'] + TopK_features + ['class']
        df_final.columns = columns_names

        output_filename = f'Cine_MitralAnnulus_Features_FULL_OR_top{self.K}_v5.csv'
        output_path = os.path.join(self.root_dir, output_filename)
        df_final.to_csv(output_path, index=False)
        
        return output_path

    def run(self):
        """
        Run the full feature selection and saving process.
        """
        df_features, target, assession = self.preprocess_data()
        TopK_features_df, _ = self.select_top_k_features(df_features, target)
        output_path = self.save_selected_features(TopK_features_df, assession, target, TopK_features_df.columns.tolist())
        
        return output_path