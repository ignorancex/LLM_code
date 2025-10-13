#!/usr/bin/env python3
# coding: utf-8

"""
Implement a ResNet-based RecSys engine using a precomputed similarity matrix for painting-to-painting recommendations.

Author:
- Bereket A. Yilma <name.surname@artaicare.com>
- Luis A. Leiva <name.surname@uni.lu>
"""

import os
import pandas as pd
import numpy as np
from engine import Engine
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ResNetEngine(Engine):
    def __init__(self):
        """Initialize the engine with the precomputed similarity matrix."""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "data")
        self.similarity_path = os.path.join(
            self.data_dir, "painting", "resnet_similarity_matrix.csv"
        )

        # Load similarity matrix
        try:
            self.similarity_matrix = pd.read_csv(self.similarity_path, index_col=0)
            self.similarity_matrix.index = self.similarity_matrix.index.astype(str)
            self.similarity_matrix.columns = self.similarity_matrix.columns.astype(str)
            logging.info(
                f"Loaded similarity matrix with {len(self.similarity_matrix)} painting IDs."
            )
        except FileNotFoundError:
            logging.error(f"Similarity matrix not found at {self.similarity_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading similarity matrix: {str(e)}")
            raise

    def painting_id_to_index(self, painting_id):
        """
        Verify if a painting ID exists in the similarity matrix.

        Args:
            painting_id (str): Painting ID (filename).

        Returns:
            str: Painting ID if found in the matrix, else raises ValueError.
        """
        if painting_id not in self.similarity_matrix.index:
            raise ValueError(
                f"Painting ID '{painting_id}' not found in similarity matrix."
            )
        return painting_id

    def painting_ids_to_indices(self, painting_ids):
        """
        Filter a list of painting IDs to those present in the similarity matrix.

        Args:
            painting_ids (list): List of painting IDs.

        Returns:
            list: List of painting IDs that were found in the matrix.
        """
        indices = []
        for painting_id in painting_ids:
            try:
                index = self.painting_id_to_index(painting_id)
                indices.append(index)
            except ValueError as e:
                logging.warning(str(e))
                continue
        return indices

    def unpack_prefs(self, preferences):
        """
        Unpack preferences into painting IDs and normalized weights.

        Args:
            preferences (dict): Dictionary of painting IDs (str) and ratings (int, 1-5).

        Returns:
            tuple: (list of painting IDs, list of normalized weights).
        """
        painting_ids = list(preferences.keys())
        ratings = [float(preferences[pid]) for pid in painting_ids]
        total = sum(ratings)
        if total == 0:
            raise ValueError("Sum of ratings cannot be zero.")
        weights = [r / total for r in ratings]
        return painting_ids, weights

    def retrieval(self, preferences, n=3):
        """
        Recommend paintings for a user based on their painting preferences using the similarity matrix.

        Args:
            preferences (dict): Dictionary of painting IDs (str) and ratings (int, 1-5).
            n (int): Number of recommendations to return.

        Returns:
            list: List of recommended painting IDs.
        """
        painting_list, weights = self.unpack_prefs(preferences)
        if not painting_list:
            logging.error("No valid painting IDs provided after unpacking preferences.")
            raise ValueError("No valid painting IDs provided.")

        painting_indices = self.painting_ids_to_indices(painting_list)
        if not painting_indices:
            logging.error(
                "None of the provided painting IDs were found in the similarity matrix."
            )
            raise ValueError("No painting IDs found in the similarity matrix.")

        # Compute weighted average of similarities (higher similarity = more similar)
        weights = np.array(weights)
        score_list = np.array(
            [self.similarity_matrix.loc[pid].values for pid in painting_indices]
        )
        weighted_similarities = np.average(score_list, axis=0, weights=weights)

        # Get top n painting IDs with highest similarities
        top_n_indices = np.argsort(weighted_similarities)[::-1][:n]  # Descending order
        top_n_painting_ids = self.similarity_matrix.columns[top_n_indices].tolist()

        return top_n_painting_ids


if __name__ == "__main__":
    # Example preferences
    preferences = {
        "57726e48edc2cb3880b69b14": 5,
        "57728619edc2cb388003099d": 4,
        "57728ad8edc2cb3880125a8d": 3,
        "5772860dedc2cb388002f60f": 5,
        "577282c7edc2cb3880f8bec4": 2,
    }

    eng = ResNetEngine()
    recommendations = eng.retrieval(preferences, n=3)
    print("Recommended paintings:", recommendations)
