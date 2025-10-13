#!/usr/bin/env python3
# coding: utf-8

"""
Implement a Haydn-based RecSys engine using a precomputed similarity matrix for music-to-painting recommendations.

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


class HaydnEngine(Engine):
    def __init__(self):
        """Initialize the engine with the precomputed similarity matrix."""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "data")
        self.similarity_path = os.path.join(
            self.data_dir, "heal_similarity_matrix_Haydn.csv"
        )

        # Load similarity matrix
        try:
            self.similarity_matrix = pd.read_csv(self.similarity_path, index_col=0)
            self.similarity_matrix.index = self.similarity_matrix.index.astype(str)
            self.similarity_matrix.columns = self.similarity_matrix.columns.astype(str)
            logging.info(
                f"Loaded similarity matrix with {len(self.similarity_matrix)} music IDs and {len(self.similarity_matrix.columns)} painting IDs."
            )
        except FileNotFoundError:
            logging.error(f"Similarity matrix not found at {self.similarity_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading similarity matrix: {str(e)}")
            raise

    def music_id_to_index(self, music_id):
        """
        Verify if a music ID exists in the similarity matrix.

        Args:
            music_id (str): Music ID.

        Returns:
            str: Music ID if found in the matrix, else raises ValueError.
        """
        if music_id not in self.similarity_matrix.index:
            raise ValueError(f"Music ID '{music_id}' not found in similarity matrix.")
        return music_id

    def music_ids_to_indices(self, music_ids):
        """
        Filter a list of music IDs to those present in the similarity matrix.

        Args:
            music_ids (list): List of music IDs.

        Returns:
            list: List of music IDs that were found in the matrix.
        """
        indices = []
        for music_id in music_ids:
            try:
                index = self.music_id_to_index(music_id)
                indices.append(index)
            except ValueError as e:
                logging.warning(str(e))
                continue
        return indices

    def unpack_prefs(self, preferences):
        """
        Unpack preferences into music IDs and normalized weights.

        Args:
            preferences (dict): Dictionary of music IDs (str) and ratings (int, 1-5).

        Returns:
            tuple: (list of music IDs, list of normalized weights).
        """
        music_ids = list(preferences.keys())
        ratings = [float(preferences[mid]) for mid in music_ids]
        total = sum(ratings)
        if total == 0:
            raise ValueError("Sum of ratings cannot be zero.")
        weights = [r / total for r in ratings]
        return music_ids, weights

    def retrieval(self, preferences, n=3):
        """
        Recommend paintings for a user based on their music preferences using the similarity matrix.

        Args:
            preferences (dict): Dictionary of music IDs (str) and ratings (int, 1-5).
            n (int): Number of recommendations to return.

        Returns:
            list: List of recommended painting IDs.
        """
        music_list, weights = self.unpack_prefs(preferences)
        if not music_list:
            logging.error("No valid music IDs provided after unpacking preferences.")
            raise ValueError("No valid music IDs provided.")

        music_indices = self.music_ids_to_indices(music_list)
        if not music_indices:
            logging.error(
                "None of the provided music IDs were found in the similarity matrix."
            )
            raise ValueError("No music IDs found in the similarity matrix.")

        # Compute weighted average of distances (smaller distance = more similar)
        weights = np.array(weights)
        score_list = np.array(
            [self.similarity_matrix.loc[mid].values for mid in music_indices]
        )
        weighted_distances = np.average(score_list, axis=0, weights=weights)

        # Get top n painting IDs with smallest distances
        top_n_indices = np.argsort(weighted_distances)[:n]
        top_n_painting_ids = self.similarity_matrix.columns[top_n_indices].tolist()

        return top_n_painting_ids


if __name__ == "__main__":
    # Example preferences
    preferences = {
        "21": 5,
        "55": 4,
        "56": 3,
        "4": 5,
        "7": 2,
    }

    eng = HaydnEngine()
    recommendations = eng.retrieval(preferences, n=3)
    print("Recommended paintings:", recommendations)
