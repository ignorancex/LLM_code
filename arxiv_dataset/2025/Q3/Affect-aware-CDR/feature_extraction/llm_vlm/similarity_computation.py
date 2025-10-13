#!/usr/bin/env python3
# coding: utf-8

"""
Computes the cross-modal similarity matrix S_LV between 239 therapeutically relevant music tracks
and 63 paintings using cosine similarity. Saves to mozart-crossmodal/data/llm_similarity_matrix.csv.

Inputs:
- Music embeddings: (256D per track)
- Painting embeddings:  (256D per painting)
- Heal music IDs: (239 song_ids)

Author:
- Bereket A. Yilma <name.surname@artaicare.com>
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # /mozart-crossmodal
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MUSIC_CSV = os.path.join(DATA_DIR, "music", "music_multi_modal_256D.csv")
PAINTING_CSV = os.path.join(DATA_DIR, "painting", "painting_multi_modal_256D.csv")
HEAL_MUSIC_CSV = os.path.join(DATA_DIR, "heal_music.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "heal_llm_similarity_matrix.csv")


# Load and filter features
def load_and_filter_features():
    try:
        # Load music embeddings (909 tracks)
        music_df = pd.read_csv(MUSIC_CSV)
        logging.info(f"Loaded music embeddings: {music_df.shape} (256D + song_id)")

        # Load painting embeddings (63 paintings)
        painting_df = pd.read_csv(PAINTING_CSV)
        logging.info(f"Loaded painting embeddings: {painting_df.shape} (256D + ID)")

        # Load heal music IDs (239 tracks)
        heal_music_df = pd.read_csv(HEAL_MUSIC_CSV)
        logging.info(
            f"Loaded heal music metadata: {heal_music_df.shape} (song_id, valence_mean, arousal_mean)"
        )

        # Filter music to 239 heal tracks
        heal_song_ids = set(heal_music_df["song_id"])
        if len(heal_song_ids) != 239:
            raise ValueError(
                f"Expected 239 unique song_ids in heal_music.csv, found {len(heal_song_ids)}"
            )
        filtered_music_df = music_df[music_df["song_id"].isin(heal_song_ids)]
        if filtered_music_df.shape[0] != 239:
            raise ValueError(
                f"Expected 239 matching songs in music embeddings, found {filtered_music_df.shape[0]}"
            )
        logging.info(
            f"Filtered music embeddings to {filtered_music_df.shape} matching heal song_ids"
        )

        # Extract feature matrices
        music_features = filtered_music_df.drop(
            columns=["song_id"]
        ).values  # 239 × 256D
        painting_features = painting_df.drop(columns=["ID"]).values  # 63 × 256D

        return (
            filtered_music_df["song_id"].tolist(),
            painting_df["ID"].tolist(),
            music_features,
            painting_features,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading or filtering features: {str(e)}")


# Compute and save similarity matrix
def compute_similarity_matrix(
    music_ids, painting_ids, music_features, painting_features
):
    # Compute cosine similarity (239 × 63)
    similarity_matrix = cosine_similarity(music_features, painting_features)
    logging.info(f"Computed similarity matrix shape: {similarity_matrix.shape}")

    # Create DataFrame with song_ids as rows, painting_ids as columns
    similarity_df = pd.DataFrame(
        similarity_matrix, index=music_ids, columns=painting_ids
    )
    similarity_df.index.name = "song_id"

    # Save to CSV
    try:
        similarity_df.to_csv(OUTPUT_CSV)
        logging.info(
            f"Similarity matrix saved to {OUTPUT_CSV} with shape {similarity_df.shape}"
        )
    except Exception as e:
        raise Exception(f"Error saving similarity matrix: {str(e)}")


# Main execution
if __name__ == "__main__":
    # Load and filter features
    music_ids, painting_ids, music_features, painting_features = (
        load_and_filter_features()
    )

    # Compute and save similarity matrix
    compute_similarity_matrix(
        music_ids, painting_ids, music_features, painting_features
    )
