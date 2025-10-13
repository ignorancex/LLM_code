#!/usr/bin/env python3
# coding: utf-8

"""
Extract music features using MERT and acoustic features, combining with valence-arousal labels.
Features are saved to mozart-crossmodal/data/music/music_features_with_embeddings.csv.

Author:
- Bereket A. Yilma <name.surname@artaicare.com>
"""

import os
import pandas as pd
import numpy as np
import torchaudio
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from torchaudio import transforms as T

# Define paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # /mozart-crossmodal
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "music")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
FILTERED_SONGS_CSV = os.path.join(DATA_DIR, "filtered_songs.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "music_features_with_embeddings.csv")

# Load CSV with song metadata and valence-arousal labels
try:
    df = pd.read_csv(FILTERED_SONGS_CSV)
    df["song_id"] = df["song_id"].astype(int)
except FileNotFoundError:
    raise FileNotFoundError(f"Filtered songs CSV not found at {FILTERED_SONGS_CSV}")
except Exception as e:
    raise Exception(f"Error loading filtered songs CSV: {str(e)}")

# Load MERT model for deep audio embeddings
try:
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True, revision="main"
    )
    model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True, revision="main"
    )
except Exception as e:
    raise Exception(f"Error loading MERT model: {str(e)}")


# Extract acoustic features from precomputed CSV
def extract_acoustic_features(song_id):
    file_path = os.path.join(FEATURES_DIR, f"{song_id}.csv")
    if not os.path.exists(file_path):
        print(f"Feature file missing for song_id {song_id}")
        return None

    try:
        feature_df = pd.read_csv(file_path, delimiter=";")
        if feature_df.shape[0] < 10:  # Ignore files with very few timestamps
            print(
                f"Skipping song_id {song_id}: Too few timestamps ({feature_df.shape[0]})"
            )
            return None
        feature_means = feature_df.iloc[:, 1:].mean().values
        feature_medians = feature_df.iloc[:, 1:].median().values
        feature_stds = feature_df.iloc[:, 1:].std().values
        final_features = np.concatenate([feature_means, feature_medians, feature_stds])

        if np.all(final_features == 0):  # Skip if all features are zero
            print(f"Skipping song_id {song_id}: All features are zero")
            return None
        return final_features
    except Exception as e:
        print(f"Error processing acoustic features for song_id {song_id}: {str(e)}")
        return None


# Extract MERT embedding
def extract_mert_embedding(song_id):
    audio_path = os.path.join(AUDIO_DIR, f"{song_id}.mp3")
    if not os.path.exists(audio_path):
        print(f"Audio file missing for song_id {song_id} at {audio_path}")
        return None

    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        print(
            f"Loaded waveform shape for song_id {song_id}: {waveform.shape}"
        )  # (channels, num_samples)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, num_samples)
        print(f"Mono waveform shape for song_id {song_id}: {waveform.shape}")

        # Resample to MERT's expected sampling rate (24kHz)
        target_sr = 24000
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        print(f"Resampled waveform shape for song_id {song_id}: {waveform.shape}")

        # Remove extra dimension
        waveform = waveform.squeeze(0)  # Should be (num_samples,)
        print(f"Final shape before MERT for song_id {song_id}: {waveform.shape}")

        # Process through MERT
        inputs = processor(waveform, return_tensors="pt", sampling_rate=target_sr)
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean-pool the embeddings
        mert_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return mert_embedding
    except Exception as e:
        print(f"Error extracting MERT embedding for song_id {song_id}: {str(e)}")
        return None


# Iterate over all songs and extract features
all_features = []
for _, row in df.iterrows():
    song_id = int(row["song_id"])

    # Extract both types of features
    acoustic_feats = extract_acoustic_features(song_id)
    mert_feats = extract_mert_embedding(song_id)

    if acoustic_feats is None or mert_feats is None:
        continue  # Skip songs with missing data

    # Concatenate features
    combined_feats = np.concatenate([acoustic_feats, mert_feats])
    print(f"Extracted features for song_id {song_id}: {combined_feats.shape}")

    # Append with valence-arousal labels
    all_features.append(
        [song_id] + list(combined_feats) + [row["valence_mean"], row["arousal_mean"]]
    )

# Ensure features were extracted
if not all_features:
    raise ValueError("Feature extraction failed. No valid features extracted.")

# Convert to DataFrame
num_acoustic_feats = len(acoustic_feats) if acoustic_feats is not None else 0
num_mert_feats = len(mert_feats) if mert_feats is not None else 0
columns = (
    ["song_id"]
    + [f"acoustic_feat_{i}" for i in range(num_acoustic_feats)]
    + [f"mert_feat_{i}" for i in range(num_mert_feats)]
    + ["valence", "arousal"]
)

feature_df = pd.DataFrame(all_features, columns=columns)

# Save the final dataset
try:
    feature_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Feature extraction complete! Saved as {OUTPUT_CSV}")
except Exception as e:
    raise Exception(f"Error saving output CSV: {str(e)}")
