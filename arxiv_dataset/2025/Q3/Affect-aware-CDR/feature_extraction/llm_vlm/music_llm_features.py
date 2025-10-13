#!/usr/bin/env python3
# coding: utf-8

"""
Extract semantic descriptions for music using GPT-4o and encode them with BERT.
Saves descriptions to mozart-crossmodal/data/music/llm_music_semantics.csv and embeddings to
mozart-crossmodal/data/music/llm_music_embeddings.csv.

Requires OPENAI_API_KEY in .env file at project root.

Author:
- Bereket A. Yilma <name.surname@artaicare.com>
"""

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from openai import OpenAI
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Limit PyTorch threads to avoid CPU overload
torch.set_num_threads(1)
logging.info(f"Set PyTorch threads to {torch.get_num_threads()}")

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
logging.info("OpenAI API key loaded successfully")

# Define paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # /mozart-crossmodal
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "music")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
FILTERED_SONGS_CSV = os.path.join(DATA_DIR, "filtered_songs.csv")
SEMANTICS_OUTPUT_CSV = os.path.join(DATA_DIR, "llm_music_semantics.csv")
EMBEDDINGS_OUTPUT_CSV = os.path.join(DATA_DIR, "llm_music_embeddings.csv")

# Initialize OpenAI client with API key from .env
client = OpenAI(api_key=OPENAI_API_KEY)
logging.info("OpenAI client initialized")

# Load BERT model and tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    model.to("cpu")  # Force CPU
    logging.info("BERT model and tokenizer loaded successfully on CPU")
    # Test BERT standalone
    test_input = tokenizer("Test sentence", return_tensors="pt").to("cpu")
    with torch.no_grad():
        test_output = model(**test_input)
    logging.info(f"BERT test output shape: {test_output.last_hidden_state.shape}")
except Exception as e:
    raise Exception(f"Error loading or testing BERT model: {str(e)}")

# Load metadata
try:
    df = pd.read_csv(FILTERED_SONGS_CSV)
    df["song_id"] = df["song_id"].astype(int)
    logging.info(f"Loaded metadata with {len(df)} songs")
except FileNotFoundError:
    raise FileNotFoundError(f"Filtered songs CSV not found at {FILTERED_SONGS_CSV}")
except Exception as e:
    raise Exception(f"Error loading filtered songs CSV: {str(e)}")


# Load and summarize acoustic features
def load_acoustic_features(song_id):
    file_path = os.path.join(FEATURES_DIR, f"{song_id}.csv")
    if not os.path.exists(file_path):
        logging.warning(f"Feature file missing for song_id {song_id}")
        return None
    try:
        feature_df = pd.read_csv(file_path, delimiter=";")
        if feature_df.shape[0] < 10:
            logging.warning(
                f"Skipping song_id {song_id}: Too few timestamps ({feature_df.shape[0]})"
            )
            return None
        summary = {
            "F0final_mean": feature_df["F0final_sma_amean"].mean(),
            "RMSenergy_mean": feature_df["pcm_RMSenergy_sma_amean"].mean(),
            "spectralCentroid_mean": feature_df[
                "pcm_fftMag_spectralCentroid_sma_amean"
            ].mean(),
            "mfcc1_mean": feature_df["pcm_fftMag_mfcc_sma[1]_amean"].mean(),
        }
        logging.info(f"Loaded acoustic features for song_id {song_id}")
        return summary
    except Exception as e:
        logging.error(
            f"Error loading acoustic features for song_id {song_id}: {str(e)}"
        )
        return None


# Generate semantic description using GPT-4o
def generate_semantic_description(song_id, metadata, acoustic_summary):
    prompt = (
        f"Generate a concise semantic description (50-100 words) of a music track with the following details:\n"
        f"Song ID: {song_id}\n"
        f"Valence Mean: {metadata['valence_mean']:.2f} (std: {metadata['valence_std']:.2f})\n"
        f"Arousal Mean: {metadata['arousal_mean']:.2f} (std: {metadata['arousal_std']:.2f})\n"
        f"Acoustic Features: Pitch (F0) mean: {acoustic_summary['F0final_mean']:.2f}, "
        f"Energy (RMS) mean: {acoustic_summary['RMSenergy_mean']:.2f}, "
        f"Spectral Centroid mean: {acoustic_summary['spectralCentroid_mean']:.2f}, "
        f"MFCC1 mean: {acoustic_summary['mfcc1_mean']:.2f}\n"
        f"Describe the emotional tone and musical characteristics based on these features."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
            timeout=10,
        )
        description = response.choices[0].message.content.strip()
        logging.info(f"Generated description for song_id {song_id}: {description}")
        return description
    except Exception as e:
        logging.error(f"Error generating description for song_id {song_id}: {str(e)}")
        return None


# Encode description with BERT
def encode_description(description):
    try:
        logging.info(f"Tokenizing description: {description[:50]}...")
        inputs = tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}  # Ensure CPU
        logging.info(f"Inputs prepared: {inputs['input_ids'].shape}")
        with torch.no_grad():
            outputs = model(**inputs)
        logging.info(f"Model output shape: {outputs.last_hidden_state.shape}")
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        logging.info(f"Encoded embedding shape: {embedding.shape}")
        return embedding
    except Exception as e:
        logging.error(f"Error encoding description: {str(e)}")
        return None


# Process songs with progress tracking
semantics_data = []
embeddings_data = []
total_songs = len(df)
with tqdm(total=total_songs, desc="Generating descriptions and embeddings") as pbar:
    for _, row in df.iterrows():
        song_id = int(row["song_id"])
        metadata = row.to_dict()

        logging.info(f"Starting processing for song_id {song_id}")

        # Load acoustic features
        acoustic_summary = load_acoustic_features(song_id)
        if acoustic_summary is None:
            logging.warning(
                f"Skipping song_id {song_id} due to missing acoustic features"
            )
            pbar.update(1)
            continue

        # Generate semantic description
        description = generate_semantic_description(song_id, metadata, acoustic_summary)
        if description is None:
            logging.warning(
                f"Skipping song_id {song_id} due to description generation failure"
            )
            pbar.update(1)
            continue
        semantics_data.append([song_id, description])

        # Encode with BERT
        embedding = encode_description(description)
        if embedding is None:
            logging.warning(f"Skipping song_id {song_id} due to embedding failure")
            pbar.update(1)
            continue
        embeddings_data.append([song_id] + embedding.tolist())

        logging.info(f"Completed processing for song_id {song_id}")
        pbar.update(1)

# Save semantic descriptions
if not semantics_data:
    logging.error("No semantic descriptions generated.")
    raise ValueError("No semantic descriptions generated.")
semantics_df = pd.DataFrame(semantics_data, columns=["song_id", "description"])
try:
    semantics_df.to_csv(SEMANTICS_OUTPUT_CSV, index=False)
    logging.info(
        f"Semantic descriptions saved to {SEMANTICS_OUTPUT_CSV} ({len(semantics_data)} songs)"
    )
except Exception as e:
    raise Exception(f"Error saving semantics CSV: {str(e)}")

# Save embeddings
if not embeddings_data:
    logging.error("No embeddings generated.")
    raise ValueError("No embeddings generated.")
embedding_columns = ["song_id"] + [f"bert_feat_{i}" for i in range(768)]
embeddings_df = pd.DataFrame(embeddings_data, columns=embedding_columns)
try:
    embeddings_df.to_csv(EMBEDDINGS_OUTPUT_CSV, index=False)
    logging.info(
        f"Embeddings saved to {EMBEDDINGS_OUTPUT_CSV} ({len(embeddings_data)} songs)"
    )
except Exception as e:
    raise Exception(f"Error saving embeddings CSV: {str(e)}")
