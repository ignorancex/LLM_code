#!/usr/bin/env python3
# coding: utf-8

"""
Extract semantic descriptions for paintings in heal_paintings/ using GPT-4o (VLM) with both metadata and images.
Saves descriptions to mozart-crossmodal/data/painting/llm_painting_semantics_<N>.csv and embeddings to
mozart-crossmodal/data/painting/llm_painting_embeddings_<N>.csv, where <N> is the number of processed paintings.
Matches heal_paintings/ filenames to metadata 'ID' column (as strings).

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
import base64
from pathlib import Path

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
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "painting")
HEAL_IMAGES_DIR = os.path.join(DATA_DIR, "heal_paintings")
WIKIART_TSV = os.path.join(DATA_DIR, "WikiArt-Emotions-All.tsv")
VALENCE_AROUSAL_CSV = os.path.join(DATA_DIR, "painting_valence_arousal.csv")
SEMANTICS_OUTPUT_CSV_BASE = os.path.join(DATA_DIR, "llm_painting_semantics")
EMBEDDINGS_OUTPUT_CSV_BASE = os.path.join(DATA_DIR, "llm_painting_embeddings")

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
    test_input = tokenizer("Test sentence", return_tensors="pt").to("cpu")
    with torch.no_grad():
        test_output = model(**test_input)
    logging.info(f"BERT test output shape: {test_output.last_hidden_state.shape}")
except Exception as e:
    raise Exception(f"Error loading or testing BERT model: {str(e)}")

# Get filenames from heal_paintings directory
heal_image_files = [f for f in os.listdir(HEAL_IMAGES_DIR) if f.endswith(".jpg")]
heal_filenames = [
    Path(f).stem for f in heal_image_files
]  # e.g., '5772873fedc2cb388006c2c0'
logging.info(
    f"Found {len(heal_filenames)} healing painting filenames in {HEAL_IMAGES_DIR}: {heal_filenames[:5]}..."
)

# Load metadata and match to heal_filenames
try:
    wikiart_df = pd.read_csv(WIKIART_TSV, sep="\t")
    valence_arousal_df = pd.read_csv(VALENCE_AROUSAL_CSV)
    df = wikiart_df.merge(valence_arousal_df, on="ID", how="inner")
    # Keep ID as string
    logging.info(f"Loaded metadata with {len(df)} paintings")
    logging.info(f"Metadata columns: {list(df.columns)}")
    logging.info(f"Sample metadata IDs: {df['ID'].head().tolist()}")

    # Filter metadata to only include IDs present in heal_filenames
    df = df[df["ID"].isin(heal_filenames)]
    if len(df) == 0:
        raise ValueError(
            "No metadata matches heal_paintings filenames. Check ID column compatibility."
        )
    logging.info(f"Filtered metadata to {len(df)} paintings matching heal_filenames")
    metadata_dict = df.set_index("ID").to_dict("index")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Metadata file not found: {str(e)}")
except Exception as e:
    raise Exception(f"Error loading or merging metadata: {str(e)}")


# Encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {str(e)}")
        return None


# Generate semantic description using GPT-4o with both metadata and image
def generate_semantic_description(painting_id, metadata, image_path):
    prompt = (
        f"Generate a concise semantic description (50-100 words) of a painting with the following details:\n"
        f"Painting ID: {painting_id}\n"
        f"Style: {metadata['Style']}\n"
        f"Category: {metadata['Category']}\n"
        f"Artist: {metadata['Artist']}\n"
        f"Title: {metadata['Title']}\n"
        f"Year: {metadata['Year']}\n"
        f"Is Painting: {metadata['Is painting']}\n"
        f"Face/Body: {metadata['Face/body']}\n"
        f"Valence: {metadata['Valence']:.2f}\n"
        f"Arousal: {metadata['Arousal']:.2f}\n"
        f"Describe the emotional tone and visual characteristics based on this metadata and the attached image."
    )
    image_base64 = encode_image(image_path)
    if not image_base64:
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.7,
            timeout=15,
        )
        description = response.choices[0].message.content.strip()
        logging.info(
            f"Generated description for painting_id {painting_id}: {description}"
        )
        return description
    except Exception as e:
        logging.error(
            f"Error generating description for painting_id {painting_id}: {str(e)}"
        )
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


# Process paintings with progress tracking
semantics_data = []
embeddings_data = []
total_paintings = len(heal_filenames)
with tqdm(total=total_paintings, desc="Generating descriptions and embeddings") as pbar:
    for filename in heal_filenames:
        painting_id = filename  # Use filename as ID
        image_path = os.path.join(HEAL_IMAGES_DIR, f"{filename}.jpg")

        if not os.path.exists(image_path):
            logging.warning(
                f"Skipping painting_id {painting_id}: Image file {image_path} not found"
            )
            pbar.update(1)
            continue

        logging.info(f"Starting processing for painting_id {painting_id}")

        # Get metadata
        metadata = metadata_dict.get(painting_id)
        if metadata is None:
            logging.warning(f"Skipping painting_id {painting_id}: No metadata found")
            pbar.update(1)
            continue

        # Generate semantic description with metadata and image
        description = generate_semantic_description(painting_id, metadata, image_path)
        if description is None:
            logging.warning(
                f"Skipping painting_id {painting_id} due to description generation failure"
            )
            pbar.update(1)
            continue
        semantics_data.append([painting_id, description])

        # Encode with BERT
        embedding = encode_description(description)
        if embedding is None:
            logging.warning(
                f"Skipping painting_id {painting_id} due to embedding failure"
            )
            pbar.update(1)
            continue
        embeddings_data.append([painting_id] + embedding.tolist())

        logging.info(f"Completed processing for painting_id {painting_id}")
        pbar.update(1)

# Save semantic descriptions
if not semantics_data:
    logging.error("No semantic descriptions generated.")
    raise ValueError("No semantic descriptions generated.")
num_paintings = len(semantics_data)
semantics_output_csv = f"{SEMANTICS_OUTPUT_CSV_BASE}_{num_paintings}.csv"
semantics_df = pd.DataFrame(semantics_data, columns=["painting_id", "description"])
try:
    semantics_df.to_csv(semantics_output_csv, index=False)
    logging.info(
        f"Semantic descriptions saved to {semantics_output_csv} ({num_paintings} paintings)"
    )
except Exception as e:
    raise Exception(f"Error saving semantics CSV: {str(e)}")

# Save embeddings
if not embeddings_data:
    logging.error("No embeddings generated.")
    raise ValueError("No embeddings generated.")
embeddings_output_csv = f"{EMBEDDINGS_OUTPUT_CSV_BASE}_{num_paintings}.csv"
embedding_columns = ["painting_id"] + [f"bert_feat_{i}" for i in range(768)]
embeddings_df = pd.DataFrame(embeddings_data, columns=embedding_columns)
try:
    embeddings_df.to_csv(embeddings_output_csv, index=False)
    logging.info(
        f"Embeddings saved to {embeddings_output_csv} ({num_paintings} paintings)"
    )
except Exception as e:
    raise Exception(f"Error saving embeddings CSV: {str(e)}")
