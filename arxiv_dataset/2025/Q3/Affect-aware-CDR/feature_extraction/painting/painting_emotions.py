#!/usr/bin/env python3
# coding: utf-8

"""
Preprocess WikiArt emotional labels to convert them into valence and arousal scores using the NRC VAD Lexicon.
Saves intermediate emotional labels to mozart-crossmodal/data/paintings/painting_emotions.csv and
final valence-arousal scores to mozart-crossmodal/data/paintings/painting_valence_arousal.csv.

Authors:
- Bereket A. Yilma <name.surname@artaicare.com>
"""


import os
import pandas as pd
import nltk
from nltk.corpus import wordnet

# Download WordNet data if not already present (run once)
try:
    nltk.data.find("corpus/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")  # Open Multilingual Wordnet for broader synonym coverage

# Define paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PAINTING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "paintings")
WIKIART_TSV = os.path.join(PAINTING_DATA_DIR, "WikiArt-Emotions-All.tsv")
NRC_VAD_TXT = os.path.join(PAINTING_DATA_DIR, "NRC-VAD-Lexicon.txt")
INTERMEDIATE_CSV = os.path.join(PAINTING_DATA_DIR, "painting_emotions.csv")
OUTPUT_CSV = os.path.join(PAINTING_DATA_DIR, "painting_valence_arousal.csv")

# Load the WikiArt emotional labels TSV file
try:
    df = pd.read_csv(WIKIART_TSV, sep="\t")
except FileNotFoundError:
    raise FileNotFoundError(f"WikiArt TSV not found at {WIKIART_TSV}")
except Exception as e:
    raise Exception(f"Error loading WikiArt TSV: {str(e)}")

# Load the NRC VAD Lexicon
try:
    nrc_vad = pd.read_csv(
        NRC_VAD_TXT, sep="\t", names=["Word", "Valence", "Arousal", "Dominance"]
    )
except FileNotFoundError:
    raise FileNotFoundError(f"NRC VAD Lexicon not found at {NRC_VAD_TXT}")
except Exception as e:
    raise Exception(f"Error loading NRC VAD Lexicon: {str(e)}")

# Convert the lexicon to a dictionary for quick lookup
vad_dict = pd.Series(
    nrc_vad[["Valence", "Arousal"]].values.tolist(), index=nrc_vad["Word"]
).to_dict()


# Function to find synonyms of a word
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms


# Initialize lists to store results
results = []
intermediate_data = []

# Process each row in the DataFrame
for _, row in df.iterrows():
    # Initialize variables to accumulate scores and count valid words
    valence_sum = 0
    arousal_sum = 0
    count = 0
    emotions = []

    # Iterate over the emotional label columns
    for col in df.columns:
        if col.startswith("Art (image+title):"):
            emotion = col.split(":")[-1].strip()
            if pd.notna(row[col]) and row[col] > 0:
                emotions.append(emotion)
                # Look up the valence and arousal values for the emotion
                if emotion in vad_dict:
                    valence, arousal = vad_dict[emotion]
                    valence_sum += valence
                    arousal_sum += arousal
                    count += 1
                else:
                    # find synonyms in the VAD lexicon
                    found = False
                    for synonym in get_synonyms(emotion):
                        if synonym in vad_dict:
                            valence, arousal = vad_dict[synonym]
                            valence_sum += valence
                            arousal_sum += arousal
                            count += 1
                            found = True
                            break
                    if not found:
                        # Assign default neutral scores if no match is found
                        valence_sum += 0.5
                        arousal_sum += 0.5
                        count += 1

    # Calculate average scores if there are valid words
    if count > 0:
        avg_valence = valence_sum / count
        avg_arousal = arousal_sum / count
    else:
        avg_valence = avg_arousal = 0.0  # Default to neutral if no valid emotions

    # Append the result
    results.append([row["ID"], avg_valence, avg_arousal])
    intermediate_data.append([row["ID"], ", ".join(emotions)])

# Create DataFrames from the results
results_df = pd.DataFrame(results, columns=["ID", "Valence", "Arousal"])
intermediate_df = pd.DataFrame(intermediate_data, columns=["ID", "Emotions"])

# Save the intermediate and final results to CSV files
try:
    intermediate_df.to_csv(INTERMEDIATE_CSV, index=False)
    print(f"Saved intermediate emotions to {INTERMEDIATE_CSV}")
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved valence-arousal scores to {OUTPUT_CSV}")
except Exception as e:
    raise Exception(f"Error saving CSV files: {str(e)}")
