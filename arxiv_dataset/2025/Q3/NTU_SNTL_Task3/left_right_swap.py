"""
Script to perform left-right channel inversion with label swapping.
For model development, we only perform this swapping on training data.

Assuming the data directory is loosely as follows:

DCASE2025_SELD_dataset
    stereo_dev
        ├── dev-train-sony
        ├── dev-train-tau
        ├── dev-test-sony
        ├── dev-test-tau
        └── dev-train-realcs   (to be created by this script)
    metadata_dev
        ├── dev-train-sony
        ├── dev-train-tau
        ├── dev-test-sony
        ├── dev-test-tau
        └── dev-train-realcs   (to be created by this script)
"""


import os
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm


def process_audio_file(audio_fp, final_audio_folder, final_meta_folder):
    """
    Process a single stereo audio file:
    - Load the stereo audio.
    - Swap the left and right channels.
    - Save the swapped audio to the final_audio_folder.
    - Load the corresponding metadata CSV, invert the azimuth, and save the updated CSV to final_meta_folder.

    Parameters:
        audio_fp (str): Path to the input .wav file.
        final_audio_folder (str): Directory to save swapped audio files.
        final_meta_folder (str): Directory to save updated metadata CSV files.
    """

    # Load the stereo audio
    audio_data, sr = librosa.load(audio_fp, sr=None, mono=False, dtype=np.float32)
    if audio_data.ndim != 2 or audio_data.shape[0] != 2:
        print(f"File {audio_fp} is not stereo. Skipping.")
        return

    # Swap channels
    swapped_audio = audio_data[[1, 0], :]

    # Construct new audio file path and write swapped audio
    filename = os.path.basename(audio_fp)
    new_audio_fp = os.path.join(final_audio_folder, filename.replace(".wav", "_swap.wav"))
    sf.write(new_audio_fp, swapped_audio.T, sr)

    # Construct corresponding CSV file path
    csv_fp = audio_fp.replace("stereo_dev", "metadata_dev").replace(".wav", ".csv")
    if not os.path.exists(csv_fp):
        print(f"Metadata CSV file {csv_fp} not found for {audio_fp}. Skipping CSV update.")
        return

    # Read CSV with headers: frame, class, source, azimuth, distance, onscreen
    df = pd.read_csv(csv_fp)

    # Invert the 'azimuth' column
    if 'azimuth' in df.columns:
        df['azimuth'] = -df['azimuth']
    else:
        print(f"'azimuth' column not found in {csv_fp}. Skipping CSV update.")
        return

    # Save the updated CSV to the final metadata folder
    new_csv_fp = os.path.join(final_meta_folder, filename.replace(".wav", "_swap.csv"))
    df.to_csv(new_csv_fp, index=False)



def process_directory(root_dir, final_audio_folder, final_meta_folder):
    """
    Recursively process all .wav files in a directory tree, processing only training files.

    Parameters:
        root_dir (str): Root directory to search for .wav files.
        final_audio_folder (str): Directory to save swapped audio files.
        final_meta_folder (str): Directory to save updated metadata CSV files.
    """
    for current_root, _, files in os.walk(root_dir, topdown=True):
        for file in tqdm(files, desc="Performing Left-Right swap...", unit="file"):
            if file.lower().endswith(".wav"):
                audio_fp = os.path.join(current_root, file)
                # Only process files with 'train' in the file path and NOT already processed by channelswap
                if "train" in audio_fp.lower() and "realcs" not in audio_fp.lower():
                    process_audio_file(audio_fp, final_audio_folder, final_meta_folder)


def main():
    # Define the upper directories for stereo audio and metadata
    stereo_dir = "DCASE2025_SELD_dataset/stereo_dev"
    metadata_dir = "DCASE2025_SELD_dataset/metadata_dev"

    # Define the output directories for the swapped files (under a new folder 'dev-train-realcs')
    final_audio_folder = os.path.join(stereo_dir, "dev-train-realcs")
    final_meta_folder = os.path.join(metadata_dir, "dev-train-realcs")

    # Create the output directories if they don't exist
    os.makedirs(final_audio_folder, exist_ok=True)
    os.makedirs(final_meta_folder, exist_ok=True)

    # Process the stereo audio files in the training set
    process_directory(stereo_dir, final_audio_folder, final_meta_folder)


if __name__ == "__main__":
    main()