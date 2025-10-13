# -*- coding: utf-8 -*-
"""

@author: seema
"""

from pydub import AudioSegment
import pandas as pd
import os

# Base paths
audio_base_path = <path/to/original/wav/files>
csv_base_path = <path/to/csv/file/with/segment/labels> 
output_base_path = <path/to/combined/wav/files>

# Ensure the output directory exists
os.makedirs(output_base_path, exist_ok=True)

#%%
# Function to process all files in a given subdirectory
def process_files(subfolder):
    audio_path = os.path.join(audio_base_path, subfolder)
    csv_path = os.path.join(csv_base_path, subfolder)
    output_path = os.path.join(output_base_path, subfolder)
    
    # Ensure the subfolder in output exists
    os.makedirs(output_path, exist_ok=True)
    
    # Get all .wav files
    for audio_file in os.listdir(audio_path):
        if audio_file.endswith(".wav"):
            # Corresponding CSV file
            file_id = os.path.splitext(audio_file)[0]  # Get the filename without extension
            csv_file = f"{file_id}.csv"
            audio_file_path = os.path.join(audio_path, audio_file)
            csv_file_path = os.path.join(csv_path, csv_file)
            
            # Check if the corresponding CSV exists
            if os.path.exists(csv_file_path):
                print(f"Processing {audio_file} with {csv_file}")
                total_files += 1
                
                # Load the CSV and audio file
                df = pd.read_csv(csv_file_path)
                audio = AudioSegment.from_wav(audio_file_path)
                
                # Filter rows where speaker is 'PAR'
                par_segments = df[df['speaker'] == 'PAR']
                
                # Check if the filtered DataFrame is empty
                if par_segments.empty:
                    print(f"No 'PAR' segments found in {audio_file}. Skipping to next file.")
                    
                else:
                    # Process the non-empty par_segments
                    print(f"Processing {audio_file} with {len(par_segments)} segments.")
    
                    # Initialize an empty AudioSegment for the combined result
                    combined_audio = AudioSegment.empty()
                    
                    # Process each segment
                    for _, row in par_segments.iterrows():
                        start_time = row['begin']  # Start time in milliseconds
                        end_time = row['end']      # End time in milliseconds
                        
                        # Extract the segment and append it to the combined audio
                        segment = audio[start_time:end_time]
                        combined_audio += segment
                    
                    # Export the combined audio to a new file
                    output_file_path = os.path.join(output_path, f"{file_id}_combined.wav")
                    combined_audio.export(output_file_path, format="wav")
                    print(f"Combined audio saved to: {output_file_path}")
                    processed_files += 1
            else:
                print(f"CSV file not found for {audio_file}")
                
#%%
# Process both 'ad' and 'cn' subdirectories
total_files = 0
processed_files = 0
process_files("ad")
print(f'\n Total # of files {total_files}.')
print(f'# files processed {processed_files}.')

total_files = 0
processed_files = 0
process_files("cn")
print(f'\n Total # of files {total_files}.')
print(f'# files processed {processed_files}.')

