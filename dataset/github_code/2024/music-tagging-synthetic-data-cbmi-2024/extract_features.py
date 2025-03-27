import argparse
import os.path
import time
from audioread.exceptions import NoBackendError

from music_tagger import collect_audio_data, extract_features
import pickle


# Set up argument parser
parser = argparse.ArgumentParser(description='Extract music-cnn data.')
parser.add_argument('--source-data', type=str, required=True, help='Path to the source (human-synthetic) embeddings directory')
parser.add_argument('--target-data', type=str, required=True, help='Path to the target (real) embeddings directory')
args = parser.parse_args()

# arguments to variables
SOURCE_AUDIO_DIR = args.source_data
TARGET_AUDIO_DIR = args.target_data

audio_src_data = collect_audio_data(SOURCE_AUDIO_DIR)
audio_trg_data = collect_audio_data(TARGET_AUDIO_DIR)
print('# src files : ', len(audio_src_data))
print('# trg files : ', len(audio_trg_data))

audio_data = audio_src_data + audio_trg_data

for k, audio_file in enumerate(audio_data):
    t_s = time.time()
    print('Processing file ', k + 1, ' / ', len(audio_data))
    trg_path = os.path.splitext(audio_file['path'])[0] + '.p'
    #if os.path.isfile(trg_path):
    #    print('-->  already processed - skipping!')
    #    continue
    try:
        features = extract_features(audio_file['path'])
        pickle.dump(features, open(trg_path, 'wb'))
        print(' -> time elapsed : ', time.time() - t_s)
    except NoBackendError:
        print(' -> [ ERROR ] - skipping!')

