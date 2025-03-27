import os
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
from music_tagger import GENRES
import argparse
import time
import torch

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description='Create synthetic music dataset.')
parser.add_argument('--descriptions-folder', type=str, required=True, help='Path where descriptions are stored')
parser.add_argument('--music-folder', type=str, required=True, help='Path where synthetic music will be stored')
args = parser.parse_args()

music_folder = args.music_folder
description_folder = args.descriptions_folder

processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
model = model.to(device)
sampling_rate = model.config.audio_encoder.sampling_rate

for genre in GENRES:
    genre_folder = os.path.join(description_folder, genre)
    genre_music_folder = os.path.join(music_folder, genre)
    if not os.path.isdir(genre_music_folder):
        os.mkdir(genre_music_folder)
    text_files = [os.path.join(genre_folder, x) for x in os.listdir(genre_folder) if x.endswith('.txt')]
    if genre == 'HipHop':
        genre_str = 'Hip Hop'
    else:
        genre_str = genre
    for k, t_file in enumerate(text_files):
        print(genre, k + 1, ' / ', len(text_files))
        with open(t_file, 'r') as fp:
            descriptions = fp.readlines()
        t_s = time.time()

        # remove occasional empty lines
        descriptions = [x for x in descriptions if len(x.replace('\n', ' ').strip()) > 1]

        # check if file names already exist and skip
        file_names = [os.path.join(genre_music_folder,
                                   os.path.splitext(os.path.basename(t_file))[0] + '_' + str(kk) + '.wav')
                      for kk in range(10)]
        skip = False
        for f in file_names:
            if os.path.isfile(f):
                skip = True
                break
        if skip:
            print(' --> already processed - skipping!')
            continue

        descriptions = [genre + ' ' + genre + ' ' + x for x in descriptions]
        inputs = processor(
            text=descriptions,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        audio_values = model.generate(**inputs, guidance_scale=10.0, max_new_tokens=512)
        audio_values = audio_values.to('cpu')
        audio_values = audio_values.numpy()
        for kk in range(audio_values.shape[0]):
            scipy.io.wavfile.write(os.path.join(genre_music_folder,
                                                os.path.splitext(os.path.basename(t_file))[0] + '_' + str(kk) + '.wav'),
                                   rate=sampling_rate,
                                   data=audio_values[kk, 0])
        print('time elapsed: ', time.time() - t_s)
