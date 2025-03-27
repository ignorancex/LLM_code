from tensorflow.keras.models import load_model
from music_tagger import extract_features, GENRES, N_MELS, N_FRAMES
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score


model = load_model('models/mcnn_src_only_split_0.h5')
'''
gtzan_folder = '/Users/nkroher/Desktop/audio-data/GTZAN/'
y_true = []
y_pred = []
for k, genre in enumerate(GENRES):
    genre_folder = os.path.join(gtzan_folder, genre)
    data_files = [os.path.join(genre_folder, x) for x in os.listdir(genre_folder) if x.endswith('.wav')]
    data = [extract_features(x)[:N_FRAMES, :] for x in data_files]
    data = np.asarray(data)
    y_true_genre = [k for _ in data_files]
    y_true.extend(y_true_genre)
    y_pred_genre = model.predict(data)
    y_pred_genre = list(np.argmax(y_pred_genre, axis=1))
    y_pred.extend(y_pred_genre)
    print(genre, accuracy_score(y_true_genre, y_pred_genre))

print(accuracy_score(y_true, y_pred))

'''
audio_path = "/Users/nkroher/Desktop/vocals/hiphop.00010/accompaniment.wav"
# audio_path = '/Users/nkroher/code/tmc2/misc/synthetic_music_mir/music/Blues/Blues_run_6_choice_2_4.wav'
features = extract_features(audio_path)
X = np.zeros((1, N_FRAMES, N_MELS))
X[0, :] = features[:N_FRAMES, :]

pred = model.predict(X)[0, :]
print(GENRES[np.argmax(pred)])
