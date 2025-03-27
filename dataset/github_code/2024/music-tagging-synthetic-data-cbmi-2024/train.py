import argparse
from music_tagger import collect_feature_data, get_simple_model, data_generator, BATCH_SIZE, data_generator_da, \
    get_da_model, collect_src_feature_data, get_tl_model, get_ft_model
import tensorflow as tf
import numpy as np
import datetime
import pickle
import os
import numpy as np


# Set up argument parser
parser = argparse.ArgumentParser(description='Train a model for genre classification using Music-cnn.')
parser.add_argument('--source-data', type=str, required=True, help='Path to the source (human-synthetic) data directory')
parser.add_argument('--target-data', type=str, required=True, help='Path to the target (real) data directory')
parser.add_argument('--mode', type=str, help='"trg_only", "src_only", "both", "TL", "FT", or "DA"')
args = parser.parse_args()

# arguments to variables
SOURCE_FEATURE_DIR = args.source_data
TARGET_FEATURE_DIR = args.target_data
MODE = args.mode

assert MODE in ["trg_only", "src_only", "both", "DA", "TL", "FT"], \
    print('Mode should be trg_only", "src_only", "both", "TL", "FT", or "DA"')

split_accs = []
split_loss = []


for split in range(3):
    print('SPLIT ', split + 1, ' / ', 3)
    if MODE != "src_only":
        src_tracks, trg_tracks, val_tracks = collect_feature_data(SOURCE_FEATURE_DIR, TARGET_FEATURE_DIR, split)
        print('# src tracks: ', len(src_tracks))
        print('# trg tracks: ', len(trg_tracks))
        print('# val tracks: ', len(val_tracks))
    else:
        train_tracks, val_tracks = collect_src_feature_data(SOURCE_FEATURE_DIR)
        print('# train tracks: ', len(train_tracks))
        print('# val tracks: ', len(val_tracks))

    if MODE == "src_only":
        num_train_steps = len(train_tracks) // BATCH_SIZE
        train_generator = data_generator(train_tracks, num_train_steps, shuffle=True, augment=True)
        num_val_steps = len(val_tracks) // BATCH_SIZE
        val_generator = data_generator(val_tracks, num_val_steps, shuffle=False, augment=False)
        model_path = 'models/mcnn_src_only_split_' + str(split) + '.h5'

    if MODE in ["trg_only", "TL", "FT"]:
        num_train_steps = len(trg_tracks) // BATCH_SIZE
        train_generator = data_generator(trg_tracks, num_train_steps, shuffle=True, augment=True)
        num_val_steps = len(val_tracks) // BATCH_SIZE
        val_generator = data_generator(val_tracks, num_val_steps, shuffle=False, augment=False)
        model_path = 'models/mcnn_' + MODE + '_split_' + str(split) + '.h5'

    if MODE == "both":
        tracks = trg_tracks + src_tracks
        num_train_steps = len(tracks) // BATCH_SIZE
        train_generator = data_generator(tracks, num_train_steps, shuffle=True, augment=True)
        num_val_steps = len(val_tracks) // BATCH_SIZE
        val_generator = data_generator(val_tracks, num_val_steps, shuffle=False, augment=False)
        model_path = 'models/mcnn_both_split_' + str(split) + '.h5'

    if MODE == "DA":
        tracks = trg_tracks + src_tracks
        num_train_steps = len(tracks) // BATCH_SIZE
        train_generator = data_generator_da(tracks, trg_tracks, num_train_steps, shuffle=True, augment=True)
        num_val_steps = len(val_tracks) // BATCH_SIZE
        val_generator = data_generator_da(val_tracks, trg_tracks, num_val_steps, shuffle=False, augment=False)
        model_path = 'models/mcnn_da_split_' + str(split) + '.h5'

    src_model_path = 'models/mcnn_src_only_split_0.h5'
    if MODE == "DA":
        model = get_da_model()
    if MODE in ["src_only", "trg_only", "both"]:
        model = get_simple_model()
    if MODE == "TL":
        model = get_tl_model(src_model_path)
    if MODE == "FT":
        model = get_ft_model(src_model_path)

    # Callbacks setup
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    if MODE == 'DA':
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_classifier_output_loss',
            patience=5,
            verbose=0
        )
    else:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=0
        )
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=num_train_steps,
                                  epochs=50,
                                  callbacks=[early_stopping_callback, checkpoint_callback],
                                  validation_data=val_generator,
                                  validation_steps=num_val_steps
                                  )

    results = history.history

    if MODE in ['trg_only', 'both', 'src_only', 'TL', 'FT']:
        min_idx = np.argmin(results['val_loss'])
        split_accs.append(results['val_categorical_accuracy'][min_idx])
        split_loss.append(min(results['val_loss']))
    else:
        min_idx = np.argmin(results['val_classifier_output_loss'])
        split_accs.append(results['val_classifier_output_categorical_accuracy'][min_idx])
        split_loss.append(min(results['val_classifier_output_loss']))

mean_loss = np.mean(np.asarray(split_loss))
mean_acc = np.mean(np.asarray(split_accs))
std_loss = np.std(np.asarray(split_loss))
std_acc = np.std(np.asarray(split_accs))

print(' ***************************************** ')
print('MEAN LOSS : ', mean_loss)
print('MEAN ACC  : ', mean_acc)
print('STD LOSS : ', std_loss)
print('STD ACC  : ', std_acc)

results_folder = 'results/'
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

now = datetime.datetime.now()
date_str = now.strftime("%d-%m-%Y_%H:%M:%S")
results_path = os.path.join(results_folder, MODE + '_' + date_str + '.txt')
with open(results_path, 'w') as fp:
    fp.write('MEAN LOSS : ' + str(mean_loss))
    fp.write('\n')
    fp.write('MEAN ACC : ' + str(mean_acc))
    fp.write('\n')
    fp.write('STD LOSS : ' + str(std_loss))
    fp.write('\n')
    fp.write('STD ACC : ' + str(std_acc))
