import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPool2D, AveragePooling2D, Flatten, Dropout, \
    Dense, Input, GlobalAveragePooling2D, Concatenate, LayerNormalization
import json
import random
import pickle
import tensorflow_addons as tfa

random.seed(666)
tf.random.set_seed(666)

GENRES = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
SR = 16000
FFT_HOP = 256
FFT_SIZE = 512
N_MELS = 96
N_FRAMES = 625
LR = 0.001
BATCH_SIZE = 4


def collect_audio_data(folder):
    tracks = []
    for genre in GENRES:
        genre_folder = os.path.join(folder, genre)
        genre_tracks = [os.path.join(genre_folder, x) for x in os.listdir(genre_folder) if x.endswith('.wav')]
        label = GENRES.index(genre)
        for g in genre_tracks:
            tracks.append({'path': g,
                           'label': label})
    return tracks


def collect_src_feature_data(src_folder):
    src_tracks = []
    for genre in GENRES:
        genre_folder = os.path.join(src_folder, genre)
        genre_tracks = [os.path.join(genre_folder, x) for x in os.listdir(genre_folder) if x.endswith('.p')]
        label = GENRES.index(genre)
        for g in genre_tracks:
            src_tracks.append({'path': g,
                               'label': label})
    random.shuffle(src_tracks)
    split_idx = int(0.9 * len(src_tracks))
    train_tracks = src_tracks[:split_idx]
    val_tracks = src_tracks[split_idx:]
    return train_tracks, val_tracks


def collect_feature_data(src_folder, trg_folder, split_idx):
    src_tracks = []
    for genre in GENRES:
        genre_folder = os.path.join(src_folder, genre)
        genre_tracks = [os.path.join(genre_folder, x) for x in os.listdir(genre_folder) if x.endswith('.p')]
        label = GENRES.index(genre)
        for g in genre_tracks:
            src_tracks.append({'path': g,
                               'label': label})
    json_dir = 'artist_filtered_splits/'
    val_files = json.load(open(os.path.join(json_dir, 'split_' + str(split_idx) + '.json'), 'r'))
    val_files = [os.path.splitext(x)[0] + '.p' for x in val_files]
    val_files = [x.replace('_', '.') for x in val_files]
    trg_tracks = []
    val_tracks = []
    for genre in GENRES:
        genre_folder = os.path.join(trg_folder, genre)
        genre_tracks = [os.path.join(genre_folder, x) for x in os.listdir(genre_folder) if x.endswith('.p')]
        label = GENRES.index(genre)
        for g in genre_tracks:
            if os.path.basename(g) in val_files:
                val_tracks.append({'path': g,
                                   'label': label})
            else:
                trg_tracks.append({'path': g,
                                   'label': label})
    return src_tracks, trg_tracks, val_tracks


def extract_features(audio_path):
    # compute the log-mel spectrogram with librosa
    audio, _ = librosa.load(audio_path, sr=SR, duration=10, mono=True)
    audio /= np.max(np.abs(audio))
    audio_rep = librosa.feature.melspectrogram(y=audio,
                                               sr=SR,
                                               hop_length=FFT_HOP,
                                               n_fft=FFT_SIZE,
                                               n_mels=N_MELS).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)
    return audio_rep


def frontend_bn_mp2d_block(feats):
    """This function assumes max-pooling over the full frequency axis (2)
    """
    feats = LayerNormalization()(feats)
    feats = MaxPool2D(pool_size=(1, feats.shape[2]),
                      strides=(1, feats.shape[2]))(feats)
    feats = tf.squeeze(feats, [2])

    return feats


def musicnn():

    x_in = Input(shape=(N_FRAMES, N_MELS))
    x_in = tf.expand_dims(x_in, 3)

    x_in = LayerNormalization()(x_in)

    x_pad = tf.pad(x_in, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

    #### front-end musicnn features

    timbre_1 = Conv2D(
        filters=204,
        kernel_size=(7,38),
        padding="valid",
        activation="relu"
    )(x_pad)
    timbre_1 = frontend_bn_mp2d_block(timbre_1)


    timbre_2 = Conv2D(
        filters=204,
        kernel_size=(7, 67),
        padding="valid",
        activation="relu"
    )(x_pad)
    timbre_2 = frontend_bn_mp2d_block(timbre_2)

    tempo_1 = Conv2D(
        filters=51,
        kernel_size=(32, 1),
        padding="same",
        activation="relu"
    )(x_in)
    tempo_1 = frontend_bn_mp2d_block(tempo_1)

    tempo_2 = Conv2D(
        filters=51,
        kernel_size=(64, 1),
        padding="same",
        activation="relu"
    )(x_in)
    tempo_2 = frontend_bn_mp2d_block(tempo_2)

    tempo_3 = Conv2D(
        filters=51,
        kernel_size=(128, 1),
        padding="same",
        activation="relu"
    )(x_in)
    tempo_3 = frontend_bn_mp2d_block(tempo_3)

    frontend = tf.concat([timbre_1, timbre_2, tempo_3, tempo_2, tempo_1], 2)  # front-end features

    #### mid-end musicnn features
    num_midend_filters = 64

    frontend_in = tf.expand_dims(frontend, 3)
    frontend_pad = tf.pad(frontend_in, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

    # conv 1 midend
    midend_1 = Conv2D(
        filters=num_midend_filters,
        kernel_size=(7, frontend_pad.shape[2]),
        padding="valid",
        activation="relu"
    )(frontend_pad)

    midend_1 = LayerNormalization()(midend_1)
    midend_1 = tf.transpose(midend_1, (0,1,3,2))
    midend_1_pad = tf.pad(midend_1, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

    # conv 2 midend
    midend_2 = Conv2D(
        filters=num_midend_filters,
        kernel_size=(7, midend_1_pad.shape[2]),
        padding="valid",
        activation="relu"
    )(midend_1_pad)

    midend_2 = LayerNormalization()(midend_2)
    midend_2 = tf.transpose(midend_2, (0, 1, 3, 2))

    res_midend_2 = tf.add(midend_1, midend_2)

    res_midend_2_pad = tf.pad(res_midend_2, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

    # conv 3 midend
    midend_3 = Conv2D(
        filters=num_midend_filters,
        kernel_size=(7, res_midend_2_pad.shape[2]),
        padding="valid",
        activation="relu"
    )(res_midend_2_pad)

    midend_3 = LayerNormalization()(midend_3)
    midend_3 = tf.transpose(midend_3, (0, 1, 3, 2))

    res_midend_3 = tf.add(midend_3, res_midend_2)
    midend = tf.concat([frontend_in, midend_1, res_midend_2, res_midend_3], 2)  # mid-end features

    #### back-end musicnn

    # midend = tf.squeeze(midend, [-1])

    backend_1_mxp = MaxPool2D(
        pool_size=(midend.shape[1], 1),
        strides=(midend.shape[1], 1)
    )(midend)

    backend_1_avgp = AveragePooling2D(
        pool_size=(midend.shape[1], 1),
        strides=(midend.shape[1], 1)
    )(midend)

    backend_1_mxp = tf.squeeze(backend_1_mxp, -1)
    backend_1_avgp = tf.squeeze(backend_1_avgp, -1)
    backend_gp = tf.concat([backend_1_mxp, backend_1_avgp], 2)

    backend_fl = Flatten()(backend_gp)
    backend_fl = LayerNormalization()(backend_fl)
    flat_pool_dropout = Dropout(rate=0.3)(backend_fl)

    ## dense embedding layer
    embed = Dense(units=512,
                  activation="relu")(flat_pool_dropout)
    embed = LayerNormalization()(embed)

    model = tf.keras.models.Model(x_in, embed)

    return model


def get_feature_idx(feature_shape, augment):
    if augment:
        rand_idx = random.randint(0, feature_shape[0] - N_FRAMES)
    else:
        rand_idx = int(feature_shape[0] * 0.5) - int(N_FRAMES * 0.5)
    return rand_idx


def data_generator(tracks, num_steps, shuffle=True, augment=True):
    if shuffle:
        random.shuffle(tracks)
    while True:
        for step in range(num_steps):
            X = []
            y = []
            batch_tracks = tracks[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            for t in batch_tracks:
                y.append(tf.keras.utils.to_categorical(t['label'], len(GENRES)))
                feat = pickle.load(open(t['path'], 'rb'))
                rand_idx = get_feature_idx(feat.shape, augment)
                feat = feat[rand_idx:rand_idx + N_FRAMES, :]
                X.append(feat)
            X = np.asarray(X)
            y = np.asarray(y)
            yield X, y


def data_generator_da(tracks, trg_tracks, num_steps, shuffle=True, augment=True):
    if shuffle:
        random.shuffle(tracks)
    while True:
        for step in range(num_steps):
            X = []
            X_pos = []
            X_neg = []
            y = []
            batch_tracks = tracks[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            for t in batch_tracks:
                y.append(tf.keras.utils.to_categorical(t['label'], len(GENRES)))
                feat = pickle.load(open(t['path'], 'rb'))
                rand_idx = get_feature_idx(feat.shape, augment)
                feat = feat[rand_idx:rand_idx + N_FRAMES, :]
                X.append(feat)
                pos = random.choice([x for x in trg_tracks if x['label'] == t['label']])
                feat_pos = pickle.load(open(pos['path'], 'rb'))
                if augment:
                    rand_idx = random.randint(0, feat_pos.shape[0] - N_FRAMES)
                else:
                    rand_idx = int(feat_pos.shape[0] * 0.5) - int(N_FRAMES * 0.5)
                feat_pos = feat_pos[rand_idx:rand_idx + N_FRAMES, :]
                X_pos.append(feat_pos)
                neg = random.choice([x for x in trg_tracks if x['label'] != t['label']])
                feat_neg = pickle.load(open(neg['path'], 'rb'))
                rand_idx = get_feature_idx(feat.shape, augment)
                feat_neg = feat_neg[rand_idx:rand_idx + N_FRAMES, :]
                X_neg.append(feat_neg)
            X = np.asarray(X)
            X_pos = np.asarray(X_pos)
            X_neg = np.asarray(X_neg)
            y_classifier = np.asarray(y)
            y_contrastive = np.zeros((y_classifier.shape[0] * 2,))
            y_contrastive[:y_classifier.shape[0]] = 1
            targets = {'classifier': y_classifier,
                       'contrastive': y_contrastive}
            yield [X, X_pos, X_neg], targets


def get_simple_model():
    src_input = Input(shape=(N_FRAMES, N_MELS), name='src_input')
    embedding_net = musicnn()
    embedding_out = embedding_net(src_input)
    classifier = Dense(len(GENRES), activation="softmax", name='classifier_output')(embedding_out)
    model = tf.keras.models.Model(inputs=src_input,
                                  outputs=classifier)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def get_tl_model(src_model_path):
    embedding_net = tf.keras.models.load_model(src_model_path)
    embedding_net = embedding_net.layers[1]
    embedding_net.trainable = False
    classifier = Dense(len(GENRES), activation="softmax", name='classifier_output')(embedding_net.output)
    model = tf.keras.models.Model(inputs=embedding_net.input,
                                  outputs=classifier)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def get_ft_model(src_model_path):
    model = tf.keras.models.load_model(src_model_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR * 0.1),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.summary()
    return model


def get_da_model():
    src_input = Input(shape=(N_FRAMES, N_MELS), name='src_input')
    pos_input = Input(shape=(N_FRAMES, N_MELS), name='pos_input')
    neg_input = Input(shape=(N_FRAMES, N_MELS), name='neg_input')

    embedding_net = musicnn()

    src_embed = embedding_net(src_input)
    classifier_output = Dense(len(GENRES), activation="softmax", name='classifier_output')(src_embed)
    pos_embed = embedding_net(pos_input)
    neg_embed = embedding_net(neg_input)

    contrastive_pos_output = tf.linalg.norm(src_embed - pos_embed, axis=1)
    contrastive_neg_output = tf.linalg.norm(src_embed - neg_embed, axis=1)

    contrastive_output = Concatenate()([contrastive_pos_output, contrastive_neg_output])

    model = tf.keras.models.Model(inputs=[src_input, pos_input, neg_input],
                                  outputs=({'classifier': classifier_output,
                                            'contrastive': contrastive_output}))

    c_loss = tfa.losses.ContrastiveLoss(margin=2)

    losses = {
        "classifier": "categorical_crossentropy",
        "contrastive": c_loss,
    }

    loss_weights = {
        "classifier": 0.3,
        "contrastive": 0.7,
    }
    # Compile the model with categorical cross-entropy for classification
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics={"classifier": "categorical_accuracy"})

    return model

