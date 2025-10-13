"""
utils.py

This module includes miscellaneous utility functions that support the project,
such as data_preprocessing, logging, file handling, and general-purpose helpers.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import os
from torch.utils.tensorboard import SummaryWriter
import time
import pickle
import librosa
import librosa.feature
import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment
import warnings


def setup(params):
    """
    Sets up the environment for training by creating directories for model checkpoints
    and logging, saving configuration parameters, and initializing a tensorboard summary writer.
    Args:
        params (dict): Dictionary containing the configuration parameters.
    Returns:
        tuple: A tuple containing the path to the checkpoints folder, output folder and the tensorboard summary writer instance.
    """
    # create dir to save model checkpoints
    reference = f"{params['net_type']}_{params['modality']}_{'multiACCDOA' if params['multiACCDOA'] else 'singleACCDOA'}{time.strftime('_%Y%m%d_%H%M%S')}"
    checkpoints_dir = os.path.join(params['checkpoints_dir'], reference)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # save the all the config/hyperparams to a pickle file
    pickle_filepath = os.path.join(str(checkpoints_dir), 'config.pkl')
    pickle_file = open(pickle_filepath, 'wb')
    pickle.dump(params, pickle_file)

    # create a tensorboard summary writer for logging and visualization
    log_dir = os.path.join(params['log_dir'], reference)
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=str(log_dir))

    # create output folder to save the predictions
    output_dir = os.path.join(params['output_dir'], reference)
    os.makedirs(output_dir, exist_ok=True)

    return checkpoints_dir, output_dir, summary_writer


def load_audio(audio_file, sampling_rate):
    """
    Loads an audio file.
    Args:
        audio_file (str): Path to the audio file.
        sampling_rate (int): Target sampling rate
    Returns:
        tuple: (audio_data, sample_rate)
    """
    audio_data, sr = librosa.load(path=audio_file, sr=sampling_rate, mono=False)
    return audio_data, sr


def extract_stereo_features(audio, sr: int = 24000, n_fft: int = 512, hop_length: int = 300, win_length: int = 512, 
                            nb_mels: int = 64, max_freq: int = 2000, use_gamma: bool = False, use_ipd: bool = False, 
                            use_iv: bool = False, use_slite: bool = False, use_ms: bool = False):
    """
    Extract stereo audio features. Mel spectrograms are always extracted.
    Optional interaural features include ILDs, coherence (gamma), and IPDs.

    Args:
        audio (np.ndarray): Stereo audio signal with shape (2, n_samples).
        sr (int): Sampling rate. Default is 24000.
        n_fft (int): Number of FFT points for STFT. Default is 512.
        hop_length (int): Hop length between successive STFT frames. Default is 300.
        win_length (int): Window length for STFT. Default is 512.
        nb_mels (int): Number of Mel bands to generate. If None, the linear frequency bins are retained. Default is 64.
        max_freq (int): Frequency threshold (in Hz) above which IPD values are set to zero. If None, all frequencies are used. Default is 2000.
        use_gamma (bool): Whether to compute the magnitude-squared coherence (MSC) feature.
        use_ipd (bool): Whether to compute interaural phase difference (IPD) features.
        use_iv (bool): Whether to compute Mid-Side Intensity Vectors (IVs)
        use_slite (bool): Whether to compute the stereo version of SALSA-Lite
        use_ms (bool): Whether to compute Mid-Side (MS) spectrograms.

    Returns:
        np.ndarray: (n_feature_channels, time_bins, freq_bins)
    """

    # Validate that the audio is stereo
    if audio.shape[0] != 2:
        raise ValueError("Input audio must have two channels (shape: (2, n_samples)).")

    if nb_mels == 0:
        W = np.zeros((200, 257), dtype=np.float32)
        for i in np.arange(192):
            W[i, i+1] = 1.0
        for i in np.arange(192, 200):
            if i < 199:
                W[i, 193 + (i - 192) * 8: 193 + (i - 192) * 8 + 8] = 1/8
            elif i == 199:
                W[i, 193 + (i - 192) * 8: 193 + (i - 192) * 8 + 7] = 1/8

    # Compute the STFT for each channel (each output has shape (F, T))
    stfts = [
        librosa.stft(y=np.asfortranarray(channel), n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window='hann', pad_mode='reflect')
        for channel in audio
    ]

    stfts = np.stack(stfts, axis=0)         # (2, F, T)
    stfts = np.transpose(stfts, (0, 2, 1))  # (2, T, F)

    # Compute the mag-squared spectrogram 
    power_spec = np.abs(stfts) ** 2         # (2, T, F)
    mel_specs = []

    # Compute the log-Mel spectrogram for each channel
    for channel_power in power_spec:
        # librosa expects a spectrogram with shape (F, T)
        pspec = channel_power.T
        if nb_mels:
            mel_spec = librosa.feature.melspectrogram(S=pspec, sr=sr, n_mels=nb_mels, fmin=50)
            log_mel = librosa.power_to_db(mel_spec)
            log_mel = log_mel.T     # (T, nb_mels)
        else:
            log_mel = librosa.power_to_db(pspec)
            log_mel = log_mel.T     # (T, F)
        mel_specs.append(log_mel)
    mel_specs = np.stack(mel_specs, axis=0)  # (2, T, nb_mels or F)

    # Compute Mid-Side spectrograms
    if use_ms:
        left, right = audio[0], audio[1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        ms_stft = [librosa.stft(y=np.asfortranarray(mid), n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window='hann', pad_mode='reflect'),
                   librosa.stft(y=np.asfortranarray(side), n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window='hann', pad_mode='reflect')]

        ms_stft = np.stack(ms_stft, axis=0)         # (2, F, T)
        ms_stft = np.transpose(ms_stft, (0, 2, 1))  # (2, T, F)

        # Compute the mag-squared spectrogram
        ms_power = np.abs(ms_stft) ** 2             # (2, T, F)
        ms_specs = []

        # Compute the log-Mel spectrogram for each channel
        for channel_power in ms_power:
            # librosa expects a spectrogram with shape (F, T)
            pspec = channel_power.T
            if nb_mels:
                mel_spec = librosa.feature.melspectrogram(S=pspec, sr=sr, n_mels=nb_mels, fmin=50)
                log_mel = librosa.power_to_db(mel_spec)
                log_mel = log_mel.T     # (T, nb_mels)
            else:
                log_mel = librosa.power_to_db(pspec)
                log_mel = log_mel.T     # (T, F)
            ms_specs.append(log_mel)
        ms_specs = np.stack(ms_specs, axis=0)
        mel_specs = np.concatenate((mel_specs, ms_specs), axis=0)

    # Compute normalized Mid-Side IVs
    if use_iv:
        i_norm = compute_msiv(audio_data=audio, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, nb_mels=nb_mels)
        mel_specs = np.concatenate((mel_specs, np.expand_dims(i_norm, axis=0)), axis=0)

    # Compute coherence between left and right channels, denoted as gamma
    if use_gamma:
        msc = compute_binaural_msc_recursive(stfts=stfts, sr=sr, n_fft=n_fft, nb_mels=nb_mels)
        msc = np.expand_dims(msc, axis=0)
        mel_specs = np.concatenate((mel_specs, msc), axis=0)

    # Compute inter-channel phase differences between left and right channels
    if use_ipd:
        ipds = compute_ipd_features(stfts, sr=sr, n_fft=n_fft, nb_mels=nb_mels, max_freq=max_freq)
        mel_specs = np.concatenate((mel_specs, ipds), axis=0)

    # Compute SALSA-Lite (Normalized Interchannel Phase Differences)
    if use_slite:
        salsalite = compute_salsa_slite(stfts, sr=sr, n_fft=n_fft, nb_mels=nb_mels, fmax=max_freq)
        mel_specs = np.concatenate((mel_specs, salsalite), axis=0)

    if nb_mels == 0:
        mel_specs = np.dot(mel_specs, W.T)

    return mel_specs



def compute_msiv(audio_data, sr: int = 24000, n_fft: int = 512, win_length: int = 512,
                 hop_length: int = 300, nb_mels: int = 64, fmin: int = 50):
    """
    Extract Mid-Side Intensity Vector. 

    Args:
        audio_data (np.ndarray): Stereo audio signal with shape (2, n_samples).
        sr (int): Sampling rate. Default is 24000.
        n_fft (int): Number of FFT points for STFT. Default is 512.
        hop_length (int): Hop length between successive STFT frames. Default is 300.
        win_length (int): Window length for STFT. Default is 512.
        nb_mels (int): Number of Mel bands to generate. If None, the linear frequency bins are retained. Default is 64.
        fmin (int): Minimum frequency (in Hz) for log-Mel spectrogram computation. Default is 50Hz.

    Returns:
        np.ndarray: (time_bins, freq_bins)
    """
    # varepsilon
    eps = 1e-8

    # extract Mid-Side STFTs
    mid = (audio_data[0] + audio_data[1]) / 2
    side = (audio_data[0] - audio_data[1]) / 2
    M_stft = librosa.stft(mid, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True)
    S_stft = librosa.stft(side, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True)

    # cross‑spectrum active intensity
    I_num = np.real(M_stft * np.conj(S_stft))   # (F, T)

    # energy denom
    E = eps + np.abs(M_stft)**2 + np.abs(S_stft)**2

    # normalized intensity
    I_norm = I_num / E      # (F, T)

    if nb_mels:
        mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=nb_mels, fmin=fmin)  # (n_mels, freq_bins)
        I_norm = mel_fb.dot(I_norm)   # (n_mels, T)

    return I_norm.T  # (T, F)



def compute_binaural_msc_recursive(stfts, lambda_: float = 0.8, sr: int = 24000, n_fft: int = 512, nb_mels: int = 64):
    """
    Compute an energy-weighted Magnitude-Squared Coherence (MSC) spectrogram
    for a binaural (stereo) audio signal using time-recursive averaging. 

    The MSC between the left and right channels is computed as:
        ``MSC(f) = |S_12(f)|^2 / (S_11(f) * S_22(f))``
    where S_11 and S_22 are the auto-power spectral densities of channels 1 and 2,
    and S_12 is the cross-power spectral density.

    Parameters:
        stfts (np.ndarray): STFTs of the two-channel stereo signal, of shape (2, time_bins, frequency_bins)
        lambda_ (float): Forgetting factor for recursive averaging (0 < lambda_ < 1). Default is 0.8. Lower
                         values allow for faster adaptation (noiser), higher values give slower (smoother).
        sr (int): Sampling rate in Hz. Default is 24000.
        n_fft (int): Number of FFT points for the STFT. Default is 512.
        nb_mels (int): If provided, convert the MSC from linear frequency scale to 
                       mel scale using this many mel bands. Default is 64.

    Returns:
        gamma (ndarray): Coherence spectrogram. Shape is (T, F) for linear scale or (T, nb_mels) for mel scale.

    Raises:
        ValueError: If lambda_ is not in the interval (0, 1).
    """
    # Validate lambda
    if not (0 < lambda_ < 1):
        raise ValueError(f"Parameter lambda_ must be in the interval (0, 1), but got lambda_={lambda_}.")

    epsilon = 1e-8    # Small constant to prevent division by zero
    X = stfts   # (2, T, F)
    T, F = X.shape[1], X.shape[2]

    # Initialize recursive estimates for auto- and cross-power spectral densities
    S_11 = np.zeros((F,), dtype=np.complex64)
    S_22 = np.zeros((F,), dtype=np.complex64)
    S_12 = np.zeros((F,), dtype=np.complex64)
    gamma = np.zeros((T, F), dtype=np.float32)

    # Loop over time frames to update the recursive estimates
    for t in range(T):
        # Extract current frame for both channels (shape: (2, F))
        X_l = X[0, t, :]
        X_r = X[1, t, :]

        # Update auto-power spectral densities for channels 1 and 2
        S_11 = lambda_ * S_11 + (1 - lambda_) * (X_l * np.conjugate(X_l))
        S_22 = lambda_ * S_22 + (1 - lambda_) * (X_r * np.conjugate(X_r))
        # Update cross-power spectral density
        S_12 = lambda_ * S_12 + (1 - lambda_) * (X_l * np.conjugate(X_r))

        # Compute MSC for the current frame: |S_12|^2 / (S_11 * S_22)
        numerator = np.abs(S_12)**2
        denominator = (S_11 * S_22).real + epsilon
        gamma[t, :] = numerator / denominator

    # Optionally convert MSC from linear frequency bins to mel bins
    if nb_mels:
        mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=nb_mels, fmin=50) # (nb_mels, F)
        gamma = gamma.dot(mel_filter_bank.T)

    return gamma



def compute_salsa_slite(stfts, sr: int = 24000, n_fft: int = 512, nb_mels: int = 64, fmax: int = 2000):
    """
    Compute the stereo version of SALSA-Lite (Single channel normalized IPD)

    Parameters:
        stfts (np.ndarray): STFTs of the two-channel stereo signal, of shape (2, time_bins, frequency_bins)
        sr (int): Sampling rate in Hz. Default is 24000.
        n_fft (int): Number of FFT points for the STFT. Default is 512.
        nb_mels (int): If provided, convert the MSC from linear frequency scale to mel scale using this many mel bands. Default is 64.
        fmax (int): Upper frequency (in Hz) to zero-out, determined by the spatial aliasing frequency. Defaults to 2000.

    Returns:
        nipd (ndarray): Normalized inter-channel phase-differences. Shape is (T, F) for linear scale or (T, nb_mels) for mel scale.
    """

    X = stfts.T  # n_bins, n_frames, n_mics

    # Normalization factor for SALSA-Lite
    c = 343 # speed of sound
    delta = 2 * np.pi * sr / (n_fft * c)
    n_bins = n_fft // 2 + 1
    freq_vector = np.arange(n_bins)
    freq_vector[0] = 1
    freq_vector = freq_vector[:, None, None] # n_bins x 1 x 1

    phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None])) # F, T, 1
    phase_vector = phase_vector / (delta * freq_vector)
    phase_vector = phase_vector.T

    # Spatial Aliasing
    if fmax is not None:
        upper_bin = int(np.floor(fmax * n_fft / float(sr)))
        phase_vector[:, :, upper_bin:] = 0

    # Optionally from linear frequency bins to mel bins
    if nb_mels:
        mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=nb_mels, fmin=50) # nb_mels, F
        phase_vector = np.dot(phase_vector, mel_filter_bank.T) # 1, T, nb_mels

    return phase_vector



def compute_ipd_features(stfts, sr: int = 24000, n_fft: int = 512, nb_mels: int = 64, max_freq: int = 2000):
    """
    Compute the interchannel phase differences (IPDs) for a binaural (stereo) audio signal 
    and represent them via their sine and cosine.

    For a stereo signal with channels [0, 1], the IPD is defined as:
        ``IPD(f) = angle(left(f)) - angle(right(f))``
    where 'left' and 'right' are the STFT representations of the two channels.

    Parameters:
        stfts (ndarray): STFT of the binaural signal with shape (2, T, F).
        sr (int): Sampling rate in Hz. Default is 24000.
        n_fft (int): Number of FFT points for the STFT. Default is 512.
        nb_mels (int): If provided, convert the MSC from linear frequency scale to mel scale using this many mel bands. Default is 64.
        fmax (int): Upper frequency (in Hz) to zero-out, determined by the spatial aliasing frequency. Defaults to 2000.

    Returns:
        ipd_features (ndarray): Sine and Cosine of the IPD, shape (2, T, F).
    """
    stft_left = stfts[0]
    stft_right = stfts[1]

    # If max_freq is specified, zero out frequency bins above this threshold.
    if max_freq is not None:
        # Get the frequencies corresponding to the STFT bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        # Create a boolean mask for bins above max_freq
        mask = freqs > max_freq
        # Set the corresponding columns to zero in both representations
        stft_left[:, mask] = 0
        stft_right[:, mask] = 0

    if nb_mels:
        mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=nb_mels, fmin=50) # (nb_mels, frequency_bins)

        stft_left = np.dot(stft_left, mel_filter_bank.T)
        stft_right = np.dot(stft_right, mel_filter_bank.T)

    # Compute the IPD
    ipd = np.angle(stft_left) - np.angle(stft_right)

    # Compute sine and cosine of IPD
    ipd_sin = np.sin(ipd)
    ipd_cos = np.cos(ipd)

    return np.stack((ipd_sin, ipd_cos), axis=0)     # (2, T, F)



def load_labels(label_file, convert_to_cartesian=True):
    label_data = {}
    with open(label_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
        for line in lines:
            values = line.strip().split(',')
            frame_idx = int(values[0])
            data_row = [int(values[1]), int(values[2]), float(values[3]), float(values[4]), int(values[5])]
            if frame_idx not in label_data:
                label_data[frame_idx] = []
            label_data[frame_idx].append(data_row)

    if convert_to_cartesian:
        label_data = convert_polar_to_cartesian(label_data)
    return label_data


def process_labels(_desc_file, _nb_label_frames, _nb_unique_classes, params=None):

    se_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    x_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    y_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    dist_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    onscreen_label = torch.zeros((_nb_label_frames, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            for active_event in active_event_list:
                se_label[frame_ind, active_event[0]] = 1
                x_label[frame_ind, active_event[0]] = active_event[2]
                y_label[frame_ind, active_event[0]] = active_event[3]
                dist_value = active_event[4] / 100.
                if params['dnorm']:
                    dist_label[frame_ind, active_event[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                else:
                    dist_label[frame_ind, active_event[0]] = dist_value
                onscreen_label[frame_ind, active_event[0]] = active_event[5]

    label_mat = torch.cat((se_label, x_label, y_label, dist_label, onscreen_label), dim=1)
    return label_mat


def process_labels_adpit(_desc_file, _nb_label_frames, _nb_unique_classes, params=None):

    se_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))  # 50, 6, 13
    x_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    y_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    dist_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    onscreen_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
            active_event_list_per_class = []
            for i, active_event in enumerate(active_event_list):
                active_event_list_per_class.append(active_event)
                if i == len(active_event_list) - 1:  # if the last
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        dist_value = active_event_a0[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 0, active_event_a0[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 0, active_event_a0[0]] = dist_value
                        onscreen_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        dist_value = active_event_b0[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 1, active_event_b0[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 1, active_event_b0[0]] = dist_value
                        onscreen_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        dist_value = active_event_b1[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 2, active_event_b1[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 2, active_event_b1[0]] = dist_value
                        onscreen_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]

                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        dist_value = active_event_c0[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 3, active_event_c0[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 3, active_event_c0[0]] = dist_value
                        onscreen_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]
                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        dist_value = active_event_c1[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 4, active_event_c1[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 4, active_event_c1[0]] = dist_value
                        onscreen_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        dist_value = active_event_c2[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 5, active_event_c2[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 5, active_event_c2[0]] = dist_value
                        onscreen_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]

                elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        dist_value = active_event_a0[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 0, active_event_a0[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 0, active_event_a0[0]] = dist_value
                        onscreen_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        dist_value = active_event_b0[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 1, active_event_b0[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 1, active_event_b0[0]] = dist_value
                        onscreen_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        dist_value = active_event_b1[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 2, active_event_b1[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 2, active_event_b1[0]] = dist_value
                        onscreen_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]
                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        dist_value = active_event_c0[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 3, active_event_c0[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 3, active_event_c0[0]] = dist_value
                        onscreen_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]
                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        dist_value = active_event_c1[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 4, active_event_c1[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 4, active_event_c1[0]] = dist_value
                        onscreen_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        dist_value = active_event_c2[4] / 100.
                        if params['dnorm']:
                            dist_label[frame_ind, 5, active_event_c2[0]] = (dist_value - params['d_mean'])/(params['d_std'] * params['d_max'])
                        else:
                            dist_label[frame_ind, 5, active_event_c2[0]] = dist_value
                        onscreen_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]
                    active_event_list_per_class = []

    label_mat = torch.stack((se_label, x_label, y_label, dist_label, onscreen_label), dim=2)  # [nb_frames, 6, 5(act+XY+dist+onscreen), max_classes]
    return label_mat


def organize_labels(input_dict, max_frames, max_tracks=10):
    """
    :param input_dict: Dictionary containing frame-wise sound event time and location information
            _pred_dict[frame-index] = [[class-index, source-index, azimuth, distance, onscreen] x events in frame]
    :param max_frames: Total number of frames in the recording
    :param max_tracks: Total number of tracks in the output dict
    :return: Dictionary containing class-wise sound event location information in each frame
            dictionary_name[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
    """
    tracks = set(range(max_tracks))
    output_dict = {x: {} for x in range(max_frames)}
    for frame_idx in range(0, max_frames):
        if frame_idx not in input_dict:
            continue
        for [class_idx, source_idx, az, dist, onscreen] in input_dict[frame_idx]:
            if class_idx not in output_dict[frame_idx]:
                output_dict[frame_idx][class_idx] = {}
            if source_idx not in output_dict[frame_idx][class_idx] and source_idx < max_tracks:
                track_idx = source_idx  # If possible, use source_idx as track_idx
            else:                       # If not, use the first one available
                try:
                    track_idx = list(set(tracks) - output_dict[frame_idx][class_idx].keys())[0]
                except IndexError:
                    warnings.warn("The number of sources of is higher than the number of tracks. "
                                  "Some events will be missed.")
                    track_idx = 0  # Overwrite one event
            output_dict[frame_idx][class_idx][track_idx] = [az, dist, onscreen]

    return output_dict


def convert_polar_to_cartesian(input_dict):
    output_dict = {}
    for frame_idx in input_dict.keys():
        if frame_idx not in output_dict:
            output_dict[frame_idx] = []
        for tmp_val in input_dict[frame_idx]:
            azi_rad = tmp_val[2]*np.pi/180
            x = np.cos(azi_rad)
            y = np.sin(azi_rad)
            output_dict[frame_idx].append(tmp_val[0:2] + [x, y] + tmp_val[3:])
    return output_dict


def convert_cartesian_to_polar(input_dict):
    output_dict = {}
    for frame_idx in input_dict.keys():
        if frame_idx not in output_dict:
            output_dict[frame_idx] = []
        for tmp_val in input_dict[frame_idx]:
            x = tmp_val[2]
            y = tmp_val[3]
            azi_rad = np.arctan2(y, x)
            azimuth = azi_rad * 180 / np.pi
            output_dict[frame_idx].append(tmp_val[0:2] + [azimuth] + tmp_val[4:])
    return output_dict


def get_accdoa_labels(logits, nb_classes, modality, params=None):
    x, y = logits[:, :, :nb_classes], logits[:, :, nb_classes:2 * nb_classes]
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, -1, 1)
    sed = torch.sqrt(x ** 2 + y ** 2) > 0.5
    distance = logits[:, :, 2 * nb_classes: 3 * nb_classes]
    if params['dnorm']:
        distance = (distance * params['d_max'] * params['d_std']) + params['d_mean']
    else:
        distance = torch.clamp(distance, min=0)

    if modality == 'audio_visual':
        on_screen = logits[:, :, 3 * nb_classes: 4 * nb_classes]
    else:
        on_screen = torch.zeros_like(distance)  # don't care for audio modality
    dummy_src_id = torch.zeros_like(distance)
    return sed, dummy_src_id, x, y, distance, on_screen


def get_multiaccdoa_labels(logits, nb_classes, modality=None, dnorm=False, params=None):
    # -------- First event slot --------
    # x0 is expected to be in [0,1] and y0 in [-1,1]
    x0, y0 = logits[:, :, :1*nb_classes], logits[:, :, 1*nb_classes:2*nb_classes]
    x0 = torch.clamp(x0, 0, 1)
    y0 = torch.clamp(y0, -1, 1)
    sed0 = torch.sqrt(x0**2 + y0**2) > 0.5

    dist0 = logits[:, :, 2*nb_classes:3*nb_classes]
    if dnorm:
        dist0 = (dist0 * params['d_max'] * params['d_std']) + params['d_mean']
    else:
        dist0 = torch.clamp(dist0, min=0)

    doa0 = logits[:, :, :2*nb_classes]
    dummy_src_id0 = torch.zeros_like(dist0)
    on_screen0 = torch.zeros_like(dist0)

    x1, y1 = logits[:, :, 3*nb_classes:4 * nb_classes], logits[:, :, 4 * nb_classes: 5 * nb_classes]
    x1 = torch.clamp(x1, 0, 1)
    y1 = torch.clamp(y1, -1, 1)
    sed1 = torch.sqrt(x1 ** 2 + y1 ** 2) > 0.5

    dist1 = logits[:, :, 5 * nb_classes:6 * nb_classes]
    if dnorm:
        dist1 = (dist1 * params['d_max'] * params['d_std']) + params['d_mean']
    else:
        dist1 = torch.clamp(dist1, min=0)

    doa1 = logits[:, :, 3*nb_classes:5 * nb_classes]
    dummy_src_id1 = torch.zeros_like(dist1)
    on_screen1 = torch.zeros_like(dist1)

    x2, y2 = logits[:, :, 6*nb_classes:7 * nb_classes], logits[:, :, 7 * nb_classes:8 * nb_classes]
    x2 = torch.clamp(x2, 0, 1)
    y2 = torch.clamp(y2, -1, 1)
    sed2 = torch.sqrt(x2 ** 2 + y2 ** 2) > 0.5

    dist2 = logits[:, :, 8 * nb_classes:9 * nb_classes]
    if dnorm:
        dist2 = (dist2 * params['d_max'] * params['d_std']) + params['d_mean']
    else:
        dist2 = torch.clamp(dist2, min=0)

    doa2 = logits[:, :, 6*nb_classes:8 * nb_classes]
    dummy_src_id2 = torch.zeros_like(dist2)
    on_screen2 = torch.zeros_like(dist2)

    return sed0, dummy_src_id0, doa0, dist0, on_screen0, sed1, dummy_src_id1, doa1, dist1, on_screen1, sed2, dummy_src_id2, doa2, dist2, on_screen2



def get_output_dict_format_single_accdoa(sed, src_id, x, y, dist, onscreen, convert_to_polar=True):
    output_dict = {}
    for frame_cnt in range(sed.shape[0]):
        for class_cnt in range(sed.shape[1]):
            if sed[frame_cnt][class_cnt] > 0.5:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                output_dict[frame_cnt].append([class_cnt, src_id[frame_cnt][class_cnt], x[frame_cnt][class_cnt], y[frame_cnt][class_cnt], dist[frame_cnt][class_cnt], onscreen[frame_cnt][class_cnt]])

    if convert_to_polar:
        output_dict = convert_cartesian_to_polar(output_dict)
    return output_dict


def distance_between_cartesian_coordinates(x1, y1, x2, y2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + 1e-10)
    x1, y1, x2, y2 = x1/N1, y1/N1, x2/N2, y2/N2

    # Compute the distance
    dist = x1*x2 + y1*y2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def fold_az_angle(az):
    """
    Project azimuth angle into the range [-90, 90]

    :param az: azimuth angle in degrees
    :return: folded angle in degrees
    """
    # Fold az angles
    az = (az + 180) % 360 - 180  # Make sure az is in the range [-180, 180)
    az_fold = az.copy()
    az_fold[np.logical_and(-180 <= az, az < -90)] = -180 - az[np.logical_and(-180 <= az, az < -90)]
    az_fold[np.logical_and(90 < az, az <= 180)] = 180 - az[np.logical_and(90 < az, az <= 180)]
    return az_fold


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def get_output_dict_format_multi_accdoa(sed0, dummy_src_id0, doa0, dist0, on_screen0, sed1, dummy_src_id1, doa1, dist1, on_screen1, sed2, dummy_src_id2, doa2, dist2, on_screen2, thresh_unify, nb_classes, convert_to_polar=True):
    output_dict = {}
    for frame_cnt in range(sed0.shape[0]):
        for class_cnt in range(sed0.shape[1]):
            flag_0sim1 = determine_similar_location(sed0[frame_cnt][class_cnt], sed1[frame_cnt][class_cnt], doa0[frame_cnt], doa1[frame_cnt], class_cnt, thresh_unify, nb_classes)
            flag_1sim2 = determine_similar_location(sed1[frame_cnt][class_cnt], sed2[frame_cnt][class_cnt], doa1[frame_cnt], doa2[frame_cnt], class_cnt, thresh_unify, nb_classes)
            flag_2sim0 = determine_similar_location(sed2[frame_cnt][class_cnt], sed0[frame_cnt][class_cnt], doa2[frame_cnt], doa0[frame_cnt], class_cnt, thresh_unify, nb_classes)

            # unify or not unify according to flag
            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                if sed0[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id0[frame_cnt][class_cnt],
                                                   doa0[frame_cnt][class_cnt],
                                                   doa0[frame_cnt][class_cnt + nb_classes],
                                                   dist0[frame_cnt][class_cnt],
                                                   on_screen0[frame_cnt][class_cnt]])

                if sed1[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id1[frame_cnt][class_cnt],
                                                   doa1[frame_cnt][class_cnt],
                                                   doa1[frame_cnt][class_cnt + nb_classes],
                                                   dist1[frame_cnt][class_cnt],
                                                   on_screen1[frame_cnt][class_cnt]])

                if sed2[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id2[frame_cnt][class_cnt],
                                                   doa2[frame_cnt][class_cnt],
                                                   doa2[frame_cnt][class_cnt + nb_classes],
                                                   dist2[frame_cnt][class_cnt],
                                                   on_screen2[frame_cnt][class_cnt]])

            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                if flag_0sim1:
                    if sed2[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id2[frame_cnt][class_cnt],
                                                       doa2[frame_cnt][class_cnt],
                                                       doa2[frame_cnt][class_cnt + nb_classes],
                                                       dist2[frame_cnt][class_cnt],
                                                       on_screen2[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa0[frame_cnt] + doa1[frame_cnt]) / 2
                    dist_pred_fc = (dist0[frame_cnt] + dist1[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen0[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id0[frame_cnt]
                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nb_classes],dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

                elif flag_1sim2:
                    if sed0[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id0[frame_cnt][class_cnt],
                                                       doa0[frame_cnt][class_cnt],
                                                       doa0[frame_cnt][class_cnt + nb_classes],
                                                       dist0[frame_cnt][class_cnt],
                                                       on_screen0[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa1[frame_cnt] + doa2[frame_cnt]) / 2
                    dist_pred_fc = (dist1[frame_cnt] + dist2[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen1[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id1[frame_cnt]

                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt],
                         doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

                elif flag_2sim0:
                    if sed1[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id1[frame_cnt][class_cnt],
                                                       doa1[frame_cnt][class_cnt],
                                                       doa1[frame_cnt][class_cnt + nb_classes],
                                                       dist1[frame_cnt][class_cnt],
                                                       on_screen1[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa2[frame_cnt] + doa0[frame_cnt]) / 2
                    dist_pred_fc = (dist2[frame_cnt] + dist0[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen2[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id2[frame_cnt]

                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt],
                         doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                doa_pred_fc = (doa0[frame_cnt] + doa1[frame_cnt] + doa2[frame_cnt]) / 3
                dist_pred_fc = (dist0[frame_cnt] + dist1[frame_cnt] + dist2[frame_cnt]) / 3

                dummy_src_id_pred_fc = dummy_src_id0[frame_cnt]
                on_screen_pred_fc = on_screen0[frame_cnt]  # TODO: How to do this?

                output_dict[frame_cnt].append(
                    [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

    if convert_to_polar:
        output_dict = convert_cartesian_to_polar(output_dict)
    return output_dict


def write_to_dcase_output_format(output_dict, output_dir, filename, split, convert_dist_to_cm=True):
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    file_path = os.path.join(output_dir,split, filename)
    with open(file_path, 'w') as f:
        f.write('frame,class,source,azimuth,distance,onscreen\n')
        # Write data
        for frame_ind, values in output_dict.items():
            for value in values:
                azimuth_rounded = round(float(value[2]))
                dist_rounded = round(float(value[3]) * 100) if convert_dist_to_cm else round(float(value[3]))
                f.write(f"{int(frame_ind)},{int(value[0])},{int(value[1])},{azimuth_rounded},{dist_rounded},{int(value[4])}\n")


def write_logits_to_dcase_format(logits, params, output_dir, filelist, split='dev-test'):
    if not params['multiACCDOA']:
        sed, dummy_src_id, x, y, dist, onscreen = get_accdoa_labels(logits, params['nb_classes'], params['modality'], params=params)
        for i in range(sed.size(0)):
            sed_i, dummy_src_id_i, x_i, y_i, dist_i, onscreen_i = sed[i].cpu().numpy(), dummy_src_id[i].cpu().numpy(), x[i].cpu().numpy(), y[i].cpu().numpy(), dist[i].cpu().numpy(), onscreen[i].cpu().numpy()
            output_dict = get_output_dict_format_single_accdoa(sed_i, dummy_src_id_i, x_i, y_i, dist_i, onscreen_i, convert_to_polar=True)
            write_to_dcase_output_format(output_dict, output_dir, os.path.basename(filelist[i])[:-3] + '.csv', split)

    else:
        (sed0, dummy_src_id0, doa0, dist0, on_screen0,
         sed1, dummy_src_id1, doa1, dist1, on_screen1,
         sed2, dummy_src_id2, doa2, dist2, on_screen2) = get_multiaccdoa_labels(logits, params['nb_classes'], params['modality'], dnorm=params['dnorm'], params=params)

        for i in range(sed0.size(0)):
            sed0_i, dummy_src_id0_i, doa0_i, dist0_i, on_screen0_i = sed0[i].cpu().numpy(), dummy_src_id0[i].cpu().numpy(), doa0[i].cpu().numpy(), dist0[i].cpu().numpy(), on_screen0[i].cpu().numpy()
            sed1_i, dummy_src_id1_i, doa1_i, dist1_i, on_screen1_i = sed1[i].cpu().numpy(), dummy_src_id1[i].cpu().numpy(), doa1[i].cpu().numpy(), dist1[i].cpu().numpy(), on_screen1[i].cpu().numpy()
            sed2_i, dummy_src_id2_i, doa2_i, dist2_i, on_screen2_i = sed2[i].cpu().numpy(), dummy_src_id2[i].cpu().numpy(), doa2[i].cpu().numpy(), dist2[i].cpu().numpy(), on_screen2[i].cpu().numpy()

            output_dict = get_output_dict_format_multi_accdoa(sed0_i, dummy_src_id0_i, doa0_i, dist0_i, on_screen0_i,
                                                              sed1_i, dummy_src_id1_i, doa1_i, dist1_i, on_screen1_i,
                                                              sed2_i, dummy_src_id2_i, doa2_i, dist2_i, on_screen2_i, params['thresh_unify'], params['nb_classes'], convert_to_polar=True)
            write_to_dcase_output_format(output_dict, output_dir, os.path.basename(filelist[i])[:-3] + '.csv', split)


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of azimuth angles. Given a set of ground truth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of ground truth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        ground truth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list: list of ground-truth azimuth angles in degrees
        :param pred_list: list of predicted azimuth angles in degrees
        :return: cost - azimuth distance (after folding them to the range [-90, 90])
        :return: row_ind - row indexes obtained from the Hungarian algorithm
        :return: col_ind - column indexes obtained from the Hungarian algorithm
    """
    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        az1, az2 = gt_list[ind_pairs[:, 0]], pred_list[ind_pairs[:, 1]]
        distances_ang = np.abs(fold_az_angle(az1) - fold_az_angle(az2))
        cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_ang

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind


def print_results(f, ang_error, dist_error, rel_dist_error, onscreen_acc, class_wise_scr, params):
    use_jackknife = params['use_jackknife']
    print('\n\n')
    print('F-score: {:0.2f}% {}'.format(
        100 * f[0] if use_jackknife else 100 * f,
        '[{:0.2f}, {:0.2f}]'.format(100 * f[1][0], 100 * f[1][1]) if use_jackknife else ''
    ))

    print('DOA error: {:0.2f} {}'.format(
        ang_error[0] if use_jackknife else ang_error,
        '[{:0.2f}, {:0.2f}]'.format(ang_error[1][0], ang_error[1][1]) if use_jackknife else ''
    ))

    print('Distance error: {:0.2f} {}'.format(
        dist_error[0] if use_jackknife else dist_error,
        '[{:0.2f}, {:0.2f}]'.format(dist_error[1][0], dist_error[1][1]) if use_jackknife else ''
    ))
    print('Relative distance error: {:0.3f} {}'.format(
        rel_dist_error[0] if use_jackknife else rel_dist_error,
        '[{:0.3f}, {:0.3f}]'.format(rel_dist_error[1][0], rel_dist_error[1][1]) if use_jackknife else ''
    ))


    if params['average'] == 'macro':
        print('Class-wise results on unseen data:')
        print('Class\tF-score\tDOA-Error\tDist-Error\tRelDist-Error')

        for cls_cnt in range(params['nb_classes']):
            print('{}\t{:0.3f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.3f} {}'.format(
                cls_cnt,
                class_wise_scr[0][0][cls_cnt] if use_jackknife else class_wise_scr[0][cls_cnt],
                '[{:0.3f}, {:0.3f}]'.format(class_wise_scr[1][0][cls_cnt][0],
                                            class_wise_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                class_wise_scr[0][1][cls_cnt] if use_jackknife else class_wise_scr[1][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][1][cls_cnt][0],
                                            class_wise_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                class_wise_scr[0][2][cls_cnt] if use_jackknife else class_wise_scr[2][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][2][cls_cnt][0],
                                            class_wise_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                class_wise_scr[0][3][cls_cnt] if use_jackknife else class_wise_scr[3][cls_cnt],
                '[{:0.3f}, {:0.3f}]'.format(class_wise_scr[1][3][cls_cnt][0],
                                            class_wise_scr[1][3][cls_cnt][1]) if use_jackknife else ''
            ))


def format_elapsed_time(elapsed: float) -> str:
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    return f"{hours}h {minutes}min {seconds:.2f}s"