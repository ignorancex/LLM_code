"""
extract_features.py

This module defines the SELDFeatureExtractor class, which provides functionality to extract features from both audio
and video data. It also processes the labels to support MultiACCDOA (ADPIT).  It includes the following key components:

Classes:
    SELDFeatureExtractor: A class that supports the extraction of audio and video features. It extracts log Mel
    spectrogram from audio files and ResNet-based features from video frames. It also processes labels for MultiACCDOA.

    Methods:
        - extract_audio_features: Extracts audio features from a specified split of the dataset.
        - extract_video_features: Extracts video features from a specified split of the dataset.
        - extract_features: A high-level function to extract features based on the modality ('audio' or 'audio_visual').
        - extract_labels: converts labels to support multiACCDOA.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import os
import glob
import torch
import utils
import joblib
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress


class SELDFeatureExtractor():
    def __init__(self, params):
        """
        Initializes the SELDFeatureExtractor with the provided parameters.
        Args:
            params (dict): A dictionary containing various parameters for audio/video feature extraction among others.
            use_mms (bool): Whether to average the Left/Right MelSpectrograms into one Mean MelSpectrogram.
            use_gamma (bool): Whether to extract Coherence features.
            use_ipd (bool): Whether to extract Inter-aural Phase Differences.
            use_ild (bool): Whether to extract Inter-aural Level Differences.
            use_iv (bool): Whether to extract the Mid-Side Intensity Vector
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.params = params
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']

        # audio feature extraction
        self.sampling_rate = params['sampling_rate']
        self.hop_length = int(self.sampling_rate * params['hop_length_s'])
        self.n_fft = 2 ** (2 * self.hop_length - 1).bit_length()
        self.win_length = self.n_fft
        self.nb_mels = params['nb_mels']
        if self.nb_mels == 0: # Cutdown to 200 Freq bins for linspec 
            self.n_fft = 512
            self.win_length = 512
        self.max_freq = params['max_freq']
        print("Feature Extraction Params:")
        print(f"\tSampling Rate: {self.sampling_rate}\n\tHop_Length: {self.hop_length}\n\tWin_Length: {self.win_length}\n\tN_FFT: {self.n_fft}\n\tN_Mels: {self.nb_mels}\n\tIPD Max Frequency: {self.max_freq}")

        # label extraction
        self.nb_label_frames    = params['label_sequence_length']
        self.nb_unique_classes  = params['nb_classes']
        print(f"Loading features from: {self.feat_dir}")


    def extract_features(self, split='dev'):
        """
        Extracts features
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """

        os.makedirs(self.feat_dir, exist_ok=True)

        if split == 'dev':
            audio_files = glob.glob(os.path.join(self.root_dir, 'stereo_dev', 'dev-*', '*.wav'))
        elif split == 'eval':
            audio_files = glob.glob(os.path.join(self.root_dir, 'stereo_eval', 'eval', '*.wav'))
        else:
            raise ValueError("Split must be either 'dev' or 'eval'.")

        output_dir = os.path.join(self.feat_dir, f'stereo_{split}')
        os.makedirs(output_dir, exist_ok=True)

        with Progress() as progress:
            task = progress.add_task(f"[cyan]Processing {len(audio_files)} audio files ({split})", total=len(audio_files))
            for audio_file in audio_files:
                filename = os.path.splitext(os.path.basename(audio_file))[0] + '.pt'
                feature_path = os.path.join(output_dir, filename)

                # Skip if the feature file already exists
                if os.path.exists(feature_path):
                    progress.update(task, advance=1)
                    continue

                # Load audio and extract stereo features
                audio, sr = utils.load_audio(audio_file, self.sampling_rate)
                audio_feat = utils.extract_stereo_features(audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, 
                                                           nb_mels=self.nb_mels, max_freq=self.max_freq, use_gamma=self.params['gamma'], use_ipd=self.params['ipd'], 
                                                           use_iv=self.params['iv'], use_slite=self.params['slite'], use_ms=self.params['ms'])

                # Convert to tensor and save
                audio_feat = torch.tensor(audio_feat, dtype=torch.float32)
                torch.save(audio_feat, feature_path)
                progress.update(task, advance=1)


    def extract_labels(self, split):

        os.makedirs(self.feat_dir, exist_ok=True)   # already created by extract_features method

        if split == 'dev':
            label_files = glob.glob(os.path.join(self.root_dir, 'metadata_dev', 'dev-*', '*.csv'))
        elif split == 'eval':  # only for organizers
            label_files = glob.glob(os.path.join(self.root_dir, 'metadata_eval', 'eval', '*.csv'))
        else:
            raise ValueError("Split must be either 'dev' or 'eval'.")

        suffix = '_adpit' if self.params['multiACCDOA'] else ''
        metadata_dir = os.path.join(self.feat_dir, f'metadata_{split}{suffix}')
        os.makedirs(metadata_dir, exist_ok=True)

        with Progress() as progress:
            task = progress.add_task(f"[cyan]Processing {len(label_files)} label files ({split})", total=len(label_files))
            for label_file in label_files:
                filename = os.path.splitext(os.path.basename(label_file))[0] + '.pt'
                label_path = os.path.join(metadata_dir, filename)

                # Skip if the label file already exists.
                if os.path.exists(label_path):
                    progress.update(task, advance=1)
                    continue

                label_data = utils.load_labels(label_file)
                if self.params['multiACCDOA']:
                    processed_labels = utils.process_labels_adpit(label_data, self.nb_label_frames, self.nb_unique_classes, params=self.params)
                else:
                    processed_labels = utils.process_labels(label_data, self.nb_label_frames, self.nb_unique_classes, params=self.params)
                torch.save(processed_labels, label_path)
                progress.update(task, advance=1)


    def preprocess_features(self, split='dev', is_eval=False):
        """
        Preprocesses feature files by normalizing them using sklearn's StandardScaler.
        This function either fits the scaler(s) on the unnormalized features (if is_eval is False)
        or loads precomputed scaler(s) (if is_eval is True), then applies the transformation and
        saves the normalized features.

        Args:
            split (str): The data split, e.g., 'dev' or 'eval'.
            is_eval (bool): If True, the scaler(s) are loaded from file; if False, they are fitted.
        """
        unnorm_dir = os.path.join(self.feat_dir, f"stereo_{split}")
        norm_dir = os.path.join(self.feat_dir, f"stereo_{split}_normalized")
        os.makedirs(norm_dir, exist_ok=True)
        scaler_file = os.path.join(self.feat_dir, f"scaler_{split}.pkl")

        # Per-channel normalization
        spec_scalers = None     # for per-channel 3D features
        n_norm_channels = 4 if self.params['ms'] else 2     # We want to maintain the inter-channel differences

        # If evaluation, load the scaler(s)
        if is_eval or os.path.exists(scaler_file):
            spec_scalers = joblib.load(scaler_file)
            print(f'Loaded scaler(s) from {scaler_file}, skipping fitting scalers!')
        else:
            file_list = os.listdir(unnorm_dir)
            with Progress() as progress:
                task = progress.add_task("[yellow]Fitting scalers...", total=len(file_list))
                for file_cnt, file_name in enumerate(file_list):

                    # Check if we have already normalized the file
                    normalized_feat_path = os.path.join(norm_dir, file_name)
                    if os.path.exists(normalized_feat_path):
                        continue

                    feat_path = os.path.join(unnorm_dir, file_name)
                    feat_tensor = torch.load(feat_path) # Load the feature tensor and convert to NumPy array
                    feat_file = feat_tensor.cpu().numpy()  # Expected shape: (channels, time, frequency)

                    if feat_file.ndim != 3:
                        raise ValueError("Feature file has unsupported dimensions: {}".format(feat_file.shape))

                    # Initialize per-channel scalers if not already done
                    if spec_scalers is None:
                        num_channels = feat_file.shape[0]
                        shared_scaler = StandardScaler()
                        spec_scalers = []
                        for ch in range(num_channels):
                            if ch < n_norm_channels: # Left and Right share the same scaler (and Mid-Side if used)
                                spec_scalers.append(shared_scaler)
                            else:
                                spec_scalers.append(StandardScaler())

                    # Fit each scaler with data from its respective channel
                    for ch, scaler in enumerate(spec_scalers):
                        channel_data = feat_file[ch].reshape(-1, feat_file.shape[2]) # Reshape channel data: (time, frequency)
                        scaler.partial_fit(channel_data)
                    del feat_file
                    progress.update(task, advance=1)

            # Save the scaler(s)
            if not os.path.exists(scaler_file):
                print(f'Initialized 1 shared scaler for the first {n_norm_channels} channels'
                        f'+ {num_channels - n_norm_channels} individual scalers.')
                joblib.dump(spec_scalers, scaler_file)
                print(f'Scaler(s) saved to {scaler_file}')

        # Normalization preprocessing
        file_list = os.listdir(unnorm_dir)
        with Progress() as progress:
            task = progress.add_task("[yellow]Normalizing features...", total=len(file_list))
            for file_name in file_list:
                # Determine filepaths
                feat_path = os.path.join(unnorm_dir, file_name)
                normalized_feat_path = os.path.join(norm_dir, file_name)

                # Check if we have already normalized the file
                if os.path.exists(normalized_feat_path):
                    progress.update(task, advance=1)
                    continue

                feat_tensor = torch.load(feat_path)
                feat_file = feat_tensor.cpu().numpy()  # Convert to NumPy for processing

                if feat_file.ndim != 3:
                    raise ValueError("Feature file has unsupported dimensions: {}".format(feat_file.shape))

                for ch, scaler in enumerate(spec_scalers):
                    channel_data = feat_file[ch].reshape(-1, feat_file.shape[2])
                    normalized_channel = scaler.transform(channel_data)
                    feat_file[ch] = normalized_channel.reshape(feat_file.shape[1], feat_file.shape[2])

                # Save the normalized features as a torch tensor
                torch.save(torch.tensor(feat_file, dtype=torch.float32), normalized_feat_path)
                del feat_file
                progress.update(task, advance=1)
        print(f'Normalized features have been saved to {norm_dir}')