#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


import os
import numpy as np
import librosa
from pydub import AudioSegment


def read_audio_generic(input_file, monochannel=False):
    """
    Function to read files containing PCM (int16 or int32) audio
    """
    import urllib
    from urllib.parse import urlparse
    input_file = urlparse(input_file, scheme='file')
    typ = os.path.splitext(input_file.path.split('/')[-1])[1][1:]
    if input_file.scheme == 'file':
        audiofile = AudioSegment.from_file(input_file.path, typ)
    else:
        from io import BytesIO
        with urllib.request.urlopen(input_file.geturl()) as response:
            with BytesIO(response.read()) as dat:
                audiofile = AudioSegment.from_file(dat, typ)

    channels = audiofile.channels
    sampling_rate = audiofile.frame_rate
    if monochannel and (channels > 1):
        audiofile = audiofile.set_channels(1)
        channels = audiofile.channels

    if audiofile.sample_width == 2:
        dtype = np.int16
        norm = 32768.0
    elif audiofile.sample_width == 4:
        dtype = np.int32
        norm = 2147483648.0
    else:
        assert False

    data = np.frombuffer(audiofile._data, dtype)

    if channels == 1:
        signal = data[:, None]
    elif data.size >= channels:
        signal = list()
        for chn in list(range(channels)):
            signal.append(data[chn::channels])
        signal = np.stack(signal, -1)
    else:
        signal = np.zeros((0, channels), dtype)

    if monochannel:
        signal = signal[:, 0]

    return sampling_rate, signal / norm


def wav_to_spectrogram(signal, sampling_rate, n_fft, window_step, window_length):
    frames = np.abs(librosa.core.stft(
        signal,
        n_fft=n_fft,
        hop_length=int(sampling_rate * window_step / 1000),
        win_length=int(sampling_rate * window_length / 1000),
    ))
    return frames.astype(np.float32).T


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    wave_dBFS = 10 * np.log10(np.mean(wav ** 2))
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def compute_spec(fileinput,
                 target_sampling_rate=16000, audio_norm_target_dBFS=-30,
                 n_fft=512, window_step=10, window_length=25):
    sampling_rate, signal = read_audio_generic(fileinput, monochannel=True)

    signal = librosa.resample(signal, orig_sr=sampling_rate, target_sr=target_sampling_rate)
    signal = normalize_volume(signal, audio_norm_target_dBFS, increase_only=True)
    spec = wav_to_spectrogram(signal, target_sampling_rate,
                              n_fft=n_fft, window_step=window_step, window_length=window_length)
    return spec


class MockFileSpec:

    def __init__(self, audiodata, fps=25, audio_window_step=10, output_key='spec'):
        self.audiodata = audiodata
        self.output_key = output_key
        factor = fps*audio_window_step
        assert (1000 % factor) == 0
        self.datastep = 1000//factor

    def reset(self, audiodata=None):
        if audiodata is not None:
            self.audiodata = audiodata
        return self

    def _get_sample(self, index):
        start = self.datastep*index
        end = start + self.datastep
        if start >= len(self.audiodata):
            return np.zeros((self.datastep,) + self.audiodata.shape[1:], dtype=self.audiodata.dtype)
        elif end > len(self.audiodata):
            out = np.zeros((self.datastep,) + self.audiodata.shape[1:], dtype=self.audiodata.dtype)
            out[:len(self.audiodata)-end] = self.audiodata[start:end]
            return out
        else:
            return self.audiodata[start:end]

    def __call__(self, inp):
        inp[self.output_key] = [self._get_sample(index) for index in inp['image_inds']]
        return inp


class IterImageInds:

    def __init__(self, total=25, stride=32, key='image_inds'):
        self.inds = list(range(total))
        self.stride = stride
        self.key = key
        self._len = int(np.ceil(len(self.inds)/self.stride))

    def __len__(self):
        return self._len

    def __enter__(self):
        return self

    def __call__(self, c):
        if c >= self._len:
            raise StopIteration
        out = self.inds[c*self.stride:(c*self.stride+self.stride)]
        return {self.key: out,
                'id_track': [0, ]*len(out),
                'boxes': [(np.nan,np.nan,np.nan,np.nan), ]*len(out) }

    def __iter__(self):
        for count in range(len(self)):
            yield self(count)

    def __exit__(self, type, value, tb):
        pass
