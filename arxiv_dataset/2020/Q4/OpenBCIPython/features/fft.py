import librosa
import numpy as np

from utils import feature_extractor as utils

class FFT:
    def __init__(self, audio, config):
        self.audio = audio
        self.dependencies = config["fft"]["dependencies"]
        self.frame_size = int(config["frame_size"])
        self.sampling_rate = int(config["sampling_rate"])
        self.number_of_bins = int(config["fft"]["number_of_bins"])
        self.is_raw_data = config["is_raw_data"]
        self.frames = int(np.ceil(len(self.audio.data) /self.frame_size))

    def __enter__(self):
        print "Initializing fft calculation..."

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "Done with calculations..."

    def compute_fft(self):
        self.fft = []
        self.logamplitude = []
        for i in range(0, self.frames):
            current_frame = utils._get_frame_array(self.audio, i, self.frame_size)
            ps = np.abs(np.fft.fft(current_frame, self.number_of_bins))
            self.fft.append(ps)
            self.logamplitude.append(librosa.logamplitude(ps ** 2))
        self.fft = np.asarray(self.fft)
        self.logamplitude = np.asarray(self.logamplitude)

    def get_fft_spectrogram(self):
        return self.fft


    def get_logamplitude(self):
        return self.logamplitude
