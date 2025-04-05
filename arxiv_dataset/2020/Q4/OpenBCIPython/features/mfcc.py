import librosa
import numpy as np
from utils import feature_extractor as utils
import matplotlib.pyplot as plt

class MFCC:
    def __init__(self, audio, dependencies=None, number_of_mfcc=13, frame=2048, sampling_rate=44000):
        self.audio = audio
        self.dependencies = dependencies
        self.frame = frame
        self.sampling_rate = sampling_rate
        self.number_of_mfcc = number_of_mfcc
        self.frames = int(np.ceil(len(self.audio.data) / 1000.0 * self.sampling_rate / self.frame))


    def __enter__(self):
        print "Initializing mfcc calculation..."

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "Done with calculations..."

    def compute_mfcc(self):
        self.melspectrogram = []
        self.logamplitude = []
        self.mfcc = []
        self.mfcc_delta = []
        self.mfcc_delta2 = []
        for i in range(0, self.frames-1):
            current_frame = utils._get_frame(self.audio, i, self.frame)
            # MFCC computation with default settings (2048 FFT window length)
            self.melspectrogram.append(librosa.feature.melspectrogram(current_frame, sr=self.sampling_rate,
                                                                 hop_length=self.frame)[0:,][0:,1])
            self.logamplitude.append(librosa.logamplitude(self.melspectrogram[i]))
            self.mfcc.append(librosa.feature.mfcc(S=self.logamplitude[i], n_mfcc=self.number_of_mfcc).transpose())
            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(self.mfcc[i], x_axis='time')
            # plt.colorbar()
            # plt.title('MFCC')
            # plt.tight_layout()
            self.mfcc_delta.append(librosa.feature.delta(self.mfcc[i]))
            self.mfcc_delta2.append(librosa.feature.delta(self.mfcc[i], order=2))
            self.logamplitude[i]=(self.logamplitude[i].T.flatten()[:, np.newaxis].T)

        self.melspectrogram = np.asarray(self.melspectrogram)
        self.logamplitude = np.asarray(self.logamplitude)
        self.mfcc = np.asarray(self.mfcc)
        self.mfcc_delta = np.asarray(self.mfcc_delta)
        self.mfcc_delta2 = np.asarray(self.mfcc_delta2)


    def get_mel_spectrogram(self):
        return self.melspectrogram

    def get_log_amplitude(self):
        return self.logamplitude

    def get_mfcc(self):
        return self.mfcc

    def get_delta_mfcc(self):
        return self.mfcc_delta

    def get_delta2_mfcc(self):
        return self.mfcc_delta2
