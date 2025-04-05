import librosa
import numpy as np

from utils import feature_extractor as utils

class Spectral:
    def __init__(self, audio, config):
        self.audio = audio
        self.dependencies = config["spectral"]["dependencies"]
        self.frame_size = int(config["frame_size"])
        self.sampling_rate = int(config["sampling_rate"])
        self.number_of_bins = int(config["spectral"]["number_of_bins"])
        self.is_raw_data = config["is_raw_data"]
        self.frames = int(np.ceil(len(self.audio.data) /self.frame_size))

    def __enter__(self):
        print "Initializing fft calculation..."

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "Done with calculations..."

    def get_current_frame(self, index):
            return utils._get_frame_array(self.audio, index, self.frame_size)

    def compute_hurst(self):
        self.hurst = []
        for k in range(0, self.frames):
            current_frame = self.get_current_frame(k)
            N = current_frame.size
            T = np.arange(1, N + 1)
            Y = np.cumsum(current_frame)
            Ave_T = Y / T
            S_T = np.zeros(N)
            R_T = np.zeros(N)
            for i in range(N):
                S_T[i] = np.std(current_frame[:i + 1])
                X_T = Y - T * Ave_T[i]
                R_T[i] = np.ptp(X_T[:i + 1])

            R_S = R_T / S_T
            R_S = np.log(R_S)[1:]
            n = np.log(T)[1:]
            A = np.column_stack((n, np.ones(n.size)))
            [m, c] = np.linalg.lstsq(A, R_S)[0]
            self.hurst.append(m)
        self.hurst = np.asarray(self.hurst)

    def get_hurst(self):
        return self.hurst
