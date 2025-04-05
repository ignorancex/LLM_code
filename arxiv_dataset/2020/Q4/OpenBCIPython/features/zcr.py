import librosa
import numpy as np
from utils import feature_extractor as utils


class ZCR:
    def __init__(self, audio, dependencies=None, frame=2048, sampling_rate=44000):
        self.sampling_rate = sampling_rate
        self.frame = frame
        self.audio = audio
        self.dependencies = dependencies
        self.frames = int(np.ceil(len(self.audio.data) / 1000.0 * self.sampling_rate / self.frame))

    def __enter__(self):
        print "Initializing zcr calculation..."

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "Done with calculations..."

    def compute_zcr(self):
        self.zcr = []
        for i in range(0, self.frames):
            current_frame = utils._get_frame(self.audio, i, self.frame)
            self.zcr.append(np.mean(0.5 * np.abs(np.diff(np.sign(current_frame)))))
        self.zcr = np.asarray(self.zcr)

    def get_zcr(self):
        return self.zcr
