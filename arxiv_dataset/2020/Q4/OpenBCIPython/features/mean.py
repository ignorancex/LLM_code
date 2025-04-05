import librosa
import numpy as np

from utils import feature_extractor as utils

class Mean:
    def __init__(self, audio, dependencies=None, frame=2048, sampling_rate=44000):
        self.audio = audio
        self.dependencies = dependencies
        self.frame = frame
        self.sampling_rate = sampling_rate
        self.frames = int(np.ceil(len(self.audio.data) / 1000.0 * self.sampling_rate / self.frame))

    def __enter__(self):
        print "Initializing mean calculation..."

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "Done with calculations..."

    def compute_mean(self):
        self.mean = []
        for i in range(0, self.frames):
            current_frame = utils._get_frame(self.audio, i, self.frame)
            sum = np.sum(current_frame ** 2)  # total frame energy
            frame_length = len(current_frame)
            self.mean.append(sum/frame_length)
        self.mean = np.asarray(self.mean)

    def compute_geometric_mean(self):
        self.geometric_mean = []
        for i in range(0, self.frames):
            current_frame = utils._get_frame(self.audio, i, self.frame)
            sum = np.sum(current_frame ** 2)  # total frame energy
            frame_length = len(current_frame)
            self.geometric_mean.append(sum/frame_length)
        self.geometric_mean = np.asarray(self.geometric_mean)