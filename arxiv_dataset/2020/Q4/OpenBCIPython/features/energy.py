import librosa
import numpy as np

from utils import feature_extractor as utils

class Energy:
    def __init__(self, audio, dependencies=None, frame=2048, sampling_rate=44000):
        self.audio = audio
        self.dependencies = dependencies
        self.frame = frame
        self.sampling_rate = sampling_rate
        self.frames = int(np.ceil(len(self.audio.data) / 1000.0 * self.sampling_rate / self.frame))

    def __enter__(self):
        print "Initializing energy calculation..."

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "Done with calculations..."

    def compute_energy(self, frame=2048, sampleing_rate=44000):
        self.energy = []
        for i in range(0, self.frames):
            current_frame = utils._get_frame(self.audio, i, frame)
            self.energy.append(np.sum(current_frame ** 2) / np.float64(len(current_frame)))
        self.energy = np.asarray(self.energy)

    def compute_energy_entropy(self):
        numOfShortBlocks = 10
        eps = 0.00000001
        self.energy_entropy = []
        for i in range(0, self.frames):
            current_frame = utils._get_frame(self.audio, i, self.frame)
            Eol = np.sum(current_frame ** 2)  # total frame energy
            L = len(current_frame)
            subWinLength = int(np.floor(L / numOfShortBlocks))
            if L != subWinLength * numOfShortBlocks:
                current_frame = current_frame[0:subWinLength * numOfShortBlocks]
            # subWindows is of size [numOfShortBlocks x L]
            subWindows = current_frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()
            # Compute normalized sub-frame energies:
            s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)
            # Compute entropy of the normalized sub-frame energies:
            entropy = -np.sum(s * np.log2(s + eps))
            self.energy_entropy.append(entropy)
        self.energy_entropy = np.asarray(self.energy_entropy)

    def get_energy(self):
        return self.energy

    def get_energy_entropy(self):
        return self.energy_entropy
