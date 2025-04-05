
import pydub
import os

import seaborn as sb


from manager import FeatureManager
from features.energy import Energy
from features.fft import FFT
from features.mean import Mean
from features.mfcc import MFCC
from features.zcr import ZCR
from utils.Audio import Audio

sb.set(style="white", palette="muted")

import random
random.seed(20150420)

class Clip:

    RATE = 44100  # All recordings in ESC are 44.1 kHz
    FRAME = 248  # Frame size in samples

    def __init__(self, filename, file_type):
        self.filename = os.path.basename(filename)
        self.path = os.path.abspath(filename)
        self.directory = os.path.dirname(self.path)
        self.category = self.directory.split('/')[-1]
        self.audio = Audio(self.path, file_type)
        with self.audio as audio:
            self.featureManager = FeatureManager()
            self.featureManager.addRegisteredFeatures(MFCC(self.audio,None, 32, self.FRAME, self.RATE), "mfcc")
            self.featureManager.addRegisteredFeatures(FFT(self.audio,None,1024, self.FRAME, self.RATE), "fft")
            # TODO recheck
            # self.featureManager.addRegisteredFeatures(Energy(self.audio,None,self.FRAME, self.RATE), "energy")
            self.featureManager.addRegisteredFeatures(ZCR(self.audio,None,self.FRAME, self.RATE), "zcr")
            self.featureManager.addRegisteredFeatures(Mean(self.audio, None, self.FRAME, self.RATE), "mean")

            self.featureManager.getRegisteredFeature("mfcc").compute_mfcc()
            self.featureManager.getRegisteredFeature("fft").compute_fft()
            #TODO recheck
            # self.featureManager.getRegisteredFeature("energy").compute_energy()
            # self.featureManager.getRegisteredFeature("energy").compute_energy_entropy()
            self.featureManager.getRegisteredFeature("zcr").compute_zcr()
            self.featureManager.getRegisteredFeature("mean").compute_mean()

            self.feature_list = self.featureManager.getRegisteredFeatures()

    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)

    def get_feature_vector(self):
        return self.featureManager.getRegisteredFeature("fft").get_logamplitude()




































