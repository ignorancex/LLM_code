from src.experiment import Experiment
class ExperimentConstants:
    def __init__(self) -> None:
        self.DETECTOR_OPENING_ANGLE = 0
        self.DETECTOR_LENGTH = 0
        self.DETECTOR_DISTANCE = 0
        self.ELECTRON_HNL_CHANNELS = []
        self.TAU_HNL_CHANNELS = [] 
        self.ELECTRON_NU_MASSLESS_FLUX = 0
        self.OBSERVED_EVENTS = 0
        self.BEAM_ENERGY = 0
        self.NUCLEON_MASS = 0
        self.POT = 0
        self.B_MESON_FRACTION =  0
        self.MUON_E_MIN = 0
        self.MT_MAX = 0

def get_experimental_constants(experiment):
    exp_const = ExperimentConstants()
    if experiment == Experiment.BEBC: 
        exp_const.DETECTOR_OPENING_ANGLE = 1.69/404
        exp_const.DETECTOR_LENGTH = 1.85 * 5.08e15 # GeV^-1
        exp_const.DETECTOR_DISTANCE = 404 * 5.08e15 # GeV^-1
        exp_const.ELECTRON_HNL_CHANNELS = ["e+pos+nu", "mu+e+nu", "e+pi", "mu+mu+nu"]
        exp_const.TAU_HNL_CHANNELS = ["e+pos+nu", "mu+mu+nu"] 
        exp_const.ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.72e18
        exp_const.OBSERVED_EVENTS = 2.3
        exp_const.BEAM_ENERGY = 400 #GeV
        exp_const.NUCLEON_MASS = 1. #GeV
        exp_const.POT = 2.72e18 # Protons on target
        exp_const.B_MESON_FRACTION = 1.6e-7
        exp_const.MUON_E_MIN = 3. #GeV
        exp_const.MT_MAX = 1.85 #GeV
        
    elif experiment == Experiment.NuTeV:
        exp_const.DETECTOR_OPENING_ANGLE = 1.69/1400 
        exp_const.DETECTOR_LENGTH = 34 * 5.08e15 # GeV^-1
        exp_const.DETECTOR_DISTANCE = 1400 * 5.08e15 # GeV^-1
        exp_const.ELECTRON_HNL_CHANNELS = ["mu+mu+nu"]
        exp_const.TAU_HNL_CHANNELS = ["mu+mu+nu"] 
        exp_const.ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.54e18
        exp_const.OBSERVED_EVENTS = 2.3
        exp_const.BEAM_ENERGY = 800 #GeV
        exp_const.NUCLEON_MASS = 1. #GeV
        exp_const.POT = 2.54e18 # Protons on target 
        exp_const.B_MESON_FRACTION = 1.6e-6
        
    elif experiment == Experiment.SHiP:
        exp_const.DETECTOR_OPENING_ANGLE = 4/50
        exp_const.DETECTOR_LENGTH = 50 * 5.08e15 # GeV^-1
        exp_const.DETECTOR_DISTANCE = 50 * 5.08e15 # GeV^-1
        exp_const.ELECTRON_HNL_CHANNELS = ["e+pos+nu", "mu+e+nu", "e+pi", "mu+mu+nu"]
        exp_const.TAU_HNL_CHANNELS = ["e+pos+nu", "mu+mu+nu"] 
        exp_const.ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.72e20
        exp_const.OBSERVED_EVENTS = 2.3
        exp_const.BEAM_ENERGY = 400 #GeV
        exp_const.NUCLEON_MASS = 1. #GeV
        exp_const.POT = 2.54e20 # Protons on target
        exp_const.B_MESON_FRACTION = 1.6e-7
        exp_const.MUON_E_MIN = 3. #GeV
        exp_const.MT_MAX = 1.85 #GeV

    return exp_const
