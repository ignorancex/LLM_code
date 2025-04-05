"""
This module runs an example on training the detector utilizing a set of trained models
"""
###############################################################################
# preample
import numpy as np
from qubitmlmodel import qubitMLmodel
from Detector import detector
from utilities import Pauli_operators
from simulator import quantumTFsim
import pickle
import sys
###############################################################################
# define simulation parameters
T                 = 1
M                 = 1024
Omega             = 12                                # qubit energy gap
static_operators  = [0.5*Pauli_operators[3]*Omega]    # drift Hamiltonian
dynamic_operators = [0.5*Pauli_operators[1]]          # control Hamiltonian 
noise_operators   = [0.5*Pauli_operators[3]]          # noise Hamiltonian
initial_states    = [
                     np.array([[0.5,0.5],[0.5,0.5]]),
                     np.array([[0.5,-0.5j],[0.5j,0.5]]),
                     np.array([[1,0],[0,0]])
                    ]                                 # intial state of qubit 
measurement_operators = Pauli_operators[1:2]          # measurement operators
n_max             = 5                                 # number of pulses per sequence
time_range        = [(0.5*T/M) + (j*T/M) for j in range(M)] 
# select one of three different scenarios
ex                = int(sys.argv[1])
if ex==0:  # 5 distant profiles, unlimited control
    noise_profiles= [0,1,2,3,4] 
    max_amp       = 100
elif ex==1:# 5 profiles with 2 near each other 
    noise_profiles= [5,1,2,3,4] 
    max_amp       = 100
else:             # 5 distant models with limited control                      
    noise_profiles= [0,1,2,3,4] 
    max_amp       = 1
# define dataset names
models            = ["G_1q_X_Z_N%d"%idx for idx in noise_profiles]
# classifying testing parameters
K                 = 1000                              # number of noise realizations
seq_len           = 10000                             # number of repitions for analyzing the confusion matrix
###############################################################################
# define the simulators
simulators = [quantumTFsim(T, M, dynamic_operators, static_operators, noise_operators, measurement_operators, initial_states, K=K, waveform="input", num_pulses=5, distortion=False, noise_profile=[profile]) for profile in noise_profiles]

# define the qubit ML models
mlmodels   = [qubitMLmodel(T, M, dynamic_operators, static_operators, noise_operators, measurement_operators, initial_states, K=4) for _ in models]

# load the pretrained models
for idx_m , m in enumerate(models):
    mlmodels[idx_m].load_model("%s"%m)
    
###############################################################################
# define the detector model
detector_mdl = detector(max_amp, n_max, mlmodels)

###############################################################################
# find the optimal discriminating pulse
detector_mdl.train_optimal_pulse(500)
optimal_pulse = detector_mdl.predict_optimal_pulse()

###############################################################################
# train the classifier
detector_mdl.train_detector(500)

###############################################################################
# save the model
detector_mdl.save_model("Example%d"%ex)

###############################################################################
# test the detector using the simulator
classification_results  = [] # list to store the results of detection

# 1) simulate a random sequnece of hopping between the different noise profiles  
active_profile = np.random.randint(low=0, high=len(noise_profiles), size=seq_len)

# loop over every hop
for idx in active_profile:
    # 2) perform an optimal measurement in the experiment [not the trained model]
    optimal_measurements = simulators[idx].simulate([np.ones((1,)), optimal_pulse])
   
    # 3) perform the classification
    classification_results.append( detector_mdl.detect(optimal_measurements)[1] )

# calculate the confusion matrix [rows: True, columns:predicted]
confusion_matrix = np.zeros( (len(noise_profiles), len(noise_profiles)) )

for idx_ex in range(seq_len):
    confusion_matrix[active_profile[idx_ex],classification_results[idx_ex]] += 1 # add 1 at the correct place

# store the confusion matrix externally
np.save("confusion%d"%ex, confusion_matrix)
