import numpy as np
from scipy.integrate import romb
import torch
from torch.autograd import Variable
from scipy.optimize import minimize


def get_model_predictions(num_treatments, test_data, model):
    x = Variable(torch.from_numpy(test_data['x']).cuda().detach()).float()
    t = Variable(torch.from_numpy(test_data['t']).cuda().detach()).float()
    d = Variable(torch.from_numpy(test_data['d']).cuda().detach()).float()
    I_logits = model(x, t, d)
    return I_logits.cpu().detach().numpy()

def get_true_dose_response_curve(meta_info, patient, treatment_idx):
    def true_dose_response_curve(dosage):
        y = get_patient_outcome(patient, meta_info, treatment_idx, dosage)
        return y

    return true_dose_response_curve

def get_patient_outcome(x, v, treatment, dosage, scaling_parameter=10):
    if (treatment == 0):
        #y = float(scaling_parameter) * (np.dot(x, v[treatment][0]) + 12.0 * dosage * (dosage - ( np.dot(x, v[treatment][1]) / np.dot(x, v[treatment][2]))) ** 2)
        y = float(scaling_parameter) * (np.dot(x, v[treatment][0]) + 12.0 * dosage * (dosage - 0.75 * ( np.dot(x, v[treatment][1]) / np.dot(x, v[treatment][2]))) ** 2)
    elif (treatment == 1):
        y = float(scaling_parameter) * (np.dot(x, v[treatment][0]) + np.sin(
            np.pi * (np.dot(x, v[treatment][1]) / np.dot(x, v[treatment][2])) * dosage))
    elif (treatment == 2):
        y = float(scaling_parameter) * (np.dot(x, v[treatment][0]) + 12.0 * (np.dot(x, v[treatment][
            1]) * dosage - np.dot(x, v[treatment][2]) * dosage ** 2))

    return y

