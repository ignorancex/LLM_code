import math
import torch as ch
import torch.nn.functional as F
import numpy as np
import os
from robustness.attack_steps import transform
from robustness.datasets import CIFAR
from statsmodels.stats.proportion import proportion_confint 

ds = CIFAR('./path/to/cifar', std = ch.tensor([0.2471, 0.2435, 0.2616]))
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm
    
# Returns the (epsilon, delta) values of the adaptive concentration inequality
# for the given parameters.
#
# n: int (number of samples)
# delta: float (parameter delta)
# return: epsilon (parameter epsilon)
def get_type(n, delta):
    n = float(n)
    b = -np.log(delta / 24.0) / 1.8
    epsilon = np.sqrt((0.6 * np.log(np.log(n)/np.log(1.1) + 1) + b) / n)
    return epsilon

# Returns the (epsilon, delta) values of the adaptive concentration inequality
# for the given parameters.
#
# n: int (number of samples)
# delta: float (parameter delta)
# return: epsilon (parameter epsilon)
def get_type_ho(n, delta):
    n = float(n)
    epsilon = np.sqrt((np.log(2)-np.log(delta)) / (2 * n))
    return epsilon

# Return required number of samples
#  tau: float (threshold of robustness)
# delta: float (parameter delta)  
def get_num_ho(tau, delta):
    n = -(np.log(delta / 2)) / (2 * (tau**2))
    return n

# Run type inference to get the type judgement for the robustness criterion.
# tau: float (threshold of robustness)
# n: int (number of samples)
# delta: float (parameter delta)
# E_Z: float (estimate of Z)
# Delta: float (threshold on inequalities)
def get_verification_type(tau, n, delta, E_Z, Delta=0):

    # Step 1: Get (epsilon, delta) values from the adaptive concentration inequality for the current number of samples
    epsilon = get_type(n, delta)

    # Step 2: Check if robustness holds
    if E_Z - epsilon >= 1-tau:
        return 1

    # Step 3: Check if robustness does not hold
    if E_Z + epsilon < 1-tau:
        return 0

    # Step 4: Check if robustness holds (ambiguously)
    if E_Z - epsilon >= 1-tau - Delta and epsilon <= Delta:
        return 1

    # Step 5: Check if robustness does not hold (ambiguously)
    if E_Z + epsilon < 1-tau + Delta and epsilon <= Delta:
        return 0

    #  Failed to verify after maximum number of samples
    return None

def get_verification_ho(tau, n, delta, E_Z, Delta=0):

    # Step 1: Get (epsilon, delta) values from the Hoeffding inequality for the number of samples
    epsilon = get_type_ho(n, delta)

    # Step 2: Check if robustness holds
    if E_Z - epsilon >= 1-tau:
        return 1

    # Step 3: Check if robustness does not hold
    if E_Z + epsilon < 1-tau:
        return 0

    # Step 4: Check if robustness holds (ambiguously)
    if E_Z - epsilon >= 1-tau - Delta and epsilon <= Delta:
        return 1

    # Step 5: Check if robustness does not hold (ambiguously)
    if E_Z + epsilon < 1-tau + Delta and epsilon <= Delta:
        return 0

    #  Failed to verify after maximum number of samples
    return None

def get_verification_AC(tau, n, delta, Z_sum, Delta=0):
    ci_low, ci_upp = proportion_confint(Z_sum, n, method='agresti_coull', alpha=delta)
    # Step 1: Check if robustness holds
    
    if ci_low >= 1-tau:
        return 1

    # Step 2: Check if robustness does not hold
    if ci_upp < 1-tau:
        return 0

    # Step 3: Check if robustness holds (ambiguously)
    if ci_low >= 1-tau - Delta:
        return 1

    # Step 4: Check if robustness does not hold (ambiguously)
    if ci_upp < 1-tau + Delta:
        return 0
    
    #  Failed to verify after maximum number of samples
    return None


def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def margin_ind(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    return ind

def p_ind(x, y, d):
    ind = ch.lt(ch.norm(x-y, p=float('inf'), dim=1), d)
    return ind

def verify(X, label, model, type="adaptive", tau=0.01, delta=1e-10, sample_limit=10000, bs=200, kwargs={
    'rot': 15,
    'trans': 0.3, 
    'scale': 0.3,
    'hue': math.pi/4,
    'satu': 0.3,
    'bright': 0.3,
    'cont': 0.3,
    'tries':1,
    'use_best': True, 
    'transform_type': 'spatial',
    'attack_type': 'random',
    'do_tqdm': False
    }):

    output_ori, _ = model(X)
    output_ori = F.softmax(output_ori, dim=1)
    output_sorted, ind_sorted = output_ori.sort(dim=1)
    d = (output_sorted[:, -1]-output_sorted[:, -2])/2  
    if type == "adaptive":
    # Iteratively sample and check whether fairness holds
        Z_sum = 0
        all_label = label.repeat(bs)
        for i in range(int(sample_limit/bs)):
            with ch.no_grad(): 
                im = X.repeat(bs, 1, 1, 1)
                _, im_spat =model(im, all_label, make_adv=True, **kwargs)

                output, _ = model(im_spat)
                all_output = F.softmax(output, dim=1)

                Z = p_ind(output_ori, all_output, d)

                Z_sum += Z.sum()
                n = (i+1) * bs
                E_Z = Z_sum / n
                t = get_verification_type(tau, n, delta, E_Z)
                # Return if converged
                if not t is None:
                    return t, n
    #  Failed to verify after maximum number of samples
        return None, n
        
    elif type == "hoeffding":
        Z_sum = 0
        sample_limit = get_num_ho(tau, delta)
        
        if sample_limit < bs:
            bs = int(sample_limit)

        all_label = label.repeat(bs)


        for i in range(round(sample_limit/bs)):
            with ch.no_grad(): 
                im = X.repeat(bs, 1, 1, 1)
                _, im_spat =model(im, all_label, make_adv=True, **kwargs)
                output, _ = model(im_spat)
                all_output = F.softmax(output, dim=1)
                Z = p_ind(output_ori, all_output, d)

                Z_sum += Z.sum()
        n = (math.ceil(sample_limit/bs)) * bs
        E_Z = Z_sum / n
        t = get_verification_ho(tau, n, delta, E_Z)
        # Return if converged
        if not t is None:
            return t, n
        else:
            return None, n
    
    elif type == "AC":
        Z_sum = 0

        all_label = label.repeat(bs)

        for i in range(math.ceil(sample_limit/bs)):
            with ch.no_grad(): 
                im = X.repeat(bs, 1, 1, 1)
                _, im_spat =model(im, all_label, make_adv=True, **kwargs)
                output, _ = model(im_spat)
                all_output = F.softmax(output, dim=1)
                Z = p_ind(output_ori, all_output, d)
                Z_sum += Z.sum()
        n = (math.ceil(sample_limit/bs)) * bs

        t = get_verification_AC(tau, n, delta, Z_sum.cpu())
        # Return if converged
        if not t is None:
            return t, n
        else:
            return None, n
