import numpy as np
import itertools
from gurobipy import *
import random
def INV_check(R,p,q,tol):
    I = np.nonzero(p)[0]
    J = np.nonzero(q)[0]
    aug = np.block([[R[I][:,J], -np.ones((len(I),1))],[np.ones(len(J)), 0]])
    return np.linalg.cond(aug) < tol*1.001

def SIISOW_check(R,p,q,iota = 1e-15, tol = 1e-15):
    n = len(p)
    m = len(q)
    I = [i for i in range(n) if p[i] > 0]
    J = [j for j in range(m) if q[j] > 0]
    Ic = [i for i in range(n) if p[i] == 0]
    Jc = [j for j in range(m) if q[j] == 0]
    val = p @ R @ q
    l1 = [0]
    l2 = [0]
    for i in I:
        ei = np.zeros(n)
        ei[i] = 1
        l1.append(np.abs(p @ R @ ei - val))
    for j in J:
        ej = np.zeros(m)
        ej[j] = 1
        l1.append(np.abs(ej @ R @ q - val))
    for i in Ic:
        ei = np.zeros(n)
        ei[i] = 1
        l2.append(-(val - ei @ R @ q - iota))
    for j in Jc:
        ej = np.zeros(m)
        ej[j] = 1
        l2.append(-(p @ R @ ej - val - iota))
    return (max(max(l1),max(l2)) <= tol)

def eRPS(p,q):
    n = len(p)
    m = len(q)
    
    # Reindex the nonzero entries to the front
    nonzero_indices_p = [i for i in range(n) if p[i] != 0]
    nonzero_indices_q = [i for i in range(m) if q[i] != 0]
    reindexed_p = [p[i] for i in nonzero_indices_p] + [0] * (n - len(nonzero_indices_p))
    reindexed_q = [q[i] for i in nonzero_indices_q] + [0] * (m - len(nonzero_indices_q))
    
    # Determine the minimum index
    k = min(len(nonzero_indices_p), len(nonzero_indices_q), n, m)
    
    RPS = np.zeros((n, m))
    
    # Calculate payoffs
    c = min([min(reindexed_p[i] * reindexed_q[(i + 1) % k], reindexed_p[i] * reindexed_q[(i + 2) % k]) for i in range(k)])
    
    if k == 1:
        RPS[0, :] = 1
        RPS[:, 0] = -1
        RPS[0, 0] = 0
    else:
        for i in range(n):
            for j in range(m):
                if max(i, j) < k and j == (i + 1) % k:
                    RPS[i, j] = -c / (reindexed_p[i] * reindexed_q[j])
                elif max(i, j) < k and j == (i + 2) % k:
                    RPS[i, j] = c / (reindexed_p[i] * reindexed_q[j])
                elif i < k and j >= k:
                    RPS[i, j] = 1
                elif i >= k and j < k:
                    RPS[i, j] = -1
    
    # Put the values back into their original positions
    reindexed_RPS = np.zeros((n, m))
    for orig_i  in range(n):
        for j, orig_j in enumerate(nonzero_indices_q):
            reindexed_RPS[orig_i, orig_j] = -1

    for i, orig_i in enumerate(nonzero_indices_p):
        for orig_j in range(m):
            if reindexed_RPS[orig_i, orig_j] == 0:
                reindexed_RPS[orig_i, orig_j] = 1
            
    for i, orig_i in enumerate(nonzero_indices_p):
         for j, orig_j in enumerate(nonzero_indices_q):
            reindexed_RPS[orig_i, orig_j] = RPS[i, j]
    return reindexed_RPS
class InvalidPolicyError(Exception):
    pass

class InvalidOriginalRewardError(Exception):
    pass

class UnequalSupportSizeError(Exception):
    pass

class ValueRangeError(Exception):
    pass

class InvalidPerturbationError(Exception):
    pass


# give a det range/ threshold parameter less than lambda
def RAP(policy_1, policy_2, original_reward, reward_bound=float('inf'), 
          value_lower_bound=float('-inf'), value_upper_bound=float('inf'), 
          value_gap=None, perturbation_bound=None, perturbation_value=None, 
          condition_number_bound=100):

    
    # Convert inputs to numpy arrays if they are lists
    if isinstance(policy_1, list):
        policy_1 = np.array(policy_1)
    if isinstance(policy_2, list):
        policy_2 = np.array(policy_2)
    if isinstance(original_reward, list):
        original_reward = np.array(original_reward)

    if value_gap is None:
        value_gap = 0.01
    if perturbation_bound is None:
        perturbation_bound = 1e-20

    #check for type error
    if not isinstance(policy_1, np.ndarray):
        raise TypeError("policy_1 must be a numpy array or a list")
    if not isinstance(policy_2, np.ndarray):
        raise TypeError("policy_2 must be a numpy array or a list")
    if not isinstance(original_reward, np.ndarray):
        raise TypeError("original_reward must be a numpy array or a list")
    if not isinstance(reward_bound, (float, int)):
        raise TypeError("reward_bound must be a number")
    if value_gap is None or not isinstance(value_gap, (float, int)) or value_gap <= 0:
        raise TypeError("value_gap must be a positive float or int")
    if perturbation_bound is None or not isinstance(perturbation_bound, (float, int)) :
        raise TypeError("perturbation_bound must be a decimal point number or None")

    #check for value error
    if not (np.isclose(np.sum(policy_1), 1) and np.isclose(np.sum(policy_2), 1)):
        raise InvalidPolicyError("Policy vectors must be probability distributions.")
    if original_reward.shape != (len(policy_1), len(policy_2)):
        raise InvalidOriginalRewardError("Original reward must match the dimensions of the policy vectors.")
    if not np.array_equal(np.nonzero(policy_1)[0], np.nonzero(policy_2)[0]):
        raise UnequalSupportSizeError("The supports of the policy vectors must be equal.")
    adjusted_reward_lower = -reward_bound + value_gap + perturbation_bound
    adjusted_reward_upper = reward_bound - value_gap - perturbation_bound
    if value_upper_bound <= adjusted_reward_lower or value_lower_bound >= adjusted_reward_upper or adjusted_reward_lower==adjusted_reward_upper:
        raise ValueRangeError("Value range and reward range do not overlap sufficiently.")

    I = np.nonzero(policy_1)[0]
    J = np.nonzero(policy_2)[0]
    Ic = np.nonzero(policy_1 == 0)[0]
    Jc = np.nonzero(policy_2 == 0)[0]
    
    # Initialize model
    nash = Model()
    nash.setParam('OutputFlag', False)
    R = nash.addMVar(original_reward.shape, name = 'R', lb = -reward_bound+perturbation_bound, ub = reward_bound-perturbation_bound)
    v = nash.addVar(name = 'v', lb = value_lower_bound, ub = value_upper_bound)
    t = nash.addMVar(original_reward.shape, name = 't')

    
    # Constraints and objective
    nash.addConstr(t + R >= original_reward)
    nash.addConstr(t - R >= -original_reward)
    nash.setObjective(t.sum(), GRB.MINIMIZE)

    #SII
    nash.addConstr(R[I,:] @ policy_2 == v)
    nash.addConstr(policy_1 @ R[:,J] == v)
    
    #SOW
    if len(Ic) > 0:
        nash.addConstr(R[Ic,:] @ policy_2 <= (v-value_gap))
        nash.addConstr(policy_1 @ R[:,Jc] >= (v+value_gap))

    nash.optimize()   
    R_values = np.array(R.X)
    # obj = nash.getObjective()
    adjusted_R = R_values
    RPS = eRPS(policy_1,policy_2)
    
    if INV_check(R_values,policy_1,policy_2,condition_number_bound)!=True:
        if perturbation_value is not None:
            R_values += perturbation_value * RPS
            if INV_check(R_values,policy_1,policy_2,condition_number_bound):
                return R_values
            else:
                raise InvalidPerturbationError("Failed to achieve a Nash equilibrium with the specified perturbations.")
        else:
            for _ in range(100):
                adjusted_R = R_values + np.random.uniform(-1, 1) * perturbation_bound * RPS
                if INV_check(adjusted_R,policy_1,policy_2,condition_number_bound):
                    return adjusted_R
            else:
                raise InvalidPerturbationError("Failed to achieve a Nash equilibrium with the specified perturbations.")
    return R_values