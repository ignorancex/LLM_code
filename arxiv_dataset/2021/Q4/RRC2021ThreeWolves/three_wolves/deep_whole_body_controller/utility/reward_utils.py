import numpy as np

def ComputeDist(p0, p1):
    return np.linalg.norm(np.subtract(p1, p0))

def FVCap(v_cap, r):
    return max(-v_cap, min(r, v_cap))

def ComputeAcc(pos_3, time_step=0.1):
    assert pos_3.shape == (3, 3)
    vel_0 = ComputeDist(pos_3[0], pos_3[1]) / time_step
    vel_1 = ComputeDist(pos_3[1], pos_3[2]) / time_step
    acc_3 = (vel_1 - vel_0) / time_step
    return acc_3

def ExpSqr(cur, tar=0, wei=-3):
    assert wei < 0
    return np.sum(np.exp(wei * np.square(np.abs(tar - cur))))

def Delta(seq):
    seq = np.array(seq)
    assert seq.ndim == 1
    _diff = seq - np.mean(seq)
    return np.sum(np.abs(_diff))

