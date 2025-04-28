import torch


def computeI1(M):
    return torch.einsum('kii->k', M)


def computeI2(M):
    trM = computeI1(M)
    trM2 = computeI1(M@M)
    return 0.5*(trM**2-trM2)


def computeI3(M):
    return torch.linalg.det(M)


def compute_cauchy_stress(dUdC, F):
    PK2 = 2.0*dUdC
    return pk2_to_cauchy(PK2, F)


def pk2_to_cauchy(PK2, F):
    Jinv = 1.0/computeI3(F)
    sigma = Jinv[:, None, None]*torch.einsum('kip,kpl,kjl->kij', F, PK2, F)
    # 00, 11, 22, 12, 02, 01
    return sigma[:, (0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)]


def compute_right_cauchy_green(deformation_gradient):
    return torch.einsum('kli,klj->kij', deformation_gradient, deformation_gradient)
