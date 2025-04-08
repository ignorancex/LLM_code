import torch
import torch.nn.functional as F


def proper_fmap_computation(method, C12, C21, features1, features2, evecs1, evecs2, evecs_trans1, evecs_trans2, T=0.07):
    """
    Compute the proper fuctional map between two shape1 and shape2.
    :param method: str, method to compute the proper feature map. Options are None, 'adjoint', 'soft'.
    :param C12: torch.Tensor, the functional map from shape1 to shape2.
    :param C21: torch.Tensor, the functional map from shape2 to shape1.
    :param features1: torch.Tensor, features of shape1 used to compute the fmap.
    :param features2: torch.Tensor, features of shape2 used to compute the fmap.
    :param evecs1: torch.Tensor, eigenvectors of shape1.
    :param evecs2: torch.Tensor, eigenvectors of shape2.
    :param evecs_trans1: torch.Tensor, inverse of eigenvectors of shape1
    :param evecs_trans2: torch.Tensor, inverse of eigenvectors of shape2
    :param T: float, temperature parameter for softmax.
    :return: C12_proper, C21_proper
    """
    if features1.ndim == 2:
        raise Exception('proper fmap computation is not implemented for 2D features, add a batch dimension')

    if method is None:
        return C12, C21

    neig = C12.size(1)
    evecs1, evecs2 = evecs1[:, :, :neig], evecs2[:, :, :neig]
    evecs_trans1, evecs_trans2 = evecs_trans1.transpose(1, 2)[:, :neig], evecs_trans2.transpose(1, 2)[:, :neig]

    if method == 'adjoint':
        pmap21_soft = (evecs2 @ C12) @ torch.transpose(evecs1, 1, 2)
        pmap21_soft = torch.softmax(pmap21_soft / T, dim=-1)
        C12_proper = evecs_trans2 @ pmap21_soft @ evecs1
        if C21 is not None:
            pmap12_soft = (evecs1 @ C21) @ torch.transpose(evecs2, 1, 2)
            pmap12_soft = torch.softmax(pmap12_soft / T, dim=-1)
            C21_proper = evecs_trans1 @ pmap12_soft @ evecs2
    elif method == 'feat_based':
        feats1 = F.normalize(features1, p=2, dim=-1)
        feats2 = F.normalize(features2, p=2, dim=-1)
        pmap21_soft = feats2 @ torch.transpose(feats1, 1, 2)
        pmap21_soft = torch.softmax(pmap21_soft / T, dim=-1)
        C12_proper = evecs_trans2 @ pmap21_soft @ evecs1
        if C21 is not None:
            pmap12_soft = feats1 @ torch.transpose(feats2, 1, 2)
            pmap12_soft = torch.softmax(pmap12_soft / T, dim=-1)
            C21_proper = evecs_trans1 @ pmap12_soft @ evecs2
    else:
        raise Exception(f'method `{method}` not implemented!')

    if C21 is None:
        C21_proper = None

    return C12_proper, C21_proper
