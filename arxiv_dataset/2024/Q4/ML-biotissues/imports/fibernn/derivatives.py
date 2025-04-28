import torch
from . import mechanics
from math import sqrt


def batch_jacobian(f, x):
    def f_sum(x): return torch.sum(f(x), axis=0)
    return torch.autograd.functional.jacobian(f_sum, x, create_graph=True)  # .squeeze()


def batch_hessian(f, x):
    def f_sum(x): return torch.sum(f(x), axis=0)
    return torch.autograd.functional.hessian(f_sum, x, create_graph=True).squeeze()


def batch_hessian2(f, x):
    def f_sum(x): return torch.sum(batch_jacobian(f, x), axis=0)
    hessian = torch.autograd.functional.jacobian(f_sum, x, create_graph=True).squeeze()
    return torch.permute(hessian, (2, 0, 1, 3, 4))


def tensor4_to_voigt(tensor):
    assert (tensor.ndim == 5)
    for i in range(1, tensor.ndim):
        assert (tensor.shape[i] == 3)
    # tensor should have minor symmetries to be converted to voigt form
    # assert(torch.allclose(tensor, torch.transpose(tensor,1,2)))
    # assert(torch.allclose(tensor, torch.transpose(tensor,3,4)))
    voigt_tensor = torch.empty(len(tensor), 6, 6)
    idxs = [(0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)]
    voigt_tensor[:, :, :]  # = tensor[:,idxs[0],idxs[1]]
    for i in range(6):
        for j in range(6):
            voigt_tensor[:, i, j] = tensor[:, idxs[0][i], idxs[1][i], idxs[0][j], idxs[1][j]]
    return voigt_tensor


def get_probe_directions():
    probe_directions = torch.empty(6, 3, 3, requires_grad=False)
    probe_directions[0, :, :] = torch.Tensor(((1, 0, 0), (0, 0, 0), (0, 0, 0)))
    probe_directions[1, :, :] = torch.Tensor(((0, 0, 0), (0, 1, 0), (0, 0, 0)))
    probe_directions[2, :, :] = torch.Tensor(((0, 0, 0), (0, 0, 0), (0, 0, 1)))
    probe_directions[3, :, :] = torch.Tensor(((0, 0, 0), (0, 1, 1), (0, 1, 1)))
    probe_directions[4, :, :] = torch.Tensor(((1, 0, 1), (0, 0, 0), (1, 0, 1)))
    probe_directions[5, :, :] = torch.Tensor(((1, 1, 0), (1, 1, 0), (0, 0, 0)))
    return probe_directions


def mandel_probe_vec(dtype=torch.float32):
    T = torch.zeros((6, 6), requires_grad=False, dtype=dtype)
    sr2 = sqrt(2.0)
    T[0, 0] = 1
    T[1, 1] = 1
    T[2, 2] = 1
    T[3, 3] = 2
    T[4, 4] = 2
    T[5, 5] = 2
    T[0, 4] = sr2
    T[0, 5] = sr2
    T[1, 3] = sr2
    T[1, 5] = sr2
    T[2, 3] = sr2
    T[2, 4] = sr2
    return T


def mandel_probe_vec_inv(dtype=torch.float32):
    T = torch.zeros((6, 6), requires_grad=False, dtype=dtype)
    sr2 = -sqrt(2.0) / 2.0
    T[0, 0] = 1
    T[1, 1] = 1
    T[2, 2] = 1
    T[3, 3] = 1. / 2.
    T[4, 4] = 1. / 2.
    T[5, 5] = 1. / 2.
    T[0, 4] = sr2
    T[0, 5] = sr2
    T[1, 3] = sr2
    T[1, 5] = sr2
    T[2, 3] = sr2
    T[2, 4] = sr2
    return T


def voigt_index(i, j):
    if i == j:
        return i
    idxs = sorted((i, j))
    # 00, 11, 22, 12, 02, 01
    #  0,  1,  2,  3,  4,  5
    if idxs == [1, 2]:
        return 3
    if idxs == [0, 2]:
        return 4
    if idxs == [0, 1]:
        return 5
    raise RuntimeError(f"{idxs} not valid")


def voigt_to_tensor4(matrix):
    assert (matrix.ndim == 3)
    assert (matrix.shape[1] == 6)
    assert (matrix.shape[2] == 6)
    tensor = torch.Tensor(len(matrix), 3, 3, 3, 3)
    for i in range(3):
        for j in range(3):
            idx0 = voigt_index(i, j)
            for k in range(3):
                for l in range(3):
                    idx1 = voigt_index(k, l)
                    tensor[:, i, j, k, l] = matrix[:, idx0, idx1]
    # assert(torch.allclose(tensor, torch.transpose(tensor,1,2)))
    # assert(torch.allclose(tensor, torch.transpose(tensor,3,4)))

    return tensor


def voigt_to_mandel(matrix):
    """
    Convert a 6x6 matrix in Voigt form to Mandel form
    This is a in place operation
    :param matrix: 6x6 matrix in Voigt form
    """
    # multiply by factors to convert to Mandel form
    matrix[:, 3:, 0:3] *= sqrt(2.)
    matrix[:, 0:3, 3:] *= sqrt(2.)
    matrix[:, 3:, 3:] *= 2.


def mandel_to_voigt(matrix):
    """
    Convert a 6x6 matrix in Mandel form to Voigt form
    This is a in place operation
    :param matrix: 6x6 matrix in Mandel form
    """
    # multiply by factors to convert to Mandel form
    matrix[:, 3:, 0:3] /= sqrt(2.)
    matrix[:, 0:3, 3:] /= sqrt(2.)
    matrix[:, 3:, 3:] /= 2.


def total_lagrangian_stiffness_to_updated_lagrangian(material_stiffness_tensor, F):
    """
    Convert a total lagrangian material stiffness to an updated lagrangian material stiffness
    i.e., C_{mnpq} = 1/J F_{mi} F_{nj} A_{ijrs} F_{pr} F_{qs}
    :param material_stiffness:
    :param F:
    :return:
    """
    # updated lagrangian material stiffness
    # fixme need to multiply by 1/J
    Jinv = 1.0 / torch.linalg.det(F)
    ul_material_stiffness_tensor = torch.einsum(
        'kmi,knj,kijrs,kpr,kqs->kmnpq', F, F, material_stiffness_tensor, F, F)
    ul_material_stiffness = Jinv[:, None, None] * tensor4_to_voigt(ul_material_stiffness_tensor)
    return ul_material_stiffness


def compute_material_stiffness(PK2, right_cauchy_green, check_symmetry=True):
    """
    compute the derivatives dPK2/dE
    Note this is is using a special technique to get the derivatives w.r.t. SPD part of C
    Otherwise this is equivalent to taking the jacobian
    :return:
    """
    probe_directions = [(0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)]
    material_stiffness = torch.empty(len(PK2), 6, 6)
    grad_output = get_probe_directions().expand((len(PK2), 6, 3, 3)).permute(1, 0, 2, 3)
    for i in range(len(grad_output)):
        grad = torch.autograd.grad(PK2, right_cauchy_green,
                                   grad_outputs=grad_output[i], retain_graph=True)[0]
        if (check_symmetry):
            assert (torch.allclose(grad, torch.transpose(grad, 1, 2)))
        # each column comes from the specific probe directions
        material_stiffness[:, :, i] = grad[:, probe_directions[0], probe_directions[1]]
    voigt_to_mandel(material_stiffness)
    # use probe inverse to get true derivatives (in Mandel form)
    material_stiffness = torch.matmul(material_stiffness, mandel_probe_vec_inv())
    mandel_to_voigt(material_stiffness)
    # at this point we have the Voigt form of dPK2/dC
    # dPK2/dE = 2*dPK2/dC (d^2u/dEdE = 4*d^2u/dCdC)...take derivative w.r.t. PK2
    material_stiffness *= 2.0
    return material_stiffness


def compute_cauchy_stress_stiffness_hand(energy, F, right_cauchy_green, check_symmetry=False):
    """
    compute the derivatives dPK2/dE
    Note this is is using a special technique to get the derivatives w.r.t. SPD part of C
    Otherwise this is equivalent to taking the jacobian
    :return:
    """
    # assert(torch.allclose(right_cauchy_green, torch.transpose(right_cauchy_green,1,2)))
    # right_cauchy_green = compute_right_cauchy_green(F)
    # 2*dU/dC
    outputs = torch.sum(energy, 0)
    PK2 = 2 * (torch.autograd.grad(outputs, right_cauchy_green, create_graph=True)[0])
    if check_symmetry:
        assert (torch.allclose(PK2, torch.transpose(PK2, 1, 2)))
    cauchy_stress = mechanics.pk2_to_cauchy(PK2, F)
    # compute the material stiffness
    material_stiffness = compute_material_stiffness(
        PK2, right_cauchy_green, check_symmetry=check_symmetry)
    if check_symmetry:
        assert (torch.allclose(material_stiffness, torch.transpose(
            material_stiffness, 1, 2), rtol=1E-4, atol=1E-3))
    # now we need the updated lagrangian material stiffness. Easiest way to avoid lots of for loops in python is to expand the TL material
    # stiffness to a 4th order tensor. Do multiplications then come back to second order tensor
    material_stiffness_tensor = voigt_to_tensor4(material_stiffness)
    ul_material_stiffness = total_lagrangian_stiffness_to_updated_lagrangian(
        material_stiffness_tensor, F)
    if check_symmetry:
        assert (torch.allclose(ul_material_stiffness, torch.transpose(
            ul_material_stiffness, 1, 2), rtol=1E-4))
    return cauchy_stress, ul_material_stiffness


def compute_cauchy_stress_stiffness(model, F, right_cauchy_green, check_symmetry=False):
    """
    compute the derivatives dPK2/dE
    Note this is is using a special technique to get the derivatives w.r.t. SPD part of C
    Otherwise this is equivalent to taking the jacobian
    :return:
    """
    # 2*dU/dC
    PK2 = 2 * batch_jacobian(model, right_cauchy_green).squeeze()
    if check_symmetry:
        assert (torch.allclose(PK2, torch.transpose(PK2, 1, 2)))
    cauchy_stress = mechanics.pk2_to_cauchy(PK2, F)
    # compute the material stiffness
    # this is much faster than calling batch_hessian when using large batch sizes
    material_stiffness = 4 * \
        batch_jacobian(lambda x: batch_jacobian(model, x).sum(
            axis=0), right_cauchy_green).permute(2, 0, 1, 3, 4)
    # need to symmetrize since we are taking tensor derivative with symmetric tensor C
    material_stiffness = 0.5 * (material_stiffness + torch.transpose(material_stiffness, 1, 2))
    if check_symmetry:
        assert (torch.allclose(material_stiffness, torch.transpose(
            material_stiffness, 1, 2), rtol=1E-4, atol=1E-3))
    ul_material_stiffness = total_lagrangian_stiffness_to_updated_lagrangian(material_stiffness, F)
    if check_symmetry:
        assert (torch.allclose(ul_material_stiffness, torch.transpose(
            ul_material_stiffness, 1, 2), rtol=1E-4))
    return cauchy_stress, ul_material_stiffness
