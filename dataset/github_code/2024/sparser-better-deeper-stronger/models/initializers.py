# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math
import numpy as np

def block_convolution(a: torch.Tensor, b: torch.Tensor):
    n = a[0][0].shape[0]
    if n != b[0][0].shape[0]:
        raise ValueError("The entries in block convolution must have the same dimensions.")

    k = a.shape[0]
    l = b.shape[0]
    output_size = k + l - 1

    result = torch.zeros((output_size, output_size, n, n))
    for i in range(output_size):
        for j in range(output_size):
            for i1 in range(min(k, i + 1)):
                for i2 in range(min(k, j + 1)):
                    if (i - i1) < l and (j - i2) < l:
                        result[i][j] += torch.matmul(a[i1][i2], b[i - i1][j - i2])

    return result

def block_orthogonal(p, q):
    if p.shape != q.shape:
        raise ValueError("The dimension of the block components have to match")
    n = p.shape[0]
    eye = torch.eye(n)

    result = torch.zeros((2, 2, n, n))
    result[0][0] = torch.matmul(p, q)
    result[0][1] = torch.matmul(p, eye - q)
    result[1][0] = torch.matmul(eye - p, q)
    result[1][1] = torch.matmul(eye - p, eye - q)
    return result

def symmetric_projection(tensor):
    """ Compute a n x n symmetric projection matrix.
    """
    q = orthogonal_matrix(tensor)
    # randomly zero out some columns
    n = tensor.size(0)
    mask = (torch.normal(torch.zeros((n,))) > 0).float()
    c = torch.multiply(q, mask)
    return torch.matmul(c, c.T)

def conv_orthogonal(tensor, gain=1.0): # algorithm 1 from [https://arxiv.org/abs/1806.05393]
    c_out, c_in, ker_x, _ = tensor.shape # assuming ker_x == ker_y

    swapped = False
    if c_in > c_out:
        print('Warning: c_in > c_out, convolution will be transposed orthogonal')
        c_in, c_out = c_out, c_in
        swapped = True

    a = torch.zeros((1, 1, c_out, c_out))
    a[0, 0, :, :] = torch.eye(c_out)

    shape_template = torch.zeros((c_out, c_out))
    p = symmetric_projection(shape_template)
    q = symmetric_projection(shape_template)
    b = block_orthogonal(p, q)

    for _ in range(ker_x - 1):
        a = block_convolution(a, b)
    
    shape_template = torch.zeros((c_out, c_out))
    h = orthogonal_matrix(shape_template)[:c_in, :]
    result = torch.zeros((c_out, c_in, ker_x, ker_x))

    for i in range(ker_x):
        for j in range(ker_x):
            result[:, :, i, j] = torch.matmul(h, a[i][j]).T
    
    if swapped:
        result = result.transpose(0, 1)

    with torch.no_grad():
        tensor[:, :, :, :] = result[:, :, :, :]
        tensor *= gain
    return tensor
           
def orthogonal_matrix(tensor):
    # Generate a random matrix
    a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.linalg.qr(a)
    # Make Q uniform
    d = torch.diag(r, 0)
    q *= d.sign()
    q = q[:, :tensor.size(1)]
    return q

def delta_orthogonal(tensor, q, gain=1.0): # algorithm 2 from [https://arxiv.org/abs/1806.05393]
    r"""Initializer that generates a delta orthogonal kernel for ConvNets.
    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`3 \leq n \leq 5`
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
        orthogonal_matrix: one can specify the orthogonal matrix by themselves
    Examples:
        >>> w = torch.empty(5, 4, 3, 3)
        >>> nn.init.conv_delta_orthogonal(w)
    """
    if tensor.ndimension() < 3 or tensor.ndimension() > 5:
        raise ValueError("The tensor to initialize must be at least "
                         "three-dimensional and at most five-dimensional")
    
    if tensor.size(1) > tensor.size(0):
        print('Warning: c_in > c_out, convolution will be transposed orthogonal')

    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 3:
            tensor[:, :, (tensor.size(2) - 1) // 2] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2) - 1) // 2, (tensor.size(3) - 1) // 2] = q
        else:
            tensor[:, :, (tensor.size(2) - 1) // 2, (tensor.size(3) - 1) // 2, (tensor.size(4) - 1) // 2] = q
        tensor *= gain
        
    return tensor


class Initialization():
    def __call__(self, w):
        pass


class Binary(Initialization):
    def __call__(self, w):
        self.binary(w)

    def binary(self, w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
            sigma = w.weight.data.std()
            w.weight.data = torch.sign(w.weight.data) * sigma


class KaimingNormal(Initialization):
    def __call__(self, w):
        self.kaiming_normal(w)

    def kaiming_normal(self, w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)


class Default(Initialization):
    def __call__(self, w):
        pass


class ScaledKaimingNormal(Initialization):
    def __init__(self, density, gain=1):
        self.density = density
        self.gain = gain

    def __call__(self, w):
        self.scaled_kaiming_normal(w)

    def scaled_kaiming_normal(self, w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            fan = nn.init._calculate_correct_fan(w.weight, mode='fan_in')
            fan = fan * self.density
            gain = self.gain
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                w.weight.data.normal_(0, std)


class KaimingUniform(Initialization):
    def __call__(self, w):
        self.kaiming_uniform(w)

    def kaiming_uniform(self, w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(w.weight)


class Orthogonal(Initialization):
    def __init__(self, sigma_w=1, sigma_b=0, conv_init_type='conv_orthogonal'):
        self.sigma_w=sigma_w
        self.sigma_b = sigma_b
        self.conv_init_type = conv_init_type

    def __call__(self, w):
        with torch.no_grad():
            self.orthogonal(w)

    def orthogonal(self, w):
        if isinstance(w, torch.nn.Linear):
            torch.nn.init.orthogonal_(w.weight, gain=self.sigma_w)
            if w.bias is not None:
                if self.sigma_b == 0:
                    torch.nn.init.constant_(w.bias, self.sigma_b)
                else:
                    torch.nn.init.normal_(w.bias, std=self.sigma_b)
            
        if isinstance(w, torch.nn.Conv2d):
            if self.conv_init_type == 'conv_orthogonal':
                conv_orthogonal(w.weight, gain=self.sigma_w)
            elif self.conv_init_type == 'delta_orthogonal':
                if w.weight.size(1) > w.weight.size(0):
                    orthogonal_q = orthogonal_matrix(w.weight.transpose(0,1)).transpose(0,1)
                else:
                    orthogonal_q = orthogonal_matrix(w.weight)
                delta_orthogonal(w.weight, orthogonal_q, gain=self.sigma_w)
            else:
                raise NotImplementedError()
            
            if w.bias is not None:
                if self.sigma_b == 0:
                    torch.nn.init.constant_(w.bias, self.sigma_b)
                else:
                    torch.nn.init.normal_(w.bias, std=self.sigma_b)


class PlusKaimingNormal(Initialization):
    def __init__(self, gain=1):
        self.gain=gain

    def __call__(self, w):
        self.plus_kaiming_normal(w)

    def plus_kaiming_normal(self, w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            fan = nn.init._calculate_correct_fan(w.weight, mode='fan_in')
            gain = self.gain
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                w.weight.data.normal_(1, std)


class VGGInitialization(Initialization):
    def __call__(self, w):
        self.vgg_initialize_weights(w)

    def vgg_initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class ScaledBimodalVGGInitialization(Initialization):
    def __init__(self,  prob, seed=None, gain=1):
        self.prob = prob
        self.seed = seed
        self.gen = np.random.default_rng(seed)
        self.gain = gain


    def __call__(self, w):
        self.scaled_binomial_vgg_initialize_weights(w)

    def scaled_binomial_vgg_initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            print(n)
            n = nn.init._calculate_correct_fan(m.weight, mode='fan_in')
            print(n)

            gain = self.gain
            squared_gain = gain ** 2

            m1 = - math.sqrt(1.5 / n)
            m2 = + math.sqrt(1.5 / n)

            rho = m1 ** 2 + m2 ** 2
            std = math.sqrt(squared_gain / n - 0.5 * rho)

            with torch.no_grad():
                idx = self.gen.binomial(1, self.prob, size=m.weight.data.shape)
                part1 = self.gen.normal(m1, std, size=m.weight.data.shape)
                part2 = self.gen.normal(m2, std, size=m.weight.data.shape)

                idx = torch.Tensor(idx).to(bool).to(m.weight.data.device)
                part1 = torch.Tensor(part1).to(m.weight.data.device)
                part2 = torch.Tensor(part2).to(m.weight.data.device)

                m.weight.data  = torch.where(idx, m.weight.data, part1)
                m.weight.data  = torch.where(~idx, m.weight.data, part2)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class ScaledBimodalKaimingNormal(Initialization):
    def __init__(self, prob, seed=None, gain=1):
        self.prob = prob
        self.seed = seed
        self.gain = gain
        self.gen = np.random.default_rng(seed)

    def __call__(self, w):
        self.bimodal_kaiming_normal(w)

    def bimodal_kaiming_normal(self, w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            fan = nn.init._calculate_correct_fan(w.weight, mode='fan_in')
            gain = self.gain
            squared_gain = gain**2

            m1 = - math.sqrt(1.5/fan)
            m2 = + math.sqrt(1.5/fan)

            rho = m1**2 + m2**2
            std = math.sqrt(squared_gain/fan - 0.5*rho)

            with torch.no_grad():
                idx = self.gen.binomial(1, self.prob, size=w.weight.data.shape)
                part1 =  self.gen.normal(m1, std, size=w.weight.data.shape)
                part2 = self.gen.normal(m2, std, size=w.weight.data.shape)

                idx = torch.Tensor(idx).to(bool).to(w.weight.data.device)
                part1 = torch.Tensor(part1).to(w.weight.data.device)
                part2 = torch.Tensor(part2).to(w.weight.data.device)

                w.weight.data = torch.where(idx, w.weight.data, part1)
                w.weight.data = torch.where(~idx, w.weight.data, part2)


class BimodalKaimingNormal(Initialization):

    def __init__(self, prob, seed=None, m1=1, m2=-1, gain=1):
        self.prob = prob
        self.seed = seed
        self.gen = np.random.default_rng(seed)
        self.m1 = m1
        self.m2 = m2
        self.gain=1

    def __call__(self, w):
        self.bimodal_kaiming_normal(w)

    def bimodal_kaiming_normal(self, w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            fan = nn.init._calculate_correct_fan(w.weight, mode='fan_in')
            gain = self.gain
            std = gain / math.sqrt(fan)

            with torch.no_grad():
                idx = self.gen.binomial(1, self.prob, size=w.weight.data.shape)
                part1 = self.gen.normal(self.m1, std, size=w.weight.data.shape)
                part2 = self.gen.normal(self.m2, std, size=w.weight.data.shape)

                idx = torch.Tensor(idx).to(bool).to(w.weight.data.device)
                part1 = torch.Tensor(part1).to(w.weight.data.device)
                part2 = torch.Tensor(part2).to(w.weight.data.device)

                torch.where(idx, w.weight.data, part1)
                torch.where(~idx, w.weight.data, part2)


def get_id(n, m):
    A = np.eye(n, m)
    if n < m:
        A[:, n:] = A[:, :(m - n)]
    elif n > m:
        A[m:, :] = A[:(n - m), :]
    return A

def apply_givens(gen, A, i, j):
    theta = gen.uniform() * 2 * np.pi
    c = np.cos(theta)
    s = np.sin(theta)
    A[:, i], A[:, j] = A[:, i] * c + A[:, j] * s, A[:, i] * (-s) + A[:, j] * c
    return A

def get_col(gen, n, m):
    if m is None:
        m = n
    A = get_id(n, m)
    A1 = [(i, i + 1) for i in range(1, n - 1, 2)]
    A2 = [(i, i + 1) for i in range(0, n - 1, 2)]
    Gtype = A1 + A2
    for pair in Gtype:
        A = apply_givens(gen, A, pair[0], pair[1])
    print((A != 0).sum() / (n * m))
    return A

def max_sparse_ortho(gen, n, m=None):
    P = gen.permutation(n)
    if n <= m:
        return get_col(gen, n, m)[P]
    else:
        return get_col(gen, m, n).T[P]

def max_sparse_ortho_copy(gen, n, m=None):
    P = gen.permutation(n)
    if n <= m:
        data = get_col(gen, n, m)
        data[:, n:] = data[:, :(m-n)]
        data = data[P]
    else:
        data = get_col(gen, m, n).T
        data[m:, :] = data[:(n-m), :]
        data = data[P]
    return data

class MaxSparseOrtho(Initialization):

    def __init__(self, seed=None, func="eye"):
        self.seed = seed
        self.gen = np.random.default_rng(seed)
        self.func = func

    def __call__(self, w):
        self.max_sparse_orthogonal(w)

    def max_sparse_orthogonal(self, w):
        if self.func == "eye":
            f_ortho = max_sparse_ortho
        elif self.func == "copy":
            f_ortho = max_sparse_ortho_copy
        else:
            raise ValueError("Unknown function")
        if isinstance(w, torch.nn.Linear):
            n, m = w.weight.shape
            with torch.no_grad():
                w.weight.data = torch.Tensor(f_ortho(self.gen, n, m)).to(w.weight.data.device)
        elif isinstance(w, torch.nn.Conv2d):
            rows = w.weight.size(0)
            cols = w.weight.numel() // rows
            cols = int(cols)
            data = torch.Tensor(f_ortho(self.gen, rows, cols)).to(w.weight.data.device)
            w.weight.data.view_as(data).copy_(data)


def initializations(init_type, density, args,  seed=None, prob=0.5):
    if init_type == 'binary':
        return Binary()
    elif init_type == 'kaiming_normal':
        return KaimingNormal()
    elif init_type == 'scaled_kaiming_normal':
        return ScaledKaimingNormal(density, gain=args.sigma_w)
    elif init_type == 'kaiming_uniform':
        return KaimingUniform()
    elif init_type == 'orthogonal':
        return Orthogonal(args.sigma_w, args.sigma_b)
    elif init_type == 'conv_orthogonal' or init_type == 'delta_orthogonal':
        return Orthogonal(args.sigma_w, args.sigma_b, conv_init_type=init_type)
    elif init_type == 'bimodal_kaiming_normal':
        return BimodalKaimingNormal(prob, seed, gain=args.sigma_w)
    elif init_type == 'plus_kaiming_normal':
        return PlusKaimingNormal(gain=args.sigma_w)
    elif init_type == 'scaled_bimodal_kaiming_normal':
        return ScaledBimodalKaimingNormal(prob, seed, gain=args.sigma_w)
    elif init_type == "vgg":
        return VGGInitialization()
    elif init_type == "scaled_bimodal_vgg":
        return ScaledBimodalVGGInitialization(prob, seed, gain=args.sigma_w)
    elif init_type == "max_sparse_orthogonal":
        return MaxSparseOrtho(seed=seed)
    elif init_type == "max_sparse_orthogonal_copy":
        return MaxSparseOrtho(seed=seed, func="copy")
    elif init_type == "default":
        return Default()
    else:
        raise ValueError("Unknown Initialization")

