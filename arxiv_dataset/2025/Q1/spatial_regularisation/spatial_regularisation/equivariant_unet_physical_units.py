import math
import torch
import numpy as np
from torch import nn
from functools import partial
import torch.nn.functional as F

from e3nn.nn import BatchNorm, Gate
from e3nn.math import soft_unit_step
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, Linear, spherical_harmonics


# class EquivUNet(SegmentationNetwork):
class EquivUNet(nn.Module):

    def __init__(self,
                 irreps_in,
                 irreps_out,
                 steps,
                 n_levels=2,
                 feat_mult=2,
                 kernel_size=5,
                 activation='relu',
                 last_activation='softmax',
                 normalization='instance',
                 lmax=2,
                 scale=2,
                 return_fmaps=False):
        """Equivariant UNet with physical units

        Parameters
        ----------
        irreps_in : str
            input representations
            example: "1x0e" when one channel of scalar values
        irreps_out : str
            output representations
            example: "4x0e" when four channels of scalar values
        kernel_size : float
            diameter of input convolution kernel in physical units
        steps : sequence
            physical dimension of a pixel in physical units
        normalization : str, optional
            normalization: can be 'batch', 'instance' or 'None'.
            by default 'instance'
        feat_mult : int, optional
            multiplication factor of number of irreps
            between successive convolution blocks, by default 2
        n_levels : int, optional
            number of levels in the UNet
        lmax : int, optional
            maximum spherical harmonics l
            by default 2
        scale : int, optional
            size of pooling diameter
            in physical units, by default 2
        """
        super().__init__()

        self.n_classes_scalar = Irreps(irreps_out).count('0e')
        self.num_classes = self.n_classes_scalar

        self.n_downsample = n_levels - 1
        # self.conv_op = nn.Conv3d  # Needed in order to use nnUnet predict_3D

        self.return_fmaps = return_fmaps

        assert normalization in ['None', 'batch', 'instance'], "batch_norm needs to be 'batch', 'instance', or 'None'"

        if activation == 'relu':
            activation = [torch.relu]
        else:
            raise NotImplementedError

        if last_activation == 'relu':
            self.last_activation = torch.nn.ReLU()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax()
        elif last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.last_activation = None

        irreps_sh = Irreps.spherical_harmonics(lmax, p=1)  # even parity because we're not considering reflexions
        diameters = [kernel_size * 2 ** i for i in range(self.n_downsample + 1)]
        scales = [scale * 2 ** i for i in range(self.n_downsample)]

        steps_array = [steps]
        for i in range(self.n_downsample):
            output_steps = []
            for step in steps:
                if step < scales[i]:
                    kernel_dim = math.floor(scales[i] / step)
                    output_steps.append(kernel_dim * step)
                else:
                    output_steps.append(step)
            steps_array.append(tuple(output_steps))

        self.down = EquivDown(n_downsample=self.n_downsample,
                              activation=activation,
                              irreps_sh=irreps_sh,
                              ne=feat_mult,
                              no=0,  # even parity because we're not considering reflexions
                              normalization=normalization,
                              irreps_in=irreps_in,
                              diameters=diameters,
                              steps=steps_array,
                              scale=scales)

        self.up = EquivUp(n_blocks_up=self.n_downsample,
                          activation=activation,
                          irreps_sh=irreps_sh,
                          ne=feat_mult * 2 ** (self.n_downsample - 1),
                          no=0,  # even parity because we're not considering reflexions
                          normalization=normalization,
                          irreps_downblock=self.down.down_irreps_out,
                          diameters=diameters[::-1][1:],
                          steps=steps_array[::-1][1:],
                          scale=scales[::-1],
                          return_fmaps=return_fmaps)
        self.out = EquivConvolutionBlock(irreps_in=self.up.up_blocks[-1].irreps_out,
                                         irreps_hidden=Irreps(irreps_out),
                                         activation=activation,
                                         irreps_sh=irreps_sh,
                                         normalization=normalization,
                                         diameter=kernel_size,
                                         steps=steps,
                                         transpose=False)

    def forward(self, x):
        if self.return_fmaps:
            return self.forward_fmaps(x)

        pad = self.pad_size(x.shape[-3:])
        x = torch.nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        down_ftrs = self.down(x)
        x = self.up(down_ftrs[-1], down_ftrs)
        x = self.out(x)

        x = x[..., pad[0]:, pad[1]:, pad[2]:]

        if self.last_activation is not None:
            x = self.last_activation(x)

        return x

    def forward_fmaps(self, x):

        pad = self.pad_size(x.shape[-3:])
        x = torch.nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        down_ftrs = self.down(x)
        up_ftrs = self.up(down_ftrs[-1], down_ftrs)
        x = self.out(up_ftrs[-1])

        if self.last_activation is not None:
            x = self.last_activation(x)

        fmaps = down_ftrs + up_ftrs + [x]
        fmaps = [f[..., pad[0]:, pad[1]:, pad[2]:] for f in fmaps]

        return fmaps

    def pad_size(self, image_shape, odd=False):

        pooling_factor = np.ones(3, dtype='int')
        for pool in self.down.down_pool:
            pooling_factor *= np.array(pool.kernel_size)

        pad = []
        for f, s in zip(pooling_factor, image_shape):
            p = 0  # padding for current dimension
            if odd:
                t = (s - 1) % f
            else:
                t = s % f

            if t != 0:
                p = f - t
            pad.append(p)

        return pad


class EquivDown(nn.Module):

    def __init__(self,
                 n_downsample,
                 activation,
                 irreps_sh,
                 ne,
                 no,
                 normalization,
                 irreps_in,
                 diameters,
                 steps,
                 scale):
        super().__init__()

        blocks = []
        self.down_irreps_out = []
        for n in range(n_downsample + 1):
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {2*no}x1o + {ne}x2e + {no}x2o").simplify()
            block = EquivConvolutionBlock(irreps_in=irreps_in,
                                          irreps_hidden=irreps_hidden,
                                          activation=activation,
                                          irreps_sh=irreps_sh,
                                          normalization=normalization,
                                          diameter=diameters[n],
                                          steps=steps[n],
                                          transpose=False)
            blocks.append(block)
            self.down_irreps_out.append(block.irreps_out)
            irreps_in = block.irreps_out
            ne *= 2
            no *= 2
        self.down_blocks = nn.ModuleList(blocks)

        pooling = []
        for n in range(n_downsample):
            pooling.append(EquivDynamicPool3d(scale=scale[n],
                                              steps=steps[n],
                                              mode='maxpool3d',
                                              irreps=self.down_irreps_out[n]))
        self.down_pool = nn.ModuleList(pooling)

    def forward(self, x):
        features = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features.append(x)
            if i < len(self.down_blocks) - 1:
                x = self.down_pool[i](x)
        return features


class EquivUp(nn.Module):
    def __init__(self,
                 n_blocks_up,
                 activation,
                 irreps_sh,
                 ne,
                 no,
                 normalization,
                 irreps_downblock,
                 diameters,
                 steps,
                 scale,
                 return_fmaps):
        super().__init__()

        self.n_blocks_up = n_blocks_up
        self.return_fmaps = return_fmaps

        irreps_in = irreps_downblock[-1]
        blocks = []
        upsample_op = []
        for n in range(n_blocks_up):
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {ne}x2e + {2*no}x1o + {no}x2o").simplify()
            block = EquivConvolutionBlock(irreps_in=irreps_in + irreps_downblock[::-1][n + 1],
                                          irreps_hidden=irreps_hidden,
                                          activation=activation,
                                          irreps_sh=irreps_sh,
                                          normalization=normalization,
                                          diameter=diameters[n],
                                          steps=steps[n],
                                          transpose=True)
            blocks.append(block)
            irreps_in = block.irreps_out
            ne //= 2
            no //= 2

            # same as pooling kernel
            upsample_scale_factor = tuple([math.floor(scale[n] / step) if step < scale[n] else 1 for step in steps[n]])
            upsample_op.append(nn.Upsample(scale_factor=upsample_scale_factor, mode='trilinear', align_corners=True))

        self.up_blocks = nn.ModuleList(blocks)
        self.upsample_ops = nn.ModuleList(upsample_op)

    def forward(self, x, down_features):
        if self.return_fmaps:
            fmaps = []
            for i in range(self.n_blocks_up):
                x = self.upsample_ops[i](x)
                x = torch.cat([x, down_features[::-1][i + 1]], dim=1)
                x = self.up_blocks[i](x)
                fmaps.append(x)
            return fmaps

        else:
            for i in range(self.n_blocks_up):
                x = self.upsample_ops[i](x)
                x = torch.cat([x, down_features[::-1][i + 1]], dim=1)
                x = self.up_blocks[i](x)
            return x


class EquivDynamicPool3d(torch.nn.Module):
    def __init__(self, scale, steps, mode, irreps):
        super().__init__()

        self.scale = scale  # in physical units
        self.steps = steps
        self.mode = mode
        self.kernel_size = tuple([math.floor(self.scale / step) if step < self.scale else 1 for step in self.steps])
        self.irreps = irreps

    def forward(self, x):

        if self.mode == 'average':
            out = F.avg_pool3d(x, self.kernel_size, stride=self.kernel_size)

        # e3nn max_pool3d implementation
        elif self.mode == 'maxpool3d':

            assert x.shape[1] == self.irreps.dim, "Shape mismatch"

            cat_list = []
            start = 0
            for i in self.irreps.ls:
                end = start + 2 * i + 1
                temp = x[:, start:end, ...]
                if i == 0:
                    pooled, indices = F.max_pool3d_with_indices(temp[:, 0, ...],
                                                                self.kernel_size,
                                                                stride=self.kernel_size,
                                                                return_indices=True)
                    cat_list.append(pooled)
                else:
                    pooled, indices = F.max_pool3d_with_indices(temp.norm(dim=1),
                                                                self.kernel_size,
                                                                stride=self.kernel_size,
                                                                return_indices=True)
                    for tensor_slice in range(2 * i + 1):
                        pooled = temp[:, tensor_slice, ...].flatten()[indices]
                        cat_list.append(pooled)
                start = end
            out = torch.stack(tuple(cat_list), dim=1)

        else:
            raise ValueError("Unknown mode '{}'".format(self.mode))

        return out


class EquivConvolutionBlock(nn.Module):
    def __init__(self,
                 irreps_in,
                 irreps_hidden,
                 activation,
                 irreps_sh,
                 normalization,
                 diameter,
                 steps,
                 transpose):
        super().__init__()

        if normalization == 'batch':
            BN = BatchNorm
        elif normalization == 'instance':
            BN = partial(BatchNorm, instance=True)
        else:
            BN = None

        irreps_scalars = Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l > 0])
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid]

        self.gate1 = Gate(irreps_scalars=irreps_scalars,
                          act_scalars=activation,
                          irreps_gates=irreps_gates,
                          act_gates=activation_gate,
                          irreps_gated=irreps_gated)
        self.conv1 = Convolution(irreps_in=irreps_in,
                                 irreps_out=self.gate1.irreps_in,
                                 irreps_sh=irreps_sh,
                                 diameter=diameter,
                                 num_radial_basis=diameter,
                                 steps=steps,
                                 transpose=transpose)
        self.batchnorm1 = BN(self.gate1.irreps_in) if BN is not None else None

        self.gate2 = Gate(irreps_scalars=irreps_scalars,
                          act_scalars=activation,
                          irreps_gates=irreps_gates,
                          act_gates=activation_gate,
                          irreps_gated=irreps_gated)
        self.conv2 = Convolution(irreps_in=self.gate1.irreps_out,
                                 irreps_out=self.gate2.irreps_in,
                                 irreps_sh=irreps_sh,
                                 diameter=diameter,
                                 num_radial_basis=diameter,
                                 steps=steps,
                                 transpose=transpose)
        self.batchnorm2 = BN(self.gate2.irreps_in) if BN is not None else None

        self.irreps_out = self.gate2.irreps_out

    def forward(self, x):

        x = self.conv1(x)
        if self.batchnorm1 is not None:
            x = self.batchnorm1(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate1(x.transpose(1, 4)).transpose(1, 4)

        x = self.conv2(x)
        if self.batchnorm2 is not None:
            x = self.batchnorm2(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate2(x.transpose(1, 4)).transpose(1, 4)

        return x


class Convolution(torch.nn.Module):
    """convolution on voxels

    Parameters
    ----------
    irreps_in : `Irreps`
        input irreps

    irreps_out : `Irreps`
        output irreps

    irreps_sh : `Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``

    diameter : float
        diameter of the filter in physical units

    num_radial_basis : int
        number of radial basis functions

    steps : tuple of float
        size of the pixel in physical units
    """

    def __init__(self,
                 irreps_in,
                 irreps_out,
                 irreps_sh,
                 diameter,
                 num_radial_basis,
                 steps=(1.0, 1.0, 1.0),
                 cutoff=False,
                 transpose=False,
                 **kwargs):
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_sh = Irreps(irreps_sh)

        self.num_radial_basis = num_radial_basis
        self.transpose = transpose

        # self-connection
        self.sc = Linear(self.irreps_in, self.irreps_out)  # first transform each input irrep to same space as output

        # connection with neighbors
        r = diameter / 2

        s = math.floor(r / steps[0])
        x = torch.arange(-s, s + 1.0) * steps[0]

        s = math.floor(r / steps[1])
        y = torch.arange(-s, s + 1.0) * steps[1]

        s = math.floor(r / steps[2])
        z = torch.arange(-s, s + 1.0) * steps[2]

        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]
        self.register_buffer('lattice', lattice)

        if 'padding' not in kwargs:
            kwargs['padding'] = tuple(s // 2 for s in lattice.shape[:3])
        self.kwargs = kwargs

        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1),  # [x, y, z]
            start=0.0,
            end=r,
            number=self.num_radial_basis,
            basis='smooth_finite',
            cutoff=cutoff,
        )  # [x, y, z, B] for smooth finite, B is 5
        self.register_buffer('emb', emb)

        sh = spherical_harmonics(
            l=self.irreps_sh,  # 1x0e, 1x1e, 1x2e (why do we use e for l=1?s) this should be 4 I think?
            x=lattice,
            normalize=True,
            normalization='component'
        )  # [x, y, z, irreps_sh.dim]
        self.register_buffer('sh', sh)  # [x,y,z,9]

        self.tp = FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False,
                                              compile_right=True)

        self.weight = torch.nn.Parameter(torch.randn(self.num_radial_basis, self.tp.weight_numel))

    def kernel(self):
        weight = self.emb @ self.weight  # [s, s, s, N], N learned radial kernels
        weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])  # normalize... why?
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim]
        kernel = torch.einsum('xyzio->oixyz', kernel)  # [irreps_out.dim, irreps_in.dim, x, y, z]
        return kernel

    def forward(self, x):
        """
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, irreps_in.dim, x, y, z)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, irreps_out.dim, x, y, z)``
        """
        sc = self.sc(x.transpose(1, 4)).transpose(1, 4)

        if self.transpose:
            out = sc + torch.nn.functional.conv_transpose3d(x, self.kernel().transpose(0, 1), **self.kwargs)
        else:
            out = sc + torch.nn.functional.conv3d(x, self.kernel(), **self.kwargs)

        return out



def soft_one_hot_linspace(x: torch.Tensor, start, end, number, basis=None, cutoff=None):
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    math::

        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    math::

        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    See the last plot below.
    Note that ``bessel`` basis cannot be normalized.

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    start : float
        minimum value span by the basis

    end : float
        maximum value span by the basis

    number : int
        number of basis functions :math:`N`

    basis : {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}
        choice of basis family; note that due to the :math:`1/x` term, ``bessel`` basis does not satisfy the normalization of other basis choices

    cutoff : bool, string
        if ``cutoff=True`` then for all :math:`x` outside the interval defined by ``(start, end)``, :math:`\forall i, \; f_i(x) \approx 0`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., N)`

    Examples
    --------

     jupyter-execute::
        :hide-code:

        import torch
        from e3nn.math import soft_one_hot_linspace
        import matplotlib.pyplot as plt

    jupyter-execute::

        bases = ['gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel']
        x = torch.linspace(-1.0, 2.0, 100)

    jupyter-execute::

        fig, axes = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axes, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(-1, 1.5)
        plt.tight_layout()

    jupyter-execute::

        fig, axes = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axes, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c).pow(2).sum(1))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(0, 2)
        plt.tight_layout()
    """
    # pylint: disable=misplaced-comparison-constant

    if cutoff not in [True, False, 'left', 'right']:
        raise ValueError("cutoff must be specified: True, False, 'left', 'right'")

    if not cutoff:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
        step = values[1] - values[0] # [0, 0.625, 1.25, 1.875, 2.5]
    elif cutoff == 'left':
        values = torch.linspace(start, end, number + 1, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:]
    elif cutoff == 'right':
        values = torch.linspace(start, end, number + 1, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[:-1]
    else: #cutoff == True
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step # shape [5,5,5,5]

    if basis == 'gaussian':
        return diff.pow(2).neg().exp().div(1.12)

    if basis == 'cosine':
        return torch.cos(math.pi/2 * diff) * (diff < 1) * (-1 < diff)

    if basis == 'smooth_finite':
        output = 1.14136 * torch.exp(torch.tensor(2.0)) * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)
        return output

    if basis == 'fourier':
        x = (x[..., None] - start) / (end - start)
        if not cutoff:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
        elif cutoff == 'left':
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x)
        elif cutoff == 'right':
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (x < 1)
        else: #cutoff == True
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)

    if basis == 'bessel':
        x = x[..., None] - start
        c = end - start
        bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
        out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

        if not cutoff:
            return out
        elif cutoff == 'left':
            return out * (0 < x)
        elif cutoff == 'right':
            return out * ((x / c) < 1)
        else:
            return out * ((x / c) < 1) * (0 < x)

    raise ValueError(f"basis=\"{basis}\" is not a valid entry")
