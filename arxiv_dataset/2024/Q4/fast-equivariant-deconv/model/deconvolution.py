import torch
import math
from .unet import GraphCNNUnet


class DeconvolutionMultiSubject(torch.nn.Module):
    def __init__(self, graphSampling, feature_in, n_equi, n_inva, filter_start, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, normalize):
        """Separate equivariant and invariant features from the deconvolved model
        Args:
            x (:obj:`torch.Tensor`): input. [B x V x out_channels x X x Y x Z]
            shellSampling (:obj:`sampling.ShellSampling`): Input sampling scheme
            graphSampling (:obj:`sampling.Sampling`): Interpolation grid scheme
            filter_start (int): First intermediate channel (then multiply by 2)
            kernel_sizeSph (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            kernel_sizeSpa (int): Size of the spatial kernel
            n_equi (int): Number of equivariant deconvolved channel
            n_inva (int): Number of invariant deconvolved channel
            normalize (bool): Normalize the output such that the sum of the SHC of order and degree 0 of the deconvolved channels is math.sqrt(4 * math.pi)
        """
        super(DeconvolutionMultiSubject, self).__init__()
        print('-'*10, ' Create Deconvolution ', '-'*10)
        print(f'Equi channels: {n_equi} - Inva channels: {n_inva}')
        print(f'Convolution name: {conv_name}')
        print(f'Normalize: {normalize}')
        print(f'Feature in: {feature_in}')
        print(f'Filter start: {filter_start}')
        print(f'Kernel size spherical: {kernel_sizeSph}')
        print(f'Kernel size spatial: {kernel_sizeSpa}')
        print(f'Isotropic spatial: {isoSpa}')

        # Convolution name
        self.conv_name = conv_name

        # Interpolation
        if self.conv_name in ['spherical', 'mixed', 'spatial_vec', 'spatial', 'bekkers']:
            self.register_buffer(f'SH2GRID', torch.Tensor(graphSampling.sampling.SH2S))

        # Deconvolution Network
        keepSphericalDim = True
        out_channels = n_equi + n_inva
        if (conv_name in ['spatial', 'spatial_vec', 'spatial_sh']) and (keepSphericalDim):
            out_channels = out_channels * graphSampling.sampling.vectors.shape[0]
        block_depth = 2
        in_depth = 1
        pooling = graphSampling.pooling
        laps = graphSampling.laps
        patch_size_list = graphSampling.patch_size_list
        vecs = graphSampling.vec
        n_vec_out = graphSampling.sampling.vectors.shape[0]
        self.deconvolve = GraphCNNUnet(feature_in, out_channels, filter_start, block_depth, in_depth, kernel_sizeSph, kernel_sizeSpa, pooling, laps, conv_name, isoSpa, keepSphericalDim, patch_size_list, vecs, n_vec_out)

        # Separate Equi and Inva tissues
        self.n_equi = n_equi
        self.n_inva = n_inva

        # Get equivariant symmetric SHC
        if self.n_equi != 0:
            self.register_buffer('GRID2SH', torch.Tensor(graphSampling.sampling.S2SH))

        # fODF Normalization
        self.normalize = normalize
        self.eps = 1e-16

    def separate(self, x):
        """Separate equivariant and invariant features from the deconvolved model
        Args:
            x (:obj:`torch.Tensor`): input. [B x out_channels x V x X x Y x Z]
        Returns:
            x_equi (:obj:`torch.Tensor`): equivariant part of the deconvolution [B x out_channels_equi x V x X x Y x Z]
            x_inva (:obj:`torch.Tensor`): invariant part of the deconvolution [B x out_channels_inva x X x Y x Z]
        """
        if self.n_equi != 0:
            x_equi = x[:, :self.n_equi]
        else:
            x_equi = None
        if self.n_inva != 0:
            x_inva = x[:, self.n_equi:]
            x_inva = torch.max(x_inva, dim=2)[0]
        else:
            x_inva = None
        return x_equi, x_inva

    def norm(self, x_equi, x_inva, b0_voxel, rf_equi_shc_0, rf_inva_shc_0):
        """Separate equivariant and invariant features from the deconvolved model
        Args:
            x_equi (:obj:`torch.Tensor`): shc equivariant part of the deconvolution [B x out_channels_equi x C x X x Y x Z]
            x_inva (:obj:`torch.Tensor`): shc invariant part of the deconvolution [B x out_channels_inva x 1 x X x Y x Z]
        Returns:
            x_equi (:obj:`torch.Tensor`): normed shc equivariant part of the deconvolution [B x out_channels_equi x C x X x Y x Z]
            x_inva (:obj:`torch.Tensor`): normed shc invariant part of the deconvolution [B x out_channels_inva x 1 x X x Y x Z]
        """
        to_norm = 0
        if self.n_equi != 0:
            to_norm = to_norm + torch.sum(x_equi[:, :, 0:1]*rf_equi_shc_0[None, :, None, None, None, None], axis=1, keepdim=True)
        if self.n_inva != 0:
            to_norm = to_norm + torch.sum(x_inva*rf_inva_shc_0[None, :, None, None, None, None], axis=1, keepdim=True)
        to_norm = to_norm / (b0_voxel[:, None] + self.eps)
        to_norm = torch.sqrt(to_norm**2 + self.eps)
        if self.n_equi != 0:
            x_equi = x_equi / to_norm
        if self.n_inva != 0:
            x_inva = x_inva / to_norm
        return x_equi, x_inva


    def forward(self, intput_features, input_signal_to_shc, input_b0, rf_equi_shc_0, rf_inva_shc_0):
        """Forward Pass.
        Args:
            intput_features (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V_grad_in x X x Y x Z]
            input_signal_to_shc (:obj:`torch.Tensor`): Matrix to compute input SHC. [B x V_grad_in x S_in x C_in]
            input_b0 (:obj:`torch.Tensor`): Input B0 for fODF normalization [B x X x Y x Z]
        Returns:
            x_deconvolved_equi_shc (:obj:`torch.Tensor`): SHC equivariant part of the deconvolution [B x out_channels_equi x C x X x Y x Z]
            x_deconvolved_inva_shc (:obj:`torch.Tensor`): SHC invariant part of the deconvolution [B x out_channels_inva x 1 x X x Y x Z]
        """
        # Interpolation of the input signal on the Graph used for the graph convolution
        intput_features = torch.einsum('bfvxyz,bvsc->bfscxyz', intput_features, input_signal_to_shc) # B x in_channels x S_in x C_in x X x Y x Z
        if self.conv_name in ['spherical', 'mixed', 'spatial_vec', 'spatial', 'bekkers']:
            min_n_shc = min(intput_features.shape[3], self.SH2GRID.shape[0])
            intput_features = torch.einsum('bfscxyz,cv->bfsvxyz', intput_features[:, :, :, :min_n_shc], self.SH2GRID[:min_n_shc]) # B x in_channels x S_in x V_grid x X x Y x Z
        intput_features = intput_features.view(intput_features.shape[0], -1, *(intput_features.size()[3:])) # B x in_channels*S_in x Rep_in x X x Y x Z, Rep_in being either C_in or V_grid
        if self.conv_name in ['spatial_vec', 'spatial_sh', 'spatial']:
            intput_features = intput_features.reshape(intput_features.shape[0], -1, 1, *(intput_features.size()[3:])) # B x in_channels*S_in*Rep_in x 1 x X x Y x Z, Rep_in being either C_in or V_grid
        # Deconvolve the input signal (compute the fODFs)
        deconvolved = self.deconvolve(intput_features) # B x out_channels x V_grid x X x Y x Z

        # Separate invariant and equivariant to rotation channels (separate white matter (equivariant) and CSF + gray matter (invariant))
        deconvolved_equi, deconvolved_inva = self.separate(deconvolved) # B x out_channels_equi x V x X x Y x Z, B x out_channels_inva x X x Y x Z
        
        # Symmetrized and get the spherical harmonic coefficients of the equivariant channels
        if self.n_equi != 0:
            x_deconvolved_equi_shc = torch.einsum('bovxyz,vc->bocxyz', deconvolved_equi, self.GRID2SH) # B x out_channels_equi x C x X x Y x Z
        else:
            x_deconvolved_equi_shc = None

        # Get the spherical harmonic coefficients of the invariant channels
        if self.n_inva != 0:
            x_deconvolved_inva_shc = (deconvolved_inva * math.sqrt(4 * math.pi))[:, :, None] # B x out_channels_inva x 1 x X x Y x Z
        else:
            x_deconvolved_inva_shc = None
        
        # Normalize
        if self.normalize:
            x_deconvolved_equi_shc, x_deconvolved_inva_shc = self.norm(x_deconvolved_equi_shc, x_deconvolved_inva_shc, input_b0, rf_equi_shc_0, rf_inva_shc_0)

        return x_deconvolved_equi_shc, x_deconvolved_inva_shc