import torch
import math


class ReconstructionMultiSubject(torch.nn.Module):
    """Building Block for spherical harmonic convolution with a polar filter
    """

    def __init__(self, polar_filter_equi, polar_filter_inva, train_rf):
        """Initialization.
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
            polar_filter_inva (:obj:`torch.Tensor`): [in_channel x S x 1] Polar filter spherical harmonic coefficients
        """
        super(ReconstructionMultiSubject, self).__init__()
        print('-'*10, ' Create Reconstruction ', '-'*10)
        print(f'Equi polar filter: {polar_filter_equi.shape if polar_filter_equi is not None else None}')
        print(f'Inva polar filter: {polar_filter_inva.shape if polar_filter_inva is not None else None}')
        print(f'Train rf: {train_rf}')
        if polar_filter_equi is None:
            self.equi = False
        else:
            self.conv_equi = IsoSHConv(polar_filter_equi, train_rf)
            self.equi = True

        if polar_filter_inva is None:
            self.inva = False
        else:
            self.conv_inv = IsoSHConv(polar_filter_inva, train_rf)
            self.inva = True

    def forward(self, x_equi_shc, x_inva_shc, output_shc_to_signal):
        """Forward pass.
        Args:
            x_equi_shc (:obj:`torch.tensor`): [B x equi_channel x C x X x Y x Z] Signal spherical harmonic coefficients.
            x_inva_shc (:obj:`torch.tensor`): [B x inva_channel x 1 x X x Y x Z] Signal spherical harmonic coefficients.
            output_shc_to_signal (:obj:`torch.tensor`): [B x S_out x C_out x V_grad_out] Matrix to compute output signal.
        Returns:
            :obj:`torch.tensor`: [B x V_grad_out x X x Y x Z] Reconstruction of the signal
        """
        x_convolved_equi, x_convolved_inva = 0, 0
        if self.equi:
            x_convolved_equi_shc = self.conv_equi(x_equi_shc) # B x equi_channel x S x C x X x Y x Z
            n_min_shc = min(x_convolved_equi_shc.shape[3], output_shc_to_signal.shape[2])
            x_convolved_equi_shc = x_convolved_equi_shc[:, :, :, :n_min_shc].reshape(*x_convolved_equi_shc.shape[:2], -1, *x_convolved_equi_shc.shape[4:]) # B x equi_channel x S*C x X x Y x Z
            x_convolved_equi = torch.einsum('btoxyz,bov->btvxyz', x_convolved_equi_shc, output_shc_to_signal[:, :, :n_min_shc].reshape(output_shc_to_signal.shape[0], -1, output_shc_to_signal.shape[-1])) # B x equi_channel x V x X x Y x Z
            x_convolved_equi = torch.sum(x_convolved_equi, axis=1) # B x V x X x Y x Z
        if self.inva:
            x_convolved_inva_shc = self.conv_inv(x_inva_shc)[:, :, :, 0] # B x inva_channel x S x X x Y x Z
            x_convolved_inva = torch.einsum('btoxyz,bov->btvxyz', x_convolved_inva_shc, output_shc_to_signal[:, :, 0]) # B x inva_channel x V x X x Y x Z
            x_convolved_inva = torch.sum(x_convolved_inva, axis=1) # B x V x X x Y x Z
        # Get reconstruction
        x_reconstructed =  x_convolved_equi + x_convolved_inva

        return x_reconstructed


class IsoSHConv(torch.nn.Module):
    """Building Block for spherical harmonic convolution with a polar filter
    """

    def __init__(self, polar_filter, learnable=False):
        """Initialization.
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
        where in_channel is the number of tissue, S is the number of shell and L is the number of odd spherical harmonic order
        C is the number of coefficients for the L odd spherical harmonic order
        """
        super(IsoSHConv, self).__init__()
        # LEARNABLE POLAR FILTER
        # Number of coefficients
        L = polar_filter.shape[2]
        # Scale by sqrt(4*pi/4*l+1))
        scale = torch.Tensor([math.sqrt(4*math.pi/(4*l+1)) for l in range(L)])[None, None, :] # 1 x 1 x L
        # Repeat each coefficient 4*l+1 times
        repeat = torch.Tensor([int(4*l+1) for l in range(L)]).type(torch.int64) # L
        # Learnable polar filter
        self.register_buffer('scale', scale) # 1 x 1 x L
        self.register_buffer('repeat', repeat) # L
        if learnable:
            self.polar_filter = torch.nn.parameter.Parameter(polar_filter) #  in_channel x S x L
        else:
            self.register_buffer("polar_filter", polar_filter) #  in_channel x S x L

    def forward(self, x):
        """Forward pass.
        Args:
            x (:obj:`torch.tensor`): [B x in_channel x C x X x Y x Z] Signal spherical harmonic coefficients.
        Returns:
            :obj:`torch.tensor`: [B x in_channel x S x C x X x Y x Z] Spherical harmonic coefficient of the output
        """        
        filter = self.scale*self.polar_filter # in_channel x S x L
        filter = filter.repeat_interleave(self.repeat, dim=2) # in_channel x S x C
        n_min_shc = min(x.shape[2], filter.shape[2])
        x = x[:, :, None, :n_min_shc]*filter[None, :, :, :n_min_shc, None, None, None] # B x in_channel x S x C x X x Y x Z
        return x