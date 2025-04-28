import torch
from .deconvolution import DeconvolutionMultiSubject
from .reconstruction import ReconstructionMultiSubject


class ModelMultiSubject(torch.nn.Module):
    def __init__(self, graphSampling, polar_filter_equi, polar_filter_inva, feature_in, filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, train_rf=False):
        super(ModelMultiSubject, self).__init__()
        print('-'*50)
        print('-'*10, ' Create Model ', '-'*10)
        if not (polar_filter_equi is None):
            n_equi = polar_filter_equi.shape[0]
        else:
            n_equi = 0
        if not (polar_filter_inva is None):
            n_inva = polar_filter_inva.shape[0]
        else:
            n_inva = 0
        self.deconvolution = DeconvolutionMultiSubject(graphSampling, feature_in, n_equi, n_inva, filter_start, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, normalize)
        self.reconstruction = ReconstructionMultiSubject(polar_filter_equi, polar_filter_inva, train_rf)
        print('-'*50)

    def forward(self, intput_features, input_b0, input_signal_to_shc, output_shc_to_signal):
        """Forward Pass.
        Args:
            intput_features (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V_grad_in x X x Y x Z]
            input_b0 (:obj:`torch.Tensor`): Input B0 for fODF normalization [B x X x Y x Z]
            input_signal_to_shc (:obj:`torch.Tensor`): Matrix to compute input SHC. [B x V_grad_in x S_in x C_in]
            output_shc_to_signal (:obj:`torch.Tensor`): Matrix to compute output signal. [B x S_out x C_out x V_grad_out]
        Returns:
            :obj:`torch.Tensor`: output [B x V' x X x Y x Z]
            :obj:`torch.Tensor`: output [B x out_channels_equi x C x X x Y x Z]
            :obj:`torch.Tensor`: output [B x out_channels_inva x 1 x X x Y x Z]
        """
        # Deconvolve the signal and get the spherical harmonic coefficients
        if self.reconstruction.equi:
            rf_equi_shc_0 = self.reconstruction.conv_equi.polar_filter[:, 0, 0]
        else:
            rf_equi_shc_0 = None
        if self.reconstruction.inva:
            rf_inva_shc_0 = self.reconstruction.conv_inv.polar_filter[:, 0, 0]
        else:
            rf_inva_shc_0 = None
        deconvolved_equi_shc, deconvolved_inva_shc = self.deconvolution(intput_features, input_signal_to_shc, input_b0, rf_equi_shc_0, rf_inva_shc_0) # B x out_channels_equi x C x X x Y x Z, B x out_channels_inva x 1 x X x Y x Z
        # Reconstruct the signal
        reconstructed = self.reconstruction(deconvolved_equi_shc, deconvolved_inva_shc, output_shc_to_signal) # B x V_grad_out x X x Y x Z
        return reconstructed, deconvolved_equi_shc, deconvolved_inva_shc
