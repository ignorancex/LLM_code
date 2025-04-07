import torch
import numpy as np
from torch import nn
from spatial_regularisation.utils import add_axis


class SpatialMoments(nn.Module):

    def __init__(self,
                 input_shape,
                 return_variance=False,
                 variance_type='anisotropic',
                 make_channels_probabilistic=False,
                 **kwargs):
        """This function takes in a NON-NEGATIVE torch tensor of shape [B, C, *], where * is the field of view.
        The function returns the spatial moments of orders 1 (center of mass) and optionally order 2 (covariance matrix)
        of each channel.
        Output tensors have shape [B, C, 3] for centers of mass and [B, C] (isotropic) or [B, C, n_dims, n_dims]
        (anisotropic) for order 2 moments.
        :param input_shape: list representing the shape of inputs (without batch dimension) [C, *]
        :param return_variance: whether to return an additional output tensor with the variance/covariance matrix of
        each input channel. The shape of the returned variance depends on the type (see variance_type below).
        :param variance_type: one of ['isotropic', 'anisotropic']. If isotropic, we return tensor [B, C] with only one
        variance for all dimensions. If anisotropic, we return a covariance matrix of size [B, C, n_dims, n_dims].
        :param make_channels_probabilistic: whether to make probabilistic each channel of the input tensor (i.e. values
        in [0, 1] and sum to 1).
        """

        super(SpatialMoments, self).__init__(**kwargs)

        # initialisation
        self.size_in = input_shape  # this doesn't include the batch size
        self.return_variance = return_variance
        self.variance_type = variance_type
        self.make_channels_probabilistic = make_channels_probabilistic

        # get input shape
        self.shape_non_chan = input_shape[1:]
        self.n_chan = input_shape[0]

        # build meshgrid of size [1, 1, n_dims, H*W*D]
        self.coord_idx_list = torch.meshgrid(*[torch.arange(0, ss) for ss in self.shape_non_chan])
        self.coord_idx_list = [torch.reshape(ten, [-1]) for ten in self.coord_idx_list]
        self.coord_idxs = add_axis(torch.stack(self.coord_idx_list), [0, 0])  # [1, 1, n_dims, H*W*D]

    def forward(self, x):

        self.coord_idxs.to(x.device)

        # reshape the tensor
        x = torch.reshape(x, [-1, self.n_chan, np.prod(self.shape_non_chan)])  # [1, C, H*W*D]
        x = add_axis(x, axis=2)  # [B, C, 1, H*W*D]

        # renormalise if necessary
        if self.make_channels_probabilistic:
            x /= (x.sum(dim=-1, keepdim=True) + 1e-12)

        # get mean coordinates, weighted by the tensor values
        means_by_channel = torch.sum(x * self.coord_idxs, dim=-1)  # [B, C, n_dims]

        if not self.return_variance:
            return means_by_channel  # [B, C, n_dims]

        else:
            if self.variance_type == 'isotropic':
                var_by_channel = torch.sum(x * (self.coord_idxs - add_axis(means_by_channel, -1)) ** 2, dim=-1)  # [B, C, n_dims]
                var_by_channel = torch.mean(var_by_channel, dim=-1)  # [B, C]
            else:
                n_dims = means_by_channel.shape[-1]
                var_by_channel = torch.stack([torch.zeros_like(means_by_channel)] * n_dims, dim=-1)
                means = add_axis(means_by_channel, -1)
                for i in range(n_dims):
                    for j in range(i, n_dims):
                        tmp_var = ((self.coord_idxs[:, :, i, :] - means[:, :, i, :]) *
                                   (self.coord_idxs[:, :, j, :] - means[:, :, j, :]))
                        tmp_var = torch.sum(x[:, :, 0, :] * tmp_var, dim=-1)
                        var_by_channel[:, :, i, j] = tmp_var
                        if i != j:
                            var_by_channel[:, :, j, i] = tmp_var
            return means_by_channel, var_by_channel  # [B, C, n_dims]  [B, C, n_dims, n_dims]

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.coord_idxs = self.coord_idxs.to(*args, **kwargs)
        for idx in range(len(self.coord_idx_list)):
            self.coord_idx_list[idx].to(*args, **kwargs)
        return self
