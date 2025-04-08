# 3p
import torch
import torch.nn as nn


class ASAPDGCNN(nn.Module):
    def __init__(self, dim_in=3, dim_out=128, k=20, emb_dims=1024, p_dropout=0.5, fixed_graph=True, neig=128):
        super().__init__()
        self.k = k
        self.fixed_graph = fixed_graph
        self.neig = neig

        self.spec1 = LaplacianBlock(64, False)
        self.spec2 = LaplacianBlock(64, False)
        self.spec3 = LaplacianBlock(64, False)

        self.conv1 = nn.Sequential(nn.Conv2d(dim_in * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192 * 2, emb_dims, kernel_size=1, bias=False),  # this is changed from 192 to 192 * 2
                                   nn.BatchNorm1d(emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(emb_dims + (64 * 3) * 2, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=p_dropout)
        self.conv9 = nn.Conv1d(256, dim_out, kernel_size=1, bias=False)

    def forward(self, x, mass, evals, evecs):
        # input data
        x = x.transpose(2, 1).contiguous()

        evecs, evals = evecs[:, :, :self.neig], evals[:, :self.neig]
        mass = mass.float()
        evecs_trans = torch.bmm(evecs.transpose(2, 1), torch.diag_embed(mass))

        num_points = x.size(2)

        x, true_idx = get_graph_feature(x, k=self.k)
        x = self.conv1(x)                                 # (batch_size, dim_in*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                                 # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]              # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1_projected = torch.bmm(evecs_trans, x1.transpose(2, 1))
        x1_smooth = torch.bmm(evecs, self.spec1(x1_projected, evals)).transpose(2, 1)
        x1 = torch.cat((x1, x1_smooth), dim=1)            # (batch_size, 64, num_points) -> (batch_size, 128, num_points)

        x, _ = get_graph_feature(x1, k=self.k, idx_unchanged=(true_idx if self.fixed_graph else None))
        #                                                                                               # -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2_projected = torch.bmm(evecs_trans, x2.transpose(2, 1))
        x2_smooth = torch.bmm(evecs, self.spec2(x2_projected, evals)).transpose(2, 1)
        x2 = torch.cat((x2, x2_smooth), dim=1)  # (batch_size, 64, num_points) -> (batch_size, 128, num_points)

        x, _ = get_graph_feature(x2, k=self.k, idx_unchanged=(true_idx if self.fixed_graph else None))
        #                                                                                                  # -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3_projected = torch.bmm(evecs_trans, x3.transpose(2, 1))
        x3_smooth = torch.bmm(evecs, self.spec3(x3_projected, evals)).transpose(2, 1)
        x3 = torch.cat((x3, x3_smooth), dim=1)  # (batch_size, 64, num_points) -> (batch_size, 128, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, emb_dims + 64*3, num_points)
        # x = torch.bmm(evecs, torch.bmm(evecs_trans, x.transpose(2, 1))).transpose(2, 1)

        x = self.conv7(x)                       # (batch_size, emb_dims + 64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, dim_out, num_points)
        # x = torch.bmm(evecs, torch.bmm(evecs_trans, x.transpose(2, 1))).transpose(2, 1)

        # output data
        x = x.transpose(2, 1).contiguous()
        return x


class LaplacianBlock(nn.Module):
    """
    Applies Laplacian powers/diffusion in the spectral domain like
        f_out = lambda_i ^ k * e ^ (lambda_i t) f_in
    with learned per-channel parameters k and t.

    Inputs:
      - values: (K,C) in the spectral domain
      - evals: (K) eigenvalues
    Outputs:
      - (K,C) transformed values in the spectral domain
    """

    def __init__(self, C_inout, with_power=True):
        super(LaplacianBlock, self).__init__()
        self.C_inout = C_inout
        self.with_power = with_power

        self.laplacian_power = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)

        if self.with_power:
            nn.init.constant_(self.laplacian_power, 0.0)
        nn.init.constant_(self.diffusion_time, 0.0001)

    def forward(self, x, evals):

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout
                )
            )

        diffusion_coefs = torch.exp(
            -evals.unsqueeze(-1) * torch.abs(self.diffusion_time).unsqueeze(0)
        )

        if self.with_power:
            lambda_coefs = torch.pow(
                evals.unsqueeze(-1),
                (2.0 * torch.sigmoid(self.laplacian_power) - 1.0).unsqueeze(0),
            )
        else:
            lambda_coefs = torch.ones_like(self.laplacian_power)

        if x.is_complex():
            y = ensure_complex(lambda_coefs * diffusion_coefs) * x
        else:
            y = lambda_coefs * diffusion_coefs * x

        return y


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx_unchanged=None):
    device = x.device
    batch_size, num_dims, num_points = x.size()

    if idx_unchanged is None:
        idx_unchanged = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_unchanged + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  ->
    # (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx_unchanged  # (batch_size, 2*num_dims, num_points, k)


def ensure_complex(arr):
    if arr.is_complex():
        return arr
    return arr.to(complex_dtype_equiv(arr.dtype))


def complex_dtype_equiv(d):
    if d == torch.float32:
        return torch.complex64
    elif d == torch.float64:
        return torch.complex128
    else:
        raise RuntimeError("unexpected type: " + str(d))
