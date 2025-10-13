from layers.lorentz_ops import LorentzAct, LorentzLinear, LorentzAgg
from torch_geometric.utils import to_torch_coo_tensor
from torch.nn import Module, Parameter
from torch import Tensor
import torch


class LorentzGIN(Module):
    def __init__(self, manifold, eps, in_features, c_in, nn, use_bias, use_att):
        super(LorentzGIN, self).__init__()

        self.c_in = c_in
        self.manifold = manifold
        self.use_att = use_att
        self.agg = LorentzAgg(manifold, self.c_in,
                              self.use_att, in_features+1, use_bias)
        self.nn = nn
        self.eps = Parameter(torch.tensor(eps))

    def forward(self, input):
        x, adj = input
        adj = to_torch_coo_tensor(adj)

        # Map to hyperboloid space
        x = self.manifold.exp_map_zero(x, self.c_in)
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.agg.forward(x=x[0], adj=adj)  # Calculate the Frechet Mean

        x_r = x[1]
        if x_r is not None:
            log_out = self.manifold.log_map_zero(out, c=self.c_in)
            log_x_r = self.manifold.log_map_zero(x_r, c=self.c_in)

            pt_xr = self.manifold.ptransp(x=x_r, y=out, v=log_x_r, c=self.c_in)
            out = self.manifold.exp_map_zero(
                dp=log_out+(1+self.eps)*pt_xr, c=self.c_in)

        return self.nn(out)

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.agg.reset_parameters()


class LorentzGNN(Module):
    def __init__(self, manifold, in_features, out_features, c_in, c_out, drop_out, act, use_bias, use_att):
        super(LorentzGNN, self).__init__()

        self.c_in = c_in
        self.linear = LorentzLinear(
            manifold, in_features, out_features, c_in, drop_out, use_bias)
        self.agg = LorentzAgg(manifold, c_in, use_att, out_features, drop_out)
        self.lorentz_act = LorentzAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.lorentz_act.forward(h)

        output = h, adj

        return output

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.agg.reset_parameters()
