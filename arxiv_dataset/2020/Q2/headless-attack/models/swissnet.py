import torch.nn as nn
import torch.nn.functional as F

class SwissNet(nn.Module):
    def __init__(self, hidden_dims=[10,20], train_dp=0, test_dp=0):
        super(SwissNet, self).__init__()
        # a MLP for binary classification on 2D data
        in_shapes = [2] + hidden_dims[:-1]
        self.hidden_layers = nn.ModuleList([nn.Linear(in_shape, out_shape)  for in_shape, out_shape in zip(in_shapes, hidden_dims)])
        self.linear = nn.Linear(hidden_dims[-1], 1)
        self.train_dp = train_dp
        self.test_dp = test_dp

    def forward(self, x, penu=False):
        out = x
        for linear in self.hidden_layers:
            out = F.relu(linear(out))
            if (self.train_dp > 0 and self.training) or self.test_dp > 0:
                dp = max(self.train_dp, self.test_dp)
                out = F.dropout(out, dp, training=True)
        if penu:
            return out
        out = self.linear(out)

        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()
