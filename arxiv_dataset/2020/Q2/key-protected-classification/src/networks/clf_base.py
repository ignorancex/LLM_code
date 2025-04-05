import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ortho_group


class Net(nn.Module):

    def __init__(self, n_classes=10, d_ft=256, d_key=-1, use_fixed_layer=False):
        """
        Args:
            d_key: int, if greater than 0, then
                   l2 normalized random class keys are generated and
                   the class logits are computed using those keys.
        """
        super().__init__()
        self.n_classes = n_classes
        self.d_ft = d_ft
        self.d_key = d_key
        self.use_fixed_layer = use_fixed_layer

        # child classes will define conv layers
        self.conv = None

        # in addition to conv layers defined by child classes,
        # we have a single fc layer
        self.fc = nn.Sequential(
            nn.Linear(d_ft, d_ft),
            nn.LeakyReLU(0.2, True))

        if d_key > 0:
            # class keys
            keys = th.randn(d_key, n_classes)
            keys = F.normalize(keys, dim=0)
            self.register_buffer("keys", keys)

            # fc layer that maps convolutional features to class keys
            if use_fixed_layer:

                # check if the parameters of the fixed layer are already
                # computed and saved
                _key_path = ".dft-{}_dkey-{}.npz".format(d_ft, d_key)
                if os.path.exists(_key_path):
                    print(f"loading precomputed fixed layer parameters from {_key_path}")
                    _key_params = np.load(_key_path)
                    key_map_W = _key_params["W"]
                    key_map_b = _key_params["b"]
                else:
                    if d_ft > d_key:
                        key_map_W = ortho_group.rvs(d_ft)[:, :d_key]
                    else:
                        key_map_W = ortho_group.rvs(d_key)[:, :d_ft].T
                    key_map_b = np.random.randn(d_key)
                    # computing big orthogonal matrices takes quite some time,
                    # it is better to save them to be re-used
                    np.savez(_key_path, W=key_map_W, b=key_map_b)

                # in case of fixed layers, we fix the norms of the weight
                # vectors in the last projection layer
                temp_W = 5. * np.sqrt(d_key / 128.)
                temp_b = 0.5 * np.sqrt(d_key / 128.)
                print("temp_W:", temp_W, flush=True)
                print("temp_b:", temp_b, flush=True)

                key_map_W = th.from_numpy(key_map_W).float()
                key_map_W *= temp_W / key_map_W.norm(p=2).item()
                key_map_b = th.from_numpy(key_map_b).float()
                key_map_b *= temp_b / key_map_b.norm(p=2).item()
                self.register_buffer("key_map_W", key_map_W)
                self.register_buffer("key_map_b", key_map_b)

            else:
                self.key_map_learned = nn.Linear(d_ft, d_key)

            # layer norm here just learns 2 * d_key parameters
            # that are used to adjust the mapping
            # this is especially useful when we use a fixed layer to
            # map features to class keys
            self.key_act = nn.Sequential(
                nn.LayerNorm(d_key),
                nn.LeakyReLU(0.0, True))

        else:
            self.class_logits = nn.Sequential(
                nn.Linear(d_ft, n_classes))


    def forward(self, x):
        x = self.conv(x)
        x = th.flatten(x, 1)
        x = self.fc(x)

        if hasattr(self, "keys"):
            if hasattr(self, "key_map_W"):
                x = x.mm(self.key_map_W) + self.key_map_b
            else:
                x = self.key_map_learned(x)
            x = self.key_act(x)
            x = F.normalize(x, dim=1)
            x = x.mm(self.keys)
        else:
            x = self.class_logits(x)

        return x

    def extra_repr(self):
        return "n_classes:{}, d_key:{}, use_fixed_layer:{}".format(self.n_classes, self.d_key, self.use_fixed_layer)

