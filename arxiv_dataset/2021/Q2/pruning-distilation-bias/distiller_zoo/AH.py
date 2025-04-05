from __future__ import print_function

import torch.nn as nn

class AllHint(nn.Module):

    def __init__(self):
        super(AllHint, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, g_s, g_t):
       return [self.crit(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]