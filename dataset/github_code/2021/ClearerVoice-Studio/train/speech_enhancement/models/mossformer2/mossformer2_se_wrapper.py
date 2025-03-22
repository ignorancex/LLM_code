from models.mossformer2.mossformer2 import MossFormer_MaskNet
import torch.nn as nn


class MossFormer2_SE_48K(nn.Module):
    def __init__(self, args):
        super(MossFormer2_SE_48K, self).__init__()
        self.model = TestNet()

    def forward(self, x):
        outputs, mask = self.model(x)
        return outputs, mask

class TestNet(nn.Module):

    def __init__(self, n_layers=18):
        super(TestNet, self).__init__()
        self.n_layers = n_layers
        self.mossformer = MossFormer_MaskNet(in_channels=180, out_channels=512, out_channels_final=961)

    def forward(self, input):
        out_list=[]
        x = input.transpose(1,2)
        mask = self.mossformer(x)
        out_list.append(mask)
        return out_list
