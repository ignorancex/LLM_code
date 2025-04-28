import itertools
import torch
import torch.nn as nn
from texture_loss import _texture_loss, _GridExtractor
from texture_loss import Self_Attn


class Opt:
    def __init__(self, texture_criterion):
        self.batch_size = 1
        self.texture_criterion = texture_criterion


class DummyModel(nn.Module):
    def __init__(self, texture_criterion='max'):
        super(DummyModel, self).__init__()
        self.opt = Opt(texture_criterion)
        self.grid_extractor = _GridExtractor()
        self.dummy_network = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        if self.opt.texture_criterion == 'attention':
            self.attention_model = Self_Attn(in_dim=1)
        self.fake_im = torch.rand((1, 1, 256, 256))  # Dummy tensor for fake image
        self.real_im = torch.rand((1, 1, 256, 256))  # Dummy tensor for real image

    def forward(self, x):
        # Example forward pass
        return self.dummy_network(x)

    def compute_texture_loss(self):
        if self.opt.texture_criterion == 'attention':
            loss, attention_map, weights = _texture_loss(self.fake_im, self.real_im, self.opt, self.grid_extractor,
                                                         self.attention_model)
            print(f"Computed Texture Loss ({self.opt.texture_criterion.capitalize()} Criterion): {loss.item()}")
        else:
            loss = _texture_loss(self.fake_im, self.real_im, self.opt, self.grid_extractor)
            print(f"Computed Texture Loss ({self.opt.texture_criterion.capitalize()} Criterion): {loss.item()}")


if __name__ == "__main__":
    # Examples 1): static aggregation picking the maximum discrepant feature
    model = DummyModel(texture_criterion='max')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.compute_texture_loss()

    # Examples 2): static aggregation picking the average discrepant feature
    model = DummyModel(texture_criterion='average')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.compute_texture_loss()

    # Examples 3): static aggregation performing the Frobenius norm
    model = DummyModel(texture_criterion='Frobenius')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.compute_texture_loss()

    # Examples 4): dynamic aggregation
    model = DummyModel(texture_criterion='attention')
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), model.attention_model.parameters()), lr=0.001)
    model.compute_texture_loss()
