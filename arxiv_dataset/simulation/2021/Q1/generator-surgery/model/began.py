import torch
import torch.nn as nn

from .helper import View


class ConvBlock(nn.Module):
    """
    All convs are created with:
    conv(in_channel, out_channel, kernel, stride, pad, bias)
    """
    def __init__(self, in_ch, out_ch, k, s, p, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=k,
                      stride=s,
                      padding=p,
                      bias=b), nn.ELU())

    def forward(self, x):
        return self.net(x)


class Generator128(nn.Module):
    rescale = False

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = 128
        self.initial_size = 8
        self.image_size = 128

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim,
                          self.initial_size**2 * self.ch,
                          bias=False),
                View((-1, self.ch, self.initial_size, self.initial_size))),
            # First conv
            # want z1 = 128 x 8 x 8, z2 = 128 x 8 x 8
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(self.ch, self.ch, 3, 1, 1, False),
            ConvBlock(self.ch, self.ch, 3, 1, 1, False),

            # Second block
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(2 * self.ch, self.ch, 3, 1, 1, False),
            ConvBlock(self.ch, self.ch, 3, 1, 1, False),

            # Third block
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(2 * self.ch, self.ch, 3, 1, 1, False),
            ConvBlock(self.ch, self.ch, 3, 1, 1, False),

            # Final conv
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(2 * self.ch, self.ch, 3, 1, 1, False),
            ConvBlock(self.ch, self.ch, 3, 1, 1, False),
            ConvBlock(self.ch, 3, 3, 1, 1, False),
        ])
        self.skips = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Upsample(scale_factor=8, mode='nearest'),
        ])

        self.input_shapes = [
            # Raw input shape
            ((self.latent_dim, ), ()),

            # Skip Linear+View()
            ((128, 8, 8), ()),

            # First conv
            ((128, 16, 16), (128, 8, 8)),
            ((128, 16, 16), (128, 8, 8)),
            ((128, 16, 16), (128, 8, 8)),

            # Second conv
            ((128, 32, 32), (128, 8, 8)),
            ((128, 32, 32), (128, 8, 8)),
            ((128, 32, 32), (128, 8, 8)),

            # Third conv
            ((128, 64, 64), (128, 8, 8)),
            ((128, 64, 64), (128, 8, 8)),
            ((128, 64, 64), (128, 8, 8)),

            # Final conv
            ((128, 128, 128), (128, 8, 8)),
            ((128, 128, 128), (128, 8, 8)),
            ((128, 128, 128), (128, 8, 8)),

            # Skip entire net
            ((3, 128, 128), ()),
        ]

        # self._check_input_shapes()

    def _check_input_shapes(self):
        for n_cuts, (x1_shape, x2_shape) in enumerate(self.input_shapes):
            print(n_cuts)
            x1 = torch.randn(1, *x1_shape)
            if n_cuts <= 1:
                x2 = None
            else:
                x2 = torch.randn(1, *x2_shape)
            res = self.forward(x1, x2, n_cuts)
            print(x1.shape, () if n_cuts <= 1 else x2.shape, res.shape[1:])

    def forward(self, x1, x2=None, n_cuts=0, end=None):
        """
        Skip inputs are provided to layers with python index 4, 7, 10

        If n_cuts <= 1, we will use skip connections (same as in training)

        Else we will provide a second input x2 that goes through skip
        connections as needed

        x2: either None, or B x 128 x 8 x 8

        Consider network with blocks B1, B2, B3:
            z_0 -B1-> z_1 -B2-> z_2 -B3-> img

        If we run .forward(n_cuts=0), we do:
            z_0 -B1-> z_1 -B2-> z_2 -B3-> img

        If we run .forward(n_cuts=1), we do:
                      z_1 -B2-> z_2 -B3-> img
        """
        if n_cuts <= 1:
            assert x2 is None
        else:
            assert x2 is not None

        upsample_layer_idx = [4, 7, 10]
        conv256_layer_idx = [5, 8, 11]
        skip_counter = [1, 1, 1]
        if end is None:
            end = len(self.layers)
        for i in range(n_cuts, end):
            layer = self.layers[i]

            if i == 1:
                x2 = x1

            # i in upsample: use skip directly after self.skips[idx]
            # i in conv256: if upsample hasn't been used; upsample and use skip
            if i in upsample_layer_idx:
                idx = upsample_layer_idx.index(i)
                x_skip = self.skips[idx](x2)
                x1 = layer(torch.cat((x1, x_skip), dim=1))
                skip_counter[idx] -= 1
            elif i in conv256_layer_idx:
                idx = conv256_layer_idx.index(i)
                # Need to check if it was upsampled already from previous layer
                if skip_counter[idx] > 0:
                    # Need to pass an extra upsample
                    x_skip = nn.Upsample(scale_factor=2, mode='nearest')(x2)
                    x_skip = self.skips[idx](x_skip)
                    x1 = layer(torch.cat((x1, x_skip), dim=1))
                    skip_counter[idx] -= 1
                else:
                    # No skipping occurs
                    x1 = layer(x1)
            else:
                # No skipping occurs
                x1 = layer(x1)

        return x1

    def __str__(self):
        return f'Began.Gen128.latent_dim={self.latent_dim}'


class Discriminator128(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = 128
        self.initial_size = 8
        self.image_size = 128

        self.encoder = nn.Sequential(
            ConvBlock(3, self.ch, 3, 1, 1, True),
            ConvBlock(self.ch, self.ch, 3, 1, 1, False),
            ConvBlock(self.ch, self.ch * 2, 3, 2, 1, False),
            ConvBlock(self.ch * 2, self.ch * 2, 3, 1, 1, False),
            ConvBlock(self.ch * 2, self.ch * 3, 3, 2, 1, False),
            ConvBlock(self.ch * 3, self.ch * 3, 3, 1, 1, False),
            ConvBlock(self.ch * 3, self.ch * 4, 3, 2, 1, False),
            ConvBlock(self.ch * 4, self.ch * 4, 3, 1, 1, False),
            ConvBlock(self.ch * 4, self.ch * 5, 3, 2, 1, False),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.ch * 5 * self.initial_size**2,
                      self.latent_dim,
                      bias=False),
        )
        self.decoder = Generator128(self.latent_dim)
        self._create_input_shapes()

    def _create_input_shapes(self):
        self.encoder_input_shapes = []
        x = torch.randn(1, 3, self.image_size, self.image_size)
        self.encoder_input_shapes.append(tuple(x.shape[1:]))
        for i, next_layer in enumerate(self.encoder):
            x = next_layer(x)
            self.encoder_input_shapes.append(tuple(x.shape[1:]))

        self.linear_input_shapes = []
        self.linear_input_shapes.append(tuple(x.shape[1:]))
        x = self.linear(x)
        self.linear_input_shapes.append(tuple(x.shape[1:]))

        self.decoder_input_shapes = [
            x1_shape for x1_shape, _ in self.decoder.input_shapes
        ]
        assert self.decoder_input_shapes[-1] == self.encoder_input_shapes[0]

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        x = self.decoder(x)
        return x

    def __str__(self):
        return f'Began.Disc128.latent_dim={self.latent_dim}'
