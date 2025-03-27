import loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        out_channels_per_layer,
        kernel_sizes_per_layer,
        maxpool_per_layer,
    ):
        super().__init__()

        self.out_channels_per_layer = out_channels_per_layer
        self.kernel_sizes_per_layer = kernel_sizes_per_layer
        self.maxpool_per_layer = maxpool_per_layer

        in_features, h, w = input_shape
        layers = nn.ModuleList()

        for i in range(len(out_channels_per_layer)):
            layers += [
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_channels_per_layer[i],
                    kernel_size=kernel_sizes_per_layer[i],
                    padding='same',
                ),
                nn.LeakyReLU(),
            ]
            if maxpool_per_layer[i]:
                layers += [nn.MaxPool2d(kernel_size=maxpool_per_layer[i])]
                h //= maxpool_per_layer[i]
                w //= maxpool_per_layer[i]

            in_features = out_channels_per_layer[i]
        
        self.input_shape = input_shape
        self.output_shape = (out_channels_per_layer[-1], h, w)

        self.layers = layers

    def forward(self, x):
        x_cat = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.MaxPool2d):
                x_cat += [x]
            x = layer(x)
        return (x, x_cat)

class CNNDecoder(nn.Module):
    def __init__(
        self,
        input_shape,
        out_channels_per_layer,
        kernel_sizes_per_layer,
        upsample_per_layer,
    ):
        super().__init__()

        self.out_channels_per_layer = out_channels_per_layer
        self.kernel_sizes_per_layer = kernel_sizes_per_layer
        self.upsample_per_layer = upsample_per_layer

        in_features, h, w = input_shape
        in_features *= 2
        layers = nn.ModuleList()

        for i in range(len(out_channels_per_layer)):
            if upsample_per_layer[i]:
                layers += [nn.Upsample(scale_factor=upsample_per_layer[i], mode='bilinear')]
                h *= upsample_per_layer[i]
                w *= upsample_per_layer[i]
            
            if i + 1 < len(out_channels_per_layer) and upsample_per_layer[i+1]:
                out_channels = out_channels_per_layer[i] // 2
            else:
                out_channels = out_channels_per_layer[i]
            layers += [
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes_per_layer[i],
                    padding='same',
                ),
                nn.LeakyReLU(),
            ]

            in_features = out_channels_per_layer[i]
        
        self.input_shape = input_shape
        self.output_shape = (in_features, h, w)

        self.layers = layers
    
    def forward(self, x, x_cat):
        x_cat = x_cat.copy()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, x_cat.pop()], dim=1)
        return x

class GenWeakSegNet(nn.Module):
    def __init__(self, classifier, num_classes=2, hparams=None):
        super().__init__()

        self.input_shape = (3, 224, 224)
        self.num_classes = num_classes
        self.hparams = hparams

        out_channels = [64, 64, 128, 128, 256, 256, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
        maxpool = [0, 2, 0, 2, 0, 2, 0, 2]

        self.encoder = CNNEncoder(
            input_shape=self.input_shape,
            out_channels_per_layer=out_channels + [1024, 512],
            kernel_sizes_per_layer=kernel_sizes + [3, 3],
            maxpool_per_layer=maxpool + [0, 0],
        )

        input_shape = self.encoder.output_shape
        out_channels = out_channels[::-1]
        kernel_sizes = kernel_sizes[::-1]
        upsample = maxpool[::-1]

        # print(out_channels)
        # print(kernel_sizes)
        # print(upsample)

        self.decoder_img = CNNDecoder(
            input_shape=input_shape,
            out_channels_per_layer=out_channels,
            kernel_sizes_per_layer=kernel_sizes,
            upsample_per_layer=upsample,
        )

        self.decoder_mask = CNNDecoder(
            input_shape=input_shape,
            out_channels_per_layer=out_channels,
            kernel_sizes_per_layer=kernel_sizes,
            upsample_per_layer=upsample,
        )

        self.op_cls_img = nn.Conv2d(out_channels[-1], 3*num_classes, kernel_size=3, padding='same')
        self.op_cls_mask = nn.Conv2d(out_channels[-1], num_classes, kernel_size=3, padding='same')

        self.recon_loss_fn = loss.ReconLoss(L=1)
        self.mask_reg_loss_fn = loss.MaskRegLoss(num_classes)

        for params in classifier.parameters():
            params.requires_grad = False
        self.cls_guide_loss_fn = loss.ClsGuideLoss(classifier)

    def forward(self, x):
        # unet forward pass
        # encode
        x, x_cat = self.encoder(x)
        
        # decode to images
        x_img = self.decoder_img(x, x_cat)
        y_img = self.op_cls_img(x_img).view(x.shape[0], self.num_classes, *self.input_shape)
        y_img = F.sigmoid(y_img)

        # decode to masks
        x_mask = self.decoder_mask(x, x_cat)
        y_mask = self.op_cls_mask(x_mask)

        return (y_img, y_mask)
    
    def loss_fn(self, x, label, y_img, y_mask):
        recon = self.recon_loss_fn(x, y_img, y_mask)
        mask_reg = self.mask_reg_loss_fn(label, y_mask)
        cls_guide = self.cls_guide_loss_fn(label, y_img, y_mask)
        
        loss = (
            self.hparams['recon']*recon +
            self.hparams['mask_reg']*mask_reg +
            self.hparams['cls_guide']*cls_guide
        )
        loss_dict = {
            'loss': loss,
            'recon': recon,
            'mask_reg': mask_reg,
            'cls_guide': cls_guide,
        }
        return (loss, loss_dict)
