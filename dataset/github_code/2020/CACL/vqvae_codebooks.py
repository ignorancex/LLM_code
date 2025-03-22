import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn
from models.networks import NLayerDiscriminator, get_norm_layer, GANLoss
from util.image_pool import ImagePool
from models.cycle_gan_model import CycleGANModel

#from mobilenet import MobileNetV2
from torchvision import models
import yaml


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch
class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Quantize_bi(nn.Module):
    def __init__(self, dim, n_embed, pos_dim, pos_embed, decay=0.99, eps=1e-5):
        super().__init__()

        decay = 0.98
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.positive_embed_channel = pos_embed # 5
        self.positive_channel = pos_dim #64

        mask = torch.ones(dim, n_embed)
        mask[(dim-self.positive_channel):dim,:(n_embed - self.positive_embed_channel)] = 0.
        mask[:(dim-self.positive_channel),(n_embed-self.positive_embed_channel):n_embed] = 0.
        mask[(dim - self.positive_channel):dim , (n_embed - self.positive_embed_channel):n_embed] = 1.

        self.mask = mask

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, bi):
        if bi == 1:
            self.embed = nn.Parameter(self.embed * self.mask.float().cuda())   #nn.Parameters

            flatten = input.reshape(-1, self.dim)
            dist = (
                    flatten.pow(2).sum(1, keepdim=True)
                    - 2 * flatten @ self.embed
                    + self.embed.pow(2).sum(0, keepdim=True)
            )
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind, self.embed)

            if self.training:
                embed_onehot_sum = embed_onehot.sum(0)
                embed_sum = flatten.transpose(0, 1) @ embed_onehot

                dist_fn.all_reduce(embed_onehot_sum)
                dist_fn.all_reduce(embed_sum)

                self.cluster_size.data.mul_(self.decay).add_(
                    embed_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = (
                        (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n)
                positive_embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)

                embed_normalized = nn.Parameter(positive_embed_normalized * self.mask.float().cuda())

                self.embed.data.copy_(embed_normalized)

            diff = (quantize.detach() - input).pow(2).mean(1).mean(1).mean(1)
            quantize = input + (quantize - input).detach()

            return quantize, diff, embed_ind, self.embed

        else:
            self.embed = nn.Parameter(self.embed * self.mask.float().cuda())
            #self.embed = self.embed[:,:n_embed].cuda() * self.mask_neg.float().cuda()#nn.Parameters

            flatten = input.reshape(-1, self.dim)
            dist = (
                    flatten.pow(2).sum(1, keepdim=True)
                    - 2 * flatten @ self.embed[:,:(self.n_embed-self.positive_embed_channel)]
                    + self.embed[:,:(self.n_embed-self.positive_embed_channel)].pow(2).sum(0, keepdim=True)
            )
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed - self.positive_embed_channel).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind, self.embed[:,:(self.n_embed-self.positive_embed_channel)])

            if 0:#self.training:
                embed_onehot_sum = embed_onehot.sum(0)
                embed_sum = flatten.transpose(0, 1) @ embed_onehot

                dist_fn.all_reduce(embed_onehot_sum)
                dist_fn.all_reduce(embed_sum)

                self.cluster_size[:(self.n_embed-self.positive_embed_channel)].data.mul_(self.decay).add_(
                    embed_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg[:,:(self.n_embed-self.positive_embed_channel)].data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size[:(self.n_embed-self.positive_embed_channel)].sum()
                cluster_size = (
                        (self.cluster_size[:(self.n_embed-self.positive_embed_channel)] + self.eps) / (n + (self.n_embed - self.positive_embed_channel) * self.eps) * n
                )
                embed_normalized = self.embed_avg[:,:(self.n_embed-self.positive_embed_channel)] / cluster_size.unsqueeze(0)

                positive_embed = torch.zeros([self.dim,self.positive_embed_channel]).cuda()
                positive_embed[(self.dim-self.positive_channel):self.dim,:] = self.embed[(self.dim - self.positive_channel):self.dim, (self.n_embed - self.positive_embed_channel):self.n_embed]

                new_embed_normalized = torch.cat([embed_normalized, positive_embed], 1)

                new_embed_normalized = nn.Parameter(new_embed_normalized * self.mask.cuda().float())

                self.embed.data.copy_(new_embed_normalized)


            diff = (quantize.detach() - input).pow(2).mean(1).mean(1).mean(1)
            quantize = input + (quantize - input).detach()

            return quantize, diff, embed_ind, self.embed

    def embed_code(self, embed_id, embed):
        return F.embedding(embed_id, embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        elif stride == 1:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 2, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        elif stride == 1:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 3, stride=1, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,        #128
        n_res_block=2,      #2
        n_res_channel=32,   #32
        embed_dim=128,      #64
        n_embed = 32,       #32
        decay=0.99,
    ):
        super().__init__()

        with open('configs/test.yaml', 'r') as f:
            temp = yaml.full_load(f.read())

        embed_dim = temp['TPARAMETER']['EMBED_DIM']
        n_embed = temp['TPARAMETER']['N_EMBED']
        pos_dim = temp['TPARAMETER']['POS_DIM']
        pos_embed = temp['TPARAMETER']['POS_EMBED']

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=2)     #4,2
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=1
        )
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=2,           #4,2
        )
        self.dec_positive = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        self.dec_negtive = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        self.quantize_bi = Quantize_bi(embed_dim, n_embed, pos_dim, pos_embed)

        input_nc = in_channel
        ndf = 2
        norm_layer = get_norm_layer(norm_type='batch')
        self.discriminator =  NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        self.fc = nn.Linear(196,2)

        #Cycle_GAN
        self.device = "cuda"
        self.criterionGAN = GANLoss('vanilla').to(self.device)
        # self.classification = MobileNetV2(num_classes = 4)
        # self.fake_positive_pool = ImagePool(64)
        # self.fake_negtive_pool = ImagePool(64)# create image buffer to store previously generated images

        self.avg = nn.AvgPool2d((n_embed, n_embed))

        """
        Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

    def forward(self, input):
        #quant_t, quant_b, diff, _, _ = self.encode(input)
        #dec = self.decode(quant_t, quant_b)

        quant_positive, quant_negtive, diff_positive, diff_negtive, id_positive, id_negtive = self.encode(input)

        #celculate avg feature
        avg_features_positive = self.avg(quant_positive)
        avg_features_negtive = self.avg(quant_negtive)

        #clip_features_positive = avg_features_positive[:,32:64,:,:]
        #clip_features_negtive = avg_features_negtive[:,32:64,:,:]

        dec_positive, dec_negtive = self.decode(quant_positive, quant_negtive)

        #return dec_positive, dec_negtive, diff_positive, diff_negtive, id_positive, id_negtive, loss_fake_G, loss_fake_D, class_positive, class_negtive, class_input, avg_features_positive, avg_features_negtive
        return dec_positive, dec_negtive, diff_positive, diff_negtive, id_positive, id_negtive, avg_features_positive, avg_features_negtive

    def encode(self, input):
        enc_b = self.enc_b(input)
        #enc_t = self.enc_t(enc_b)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)

        #use both two code books
        quant_negtive, diff_negtive, id_negtive, codebook_negtive = self.quantize_bi(quant_b, 0)
        quant_positive, diff_positive, id_positive, codebook_positive = self.quantize_bi(quant_b, 1)

        #quant_positive, diff_positive, id_positive, codebook_positive = self.quantize_bi(quant_b,1)

        quant_positive = quant_positive.permute(0, 3, 1, 2)
        #diff_positive = diff_positive.unsqueeze(0)
        quant_negtive = quant_negtive.permute(0, 3, 1, 2)
        #diff_negtive = diff_negtive.unsqueeze(0)
        #return quant_t, quant_b, diff_t + diff_b, id_t, id_b

        return quant_positive, quant_negtive, diff_positive, diff_negtive, id_positive, id_negtive

    def decode(self, quant_positive, quant_negtive):
        #upsample_t = self.upsample_t(quant_t)
        #quant = torch.cat([upsample_t, quant_b], 1)
        dec_positive = self.dec(quant_positive)
        dec_negtive = self.dec(quant_negtive)

        return dec_positive, dec_negtive

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        #self.classification = MobileNetV2(num_classes=4)
        self.classification = models.resnet18(pretrained = True)
        n_feature = self.classification.fc.in_features
        n_classes = 4  #4
        self.classification.fc = nn.Linear(n_feature, n_classes)
    def forward(self, image, grad):
        if grad == True:
            for param in self.classification.parameters():
                param.requires_grad = True

            output = self.classification(image)

        else:
            for param in self.classification.parameters():
                param.requires_grad = False

            output = self.classification(image)
        return output

