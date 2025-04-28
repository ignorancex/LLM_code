import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG,img_size, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'res_cnn':
        netG = Res_CNN(input_dim= input_nc, img_size=img_size, output_dim=output_nc, vis=False)        
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(input_nc, ndf, which_model_netD,img_size,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
        
    elif which_model_netD == 'aad':
        netD = AggregatedAttn_Discriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def define_C(norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[], mode=0):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    net = SComponent(norm_layer, mode)
    # net = SComponent(mode)
    

    if use_gpu:
        net.cuda(gpu_ids[0])
    init_weights(net, init_type=init_type)

    return net  #init_net(net, init_type, init_gain, gpu_ids)



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class Encoder_Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect',down_samp=1,gated_fusion=0):
        super(Encoder_Decoder, self).__init__()        
        self.output_nc = output_nc      
        self.encoders=2
        latent_size=16
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        #Encoders
        for ii in range(2):
            model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                     norm_layer(ngf), nn.ReLU(True)]   
            n_downsampling = 2 
            
            ### downsample
            for i in range(n_downsampling):
                mult = 2**i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * mult * 2), nn.ReLU(True)]
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            #model += [nn.ReflectionPad2d(1)]              
            model += [nn.Conv2d(ngf * mult, latent_size, kernel_size=3, padding=1), 
                     norm_layer(latent_size), nn.ReLU(True)]   
            setattr(self, 'model_enc_'+str(ii), nn.Sequential(*model))
        #Decoder
        #model += [nn.ReflectionPad2d(3)] 
        model = [nn.Conv2d(latent_size*2, 256, kernel_size=3, padding=1), 
                 norm_layer(256), nn.ReLU(True)]  
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1,bias=use_bias),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(2), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        setattr(self, 'model_dec', nn.Sequential(*model))

            
            
    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        encoded=[]
        for ii in range(self.encoders):
            encoded.append( getattr(self, 'model_enc_'+str(ii))(input[:,ii,:,:]))
        decoded=self.model_dec(torch.cat((encoded[0],encoded[1]),1))
        return decoded
#        else:
#            return self.model(input)







# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect',down_samp=1):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.down_samp=down_samp
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
############################################################################################
#Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'model_1', nn.Sequential(*model)) 
############################################################################################
#Layer2-Encoder2
        n_downsampling = 2
        model = []
        i=0
        mult = 2**i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
        setattr(self, 'model_2', nn.Sequential(*model))
############################################################################################
#Layer3-Encoder3
        model = []
        i=1
        mult = 2**i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                        stride=2, padding=1, bias=use_bias),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)]
        setattr(self, 'model_3', nn.Sequential(*model))
############################################################################################
#Layer4-Residual1
        mult = 2**n_downsampling
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_4', nn.Sequential(*model))
############################################################################################
#Layer5-Residual2
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_5', nn.Sequential(*model))
############################################################################################
#Layer6-Residual3
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_6', nn.Sequential(*model))
############################################################################################
#Layer7-Residual4
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_7', nn.Sequential(*model))
############################################################################################
#Layer8-Residual5
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_8', nn.Sequential(*model))
############################################################################################
#Layer9-Residual6
        model = []
        use_dropout = norm_layer == nn.InstanceNorm2d
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_9', nn.Sequential(*model))
############################################################################################
#Layer10-Residual7
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_10', nn.Sequential(*model))
############################################################################################
#Layer11-Residual8
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_11', nn.Sequential(*model))
############################################################################################
#Layer12-Residual9
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_12', nn.Sequential(*model))
############################################################################################
#Layer13-Decoder1
        i = 0
        mult = 2**(n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                                    norm_layer(int(ngf * mult / 2)),
                                    nn.ReLU(True)]
        setattr(self, 'model_13', nn.Sequential(*model))
############################################################################################
#Layer14-Decoder2
        i = 1
        mult = 2**(n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                                    norm_layer(int(ngf * mult / 2)),
                                    nn.ReLU(True)]
        setattr(self, 'model_14', nn.Sequential(*model))
############################################################################################
#Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'model_15', nn.Sequential(*model))
############################################################################################


    def forward(self, input):
        x1 = self.model_1(input)
        x2 = self.model_2(x1)
        x3 = self.model_3(x2)
        x4 = self.model_4(x3)
        x5 = self.model_5(x4)
        x6 = self.model_6(x5)
        x7 = self.model_7(x6)
        x8 = self.model_8(x7)
        x9 = self.model_9(x8)
        x10 = self.model_10(x9)
        x11 = self.model_11(x10)
        x12 = self.model_12(x11)
        x13 = self.model_13(x12)
        x14 = self.model_14(x13)
        x15 = self.model_15(x14)
        return x15
        

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            print(self.model(input).size())
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)
        

###################### Aggregated Attention Discriminator ################################################

class attn_module(nn.Module):
    '''Auxiliary attention module for parallel trainable attention network'''


    def __init__(self, in_ch, out_ch, s1, s2):  #(32->(20,20) for image size 160,(8,8) for image size 60
        self.s1, self.s2 = s1, s2
        super(attn_module, self).__init__()
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (128 * 128) -> (64 * 64)
        self.d1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (64 * 64) -> (32 * 32)
        self.d2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (32 * 32) -> (16 * 16)

        #experimenting wihout maxpooling layer
        # self.d1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False)
        # self.d2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False)
        # self.skip2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        # self.d3 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.mid = nn.Sequential(*[
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        ])

        self.u2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.u1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        
        if self.s2 == (20,20):
            self.last = nn.Sequential(*[
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 1, 1, bias=False),
                nn.Sigmoid()
            ])
        elif self.s2 == (8,8):
            self.last = nn.Sequential(*[
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 2, 1, bias=False),  #kernel 1-->2
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 1, 1, bias=False),
                nn.Sigmoid()
            ])
            

    def forward(self, x):
        out = F.relu(self.d1(self.mp1(x)))
        out = F.relu(self.d2(self.mp2(out)))
        skip2 = F.relu(self.skip2(out))
        out = self.mid(out)
        out = F.interpolate(out, size=self.s2, mode='bilinear', align_corners=True) + skip2  #bilinear
        # (14 * 14 -> 28 * 28)
        out = self.last(self.u2(out))
        return out

class AttnDiscriminator(nn.Module):
    # The size of input is assumed to be 256 * 256
    def __init__(self, in_ch, int_ch,img_size=160, n_layers=3, mask_shape=160, norm_layer=nn.BatchNorm2d, inner_rescale=True,
                 inner_s1=None, inner_s2=None):
        super(AttnDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        if img_size == 160:
            mask_shape = 160
        elif img_size == 60:
            mask_shape = 60

        kw, padw = 4, 1
        self.fe = nn.Sequential(*[
            nn.Conv2d(in_ch, int_ch, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(.2, True)
        ])

        trunk_model = list()
        nf_mult, nf_mult_prev = 1, 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)

            trunk_model += [
                nn.Conv2d(int_ch * nf_mult_prev, int_ch * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(int_ch * nf_mult),
                nn.LeakyReLU(.2, True)
            ]

        self.trunk_brunch = nn.Sequential(*trunk_model)        
        if img_size == 160:
            
            if not inner_s1 and not inner_s2:
                self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, s1 =(64,64), s2=(20,20))
            else:
                self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, inner_s1, inner_s2)

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            
            self.fin_phase = nn.Sequential(*[
                nn.Conv2d(int_ch * nf_mult_prev, int_ch * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(int_ch * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(int_ch * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
                nn.Flatten(),                                #adding the last layer according to the reconglgan
                nn.Linear(1*18*18,64),
                nn.LeakyReLU(0.2, True)
            ])
            self.inner_rescale = inner_rescale
            self.mask_shape = mask_shape
            
        elif img_size == 60:
            
            if not inner_s1 and not inner_s2:
                self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, s1 =(64,64), s2=(8,8))
            else:
                self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, inner_s1, inner_s2)

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            
            self.fin_phase = nn.Sequential(*[
                nn.Conv2d(int_ch * nf_mult_prev, int_ch * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(int_ch * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(int_ch * nf_mult, 32, kernel_size=kw, stride=1, padding=padw),
                nn.Flatten(),                                #adding the last layer to make it to same size[1,64]
                nn.Linear(1*5*5*32,512),
                nn.LeakyReLU(0.2, True),
                nn.Linear(512,64),
                nn.LeakyReLU(0.2, True)
            ])
            self.inner_rescale = inner_rescale
            self.mask_shape = mask_shape
            

    def forward(self, x):
        feature = self.fe(x)
        trunk = self.trunk_brunch(feature)
        mask = self.mask_brunch(feature)
        expand_mask = self._mask_rescale(mask, self.inner_rescale)
        final = self.fin_phase((mask+1)*trunk)
        return self.fin_phase((mask + 1) * trunk), expand_mask

    def _mask_rescale(self, mask_tensor, final_scale=False):
        mask_tensor = torch.mean(mask_tensor, dim=1).unsqueeze(1)

        if final_scale:
            # t_max, t_min = torch.max(mask_tensor), torch.min(mask_tensor)
            # mask_tensor = (mask_tensor - t_min) / (t_max - t_min)
            # return F.interpolate(mask_tensor, (self.mask_shape, self.mask_shape), mode='bilinear')
            return F.interpolate(mask_tensor, (self.mask_shape, self.mask_shape), mode='nearest')
        else:
            return mask_tensor

class Concatenate_attn(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate_attn, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class AggregatedAttn_Discriminator(nn.Module):
    def __init__(self):
        super(AggregatedAttn_Discriminator, self).__init__()
        self.model_gd = AttnDiscriminator(in_ch =2, int_ch =64, img_size=160)
        self.model_ld = AttnDiscriminator(in_ch = 2, int_ch =64, img_size=60)
        self.concat1 = Concatenate_attn(dim=-1)
        self.linear1 = nn.Linear(128, 1)

    def forward(self, x_local, x_global):
        x_ld, local_attn = self.model_ld(x_local)
        # print('---local shape---', x_ld.shape)
        x_gd, global_attn = self.model_gd(x_global)
        # print('----global shape-----', x_gd.shape)
        x = self.linear1(self.concat1([x_ld, x_gd]))

        #local attn embedding onto gloabl attn
        new_attn = global_attn.detach().cpu().numpy()
        new_attn[:,:,50:110,60:120] = local_attn.detach().cpu().numpy()
        tensor_new_attn = torch.from_numpy(new_attn).to(global_attn.device)
  


        # # local attention added to the global attention at the ROI
        # new_attn = global_attn.detach().cpu().numpy()
        # added_attn = global_attn[:,:,50:110,60:120].detach().cpu().numpy() + local_attn.detach().cpu().numpy()
        # new_attn[:,:,50:110,60:120]= added_attn
        # tensor_new_attn = torch.from_numpy(new_attn).to(global_attn.device)

        # #multiplication of the loacl and the global attention at the ROI
        # new_attn = global_attn.detach().cpu().numpy()
        # average_attn = (global_attn[:,:,50:110,60:120].detach().cpu().numpy()* local_attn.detach().cpu().numpy())
        # new_attn[:,:,50:110,60:120]= average_attn
        # tensor_new_attn = torch.from_numpy(new_attn).to(global_attn.device)
        
        return x, tensor_new_attn #global_attn # tensor_new_attn

class SComponent(nn.Module):
    
    def __init__(self, norm_layer, mode=0, gpu_ids=[]):
        super(SComponent, self).__init__()
        self.gpu_ids = gpu_ids
        # norm_layer = nn.BatchNorm2d
        if 0 == mode:
            sequence = [nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        norm_layer(64),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
                        norm_layer(128),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(128, 1, kernel_size=3, stride=1, padding=1,bias=True),
                        norm_layer(1),
                        nn.Sigmoid()]
        else:
            sequence = [nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        norm_layer(64),
                        nn.ReLU(True),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
                        norm_layer(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True),
                        norm_layer(1),
                        nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


################################################## Residual CNN ####################################

class Residual_block(nn.Module):
    def __init__(self, input_dim, img_size=224,transformer = None):
        super(Residual_block, self).__init__()
        ngf = 64
        mult = 4
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        # Residual CNN
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                             use_bias=use_bias)]
        setattr(self, 'residual_cnn', nn.Sequential(*model))

    def forward(self, x):
        # residual CNN
        x = self.residual_cnn(x)
        return x

########Generator############
class Res_CNN(nn.Module):
    def __init__(self, input_dim, img_size=224, output_dim=3, vis=False):
        super(Res_CNN, self).__init__()
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ####################################ART Blocks##############################################
        mult = 4
        self.res1 = Residual_block(input_dim, img_size)
        self.res2 = Residual_block(input_dim, img_size)
        self.res3 = Residual_block(input_dim, img_size)
        self.res4 = Residual_block(input_dim, img_size)
        self.res5 = Residual_block(input_dim, img_size)
        self.res6 = Residual_block(input_dim, img_size)
        self.res7 = Residual_block(input_dim, img_size)
        self.res8 = Residual_block(input_dim, img_size)
        self.res9 = Residual_block(input_dim, img_size)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

    ############################################################################################

    def forward(self, x):
        # Encoder
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        # Information bottleneck
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        # Decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x
        

