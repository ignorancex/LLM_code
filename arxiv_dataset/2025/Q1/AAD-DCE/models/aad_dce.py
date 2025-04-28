import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import torchvision.transforms as transforms
from data import aux_dataset
import itertools
import cv2 as cv
import numpy as np

# from loss.loss_provider import LossProvider


class aad_dce_model(BaseModel):
    def name(self):
        return 'aad_dce_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
 
        self.netG = networks_selfattn.define_G(opt.input_nc, opt.output_nc, opt.ngf,opt.which_model_netG,opt.vit_name,opt.fineSize,opt.pre_trained_path, opt.norm,
                                      not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet)

        self.netSC = networks_selfattn.define_C(opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netSC = networks_selfattn.define_C(opt.init_type, opt.init_gain, self.gpu_ids)
        self.aux_data = aux_dataset.AuxAttnDataset(7000,7000, self.gpu_ids[0], mask_size =160)
        self.zero_attn_holder = torch.zeros((1,1, opt.mask_size, opt.mask_size), dtype = torch.float32) #.to(self.device)
        self.ones_attn_holder = torch.ones((1,1, opt.mask_size, opt.mask_size), dtype = torch.float32) #.to(self.device)


        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            self.netD = networks_selfattn.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,opt.vit_name,opt.fineSize,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain) # opt.mask_size, opt.s1, opt.s2)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks_selfattn.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.bceloss = torch.nn.BCELoss()
            self.MSE_Loss = torch.nn.MSELoss()
           
            
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netSC.parameters()),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))  #changed according to TAM
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks_selfattn.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B'].to(torch.float)
        input_B = input['B' if AtoB else 'A'].to(torch.float)
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
#             input_C = input_C.cuda(self.gpu_ids[0])
            
        self.input_A = input_A
        self.input_B = input_B
        
        self.attn_A_index = input['DX'] 
        self.attn_A, _= self.aux_data.get_attn_map(self.attn_A_index, 0)


    def forward(self):
        self.real_A = Variable(self.input_A)
        # concat_attn_A = self.netSC(self.attn_A)   #using netSC
        concat_attn_A = self.attn_A
        self.fake_B= self.netG(self.real_A*(1. + concat_attn_A))                      
        self.o_crop = self.fake_B[:,:,50:110,60:120]  # this is cropping to the size 60 which takes only the prostate
        self.real_B = Variable(self.input_B)                      #target
        self.t_crop = self.real_B[:,:,50:110,60:120]  #cropping taking only the prostate
    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            # concat_attn_A = self.netSC(self.attn_A)
            concat_attn_A = self.attn_A
            self.fake_B= self.netG(self.real_A*(1. + concat_attn_A)) 
            self.o_crop = self.fake_B[:,:,50:110,60:120]  # this is cropping to the size 60 which takes only the prostate
            self.real_B = Variable(self.input_B)
            self.t_crop = self.real_B[:,:,50:110,60:120]  #cropping taking only the prostate

    def backward_D(self):
        # Fake
        pred_fake, _ = self.netD(self.o_crop.detach(), self.fake_B.detach())
        fake_label = torch.zeros(pred_fake.shape).cuda(self.gpu_ids[0]) 
        self.loss_D_fake = self.MSE_Loss(pred_fake, fake_label) 

       # Real
        pred_real, _ = self.netD(self.t_crop, self.real_B)
        real_label = torch.ones(pred_real.shape).cuda(self.gpu_ids[0]) 
        self.loss_D_real = self.MSE_Loss(pred_real, real_label) 
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()
       
    def backward_G(self):

        pred_fake, self.tmp_attn = self.netD(self.o_crop, self.fake_B)
        pred_real, _ = self.netD(self.t_crop, self.real_B)
        real_label = torch.ones(pred_real.shape).cuda(self.gpu_ids[0]) 
        self.loss_G_GAN = self.MSE_Loss(pred_fake, real_label)*self.opt.lambda_adv
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) *self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*1 
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        #update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step() 
        #update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        #attn parameters
        self.aux_data.update_attn_map(self.attn_A_index, self.tmp_attn.detach(), True)

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())

                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
