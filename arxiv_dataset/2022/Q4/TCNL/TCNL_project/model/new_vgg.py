from model.vgg_backbone import *
from utils.loss_utils import SimilarityCriterion, GANCriterion, ClassificationCriterion
from model.nets_factory import set_requires_grad
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools
from utils.loss_utils import ClassificationCriterion, SimilarityCriterion, GANCriterion
from utils.vis_utils import array_to_image

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }


class new_vgg(nn.Module):
    def __init__(self, args: dict):
        super(new_vgg, self).__init__()
        # block
        self.shallow_backbone = vgg_shallow(args['num_classes'])
        self.head_feature_extractor = vgg_deep_head(args['num_classes'])
        self.shape_feature_extractor = vgg_deep_shape(num_classes=args['num_classes'])
        self.classifier = VGG_Classifier(num_classes=args['num_classes'])
        self.reconstructor_shape = VGG_Reconstructor_shape()
        self.discriminator_shape = VGG_Discriminator()
        self.reconstructor_head = VGG_Reconstructor_head()
        self.discriminator_head = VGG_Discriminator()
        # criterion
        self.GAN_criterion = GANCriterion()
        self.GAN_L1_criterion = nn.L1Loss()
        self.classification_criterion = ClassificationCriterion()
        self.similarity_criterion = SimilarityCriterion()
        # optimizer
        self.optimizer_R_shape = torch.optim.Adam(self.reconstructor_shape.parameters(), lr=args['lr_R'])
        self.optimizer_D_shape = torch.optim.SGD(self.discriminator_shape.parameters(), lr=args['lr_D'])
        self.optimizer_R_head = torch.optim.Adam(self.reconstructor_head.parameters(), lr=args['lr_R'])
        self.optimizer_D_head = torch.optim.SGD(self.discriminator_head.parameters(), lr=args['lr_D'])
        self.optimizer_shallow_backbone = torch.optim.SGD(self.shallow_backbone.parameters(), lr=args['lr_resnet'])
        self.optimizer_shape_feature_extractor = torch.optim.SGD(self.shape_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_head_feature_extractor = torch.optim.SGD(self.head_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_classifier = torch.optim.SGD(self.classifier.parameters(), lr=args['lr_resnet'])


    def feature_concat(self, x1, x2):
        feature = torch.cat([x1, x2], 1)
        return feature

    
    def forward(self, x, label, object, head):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        shape_feature, shape_feature_pooled = self.shape_feature_extractor(shallow_feature)
        # classifying
        feature = self.feature_concat(shape_feature_pooled, head_feature)
        pred_class = self.classifier(feature)
        # reconstructing
        reconstruct_shape = self.reconstructor_shape(shape_feature)
        reconstruct_head = self.reconstructor_head(head_feature)
        # discriminating
        shape_pred_real = self.discriminator_shape(object)
        shape_pred_fake = self.discriminator_shape(reconstruct_shape)
        head_pred_real = self.discriminator_head(head)
        head_pred_fake = self.discriminator_head(reconstruct_head)

        return pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, reconstruct_shape, reconstruct_head, object, head

    def backward_for_discriminator(self, pred_fake, pred_real):
        loss_D_fake = self.GAN_criterion(pred_fake, False)
        loss_D_real = self.GAN_criterion(pred_real, True)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward(retain_graph=True)
        return loss_D

    def backward_for_reconstructor(self, pred_fake, reconstruct_image, ori_image):
        loss_R_GAN = self.GAN_criterion(pred_fake, True)
        loss_R_similarity = self.similarity_criterion(reconstruct_image, ori_image)
        # 测试放弃鉴别器损失
        # loss_R = loss_R_similarity
        loss_R = loss_R_GAN + loss_R_similarity
        loss_R.backward(retain_graph=True)
        return loss_R

    def backward_for_backbone(self, pred_class, label, reg):
        loss_C = self.classification_criterion(pred_class, label) + reg
        loss_C = loss_C
        loss_C.backward()
        return loss_C

    def optimize_parameters(self, pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, reconstruct_shape, reconstruct_head, object, head, label, optimize_flag):
        # pay attention to the order of block during backward
        # discriminator -> reconstructor & backbone -> backbone & classifier

        set_requires_grad(self.shallow_backbone, False)
        set_requires_grad(self.head_feature_extractor, False)
        set_requires_grad(self.shape_feature_extractor, False)
        set_requires_grad(self.reconstructor_head, False)
        set_requires_grad(self.reconstructor_shape, False)
        loss_D_shape = self.backward_for_discriminator(shape_pred_fake, shape_pred_real)
        loss_D_head = self.backward_for_discriminator(head_pred_fake, head_pred_real)
        # print('successfully backward through discriminator')
        
        set_requires_grad(self.shallow_backbone, True)
        set_requires_grad(self.shape_feature_extractor, True)
        set_requires_grad(self.head_feature_extractor, True)
        set_requires_grad(self.reconstructor_head, True)
        set_requires_grad(self.reconstructor_shape, True)

        loss_R_shape = self.backward_for_reconstructor(shape_pred_fake, reconstruct_shape, object)
        loss_R_head = self.backward_for_reconstructor(head_pred_fake, reconstruct_head, head)
        # print('successfully backward through reconstructor&backbone')

        # set_requires_grad(self.backbone, True)
        # set_requires_grad(self.reconstructor,False)
        # set_requires_grad(self.discriminator, False)
        # set_requires_grad(self.classifier, True)
        loss_C = self.backward_for_backbone(pred_class, label, 0)
        # print('successfully backward through backbone&classifier')

        # optimizer的更新统一放在最后，避免optimizer的内部操作对中间变量的改变导致无法顺利反向传播
        # self.optimizer_resnet.step()
        # self.optimizer_R.step()
        # self.optimizer_D.step()

        if optimize_flag == 'all':
            # 清空本次batch运算的梯度
            self.optimizer_shallow_backbone.step()
            self.optimizer_shallow_backbone.zero_grad()
            self.optimizer_shape_feature_extractor.step()
            self.optimizer_shape_feature_extractor.zero_grad()
            self.optimizer_head_feature_extractor.step()
            self.optimizer_head_feature_extractor.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_classifier.step()
            self.optimizer_classifier.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_R_shape.step()
            self.optimizer_R_shape.zero_grad()
            self.optimizer_R_head.step()
            self.optimizer_R_head.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_D_shape.step()
            self.optimizer_D_shape.zero_grad()
            self.optimizer_D_head.step()
            self.optimizer_D_head.zero_grad()
        elif optimize_flag == 'classification':
            # 清空本次batch运算的梯度
            self.optimizer_shallow_backbone.step()
            self.optimizer_shallow_backbone.zero_grad()
            self.optimizer_shape_feature_extractor.step()
            self.optimizer_shape_feature_extractor.zero_grad()
            self.optimizer_head_feature_extractor.step()
            self.optimizer_head_feature_extractor.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_classifier.step()
            self.optimizer_classifier.zero_grad()
        elif optimize_flag == 'reconstruction_discrimination':
             # 清空本次batch运算的梯度
            self.optimizer_R_shape.step()
            self.optimizer_R_shape.zero_grad()
            self.optimizer_R_head.step()
            self.optimizer_R_head.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_D_shape.step()
            self.optimizer_D_shape.zero_grad()
            self.optimizer_D_head.step()
            self.optimizer_D_head.zero_grad()
        elif optimize_flag == 'reconstruction': 
            # 清空本次batch运算的梯度
            self.optimizer_R_shape.step()
            self.optimizer_R_shape.zero_grad()
            self.optimizer_R_head.step()
            self.optimizer_R_head.zero_grad()

        
        return loss_C.cpu().data.numpy(), loss_R_shape.cpu().data.numpy(), loss_D_shape.cpu().data.numpy(), loss_R_head.cpu().data.numpy(), loss_D_head.cpu().data.numpy()

    def validation_check(self, pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, reconstruct_shape, reconstruct_head, object, head, label):
        loss_R_GAN_shape = self.GAN_criterion(shape_pred_fake, True)
        loss_R_GAN_head = self.GAN_criterion(head_pred_fake, True)
        loss_R_similarity_shape = self.similarity_criterion(reconstruct_shape, object)
        loss_R_similarity_head = self.similarity_criterion(reconstruct_head, head)
        loss_R_shape = loss_R_GAN_shape + loss_R_similarity_shape
        loss_R_head = loss_R_GAN_head + loss_R_similarity_head
        reg = 0
        loss_C = self.classification_criterion(pred_class, label) + reg
        loss_D_fake_shape = self.GAN_criterion(shape_pred_fake, False)
        loss_D_real_shape = self.GAN_criterion(shape_pred_real, True)
        loss_D_fake_head = self.GAN_criterion(head_pred_fake, False)
        loss_D_real_head = self.GAN_criterion(head_pred_real, True)
        loss_D_shape = (loss_D_real_shape + loss_D_fake_shape) * 0.5
        loss_D_head = (loss_D_real_head + loss_D_fake_head) * 0.5

        return loss_C.cpu().data.numpy(), loss_R_shape.cpu().data.numpy(), loss_D_shape.cpu().data.numpy(), loss_R_head.cpu().data.numpy(), loss_D_head.cpu().data.numpy()


class vgg_multi_concept_for_mammal(nn.Module):
    def __init__(self, args: dict):
        super(vgg_multi_concept_for_mammal, self).__init__()
        # block
        self.shallow_backbone = vgg_shallow(args['num_classes'])
        self.head_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        self.shape_feature_extractor = vgg_deep_shape(num_classes=args['num_classes'])
        self.torso_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        self.leg_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        self.classifier = VGG_Classifier(num_classes=args['num_classes'])
        self.reconstructor_shape = VGG_Reconstructor_shape()
        self.discriminator_shape = VGG_Discriminator()
        self.reconstructor_head = VGG_Reconstructor()
        self.discriminator_head = VGG_Discriminator()
        self.reconstructor_torso = VGG_Reconstructor()
        self.discriminator_torso = VGG_Discriminator()
        self.reconstructor_leg = VGG_Reconstructor()
        self.discriminator_leg = VGG_Discriminator()
        # criterion
        self.GAN_criterion = GANCriterion()
        self.GAN_L1_criterion = nn.L1Loss()
        self.classification_criterion = ClassificationCriterion()
        self.similarity_criterion = SimilarityCriterion()
        # optimizer
        self.optimizer_R_shape = torch.optim.Adam(self.reconstructor_shape.parameters(), lr=args['lr_R'])
        self.optimizer_D_shape = torch.optim.SGD(self.discriminator_shape.parameters(), lr=args['lr_D'])
        self.optimizer_R_head = torch.optim.Adam(self.reconstructor_head.parameters(), lr=args['lr_R'])
        self.optimizer_D_head = torch.optim.SGD(self.discriminator_head.parameters(), lr=args['lr_D'])
        self.optimizer_R_torso = torch.optim.Adam(self.reconstructor_torso.parameters(), lr=args['lr_R'])
        self.optimizer_D_torso = torch.optim.SGD(self.discriminator_torso.parameters(), lr=args['lr_D'])
        self.optimizer_R_leg = torch.optim.Adam(self.reconstructor_leg.parameters(), lr=args['lr_R'])
        self.optimizer_D_leg = torch.optim.SGD(self.discriminator_leg.parameters(), lr=args['lr_D'])
        self.optimizer_shallow_backbone = torch.optim.SGD(self.shallow_backbone.parameters(), lr=args['lr_resnet'])
        self.optimizer_shape_feature_extractor = torch.optim.SGD(self.shape_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_head_feature_extractor = torch.optim.SGD(self.head_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_torso_feature_extractor = torch.optim.SGD(self.torso_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_leg_feature_extractor = torch.optim.SGD(self.leg_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_classifier = torch.optim.SGD(self.classifier.parameters(), lr=args['lr_resnet'])


    def feature_concat(self, x1, x2, x3, x4):
        feature = torch.cat([x1, x2, x3, x4], 1)
        return feature


    def forward(self, x, label, head, torso, leg, shape):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        # print(head_feature)
        torso_feature = self.torso_feature_extractor(shallow_feature)
        leg_feature = self.leg_feature_extractor(shallow_feature)
        shape_feature, shape_feature_pooled = self.shape_feature_extractor(shallow_feature)
        # classifying
        feature = self.feature_concat(head_feature, torso_feature, leg_feature, shape_feature_pooled)
        pred_class = self.classifier(feature)
        # reconstructing
        reconstruct_shape = self.reconstructor_shape(shape_feature)
        reconstruct_head = self.reconstructor_head(head_feature)
        reconstruct_torso = self.reconstructor_torso(torso_feature)
        reconstruct_leg = self.reconstructor_leg(leg_feature)
        reconstrut_result = {
            'head': reconstruct_head.cpu().data.numpy(),
            'shape': reconstruct_shape.cpu().data.numpy(),
            'torso': reconstruct_torso.cpu().data.numpy(),
            'leg': reconstruct_leg.cpu().data.numpy()
        }

        # discriminating
        shape_pred_real = self.discriminator_shape(shape)
        shape_pred_fake = self.discriminator_shape(reconstruct_shape)
        head_pred_real = self.discriminator_head(head)
        head_pred_fake = self.discriminator_head(reconstruct_head)
        torso_pred_real = self.discriminator_torso(torso)
        torso_pred_fake = self.discriminator_torso(reconstruct_torso)
        leg_pred_real = self.discriminator_leg(leg)
        leg_pred_fake = self.discriminator_leg(reconstruct_leg)
        # calculate loss
        loss_C = self.calc_loss_C(pred_class, label)
        loss_R_head = self.calc_loss_R(head_pred_fake, reconstruct_head, head)
        loss_D_head = self.calc_loss_D(head_pred_fake, head_pred_real)
        loss_R_shape = self.calc_loss_R(shape_pred_fake, reconstruct_shape, shape)
        loss_D_shape = self.calc_loss_D(shape_pred_fake, shape_pred_real)
        loss_R_torso = self.calc_loss_R(torso_pred_fake, reconstruct_torso, torso)
        loss_D_torso = self.calc_loss_D(torso_pred_fake, torso_pred_real)
        loss_R_leg = self.calc_loss_R(leg_pred_fake, reconstruct_leg, leg)
        loss_D_leg = self.calc_loss_D(leg_pred_fake, leg_pred_real)

        return pred_class, reconstrut_result, loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg

    
    def inference(self, x):
         # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        # print(head_feature)
        torso_feature = self.torso_feature_extractor(shallow_feature)
        leg_feature = self.leg_feature_extractor(shallow_feature)
        shape_feature, shape_feature_pooled = self.shape_feature_extractor(shallow_feature)
        feature_dict = {
            'head': head_feature.cpu(),
            'torso': torso_feature.cpu(),
            'leg': leg_feature.cpu(),
            'shape': shape_feature_pooled.cpu()
        }
        # classifying
        feature = self.feature_concat(head_feature, torso_feature, leg_feature, shape_feature_pooled)
        pred_class = self.classifier(feature)
        # reconstructing
        reconstruct_shape = self.reconstructor_shape(shape_feature)
        reconstruct_head = self.reconstructor_head(head_feature)
        reconstruct_torso = self.reconstructor_torso(torso_feature)
        reconstruct_leg = self.reconstructor_leg(leg_feature)
        reconstrut_result = {
            'head': reconstruct_head.cpu().data.numpy(),
            'shape': reconstruct_shape.cpu().data.numpy(),
            'torso': reconstruct_torso.cpu().data.numpy(),
            'leg': reconstruct_leg.cpu().data.numpy()
        }
        return feature_dict, pred_class, reconstrut_result


    def calc_loss_D(self, pred_fake, pred_real):
        loss_D_fake = self.GAN_criterion(pred_fake, False)
        loss_D_real = self.GAN_criterion(pred_real, True)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D


    def calc_loss_R(self, pred_fake, reconstruct_image, ori_image):
        loss_R_GAN = self.GAN_criterion(pred_fake, True)
        loss_R_similarity = self.similarity_criterion(reconstruct_image, ori_image)
        # 测试放弃鉴别器损失
        loss_R = loss_R_similarity
        # loss_R = loss_R_GAN + loss_R_similarity
        return loss_R


    def calc_loss_C(self, pred_class, label):
        loss_C = self.classification_criterion(pred_class, label)
        return loss_C


    def backward_for_discriminator(self, loss_D):
        loss_D.backward(retain_graph=True)


    def backward_for_reconstructor(self, loss_R):
        loss_R.backward(retain_graph=True)


    def backward_for_backbone(self, loss_C):
        loss_C.backward()


    def optimize_parameters(self, loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg, flag):
        # pay attention to the order of block during backward
        # discriminator -> reconstructor & backbone -> backbone & classifier
        if flag == 'all':
            set_requires_grad(self.shallow_backbone, False)
            set_requires_grad(self.head_feature_extractor, False)
            set_requires_grad(self.shape_feature_extractor, False)
            set_requires_grad(self.torso_feature_extractor, False)
            set_requires_grad(self.leg_feature_extractor, False)
            set_requires_grad(self.reconstructor_head, False)
            set_requires_grad(self.reconstructor_shape, False)
            set_requires_grad(self.reconstructor_torso, False)
            set_requires_grad(self.reconstructor_leg, False)
            self.backward_for_discriminator(loss_D_head)
            self.backward_for_discriminator(loss_D_shape)
            self.backward_for_discriminator(loss_D_torso)
            self.backward_for_discriminator(loss_D_leg)
            # print('successfully backward through discriminator')
            
            set_requires_grad(self.shallow_backbone, True)
            set_requires_grad(self.head_feature_extractor, True)
            set_requires_grad(self.shape_feature_extractor, True)
            set_requires_grad(self.torso_feature_extractor, True)
            set_requires_grad(self.leg_feature_extractor, True)
            set_requires_grad(self.reconstructor_head, True)
            set_requires_grad(self.reconstructor_shape, True)
            set_requires_grad(self.reconstructor_torso, True)
            set_requires_grad(self.reconstructor_leg, True)

            self.backward_for_reconstructor(loss_R_head)
            self.backward_for_reconstructor(loss_R_shape)
            self.backward_for_reconstructor(loss_R_torso)
            self.backward_for_reconstructor(loss_R_leg)
            # print('successfully backward through reconstructor&backbone')

            self.backward_for_backbone(loss_C)
            # print('successfully backward through backbone&classifier')

            # 清空本次batch运算的梯度
            self.optimizer_shallow_backbone.step()
            self.optimizer_shallow_backbone.zero_grad()
            self.optimizer_shape_feature_extractor.step()
            self.optimizer_shape_feature_extractor.zero_grad()
            self.optimizer_head_feature_extractor.step()
            self.optimizer_head_feature_extractor.zero_grad()
            self.optimizer_torso_feature_extractor.step()
            self.optimizer_torso_feature_extractor.zero_grad()
            self.optimizer_leg_feature_extractor.step()
            self.optimizer_leg_feature_extractor.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_classifier.step()
            self.optimizer_classifier.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_R_shape.step()
            self.optimizer_R_shape.zero_grad()
            self.optimizer_R_head.step()
            self.optimizer_R_head.zero_grad()
            self.optimizer_R_torso.step()
            self.optimizer_R_torso.zero_grad()
            self.optimizer_R_leg.step()
            self.optimizer_R_leg.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_D_shape.step()
            self.optimizer_D_shape.zero_grad()
            self.optimizer_D_head.step()
            self.optimizer_D_head.zero_grad()
            self.optimizer_D_torso.step()
            self.optimizer_D_torso.zero_grad()
            self.optimizer_D_leg.step()
            self.optimizer_D_leg.zero_grad()
        if flag == 'split_train':
            set_requires_grad(self.shallow_backbone, False)
            set_requires_grad(self.head_feature_extractor, False)
            set_requires_grad(self.shape_feature_extractor, False)
            set_requires_grad(self.torso_feature_extractor, False)
            set_requires_grad(self.leg_feature_extractor, False)
            set_requires_grad(self.reconstructor_head, False)
            set_requires_grad(self.reconstructor_shape, False)
            set_requires_grad(self.reconstructor_torso, False)
            set_requires_grad(self.reconstructor_leg, False)
            self.backward_for_discriminator(loss_D_head)
            self.backward_for_discriminator(loss_D_shape)
            self.backward_for_discriminator(loss_D_torso)
            self.backward_for_discriminator(loss_D_leg)

            set_requires_grad(self.reconstructor_head, True)
            set_requires_grad(self.reconstructor_shape, True)
            set_requires_grad(self.reconstructor_torso, True)
            set_requires_grad(self.reconstructor_leg, True)
            self.backward_for_reconstructor(loss_R_head)
            self.backward_for_reconstructor(loss_R_shape)
            self.backward_for_reconstructor(loss_R_torso)
            self.backward_for_reconstructor(loss_R_leg)

            self.backward_for_backbone(loss_C)

            # 清空本次batch运算的梯度
            self.optimizer_shallow_backbone.step()
            self.optimizer_shallow_backbone.zero_grad()
            self.optimizer_shape_feature_extractor.step()
            self.optimizer_shape_feature_extractor.zero_grad()
            self.optimizer_head_feature_extractor.step()
            self.optimizer_head_feature_extractor.zero_grad()
            self.optimizer_torso_feature_extractor.step()
            self.optimizer_torso_feature_extractor.zero_grad()
            self.optimizer_leg_feature_extractor.step()
            self.optimizer_leg_feature_extractor.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_classifier.step()
            self.optimizer_classifier.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_R_shape.step()
            self.optimizer_R_shape.zero_grad()
            self.optimizer_R_head.step()
            self.optimizer_R_head.zero_grad()
            self.optimizer_R_torso.step()
            self.optimizer_R_torso.zero_grad()
            self.optimizer_R_leg.step()
            self.optimizer_R_leg.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_D_shape.step()
            self.optimizer_D_shape.zero_grad()
            self.optimizer_D_head.step()
            self.optimizer_D_head.zero_grad()
            self.optimizer_D_torso.step()
            self.optimizer_D_torso.zero_grad()
            self.optimizer_D_leg.step()
            self.optimizer_D_leg.zero_grad()
        if flag == 'C':
            self.backward_for_backbone(loss_C)
            self.optimizer_shallow_backbone.step()
            self.optimizer_shallow_backbone.zero_grad()
            self.optimizer_shape_feature_extractor.step()
            self.optimizer_shape_feature_extractor.zero_grad()
            self.optimizer_head_feature_extractor.step()
            self.optimizer_head_feature_extractor.zero_grad()
            self.optimizer_torso_feature_extractor.step()
            self.optimizer_torso_feature_extractor.zero_grad()
            self.optimizer_leg_feature_extractor.step()
            self.optimizer_leg_feature_extractor.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_classifier.step()
            self.optimizer_classifier.zero_grad()
        if flag == 'R_D':
            set_requires_grad(self.shallow_backbone, False)
            set_requires_grad(self.head_feature_extractor, False)
            set_requires_grad(self.shape_feature_extractor, False)
            set_requires_grad(self.torso_feature_extractor, False)
            set_requires_grad(self.leg_feature_extractor, False)
            set_requires_grad(self.reconstructor_head, False)
            set_requires_grad(self.reconstructor_shape, False)
            set_requires_grad(self.reconstructor_torso, False)
            set_requires_grad(self.reconstructor_leg, False)
            self.backward_for_discriminator(loss_D_head)
            self.backward_for_discriminator(loss_D_shape)
            self.backward_for_discriminator(loss_D_torso)
            self.backward_for_discriminator(loss_D_leg)

            set_requires_grad(self.reconstructor_head, True)
            set_requires_grad(self.reconstructor_shape, True)
            set_requires_grad(self.reconstructor_torso, True)
            set_requires_grad(self.reconstructor_leg, True)
            self.backward_for_reconstructor(loss_R_head)
            self.backward_for_reconstructor(loss_R_shape)
            self.backward_for_reconstructor(loss_R_torso)
            self.backward_for_reconstructor(loss_R_leg)
            
            # 清空本次batch运算的梯度
            self.optimizer_R_shape.step()
            self.optimizer_R_shape.zero_grad()
            self.optimizer_R_head.step()
            self.optimizer_R_head.zero_grad()
            self.optimizer_R_torso.step()
            self.optimizer_R_torso.zero_grad()
            self.optimizer_R_leg.step()
            self.optimizer_R_leg.zero_grad()
            # 清空本次batch运算的梯度
            self.optimizer_D_shape.step()
            self.optimizer_D_shape.zero_grad()
            self.optimizer_D_head.step()
            self.optimizer_D_head.zero_grad()
            self.optimizer_D_torso.step()
            self.optimizer_D_torso.zero_grad()
            self.optimizer_D_leg.step()
            self.optimizer_D_leg.zero_grad()


        return loss_C.cpu().data.numpy(), loss_R_head.cpu().data.numpy(), loss_D_head.cpu().data.numpy(), loss_R_shape.cpu().data.numpy(), loss_D_shape.cpu().data.numpy(), loss_R_torso.cpu().data.numpy(), loss_D_torso.cpu().data.numpy(), loss_R_leg.cpu().data.numpy(), loss_D_leg.cpu().data.numpy()


    def validation_check(self, loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg):
        return loss_C.cpu().data.numpy(), loss_R_head.cpu().data.numpy(), loss_D_head.cpu().data.numpy(), loss_R_shape.cpu().data.numpy(), loss_D_shape.cpu().data.numpy(), loss_R_torso.cpu().data.numpy(), loss_D_torso.cpu().data.numpy(), loss_R_leg.cpu().data.numpy(), loss_D_leg.cpu().data.numpy()


    def learning(self, x, label, head, torso, leg, shape, flag):
        pred_class, reconstruct_result, loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg = self.forward(x, label, head, torso, leg, shape)
        loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg = self.optimize_parameters(loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg, flag)
        loss_curr = loss_C + loss_R_head + loss_D_head + loss_R_torso + loss_D_torso + loss_R_shape + loss_R_shape + loss_R_leg + loss_D_leg
        return pred_class, reconstruct_result, loss_curr


    def validating(self, x, label, head, torso, leg, shape):
        pred_class, reconstruct_result, loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg = self.forward(x, label, head, torso, leg, shape)
        loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg = self.validation_check(loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg)
        loss_curr = loss_C + loss_R_head + loss_D_head + loss_R_torso + loss_D_torso + loss_R_shape + loss_R_shape + loss_R_leg + loss_D_leg
        return pred_class, reconstruct_result, loss_curr 



