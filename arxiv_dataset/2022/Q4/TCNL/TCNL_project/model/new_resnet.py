from model.resnet_backbone import *
from utils.loss_utils import SimilarityCriterion, GANCriterion, ClassificationCriterion, MammalContrastiveCriterion
from model.nets_factory import set_requires_grad, bilateral_prompt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools



class new_resnet(nn.Module):
    def __init__(self, args: dict):
        super(new_resnet, self).__init__()
        # block
        self.shallow_backbone = resnet_shallow(args['num_classes'])
        self.head_feature_extractor = resnet_deep_head(num_classes=args['num_classes'])
        self.shape_feature_extractor = resnet_deep_shape(num_classes=args['num_classes'])
        self.torso_feature_extractor = resnet_deep_torso(num_classes=args['num_classes'])
        self.leg_feature_extractor = resnet_deep_leg(num_classes=args['num_classes'])
        self.classifier = ResNet_Classifier(num_classes=args['num_classes'])
        self.reconstructor_shape = ResNet_Reconstructor_shape()
        self.discriminator_shape = ResNet_Discriminator()
        self.reconstructor_head = ResNet_Reconstructor_head()
        self.discriminator_head = ResNet_Discriminator()
        self.reconstructor_torso = ResNet_Reconstructor_torso()
        self.discriminator_torso = ResNet_Discriminator()
        self.reconstructor_leg = ResNet_Reconstructor_leg()
        self.discriminator_leg = ResNet_Discriminator()
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
        shallow_feature = self.shallow_backbone(x, label)
        head_feature = self.head_feature_extractor(shallow_feature)
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
        # reconstrut_result = {
        #     'head': reconstruct_head.cpu().data.numpy(),
        #     'shape': reconstruct_shape.cpu().data.numpy(),
        #     'torso': reconstruct_torso.cpu().data.numpy(),
        #     'leg': reconstruct_leg.cpu().data.numpy()
        # }
        # discriminating
        shape_pred_real = self.discriminator_shape(shape)
        shape_pred_fake = self.discriminator_shape(reconstruct_shape)
        head_pred_real = self.discriminator_head(head)
        head_pred_fake = self.discriminator_head(reconstruct_head)
        torso_pred_real = self.discriminator_torso(torso)
        torso_pred_fake = self.discriminator_torso(reconstruct_torso)
        leg_pred_real = self.discriminator_leg(leg)
        leg_pred_fake = self.discriminator_leg(reconstruct_leg)

        return pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, torso_pred_fake, torso_pred_real, leg_pred_fake, leg_pred_real, reconstruct_shape, reconstruct_head, reconstruct_torso, reconstruct_leg, shape, head, torso, leg

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

    def optimize_parameters(self, pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, torso_pred_fake, torso_pred_real, leg_pred_fake, leg_pred_real, reconstruct_shape, reconstruct_head, reconstruct_torso, reconstruct_leg, shape, head, torso, leg, label, optimize_flag):
        # pay attention to the order of block during backward
        # discriminator -> reconstructor & backbone -> backbone & classifier

        set_requires_grad(self.shallow_backbone, False)
        set_requires_grad(self.head_feature_extractor, False)
        set_requires_grad(self.shape_feature_extractor, False)
        set_requires_grad(self.torso_feature_extractor, False)
        set_requires_grad(self.leg_feature_extractor, False)
        set_requires_grad(self.reconstructor_head, False)
        set_requires_grad(self.reconstructor_shape, False)
        set_requires_grad(self.reconstructor_torso, False)
        set_requires_grad(self.reconstructor_leg, False)
        loss_D_shape = self.backward_for_discriminator(shape_pred_fake, shape_pred_real)
        loss_D_head = self.backward_for_discriminator(head_pred_fake, head_pred_real)
        loss_D_torso = self.backward_for_discriminator(torso_pred_fake, torso_pred_real)
        loss_D_leg = self.backward_for_discriminator(leg_pred_fake, leg_pred_real)
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

        loss_R_shape = self.backward_for_reconstructor(shape_pred_fake, reconstruct_shape, shape)
        loss_R_head = self.backward_for_reconstructor(head_pred_fake, reconstruct_head, head)
        loss_R_torso = self.backward_for_reconstructor(torso_pred_fake, reconstruct_torso, torso)
        loss_R_leg = self.backward_for_reconstructor(leg_pred_fake, reconstruct_leg, leg)

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
        print(loss_C.cpu().data.numpy(), loss_R_shape.cpu().data.numpy(), loss_D_shape.cpu().data.numpy(), loss_R_head.cpu().data.numpy(), loss_D_head.cpu().data.numpy(), loss_R_torso.cpu().data.numpy(), loss_D_torso.cpu().data.numpy(), loss_R_leg.cpu().data.numpy(), loss_D_leg.cpu().data.numpy())
        return loss_C.cpu().data.numpy() + loss_R_shape.cpu().data.numpy() + loss_D_shape.cpu().data.numpy() + loss_R_head.cpu().data.numpy() + loss_D_head.cpu().data.numpy() + loss_R_torso.cpu().data.numpy() + loss_D_torso.cpu().data.numpy() + loss_R_leg.cpu().data.numpy() + loss_D_leg.cpu().data.numpy()
        # return loss_C.cpu().data.numpy(), loss_R_shape.cpu().data.numpy(), loss_D_shape.cpu().data.numpy(), loss_R_head.cpu().data.numpy(), loss_D_head.cpu().data.numpy(), loss_R_torso.cpu().data.numpy(), loss_D_torso.cpu().data.numpy(), loss_R_leg.cpu().data.numpy(), loss_D_leg.cpu().data.numpy()

    def validation_check(self, pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, torso_pred_fake, torso_pred_real, leg_pred_fake, leg_pred_real, reconstruct_shape, reconstruct_head, reconstruct_torso, reconstruct_leg, shape, head, torso, leg, label):
        loss_R_GAN_shape = self.GAN_criterion(shape_pred_fake, True)
        loss_R_GAN_head = self.GAN_criterion(head_pred_fake, True)
        loss_R_GAN_torso = self.GAN_criterion(torso_pred_fake, True)
        loss_R_GAN_leg = self.GAN_criterion(leg_pred_fake, True)
        loss_R_similarity_shape = self.similarity_criterion(reconstruct_shape, shape)
        loss_R_similarity_head = self.similarity_criterion(reconstruct_head, head)
        loss_R_similarity_torso = self.similarity_criterion(reconstruct_torso, torso)
        loss_R_similarity_leg = self.similarity_criterion(reconstruct_leg, leg)
        loss_R_shape = loss_R_GAN_shape + loss_R_similarity_shape
        loss_R_head = loss_R_GAN_head + loss_R_similarity_head
        loss_R_torso = loss_R_GAN_torso + loss_R_similarity_torso
        loss_R_leg = loss_R_GAN_leg + loss_R_similarity_leg
        reg = 0
        loss_C = self.classification_criterion(pred_class, label) + reg
        loss_D_fake_shape = self.GAN_criterion(shape_pred_fake, False)
        loss_D_real_shape = self.GAN_criterion(shape_pred_real, True)
        loss_D_fake_head = self.GAN_criterion(head_pred_fake, False)
        loss_D_real_head = self.GAN_criterion(head_pred_real, True)
        loss_D_fake_torso = self.GAN_criterion(torso_pred_fake, False)
        loss_D_real_torso = self.GAN_criterion(torso_pred_real, True)
        loss_D_fake_leg = self.GAN_criterion(leg_pred_fake, False)
        loss_D_real_leg = self.GAN_criterion(leg_pred_real, True)
        loss_D_shape = (loss_D_real_shape + loss_D_fake_shape) * 0.5
        loss_D_head = (loss_D_real_head + loss_D_fake_head) * 0.5
        loss_D_torso = (loss_D_real_torso + loss_D_fake_torso) * 0.5
        loss_D_leg = (loss_D_real_leg + loss_D_fake_leg) * 0.5
        return loss_C.cpu().data.numpy() + loss_R_shape.cpu().data.numpy() + loss_D_shape.cpu().data.numpy() + loss_R_head.cpu().data.numpy() + loss_D_head.cpu().data.numpy() + loss_R_torso.cpu().data.numpy() + loss_D_torso.cpu().data.numpy() + loss_R_leg.cpu().data.numpy() + loss_D_leg.cpu().data.numpy()
        # return loss_C.cpu().data.numpy(), loss_R_shape.cpu().data.numpy(), loss_D_shape.cpu().data.numpy(), loss_R_head.cpu().data.numpy(), loss_D_head.cpu().data.numpy()


class resnet_multi_concept_for_mammal(nn.Module):
    def __init__(self, args: dict):
        super(resnet_multi_concept_for_mammal, self).__init__()
        # block
        self.shallow_backbone = resnet_shallow(args['num_classes'])
        self.head_feature_extractor = resnet_deep_head(num_classes=args['num_classes'])
        self.shape_feature_extractor = resnet_deep_shape(num_classes=args['num_classes'])
        self.torso_feature_extractor = resnet_deep_torso(num_classes=args['num_classes'])
        self.leg_feature_extractor = resnet_deep_leg(num_classes=args['num_classes'])
        self.classifier = ResNet_Classifier(num_classes=args['num_classes'])
        self.reconstructor_shape = ResNet_Reconstructor_shape()
        self.discriminator_shape = ResNet_Discriminator()
        self.reconstructor_head = ResNet_Reconstructor_head()
        self.discriminator_head = ResNet_Discriminator()
        self.reconstructor_torso = ResNet_Reconstructor_torso()
        self.discriminator_torso = ResNet_Discriminator()
        self.reconstructor_leg = ResNet_Reconstructor_leg()
        self.discriminator_leg = ResNet_Discriminator()
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
        print(feature.shape)
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
        # loss_R = loss_R_similarity
        loss_R = loss_R_GAN + loss_R_similarity
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
        print(loss_C)
        loss_curr = loss_C + loss_R_head + loss_D_head + loss_R_torso + loss_D_torso + loss_R_shape + loss_R_shape + loss_R_leg + loss_D_leg
        return pred_class, reconstruct_result, loss_curr


    def validating(self, x, label, head, torso, leg, shape):
        pred_class, reconstruct_result, loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg = self.forward(x, label, head, torso, leg, shape)
        loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg = self.validation_check(loss_C, loss_R_head, loss_D_head, loss_R_shape, loss_D_shape, loss_R_torso, loss_D_torso, loss_R_leg, loss_D_leg)
        loss_curr = loss_C + loss_R_head + loss_D_head + loss_R_torso + loss_D_torso + loss_R_shape + loss_R_shape + loss_R_leg + loss_D_leg
        return pred_class, reconstruct_result, loss_curr


class resnet_multi_concept_for_scene(nn.Module):
    def __init__(self, args: dict):
        super(resnet_multi_concept_for_scene, self).__init__()
        # block
        self.shallow_backbone = resnet_shallow(args['num_classes'])
        self.bed_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        # self.bedsidetable_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        # self.lamp_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        self.sofa_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        # self.chair_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        # self.table_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        self.shelf_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.seat_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        # self.screen_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        # self.stage_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        # self.rest_feature_extractor = resnet_deep(num_classes=args['num_classes'])

        self.classifier = ResNet_Classifier(num_classes=args['num_classes'])
        self.reconstructor_bed = ResNet_Reconstructor()
        self.discriminator_bed = ResNet_Discriminator()
        # self.reconstructor_bedsidetable = VGG_Reconstructor()
        # self.discriminator_bedsidetable = VGG_Discriminator()
        # self.reconstructor_lamp = VGG_Reconstructor()
        # self.discriminator_lamp = VGG_Discriminator()
        self.reconstructor_sofa = ResNet_Reconstructor()
        self.discriminator_sofa = ResNet_Discriminator()
        # self.reconstructor_chair = VGG_Reconstructor()
        # self.discriminator_chair = VGG_Discriminator()
        # self.reconstructor_table = VGG_Reconstructor()
        # self.discriminator_table = VGG_Discriminator()
        self.reconstructor_shelf = ResNet_Reconstructor()
        self.discriminator_shelf = ResNet_Discriminator()
        self.reconstructor_seat = ResNet_Reconstructor()
        self.discriminator_seat = ResNet_Discriminator()
        # self.reconstructor_screen = VGG_Reconstructor()
        # self.discriminator_screen = VGG_Discriminator()
        # self.reconstructor_stage = VGG_Reconstructor()
        # self.discriminator_stage = VGG_Discriminator()
        # criterion
        self.GAN_criterion = GANCriterion()
        self.GAN_L1_criterion = nn.L1Loss()
        self.classification_criterion = ClassificationCriterion()
        self.similarity_criterion = SimilarityCriterion()
        # optimizer
        self.optimizer_R_bed = torch.optim.Adam(self.reconstructor_bed.parameters(), lr=args['lr_R'])
        self.optimizer_D_bed = torch.optim.SGD(self.discriminator_bed.parameters(), lr=args['lr_D'])
        # self.optimizer_R_bedsidetable = torch.optim.Adam(self.reconstructor_bedsidetable.parameters(), lr=args['lr_R'])
        # self.optimizer_D_bedsidetable = torch.optim.SGD(self.discriminator_bedsidetable.parameters(), lr=args['lr_D'])
        # self.optimizer_R_lamp = torch.optim.Adam(self.reconstructor_lamp.parameters(), lr=args['lr_R'])
        # self.optimizer_D_lamp = torch.optim.SGD(self.discriminator_lamp.parameters(), lr=args['lr_D'])
        self.optimizer_R_sofa = torch.optim.Adam(self.reconstructor_sofa.parameters(), lr=args['lr_R'])
        self.optimizer_D_sofa = torch.optim.SGD(self.discriminator_sofa.parameters(), lr=args['lr_D'])
        # self.optimizer_R_chair = torch.optim.Adam(self.reconstructor_chair.parameters(), lr=args['lr_R'])
        # self.optimizer_D_chair = torch.optim.SGD(self.discriminator_chair.parameters(), lr=args['lr_D'])
        # self.optimizer_R_table = torch.optim.Adam(self.reconstructor_table.parameters(), lr=args['lr_R'])
        # self.optimizer_D_table = torch.optim.SGD(self.discriminator_table.parameters(), lr=args['lr_D'])
        self.optimizer_R_shelf = torch.optim.Adam(self.reconstructor_shelf.parameters(), lr=args['lr_R'])
        self.optimizer_D_shelf = torch.optim.SGD(self.discriminator_shelf.parameters(), lr=args['lr_D'])
        self.optimizer_R_seat = torch.optim.Adam(self.reconstructor_seat.parameters(), lr=args['lr_R'])
        self.optimizer_D_seat = torch.optim.SGD(self.discriminator_seat.parameters(), lr=args['lr_D'])
        # self.optimizer_R_screen = torch.optim.Adam(self.reconstructor_screen.parameters(), lr=args['lr_R'])
        # self.optimizer_D_screen = torch.optim.SGD(self.discriminator_screen.parameters(), lr=args['lr_D'])
        # self.optimizer_R_stage = torch.optim.Adam(self.reconstructor_stage.parameters(), lr=args['lr_R'])
        # self.optimizer_D_stage = torch.optim.SGD(self.discriminator_stage.parameters(), lr=args['lr_D'])
        self.optimizer_shallow_backbone = torch.optim.SGD(self.shallow_backbone.parameters(), lr=args['lr_resnet'])
        self.optimizer_bed_feature_extractor = torch.optim.SGD(self.bed_feature_extractor.parameters(), lr=args['lr_resnet'])
        # self.optimizer_bedsidetable_feature_extractor = torch.optim.SGD(self.bedsidetable_feature_extractor.parameters(), lr=args['lr_resnet'])
        # self.optimizer_lamp_feature_extractor = torch.optim.SGD(self.lamp_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_sofa_feature_extractor = torch.optim.SGD(self.sofa_feature_extractor.parameters(), lr=args['lr_resnet'])
        # self.optimizer_chair_feature_extractor = torch.optim.SGD(self.chair_feature_extractor.parameters(), lr=args['lr_resnet'])
        # self.optimizer_table_feature_extractor = torch.optim.SGD(self.table_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_shelf_feature_extractor = torch.optim.SGD(self.shelf_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_seat_feature_extractor = torch.optim.SGD(self.seat_feature_extractor.parameters(), lr=args['lr_resnet'])
        # self.optimizer_screen_feature_extractor = torch.optim.SGD(self.screen_feature_extractor.parameters(), lr=args['lr_resnet'])
        # self.optimizer_stage_feature_extractor = torch.optim.SGD(self.stage_feature_extractor.parameters(), lr=args['lr_resnet'])
        # self.optimizer_rest_feature_extractor = torch.optim.SGD(self.rest_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_classifier = torch.optim.SGD(self.classifier.parameters(), lr=args['lr_resnet'])


    def feature_concat(self, x1, x2, x3, x4):
        feature = torch.cat([x1, x2, x3, x4], 1)
        return feature


    def forward(self, x, label, bed, sofa, shelf, seat):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        bed_feature = self.bed_feature_extractor(shallow_feature)
        # bedsidetable_feature = self.bedsidetable_feature_extractor(shallow_feature)
        # lamp_feature = self.lamp_feature_extractor(shallow_feature)
        sofa_feature = self.sofa_feature_extractor(shallow_feature)
        # chair_feature = self.chair_feature_extractor(shallow_feature)
        # table_feature = self.table_feature_extractor(shallow_feature)
        shelf_feature = self.shelf_feature_extractor(shallow_feature)
        seat_feature = self.seat_feature_extractor(shallow_feature)
        # screen_feature = self.screen_feature_extractor(shallow_feature)
        # stage_feature = self.stage_feature_extractor(shallow_feature)
        rest_feature =  self.rest_feature_extractor(shallow_feature)
        # classifying
        feature = self.feature_concat(bed_feature, sofa_feature, shelf_feature, seat_feature, rest_feature)
        pred_class = self.classifier(feature)
        # reconstructing
        reconstruct_bed = self.reconstructor_bed(bed_feature)
        # reconstruct_bedsidetable = self.reconstructor_bedsidetable(bedsidetable_feature)
        # reconstruct_lamp = self.reconstructor_lamp(lamp_feature)
        reconstruct_sofa = self.reconstructor_sofa(sofa_feature)
        # reconstruct_chair = self.reconstructor_chair(chair_feature)
        # reconstruct_table = self.reconstructor_table(table_feature)
        reconstruct_shelf = self.reconstructor_shelf(shelf_feature)
        reconstruct_seat = self.reconstructor_seat(seat_feature)
        # reconstruct_screen = self.reconstructor_screen(screen_feature)
        # reconstruct_stage = self.reconstructor_stage(stage_feature)
        reconstruct_result = {
            'bed': reconstruct_bed.cpu().data.numpy(),
            # 'bedsidetable': reconstruct_bedsidetable.cpu().data.numpy(),
            # 'lamp': reconstruct_lamp.cpu().data.numpy(),
            'sofa': reconstruct_sofa.cpu().data.numpy(),
            # 'chair': reconstruct_chair.cpu().data.numpy(),
            # 'table': reconstruct_table.cpu().data.numpy(),
            'shelf': reconstruct_shelf.cpu().data.numpy(),
            'seat': reconstruct_seat.cpu().data.numpy(),
            # 'screen': reconstruct_screen.cpu().data.numpy(),
            # 'stage': reconstruct_stage.cpu().data.numpy()
        }

        # discriminating
        bed_pred_real = self.discriminator_bed(bed)
        bed_pred_fake = self.discriminator_bed(reconstruct_bed)
        # bedsidetable_pred_real = self.discriminator_bedsidetable(bedsidetable)
        # bedsidetable_pred_fake = self.discriminator_bedsidetable(bedsidetable)
        # lamp_pred_real = self.discriminator_lamp(lamp)
        # lamp_pred_fake = self.discriminator_lamp(reconstruct_lamp)
        sofa_pred_real = self.discriminator_sofa(sofa)
        sofa_pred_fake = self.discriminator_sofa(reconstruct_sofa)
        # chair_pred_real = self.discriminator_chair(chair)
        # chair_pred_fake = self.discriminator_chair(reconstruct_chair)
        # table_pred_real = self.discriminator_table(table)
        # table_pred_fake = self.discriminator_table(reconstruct_table)
        shelf_pred_real = self.discriminator_shelf(shelf)
        shelf_pred_fake = self.discriminator_shelf(reconstruct_shelf)
        seat_pred_real = self.discriminator_seat(seat)
        seat_pred_fake = self.discriminator_seat(reconstruct_seat)
        # screen_pred_real = self.discriminator_screen(screen)
        # screen_pred_fake = self.discriminator_screen(reconstruct_screen)
        # stage_pred_real = self.discriminator_stage(stage)
        # stage_pred_fake = self.discriminator_stage(reconstruct_stage)

        # calculate loss
        loss_C = self.calc_loss_C(pred_class, label)
        loss_R_bed = self.calc_loss_R(bed_pred_fake, reconstruct_bed, bed)
        loss_D_bed = self.calc_loss_D(bed_pred_fake, bed_pred_real)
        # loss_R_bedsidetable = self.calc_loss_R(bedsidetable_pred_fake, reconstruct_bedsidetable, bedsidetable)
        # loss_D_bedsidetable = self.calc_loss_D(bedsidetable_pred_fake, bedsidetable_pred_real)
        # loss_R_lamp = self.calc_loss_R(lamp_pred_fake, reconstruct_lamp, lamp)
        # loss_D_lamp = self.calc_loss_D(lamp_pred_fake, lamp_pred_real)
        loss_R_sofa = self.calc_loss_R(sofa_pred_fake, reconstruct_sofa, sofa)
        loss_D_sofa = self.calc_loss_D(sofa_pred_fake, sofa_pred_real)
        # loss_R_chair = self.calc_loss_R(chair_pred_fake, reconstruct_chair, chair)
        # loss_D_chair = self.calc_loss_D(chair_pred_fake, chair_pred_real)
        # loss_R_table = self.calc_loss_R(table_pred_fake, reconstruct_table, table)
        # loss_D_table = self.calc_loss_D(table_pred_fake, table_pred_real)
        loss_R_shelf = self.calc_loss_R(shelf_pred_fake, reconstruct_shelf, shelf)
        loss_D_shelf = self.calc_loss_D(shelf_pred_fake, shelf_pred_real)
        loss_R_seat = self.calc_loss_R(seat_pred_fake, reconstruct_seat, seat)
        loss_D_seat = self.calc_loss_D(seat_pred_fake, seat_pred_real)
        # loss_R_screen = self.calc_loss_R(screen_pred_fake, reconstruct_screen, screen)
        # loss_D_screen = self.calc_loss_D(screen_pred_fake, screen_pred_real)
        # loss_R_stage = self.calc_loss_R(stage_pred_fake, reconstruct_stage, stage)
        # loss_D_stage = self.calc_loss_D(stage_pred_fake, stage_pred_real)


        return (pred_class, reconstruct_result, loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, 
        loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat)

    
    def inference(self, x):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        bed_feature = self.bed_feature_extractor(shallow_feature)
        # print(head_feature)
        sofa_feature = self.sofa_feature_extractor(shallow_feature)
        seat_feature = self.seat_feature_extractor(shallow_feature)
        shelf_feature = self.shelf_feature_extractor(shallow_feature)
        # rest_feature = self.rest_feature_extractor(shallow_feature)
        feature_dict = {
            'bed': bed_feature.cpu(),
            'sofa': sofa_feature.cpu(),
            'seat': seat_feature.cpu(),
            'shelf': shelf_feature.cpu()
        }
        # classifying
        feature = self.feature_concat(bed_feature, sofa_feature, shelf_feature, seat_feature)
        pred_class = self.classifier(feature)
  
        reconstruct_bed = self.reconstructor_bed(bed_feature)
        reconstruct_sofa = self.reconstructor_sofa(sofa_feature)
        reconstruct_shelf = self.reconstructor_shelf(shelf_feature)
        reconstruct_seat = self.reconstructor_seat(seat_feature)
        reconstruct_result = {
            'bed': reconstruct_bed.cpu().data.numpy(),
            'sofa': reconstruct_sofa.cpu().data.numpy(),
            'shelf': reconstruct_shelf.cpu().data.numpy(),
            'seat': reconstruct_seat.cpu().data.numpy()
        }
        return feature_dict, pred_class, reconstruct_result



    def calc_loss_D(self, pred_fake, pred_real):
        loss_D_fake = self.GAN_criterion(pred_fake, False)
        loss_D_real = self.GAN_criterion(pred_real, True)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D


    def calc_loss_R(self, pred_fake, reconstruct_image, ori_image):
        loss_R_GAN = self.GAN_criterion(pred_fake, True)
        loss_R_similarity = self.similarity_criterion(reconstruct_image, ori_image)
        # 测试放弃鉴别器损失
        # loss_R = loss_R_similarity
        loss_R = loss_R_GAN + loss_R_similarity
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


    def optimize_parameters(self, loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat):
        # pay attention to the order of block during backward
        # discriminator -> reconstructor & backbone -> backbone & classifier

        set_requires_grad(self.shallow_backbone, False)
        set_requires_grad(self.bed_feature_extractor, False)
        # set_requires_grad(self.bedsidetable_feature_extractor, False)
        # set_requires_grad(self.lamp_feature_extractor, False)
        set_requires_grad(self.sofa_feature_extractor, False)
        # set_requires_grad(self.chair_feature_extractor, False)
        # set_requires_grad(self.table_feature_extractor, False)
        set_requires_grad(self.shelf_feature_extractor, False)
        set_requires_grad(self.seat_feature_extractor, False)
        # set_requires_grad(self.screen_feature_extractor, False)
        # set_requires_grad(self.stage_feature_extractor, False)
        set_requires_grad(self.reconstructor_bed, False)
        # set_requires_grad(self.reconstructor_bedsidetable, False)
        # set_requires_grad(self.reconstructor_lamp, False)
        set_requires_grad(self.reconstructor_sofa, False)
        # set_requires_grad(self.reconstructor_chair, False)
        # set_requires_grad(self.reconstructor_table, False)
        set_requires_grad(self.reconstructor_shelf, False)
        set_requires_grad(self.reconstructor_seat, False)
        # set_requires_grad(self.reconstructor_screen, False)
        # set_requires_grad(self.reconstructor_stage, False)
        self.backward_for_discriminator(loss_D_bed)
        # self.backward_for_discriminator(loss_D_bedsidetable)
        # self.backward_for_discriminator(loss_D_lamp)
        self.backward_for_discriminator(loss_D_sofa)
        # self.backward_for_discriminator(loss_D_chair)
        # self.backward_for_discriminator(loss_D_table)
        self.backward_for_discriminator(loss_D_shelf)
        self.backward_for_discriminator(loss_D_seat)
        # self.backward_for_discriminator(loss_D_screen)
        # self.backward_for_discriminator(loss_D_stage)

        set_requires_grad(self.shallow_backbone, True)
        set_requires_grad(self.bed_feature_extractor, True)
        # set_requires_grad(self.bedsidetable_feature_extractor, True)
        # set_requires_grad(self.lamp_feature_extractor, True)
        set_requires_grad(self.sofa_feature_extractor, True)
        # set_requires_grad(self.chair_feature_extractor, True)
        # set_requires_grad(self.table_feature_extractor, True)
        set_requires_grad(self.shelf_feature_extractor, True)
        set_requires_grad(self.seat_feature_extractor, True)
        # set_requires_grad(self.screen_feature_extractor, True)
        # set_requires_grad(self.stage_feature_extractor, True)
        set_requires_grad(self.reconstructor_bed, True)
        # set_requires_grad(self.reconstructor_bedsidetable, True)
        # set_requires_grad(self.reconstructor_lamp, True)
        set_requires_grad(self.reconstructor_sofa, True)
        # set_requires_grad(self.reconstructor_chair, True)
        # set_requires_grad(self.reconstructor_table, True)
        set_requires_grad(self.reconstructor_shelf, True)
        set_requires_grad(self.reconstructor_seat, True)
        # set_requires_grad(self.reconstructor_screen, True)
        # set_requires_grad(self.reconstructor_stage, True)
        self.backward_for_reconstructor(loss_R_bed)
        # self.backward_for_reconstructor(loss_R_bedsidetable)
        # self.backward_for_reconstructor(loss_R_lamp)
        self.backward_for_reconstructor(loss_R_sofa)
        # self.backward_for_reconstructor(loss_R_chair)
        # self.backward_for_reconstructor(loss_R_table)
        self.backward_for_reconstructor(loss_R_shelf)
        self.backward_for_reconstructor(loss_R_seat)
        # self.backward_for_reconstructor(loss_R_screen)
        # self.backward_for_reconstructor(loss_R_stage)

        self.backward_for_backbone(loss_C)
        # print('successfully backward through backbone&classifier')

        # 清空本次batch运算的梯度
        self.optimizer_shallow_backbone.step()
        self.optimizer_shallow_backbone.zero_grad()
        self.optimizer_bed_feature_extractor.step()
        self.optimizer_bed_feature_extractor.zero_grad()
        # self.optimizer_bedsidetable_feature_extractor.step()
        # self.optimizer_bedsidetable_feature_extractor.zero_grad()
        # self.optimizer_lamp_feature_extractor.step()
        # self.optimizer_lamp_feature_extractor.zero_grad()
        self.optimizer_sofa_feature_extractor.step()
        self.optimizer_sofa_feature_extractor.zero_grad()
        # self.optimizer_chair_feature_extractor.step()
        # self.optimizer_chair_feature_extractor.zero_grad()
        # self.optimizer_table_feature_extractor.step()
        # self.optimizer_table_feature_extractor.zero_grad()
        self.optimizer_shelf_feature_extractor.step()
        self.optimizer_shelf_feature_extractor.zero_grad()
        self.optimizer_seat_feature_extractor.step()
        self.optimizer_seat_feature_extractor.zero_grad()
        # self.optimizer_screen_feature_extractor.step()
        # self.optimizer_screen_feature_extractor.zero_grad()
        # self.optimizer_stage_feature_extractor.step()
        # self.optimizer_stage_feature_extractor.zero_grad()

        # 清空本次batch运算的梯度
        self.optimizer_classifier.step()
        self.optimizer_classifier.zero_grad()
        # 清空本次batch运算的梯度
        self.optimizer_R_bed.step()
        self.optimizer_R_bed.zero_grad()
        # self.optimizer_R_bedsidetable.step()
        # self.optimizer_R_bedsidetable.zero_grad()
        # self.optimizer_R_lamp.step()
        # self.optimizer_R_lamp.zero_grad()
        self.optimizer_R_sofa.step()
        self.optimizer_R_sofa.zero_grad()
        # self.optimizer_R_chair.step()
        # self.optimizer_R_chair.zero_grad()
        # self.optimizer_R_table.step()
        # self.optimizer_R_table.zero_grad()
        self.optimizer_R_shelf.step()
        self.optimizer_R_shelf.zero_grad()
        self.optimizer_R_seat.step()
        self.optimizer_R_seat.zero_grad()
        # self.optimizer_R_screen.step()
        # self.optimizer_R_screen.zero_grad()
        # self.optimizer_R_stage.step()
        # self.optimizer_R_stage.zero_grad()

        # 清空本次batch运算的梯度
        self.optimizer_D_bed.step()
        self.optimizer_D_bed.zero_grad()
        # self.optimizer_D_bedsidetable.step()
        # self.optimizer_D_bedsidetable.zero_grad()
        # self.optimizer_D_lamp.step()
        # self.optimizer_D_lamp.zero_grad()
        self.optimizer_D_sofa.step()
        self.optimizer_D_sofa.zero_grad()
        # self.optimizer_D_chair.step()
        # self.optimizer_D_chair.zero_grad()
        # self.optimizer_D_table.step()
        # self.optimizer_D_table.zero_grad()
        self.optimizer_D_shelf.step()
        self.optimizer_D_shelf.zero_grad()
        self.optimizer_D_seat.step()
        self.optimizer_D_seat.zero_grad()
        # self.optimizer_D_screen.step()
        # self.optimizer_D_screen.zero_grad()
        # self.optimizer_D_stage.step()
        # self.optimizer_D_stage.zero_grad()
           

      
        return (loss_C.cpu().data.numpy(), loss_R_bed.cpu().data.numpy(), loss_D_bed.cpu().data.numpy(),
        loss_R_sofa.cpu().data.numpy(), loss_D_sofa.cpu().data.numpy(), loss_R_shelf.cpu().data.numpy(), loss_D_shelf.cpu().data.numpy(), 
        loss_R_seat.cpu().data.numpy(), loss_D_seat.cpu().data.numpy())


    def validation_check(self, loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat):
        return (loss_C.cpu().data.numpy(), loss_R_bed.cpu().data.numpy(), loss_D_bed.cpu().data.numpy(),
        loss_R_sofa.cpu().data.numpy(), loss_D_sofa.cpu().data.numpy(), loss_R_shelf.cpu().data.numpy(), loss_D_shelf.cpu().data.numpy(), 
        loss_R_seat.cpu().data.numpy(), loss_D_seat.cpu().data.numpy())


    def learning(self, x, label, bed, sofa, shelf, seat):
        pred_class, reconstruct_result, loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat = self.forward(x, label, bed, sofa, shelf, seat)
        loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat = self.optimize_parameters(loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat)
        loss_curr = loss_C + loss_R_bed + loss_D_bed + loss_R_sofa + loss_D_sofa + loss_R_shelf + loss_D_shelf + loss_R_seat + loss_D_seat
        return pred_class, reconstruct_result, loss_curr


    def validating(self, x, label, bed, sofa, shelf, seat):
        pred_class, reconstruct_result, loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat = self.forward(x, label, bed, sofa, shelf, seat)
        loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat = self.validation_check(loss_C, loss_R_bed, loss_D_bed, loss_R_sofa, loss_D_sofa, loss_R_shelf, loss_D_shelf, loss_R_seat, loss_D_seat)
        loss_curr = loss_C + loss_R_bed + loss_D_bed + loss_R_sofa + loss_D_sofa + loss_R_shelf + loss_D_shelf + loss_R_seat + loss_D_seat
        return pred_class, reconstruct_result, loss_curr


class resnet_multi_text_concept_for_mammal(nn.Module):
    def __init__(self, args: dict):
        super(resnet_multi_text_concept_for_mammal, self).__init__()
        # block
        self.shallow_backbone = resnet_shallow(args['num_classes'])
        self.head_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.head_projector = ResNet_Projector()
        self.shape_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.shape_projector = ResNet_Projector()
        self.torso_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.torso_projector = ResNet_Projector()
        self.leg_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.leg_projector = ResNet_Projector()
        self.classifier = ResNet_Classifier(num_classes=args['num_classes'])
        # criterion
        self.contrast_criterion = MammalContrastiveCriterion()
        self.classification_criterion = ClassificationCriterion()
        # self.similarity_criterion = SimilarityCriterion()
        # optimizer
        self.optimizer_shallow_backbone = torch.optim.SGD(self.shallow_backbone.parameters(), lr=args['lr_resnet'])
        self.optimizer_shape_feature_extractor = torch.optim.SGD(self.shape_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_shape_projector = torch.optim.SGD(self.shape_projector.parameters(), lr=args['lr_resnet'])
        self.optimizer_head_feature_extractor = torch.optim.SGD(self.head_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_head_projector = torch.optim.SGD(self.head_projector.parameters(), lr=args['lr_resnet'])
        self.optimizer_torso_feature_extractor = torch.optim.SGD(self.torso_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_torso_projector = torch.optim.SGD(self.torso_projector.parameters(), lr=args['lr_resnet'])
        self.optimizer_leg_feature_extractor = torch.optim.SGD(self.leg_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_leg_projector = torch.optim.SGD(self.leg_projector.parameters(), lr=args['lr_resnet'])
        self.optimizer_classifier = torch.optim.SGD(self.classifier.parameters(), lr=args['lr_resnet'])


    def feature_concat(self, x1, x2, x3, x4):
        feature = torch.cat([x1, x2, x3, x4], 1)
        return feature


    def forward(self, x, label, text_concept):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        head_feature = self.head_projector(head_feature)
        # print(head_feature)
        torso_feature = self.torso_feature_extractor(shallow_feature)
        torso_feature = self.torso_projector(torso_feature)
        leg_feature = self.leg_feature_extractor(shallow_feature)
        leg_feature = self.leg_projector(leg_feature)
        shape_feature = self.shape_feature_extractor(shallow_feature)
        shape_feature = self.shape_projector(shape_feature)
        # classifying
        feature = self.feature_concat(head_feature, torso_feature, leg_feature, shape_feature)
        pred_class = self.classifier(feature)

        # calculate loss
        loss_C = self.calc_loss_C(pred_class, label)
        loss_head = self.calc_loss_contrast(head_feature,text_concept, 0)
        loss_torso = self.calc_loss_contrast(torso_feature, text_concept, 1)
        loss_leg = self.calc_loss_contrast(leg_feature, text_concept, 2)
        loss_shape = self.calc_loss_contrast(shape_feature, text_concept, 3)
        
        
        return pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape

    
    def inference(self, x):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        head_feature = self.head_projector(head_feature)
        # print(head_feature)
        torso_feature = self.torso_feature_extractor(shallow_feature)
        torso_feature = self.torso_projector(torso_feature)
        leg_feature = self.leg_feature_extractor(shallow_feature)
        leg_feature = self.leg_projector(leg_feature)
        shape_feature = self.shape_feature_extractor(shallow_feature)
        shape_feature = self.shape_projector(shape_feature)
        # classifying
        feature = self.feature_concat(head_feature, torso_feature, leg_feature, shape_feature)
        pred_class = self.classifier(feature)
        # reconstructing
        feature_dict = {
            'head': head_feature.cpu(),
            'torso': torso_feature.cpu(),
            'leg': leg_feature.cpu(),
            'shape': shape_feature.cpu()
        }
        return feature_dict, pred_class



    def calc_loss_C(self, pred_class, label):
        loss_C = self.classification_criterion(pred_class, label)
        return loss_C
    

    def calc_loss_contrast(self, x, text_concept, concept_flag):
        loss_contrast = self.contrast_criterion(x, text_concept, concept_flag)
        return loss_contrast


    def backward_for_backbone(self, loss_C):
        loss_C.backward()

    def backward_for_contrastive_loss(self, loss):
        loss.backward(retain_graph=True)


    def optimize_parameters(self, loss_C, loss_head, loss_torso, loss_leg, loss_shape):
        # self.backward_for_contrastive_loss(loss_head)
        # self.backward_for_contrastive_loss(loss_torso)
        # self.backward_for_contrastive_loss(loss_leg)
        # self.backward_for_contrastive_loss(loss_shape)
        self.backward_for_backbone(loss_C)

        self.optimizer_shallow_backbone.step()
        self.optimizer_shallow_backbone.zero_grad()
        self.optimizer_head_feature_extractor.step()
        self.optimizer_head_feature_extractor.zero_grad()
        self.optimizer_head_projector.step()
        self.optimizer_head_projector.zero_grad()
        self.optimizer_torso_feature_extractor.step()
        self.optimizer_torso_feature_extractor.zero_grad()
        self.optimizer_torso_projector.step()
        self.optimizer_torso_projector.zero_grad()
        self.optimizer_leg_feature_extractor.step()
        self.optimizer_leg_feature_extractor.zero_grad()
        self.optimizer_leg_projector.step()
        self.optimizer_leg_projector.zero_grad()
        self.optimizer_shape_feature_extractor.step()
        self.optimizer_shape_feature_extractor.zero_grad()
        self.optimizer_shape_projector.step()
        self.optimizer_shape_projector.zero_grad()
        self.optimizer_classifier.step()
        self.optimizer_classifier.zero_grad()

        return loss_C.cpu().data.numpy(), loss_head.cpu().data.numpy(), loss_torso.cpu().data.numpy(), loss_leg.cpu().data.numpy(), loss_shape.cpu().data.numpy()



    def validation_check(self, loss_C, loss_head, loss_torso, loss_leg, loss_shape):
        return loss_C.cpu().data.numpy(), loss_head.cpu().data.numpy(), loss_torso.cpu().data.numpy(), loss_leg.cpu().data.numpy(), loss_shape.cpu().data.numpy()


    def learning(self, x, label, text_concept, flag):
        pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.forward(x, label, text_concept)
        loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.optimize_parameters(loss_C, loss_head, loss_torso, loss_leg, loss_shape)
        loss_curr = loss_C + loss_head + loss_torso + loss_leg + loss_shape
        return pred_class, loss_curr


    def validating(self, x, label, text_concept):
        pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.forward(x, label, text_concept)
        loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.validation_check(loss_C, loss_head, loss_torso, loss_leg, loss_shape)
        loss_curr = loss_C + loss_head + loss_torso + loss_leg + loss_shape
        return pred_class, loss_curr


class new_resnet_multi_text_concept_for_mammal(nn.Module):
    def __init__(self, args: dict):
        super(new_resnet_multi_text_concept_for_mammal, self).__init__()
        # block
        self.shallow_backbone = resnet_shallow(args['num_classes'])
        self.head_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.shape_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.torso_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.leg_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.classifier = ResNet_Classifier_text(num_classes=args['num_classes'])
        # criterion
        self.contrast_criterion = MammalContrastiveCriterion()
        self.classification_criterion = ClassificationCriterion()
        # self.similarity_criterion = SimilarityCriterion()
        # optimizer
        self.optimizer_shallow_backbone = torch.optim.SGD(self.shallow_backbone.parameters(), lr=args['lr_resnet'])
        self.optimizer_shape_feature_extractor = torch.optim.SGD(self.shape_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_head_feature_extractor = torch.optim.SGD(self.head_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_torso_feature_extractor = torch.optim.SGD(self.torso_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_leg_feature_extractor = torch.optim.SGD(self.leg_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_classifier = torch.optim.SGD(self.classifier.parameters(), lr=args['lr_resnet'])

        # self.optimizer_shallow_backbone = torch.optim.Adam(self.shallow_backbone.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_shape_feature_extractor = torch.optim.Adam(self.shape_feature_extractor.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_head_feature_extractor = torch.optim.Adam(self.head_feature_extractor.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_torso_feature_extractor = torch.optim.Adam(self.torso_feature_extractor.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_leg_feature_extractor = torch.optim.Adam(self.leg_feature_extractor.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        

    def feature_concat(self, x1, x2, x3, x4):
        feature = torch.cat([x1, x2, x3, x4], 1)
        return feature


    def forward(self, x, label, text_concept):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        torso_feature = self.torso_feature_extractor(shallow_feature)
        leg_feature = self.leg_feature_extractor(shallow_feature)
        shape_feature = self.shape_feature_extractor(shallow_feature)
        
        # classifying
        feature = self.feature_concat(head_feature, torso_feature, leg_feature, shape_feature)
        pred_class = self.classifier(feature)

        # calculate loss
        loss_C = self.calc_loss_C(pred_class, label)
        loss_head = self.calc_loss_contrast(head_feature,label, text_concept, 0)
        loss_torso = self.calc_loss_contrast(torso_feature, label, text_concept, 1)
        loss_leg = self.calc_loss_contrast(leg_feature, label, text_concept, 2)
        loss_shape = self.calc_loss_contrast(shape_feature, label, text_concept, 3)
        print(loss_C)
        print(loss_head)
        # loss = 0.6 * loss_C + 0.1 * (loss_head + loss_torso + loss_leg + loss_shape)
        # loss = loss_C + loss_head + loss_torso + loss_leg + loss_shape
        
        
        return pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape

    
    def inference(self, x):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        # print(head_feature)
        torso_feature = self.torso_feature_extractor(shallow_feature)
        leg_feature = self.leg_feature_extractor(shallow_feature)
        shape_feature = self.shape_feature_extractor(shallow_feature)
        # classifying
        feature = self.feature_concat(head_feature, torso_feature, leg_feature, shape_feature)
        pred_class = self.classifier(feature)
        # reconstructing
        feature_dict = {
            'head': head_feature.cpu(),
            'torso': torso_feature.cpu(),
            'leg': leg_feature.cpu(),
            'shape': shape_feature.cpu()
        }
        return feature_dict, pred_class



    def calc_loss_C(self, pred_class, label):
        loss_C = self.classification_criterion(pred_class, label)
        return loss_C
    

    def calc_loss_contrast(self, x, label, text_concept, concept_flag):
        loss_contrast = self.contrast_criterion(x, label, text_concept, concept_flag)
        return loss_contrast


    def backward_for_backbone(self, loss_C):
        loss_C.backward()


    def backward_for_contrastive_loss(self, loss):
        loss.backward(retain_graph=True)


    def optimize_parameters(self, loss_C, loss_head, loss_torso, loss_leg, loss_shape):
        self.backward_for_contrastive_loss(loss_head)
        self.backward_for_contrastive_loss(loss_torso)
        self.backward_for_contrastive_loss(loss_leg)
        self.backward_for_contrastive_loss(loss_shape)
        self.backward_for_backbone(loss_C)

        self.optimizer_shallow_backbone.step()
        self.optimizer_shallow_backbone.zero_grad()
        self.optimizer_head_feature_extractor.step()
        self.optimizer_head_feature_extractor.zero_grad()
        self.optimizer_torso_feature_extractor.step()
        self.optimizer_torso_feature_extractor.zero_grad()
        self.optimizer_leg_feature_extractor.step()
        self.optimizer_leg_feature_extractor.zero_grad()
        self.optimizer_shape_feature_extractor.step()
        self.optimizer_shape_feature_extractor.zero_grad()
        self.optimizer_classifier.step()
        self.optimizer_classifier.zero_grad()

        return loss_C.cpu().data.numpy(), loss_head.cpu().data.numpy(), loss_torso.cpu().data.numpy(), loss_leg.cpu().data.numpy(), loss_shape.cpu().data.numpy()
        # return loss_C.cpu().data.numpy()



    def validation_check(self, loss_C, loss_head, loss_torso, loss_leg, loss_shape):
        return loss_C.cpu().data.numpy(), loss_head.cpu().data.numpy(), loss_torso.cpu().data.numpy(), loss_leg.cpu().data.numpy(), loss_shape.cpu().data.numpy()
        # return loss_C.cpu().data.numpy()


    def learning(self, x, label, text_concept):
        pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.forward(x, label, text_concept)
        # pred_class, loss_C = self.forward(x, label, text_concept)
        loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.optimize_parameters(loss_C, loss_head, loss_torso, loss_leg, loss_shape)
        # loss_C = self.optimize_parameters(loss_C, 0, 0, 0, 0)
        loss_curr = loss_C + loss_head + loss_torso + loss_leg + loss_shape
        # loss_curr = loss_C
        return pred_class, loss_curr


    def validating(self, x, label, text_concept):
        pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.forward(x, label, text_concept)
        # pred_class, loss_C = self.forward(x, label, text_concept)
        loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.validation_check(loss_C, loss_head, loss_torso, loss_leg, loss_shape)
        # loss_C = self.validation_check(loss_C, 0, 0, 0, 0)
        loss_curr = loss_C + loss_head + loss_torso + loss_leg + loss_shape
        # loss_curr = loss_C
        return pred_class, loss_curr


class TCNL_bilateral_attention(nn.Module):
    def __init__(self, args: dict):
        super(TCNL_bilateral_attention, self).__init__()
        # block
        self.shallow_backbone = resnet_shallow(args['num_classes'])
        self.head_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.head_attn = bilateral_prompt(vis_chans=1024, lan_chans=1024)
        self.shape_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.shape_attn = bilateral_prompt(vis_chans=1024, lan_chans=1024)
        self.torso_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.torso_attn = bilateral_prompt(vis_chans=1024, lan_chans=1024)
        self.leg_feature_extractor = resnet_deep(num_classes=args['num_classes'])
        self.leg_attn = bilateral_prompt(vis_chans=1024, lan_chans=1024)
        self.classifier = ResNet_Classifier_text(num_classes=args['num_classes'])
        # criterion
        self.contrast_criterion = MammalContrastiveCriterion()
        self.classification_criterion = ClassificationCriterion()
        # self.similarity_criterion = SimilarityCriterion()
        # optimizer
        self.optimizer_shallow_backbone = torch.optim.SGD(self.shallow_backbone.parameters(), lr=args['lr_resnet'])
        self.optimizer_shape_feature_extractor = torch.optim.SGD(self.shape_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_head_feature_extractor = torch.optim.SGD(self.head_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_torso_feature_extractor = torch.optim.SGD(self.torso_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_leg_feature_extractor = torch.optim.SGD(self.leg_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_classifier = torch.optim.SGD(self.classifier.parameters(), lr=args['lr_resnet'])

        # self.optimizer_shallow_backbone = torch.optim.Adam(self.shallow_backbone.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_shape_feature_extractor = torch.optim.Adam(self.shape_feature_extractor.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_head_feature_extractor = torch.optim.Adam(self.head_feature_extractor.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_torso_feature_extractor = torch.optim.Adam(self.torso_feature_extractor.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_leg_feature_extractor = torch.optim.Adam(self.leg_feature_extractor.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        # self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr=args['lr_resnet'], betas=[0.9, 0.999])
        

    def feature_concat(self, x1, x2, x3, x4):
        feature = torch.cat([x1, x2, x3, x4], 1)
        return feature


    def forward(self, x, label, text_concept):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        new_head_vis_feature, new_head_text_feature = self.head_attn(head_feature, text_concept[0])
        torso_feature = self.torso_feature_extractor(shallow_feature)
        new_torso_vis_feature, new_torso_text_feature = self.torso_attn(torso_feature, text_concept[1])
        leg_feature = self.leg_feature_extractor(shallow_feature)
        new_leg_vis_feature, new_leg_text_feature = self.leg_attn(leg_feature, text_concept[2])
        shape_feature = self.shape_feature_extractor(shallow_feature)
        new_shape_vis_feature, new_shape_text_feature = self.shape_attn(shape_feature, text_concept[3])
        
        # classifying
        feature = self.feature_concat(new_head_vis_feature, new_torso_vis_feature, new_leg_vis_feature, new_shape_vis_feature)
        pred_class = self.classifier(feature)

        # calculate loss
        new_text_concept_feature = self.feature_concat(new_head_text_feature, new_torso_text_feature, new_leg_text_feature, new_shape_text_feature)
        loss_C = self.calc_loss_C(pred_class, label)
        loss_head = self.calc_loss_contrast(head_feature,label, new_text_concept_feature, 0)
        loss_torso = self.calc_loss_contrast(torso_feature, label, new_text_concept_feature, 1)
        loss_leg = self.calc_loss_contrast(leg_feature, label, new_text_concept_feature, 2)
        loss_shape = self.calc_loss_contrast(shape_feature, label, new_text_concept_feature, 3)

        # loss = 0.6 * loss_C + 0.1 * (loss_head + loss_torso + loss_leg + loss_shape)
        # loss = loss_C + loss_head + loss_torso + loss_leg + loss_shape
        
        
        return pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape

    
    def inference(self, x):
        # feature extracting
        shallow_feature = self.shallow_backbone(x)
        head_feature = self.head_feature_extractor(shallow_feature)
        # print(head_feature)
        torso_feature = self.torso_feature_extractor(shallow_feature)
        leg_feature = self.leg_feature_extractor(shallow_feature)
        shape_feature = self.shape_feature_extractor(shallow_feature)
        # classifying
        feature = self.feature_concat(head_feature, torso_feature, leg_feature, shape_feature)
        pred_class = self.classifier(feature)
        # reconstructing
        feature_dict = {
            'head': head_feature.cpu(),
            'torso': torso_feature.cpu(),
            'leg': leg_feature.cpu(),
            'shape': shape_feature.cpu()
        }
        return feature_dict, pred_class



    def calc_loss_C(self, pred_class, label):
        loss_C = self.classification_criterion(pred_class, label)
        return loss_C
    

    def calc_loss_contrast(self, x, label, text_concept, concept_flag):
        loss_contrast = self.contrast_criterion(x, label, text_concept, concept_flag)
        return loss_contrast


    def backward_for_backbone(self, loss_C):
        loss_C.backward()


    def backward_for_contrastive_loss(self, loss):
        loss.backward(retain_graph=True)


    def optimize_parameters(self, loss_C, loss_head, loss_torso, loss_leg, loss_shape):
        self.backward_for_contrastive_loss(loss_head)
        self.backward_for_contrastive_loss(loss_torso)
        self.backward_for_contrastive_loss(loss_leg)
        self.backward_for_contrastive_loss(loss_shape)
        self.backward_for_backbone(loss_C)

        self.optimizer_shallow_backbone.step()
        self.optimizer_shallow_backbone.zero_grad()
        self.optimizer_head_feature_extractor.step()
        self.optimizer_head_feature_extractor.zero_grad()
        self.optimizer_torso_feature_extractor.step()
        self.optimizer_torso_feature_extractor.zero_grad()
        self.optimizer_leg_feature_extractor.step()
        self.optimizer_leg_feature_extractor.zero_grad()
        self.optimizer_shape_feature_extractor.step()
        self.optimizer_shape_feature_extractor.zero_grad()
        self.optimizer_classifier.step()
        self.optimizer_classifier.zero_grad()

        return loss_C.cpu().data.numpy(), loss_head.cpu().data.numpy(), loss_torso.cpu().data.numpy(), loss_leg.cpu().data.numpy(), loss_shape.cpu().data.numpy()
        # return loss_C.cpu().data.numpy()



    def validation_check(self, loss_C, loss_head, loss_torso, loss_leg, loss_shape):
        return loss_C.cpu().data.numpy(), loss_head.cpu().data.numpy(), loss_torso.cpu().data.numpy(), loss_leg.cpu().data.numpy(), loss_shape.cpu().data.numpy()
        # return loss_C.cpu().data.numpy()


    def learning(self, x, label, text_concept):
        pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.forward(x, label, text_concept)
        # pred_class, loss_C = self.forward(x, label, text_concept)
        loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.optimize_parameters(loss_C, loss_head, loss_torso, loss_leg, loss_shape)
        # loss_C = self.optimize_parameters(loss_C, 0, 0, 0, 0)
        loss_curr = loss_C + loss_head + loss_torso + loss_leg + loss_shape
        # loss_curr = loss_C
        return pred_class, loss_curr


    def validating(self, x, label, text_concept):
        pred_class, loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.forward(x, label, text_concept)
        # pred_class, loss_C = self.forward(x, label, text_concept)
        loss_C, loss_head, loss_torso, loss_leg, loss_shape = self.validation_check(loss_C, loss_head, loss_torso, loss_leg, loss_shape)
        # loss_C = self.validation_check(loss_C, 0, 0, 0, 0)
        loss_curr = loss_C + loss_head + loss_torso + loss_leg + loss_shape
        # loss_curr = loss_C
        return pred_class, loss_curr

