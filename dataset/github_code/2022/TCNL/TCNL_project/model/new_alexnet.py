from model.alexnet_backbone import *
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



class alexnet_multi_concept_for_scene(nn.Module):
    def __init__(self, args: dict):
        super(alexnet_multi_concept_for_scene, self).__init__()
        # block
        self.shallow_backbone = alexnet_shallow(args['num_classes'])
        self.bed_feature_extractor = alexnet_deep(num_classes=args['num_classes'])
        # self.bedsidetable_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        # self.lamp_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        self.sofa_feature_extractor = alexnet_deep(num_classes=args['num_classes'])
        # self.chair_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        # self.table_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        self.shelf_feature_extractor = alexnet_deep(num_classes=args['num_classes'])
        self.seat_feature_extractor = alexnet_deep(num_classes=args['num_classes'])
        # self.screen_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        # self.stage_feature_extractor = vgg_deep(num_classes=args['num_classes'])
        self.rest_feature_extractor = alexnet_deep(num_classes=args['num_classes'])

        self.classifier = AlexNet_Classifier(num_classes=args['num_classes'])
        self.reconstructor_bed = AlexNet_Reconstructor()
        self.discriminator_bed = AlexNet_Discriminator()
        # self.reconstructor_bedsidetable = VGG_Reconstructor()
        # self.discriminator_bedsidetable = VGG_Discriminator()
        # self.reconstructor_lamp = VGG_Reconstructor()
        # self.discriminator_lamp = VGG_Discriminator()
        self.reconstructor_sofa = AlexNet_Reconstructor()
        self.discriminator_sofa = AlexNet_Discriminator()
        # self.reconstructor_chair = VGG_Reconstructor()
        # self.discriminator_chair = VGG_Discriminator()
        # self.reconstructor_table = VGG_Reconstructor()
        # self.discriminator_table = VGG_Discriminator()
        self.reconstructor_shelf = AlexNet_Reconstructor()
        self.discriminator_shelf = AlexNet_Discriminator()
        self.reconstructor_seat = AlexNet_Reconstructor()
        self.discriminator_seat = AlexNet_Discriminator()
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
        self.optimizer_rest_feature_extractor = torch.optim.SGD(self.rest_feature_extractor.parameters(), lr=args['lr_resnet'])
        self.optimizer_classifier = torch.optim.SGD(self.classifier.parameters(), lr=args['lr_resnet'])


    def feature_concat(self, x1, x2, x3, x4, x5):
        feature = torch.cat([x1, x2, x3, x4, x5], 1)
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
        rest_feature = self.rest_feature_extractor(shallow_feature)
        feature_dict = {
            'bed': bed_feature.cpu(),
            'sofa': sofa_feature.cpu(),
            'seat': seat_feature.cpu(),
            'shelf': shelf_feature.cpu()
        }
        # classifying
        feature = self.feature_concat(bed_feature, sofa_feature, shelf_feature, seat_feature, rest_feature)
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
