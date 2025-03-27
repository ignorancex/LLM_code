import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as f


class ClassificationCriterion(nn.Module):
    def __init__(self):
        super(ClassificationCriterion, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, output, label):
        loss = self.loss_fn(output, label)
        return loss


class SimilarityCriterion(nn.Module):
    def __init__(self):
        super(SimilarityCriterion, self).__init__()
        self.loss_fn = nn.MSELoss()

    def __call__(self, reconstruct_image, main_object):
        if (main_object.shape[-1] != reconstruct_image.shape[-1]):
            main_object = F.interpolate(main_object, size=(reconstruct_image.shape[-2], reconstruct_image.shape[-1]))
        loss = self.loss_fn(reconstruct_image, main_object)
        return loss


class GANCriterion(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANCriterion, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss_fn = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss_fn(prediction, target_tensor)
        return loss


class MammalContrastiveCriterion(nn.Module):
    def __init__(self):
        super(MammalContrastiveCriterion, self).__init__()
    
    def contrastive_loss(self, x, label, text_concept_group, concept_flag):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, x.shape[1])
        x = F.normalize(x, dim=1)
        loss = 0
        for i in range(batch_size):
            # if label[i] == 0:
            #     text_concept = text_concept_group['cat']
            # if label[i] == 1:
            #     text_concept = text_concept_group['cow']
            # if label[i] == 2:
            #     text_concept = text_concept_group['dog']
            # if label[i] == 3:
            #     text_concept = text_concept_group['panda']
            # if label[i] == 4:
            #     text_concept = text_concept_group['horse']
            text_concept = text_concept.to('cuda:2')
            new_x = torch.cat([x[i].expand(size=(1, 1024)), x[i].expand(size=(1, 1024)), x[i].expand(size=(1, 1024)), x[i].expand(size=(1, 1024))])
            sim = F.cosine_similarity(new_x, text_concept, dim=1)
            loss += torch.exp(sim[concept_flag]) / torch.sum(torch.exp(sim))
        return loss / batch_size
    
    def __call__(self, x, label, text_concept, concept_flag):
        loss = self.contrastive_loss(x, label, text_concept, concept_flag)
        return loss

    
