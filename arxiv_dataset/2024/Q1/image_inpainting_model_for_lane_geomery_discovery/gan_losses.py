from torch.nn.functional import mse_loss
from PIL import Image
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt


def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)


#----------------------------------------------------------------
# Generator's loss function in phase 1
# only use loss_content from class PerceptualLoss()
# 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)
#----------------------------------------------------------------

def contentFunc():
    conv_3_3_layer = 14
    cnn = models.vgg19(pretrained=True).features
    cnn = cnn.cuda()
    model = nn.Sequential()
    model = model.cuda()
    model = model.eval()
    for i, layer in enumerate(list(cnn)):
        model.add_module(str(i), layer)
        if i == conv_3_3_layer:
            break
    return model


self_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def tensor_to_image(images_tensor, image_name):
  img1 = images_tensor[0]
  save_image(img1, image_name+'.png') 
  

def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    

def completion_network_phase_one_loss(fakeIm, realIm):
    
    fakeIm = (fakeIm + 1) / 2.0
    realIm = (realIm + 1) / 2.0
    
    fakeIm[0, :, :, :] = self_transform(fakeIm[0, :, :, :])
    
    # realIm's shape is [1, 4, 256, 256]
    #org realIm[0, :, :, :] = self_transform(realIm[0, :, :, :])  
    realIm[0, :3, :, :] = self_transform(realIm[0, :3, :, :])  
    f_fake = contentFunc().forward(fakeIm)
   
    #org f_real = contentFunc().forward(realIm[:, :, :, :]) 
    f_real = contentFunc().forward(realIm[:, :3, :, :]) 
    f_real_no_grad = f_real.detach()
    self_criterion = nn.MSELoss()
    
    loss = self_criterion(f_fake, f_real_no_grad)
    #org loss_content = 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm[:, :, :, :])
    loss_content = 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm[:, :3, :, :])
    return loss_content 


#-----------------------------------------------------------------------------------------------------
# Discriminator's loss function for phase 2
# self.loss_D = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 1) +
#                self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 0)) / 2
#-----------------------------------------------------------------------------------------------------

def _RelativisticDiscLossLS(fakeImg, realImg, self_fake_pool, self_real_pool):
 
    self_fake_pool.add(fakeImg)
    self_real_pool.add(realImg)
    
    errD = (torch.mean((realImg - torch.mean(self_fake_pool.query()) - 1) ** 2) 
            + torch.mean((fakeImg - torch.mean(self_real_pool.query()) + 1) ** 2)) / 2
    
    errD.type(torch.cuda.FloatTensor)
     
    return errD 


def discriminator_network_phase_two_loss(pred, gt, self_fake_pool, self_real_pool):
 
    loss = _RelativisticDiscLossLS(pred, gt, self_fake_pool, self_real_pool)
    return loss     
    

def discriminator_network_phase_three_loss():     
    return None 



#--------------------------------------------------------------   
# completion network loss for phase 3
# In addition to content loss, add weighted adversarial loss 
# Loss_G = loss_content + self.adv_lambda * loss_adv
#--------------------------------------------------------------   
def completion_network_phase_three_loss(): 
    return None #ToDo 


 
