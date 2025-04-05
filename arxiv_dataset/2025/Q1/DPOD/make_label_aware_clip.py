import json  
from PIL import Image
from torch.utils.data import Dataset
import os 
import torch
import torchvision
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
import argparse
import io
# import clip_classifier
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time 
import clip 
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


from __future__ import print_function

import torch
import torch.nn as nn
device = "cuda:1" if torch.cuda.is_available() else "cuda:1" # If using GPU then use mixed precision training.
# device = "cuda:0"




class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        # print(self.temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        # print(self.base_temperature)

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda:1')
        #           if features.is_cuda
        #           else torch.device('cuda:0'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print(logits_max)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print(mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




class NewsClip(Dataset):
    def __init__(self, data_paths_nc,transform, nviews=1):

        self.data_paths_nc = data_paths_nc
        # self.visual_news_root_dir = visual_news_root_dir
        # self.news_clip_root_dir = news_clip_root_dir
        self.transform = transform
        self.nviews = nviews

        
    def __len__(self):
        return len(self.data_paths_nc)   

    def load_img_pil(self,image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
                       
    def load_queries(self,visual_news_caption_item, visual_news_image_item):
        # caption = visual_news_caption_item['caption']
        caption = visual_news_caption_item
        image_path = visual_news_image_item
        caption_tokenized = clip.tokenize(caption,truncate=True).squeeze(0)
        # image_path = os.path.join(self.visual_news_root_dir, visual_news_image_item['image_path'])
        pil_img = self.load_img_pil(image_path)
        # print("Length of nviews is : ", nviews)
        transform_img = [self.transform(pil_img) for i in range(self.nviews)]
        # print("Length of transformed image is : ", len(transform_img))
        return transform_img, caption_tokenized

    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        # print(idx)
        label = torch.as_tensor(0) if self.data_paths_nc[str(idx)]['real'] else torch.as_tensor(1)

        visual_news_caption_item = self.data_paths_nc[str(idx)]["caption"]

        
        visual_news_image_item = self.data_paths_nc[str(idx)]["image_path"]



        qImg, qCap = self.load_queries(visual_news_caption_item, visual_news_image_item)

        return qImg, qCap , label
        
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 



##############################################################################################################################################################################

model,preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training


transform1 = transforms.Compose([
    transforms.Resize(size=224, max_size=None, antialias=None),
    transforms.RandomCrop(224,224),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(10),
    transforms.ColorJitter(),

    # transforms.RandomResizedCrop(size=224,interpolation=bicubic),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])


batch_size= 64
optimizer =torch.optim.Adam(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

data_paths = json.load(open("")) # path to your training data



train_dataset = NewsClip(data_paths  , transform=transform1, nviews=4)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=6, pin_memory=True)




epochs = 40
temp = 0.05
criterion = SupConLoss(temperature=temp)


            
total_loss = 0
for epoch in range(40):
    total_loss = 0
    for (idx, batch) in enumerate(train_dataloader):
        imgs = batch[0]
        

        captions = batch[1].to(device=device)
        labels = batch[2].to(device=device)
  
    
        encoded_img0 = model.encode_image(imgs[0].to(device=device))   
        encoded_img1 = model.encode_image(imgs[1].to(device=device))  
        encoded_img2 = model.encode_image(imgs[2].to(device=device))  
        encoded_img3 = model.encode_image(imgs[3].to(device=device))  

        encoded_text = model.encode_text(captions)  





        encoded_img0 = encoded_img0 / encoded_img0.norm(dim=-1, keepdim=True) 
        encoded_img1 = encoded_img1 / encoded_img1.norm(dim=-1, keepdim=True)
        encoded_img2 = encoded_img2 / encoded_img2.norm(dim=-1, keepdim=True) 
        encoded_img3 = encoded_img3 / encoded_img3.norm(dim=-1, keepdim=True) 


        encoded_text = encoded_text / encoded_text.norm(dim=-1, keepdim=True)

        aug0=[]
        aug1=[]
        aug2=[]
        aug3=[]
        tex_aug=[]

        real0=[]
        real1=[]
        real2=[]
        real_text=[]

        for i,label in enumerate(labels):
            # print("I is: ", i )
            if label==1:
                aug0.append(encoded_img0[i])
                aug1.append(encoded_img1[i])
                aug2.append(encoded_img2[i])
                aug3.append(encoded_img3[i])
                tex_aug.append(encoded_text[i])
            if label==0:
                real0.append(encoded_img0[i])
                real1.append(encoded_img1[i])
                real2.append(encoded_img2[i])
                real_text.append(encoded_text[i])
       
        aug0=torch.stack(aug0,dim=0)
        aug1=torch.stack(aug1,dim=0)
        aug2=torch.stack(aug2,dim=0)
        aug3=torch.stack(aug3,dim=0)
        tex_aug=torch.stack(tex_aug,dim=0)
        


        real0=torch.stack(real0,dim=0)
        real1=torch.stack(real1,dim=0)
        real2=torch.stack(real2,dim=0)
        real_text=torch.stack(real_text,dim=0)


        features1 = torch.stack([aug0,aug1,aug2,aug3], dim=1)
        features2 = torch.stack([tex_aug,tex_aug,tex_aug,tex_aug], dim=1)
        features3 = torch.stack([real0,real1,real2,real_text], dim=1)

        img_text_aug_feat = torch.cat([features1,features2], dim=0)

        loss = 1.5*criterion(img_text_aug_feat)  + criterion(features3)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        # convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)

