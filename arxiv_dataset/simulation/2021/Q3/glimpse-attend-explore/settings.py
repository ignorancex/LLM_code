#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch as t
t.set_default_tensor_type('torch.cuda.FloatTensor')
channels=3
nglimpses=8
glimpse_size=48
retina_size=glimpse_size//3
checkpoint_iteration=2
batch_size=2
iters=300
img_size_height=128
img_size_width=256
#we have 3 different scales in the retina with different resolutions, therefore we allow overlapping glimpses excepet for th inside 1/3 of the glimpse which is taken in full resolution
nb_height_patches=img_size_height//(glimpse_size//3)
nb_width_patches=img_size_width//(glimpse_size//3)


# In[ ]:




