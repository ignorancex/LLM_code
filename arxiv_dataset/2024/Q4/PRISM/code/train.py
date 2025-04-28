### ====== PRISM TRAINING ======

import torch
from model import PRISM
from dataset import PRISM_MRI_Dataset

torch.autograd.set_detect_anomaly(True)

out_dir = '/kaggle/working/'

# Guys
# train_path_g = '/kaggle/input/ixi-guys-train-stripped/IXI-Guys-train.pt'
# test_path_g = '/kaggle/input/ixi-guys-test-stripped/IXI-Guys-test.pt'

# HH
train_path_h = '/kaggle/input/ixi-hh-train-stripped/IXI-HH-train.pt'
test_path_h = '/kaggle/input/ixi-hh-test-stripped/IXI-HH-test.pt'

# ADNI
# train_path_a = '/kaggle/input/adni1-train/ADNI1-train.pt'
# test_path_a = '/kaggle/input/adni1-test/ADNI1-test.pt'

# IOP
# train_path_i = '/kaggle/input/ixi-iop-train-stripped/IXI-IOP-train.pt'
# test_path_i = '/kaggle/input/ixi-iop-test-stripped/IXI-IOP-test.pt'

# Hyperparams
lr = 5e-3
batch_size = 8
epochs = 50
gpu_id = 0

print('======== PRISM training starts ========')

# ====== 1. INITIALIZE MODEL ======
prism = PRISM(intensity_levels=5, latent_dim=2, num_sites=1, gpu_id=gpu_id, modality='T2', modalities = ['T2'])

# ====== 2. LOAD DATASETS ======
prism.load_dataset_from_pt(batch_size, train_path=train_path_h, test_path=test_path_h)

# ====== 3. INITIALIZE TRAINING ======
prism.init_training(out_dir=out_dir, lr=lr, vgg_path='/kaggle/input/vgg16-imagenet/pytorch/default/1/vgg16_imagenet.pth')

# ====== 4. BEGIN TRAINING ======
prism.train(epochs=epochs)

# ====== 5. SAVE MODELS ======

# Site HH
torch.save(prism.anatomy_encoder.state_dict(), f'{prism.out_dir}/prism-anatomy-encoder_hh.pth')
torch.save(prism.style_encoder.state_dict(), f'{prism.out_dir}/prism-style-encoder_hh.pth')
torch.save(prism.decoder.state_dict(), f'{prism.out_dir}/prism-decoder_hh.pth')

print('======== PRISM training ends ========')