### ====== PRISM RECONSTRUCTION EVALUATION ======

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import PRISM
from dataset import PRISM_MRI_Dataset

# Testing the PRISM pipeline's reconstruction performance on unseen data from the same site

site = "hh" # hh, guys, adni, iop, etc..
model_dir = "/kaggle/working"
data_dir = "/kaggle/input"

prism = PRISM(intensity_levels=5, latent_dim=2, num_sites=3, gpu_id=0, modality='T2', modalities = ['T2'])
prism.anatomy_encoder.load_state_dict(torch.load(f"{model_dir}/prism-anatomy-encoder_{site}.pth", weights_only=True))
prism.style_encoder.load_state_dict(torch.load(f"{model_dir}/prism-style-encoder_{site}.pth", weights_only=True))
prism.decoder.load_state_dict(torch.load(f"{model_dir}/prism-decoder_{site}.pth", weights_only=True))

test_ds = torch.load(f"{data_dir}/ixi-{site}-test-stripped/IXI-{site.upper()}-test.pt")

mae_scores, mse_scores, ssim_scores, psnr_scores = [], [], [], []

with torch.set_grad_enabled(False):
    prism.anatomy_encoder.eval()
    prism.style_encoder.eval()
    prism.decoder.eval()
    for subject in test_ds:
        image = subject[prism.modality]['image'].to(prism.device).unsqueeze(1)
        mask = subject[prism.modality]['mask'].to(prism.device).unsqueeze(1)
        _, anatomy = prism.get_anatomy_representations(image, mask)
        style_code, _, _ = prism.get_style_code(image)
        rec_image = prism.decode(anatomy, style_code, mask)
        
        image = image.squeeze().cpu().numpy()
        rec_image = rec_image.squeeze().cpu().numpy()
        
        # Compute pixel-wise mean absolute error (MAE) and mean squared error (MSE)
        mae_scores.append(np.mean(np.abs(image - rec_image)))
        mse_scores.append(np.mean((image - rec_image) ** 2))
        
        # Compute SSIM and PSNR
        ssim_scores.append(ssim(image, rec_image, data_range=image.max() - image.min(), multichannel=False))
        psnr_scores.append(psnr(image, rec_image, data_range=image.max() - image.min()))

print(f"Site {site.upper()}: Reconstruction metrics on test (unseen) images\n")                    
print(f"MAE of test set: mean: {np.mean(mae_scores)}, std: {np.std(mae_scores)}")
print(f"MSE of test set: mean: {np.mean(mse_scores)}, std: {np.std(mse_scores)}")
print(f"SSIM of test set: mean: {np.mean(ssim_scores)}, std: {np.std(ssim_scores)}")
print(f"PSNR of test set: mean: {np.mean(psnr_scores)}, std: {np.std(psnr_scores)}")