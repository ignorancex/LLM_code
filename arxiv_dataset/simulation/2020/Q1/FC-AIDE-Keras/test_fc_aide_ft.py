from core.test_ft import Fine_tuning as test_ft
from scipy import misc
import numpy as np

file_name = 'cman'
file_name_clean = file_name + '_clean.png'
file_path = './data/'

noise_mean = 0
noise_sigma = 30

if noise_sigma not in [15, 25, 30, 50, 75]:
    print ('No weight file')
    exit()
    
clean_image = misc.imread(file_path + file_name_clean)
noisy_image = clean_image + np.random.normal(noise_mean, noise_sigma, clean_image.shape)

t_ft = test_ft(clean_image, noisy_image, noise_sigma)
denoised_img, psnr, ssim = t_ft.fine_tuning()

misc.imsave(file_name +'_denoised_ft.png', denoised_img)

print ('PSNR : ' + str(round(psnr,2)) + '\nSSIM : ' + str(round(ssim,4)))