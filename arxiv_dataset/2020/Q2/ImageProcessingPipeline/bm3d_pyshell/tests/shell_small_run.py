import numpy as np
from skimage import data, io
from skimage.metrics import peak_signal_noise_ratio

import bm3d_pyshell


noise_std_dev = 5.0
img = data.astronaut()

print("Adding noise")
noise = np.random.normal(scale=noise_std_dev,
                         size=img.shape)
print("Noise min: ", noise.min())
print("Noise max: ", noise.max())
#unique, counts = np.unique(noise, return_counts=True)

#noisy_img = img + noise
noisy_img = io.imread("../bm3d/runBuild/astronaut_noisy.png")
#noisy_img = np.zeros(img.shape, dtype=img.dtype)
noisy_img = noisy_img[:,:,0]
io.imsave("../bm3d/runBuild/astronaut_noisy_grayscale.png", noisy_img)
noisy_img = np.atleast_3d(noisy_img)
#noisy_img[:,:,2] = noisy_img[:,:,0]
#noisy_img[:,:,0] = 200
#print(type(noisy_img[0,0,0]))
print("Denoising with BM3D")
#print("THE FUCK IS THE TYPE NOW ", type(noisy_img[0,0,0]))
#print("Noisy_img max: ", noisy_img.max())
#print("Noisy_img min: ", noisy_img.min())
dtype_info = np.iinfo(np.uint8)

noisy_img = np.rint(noisy_img)
noisy_img = noisy_img.clip(0, 255).astype(np.uint8)

print(noisy_img.shape)
out6 = bm3d_pyshell.simple_bm3d(noisy_img).astype(np.uint8)
print('out min: ', out6.min())
print('out max: ', out6.max())
#out6 = out6.clip(dtype_info.min, dtype_info.max)

#noise_psnr = peak_signal_noise_ratio(img, noisy_img)
#out_psnr = peak_signal_noise_ratio(img, out6)

#print("PSNR of noisy image: ", noise_psnr)
#print("PSNR of reconstructed image: ", out_psnr)
io.imshow(img)
io.show()

io.imshow(noisy_img[:,:,0])
io.show()
io.imshow(out6[:,:,0])


io.show()
