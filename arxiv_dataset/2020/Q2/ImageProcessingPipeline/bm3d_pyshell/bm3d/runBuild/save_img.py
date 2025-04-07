import numpy as np
from skimage import data, io
from skimage.metrics import peak_signal_noise_ratio

img = data.astronaut()
io.imsave('astronaut.png', img)
noise = np.random.normal(scale=40,
                         size=img.shape)
print('img_max  :', img.max(), "|min:", img.min())
print('noise_max:', noise.max(), "|min:", noise.min())
img_noisy = img + noise
img_noisy = np.clip(img_noisy, 0, 255)
img_noisy = img_noisy.astype('uint8')
io.imsave('astronaut_noisy.png', img_noisy)
