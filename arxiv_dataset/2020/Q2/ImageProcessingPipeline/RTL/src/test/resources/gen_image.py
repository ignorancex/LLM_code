import numpy as np
from skimage import data, io
from skimage.metrics import peak_signal_noise_ratio

img = data.chelsea()
print("img shape:", img.shape)

img.tofile('frame_cat_base')
print('dtype:', img.dtype)
# io.imshow(img)
# io.show()
gen_noisy = True
gen_filtered = False
print(img.shape)

if(gen_filtered):
    imgClear = img.copy()
    for i in range(0, (1 << 3) - 1):
        imgClear[:] = img
        filename = 'frame_cat_' + format(i, '03b')

        if(i & (1 << 2) == 0):
            print("Clear red")
            imgClear[:,:,0] = 0

        if(i & (1 << 1) == 0):
            print("Clear green")
            imgClear[:,:,1] = 0

        if(i & (1 << 0) == 0):
            print("Clear blue")
            imgClear[:,:,2] = 0

        # io.imshow(imgClear)
        # io.show()
        print('save image:', i, ':', filename)
        imgClear.tofile(filename)

if(gen_noisy):
    for i in range(0, 5):
        noise_std = i*5
        noise_img = np.random.normal(scale=noise_std,
                                     size=img.shape)
        noisy_img = img + noise_img
        noisy_img = noisy_img.clip(0, 255).astype(np.uint8)
        noisy_img.tofile('frame_cat_noisy_' + format(noise_std))
