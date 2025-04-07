import numpy as np
from skimage import data, io

# load image as pixel array
image = io.imread('astronaut_noisy.png')
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
print(image.transpose((2,0,1)).flatten().astype(np.float32)[:10])
