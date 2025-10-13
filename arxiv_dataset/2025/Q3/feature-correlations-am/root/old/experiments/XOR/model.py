import numpy as np
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity

xor = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]], dtype=np.float32)
xand = np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]], dtype=np.float32)

rows = xor.shape[0]
cols = xor.shape[1]

xorcap = get_memorization_capacity(xor)
print(xorcap)

xandcap = get_memorization_capacity(xand)
print(xandcap)

#It's the same!