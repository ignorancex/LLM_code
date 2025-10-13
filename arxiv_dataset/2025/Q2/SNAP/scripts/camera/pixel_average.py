import sys

import numpy as np
from PIL import Image 

im = np.array(Image.open(sys.argv[1]))
av = np.average(im)

