import glob, os
import numpy as np
from skimage import io

filelist = glob.glob(os.path.join('.', 'frame*'))
# filelist = sort(filelist)
fileidx = map(lambda e: str('[' + str(e) + '] :'), range(len(filelist)))
filesindexed = list(zip(fileidx, filelist))
filesstr = list(map(lambda e: ' '.join(e), filesindexed))

print("List of files: ")
print(*filesstr, sep='\n')

idx = int(input("Please enter file to show: "))
filename = filelist[idx]
hight = int(input("Please enter img hight : "))
width = int(input("Please enter img width : "))
chans = int(input("Please enter img channels : "))

print("You are showing: " + filename)
img = np.reshape(np.fromfile(filename, np.uint8), (hight, width, chans))
io.imshow(img)
io.show()
