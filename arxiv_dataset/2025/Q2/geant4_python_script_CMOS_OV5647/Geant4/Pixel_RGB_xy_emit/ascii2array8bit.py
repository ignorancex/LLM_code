# Escribe tu código aquí :-)
# import numpy library as np
import numpy as np
import matplotlib.pyplot as plt
import re

# numerical data file
filename="miki_pixel_edep.csv"
#filename="/home/mik/Geant4-10.05/g4_work/rdecay02/build/pixel_edep.csv"
#filename="/media/mik/EAC2174FC2171EFF/pixel_edep_sr90.csv"
#filename="/home/mik/Documents/test.csv"


# load the data with NumPy function loadtxt
#data = np.loadtxt(filename, delimiter=",")
# use skiprows to skip rows
#data = np.loadtxt(filename, delimiter=",", skiprows=1)
# usecols select columns
#data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[1,2,3])
# Load data from tsv file: data
data = np.loadtxt(filename, delimiter="\t", skiprows=1, usecols=[0,1,2])
#dat = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[1,2,3])

height= np.int32(data[0][1])
width= np.int32(data[0][0])

data = np.delete(data, 0, axis=0)

dmax=4.31
#dmax=np.amax(data,0)[2]
dmin=np.amin(data,0)[2]
thresh=4.3

dat_b=(data[:,2]-thresh)*255/(dmax-thresh)
dat_b[dat_b<0]=0
dat_b[dat_b>255]=255

#height=1944
#width=2592
#height=3
#width=4
data_2d = np.zeros((height, width))
data_2d_full= np.zeros((height, width))
data_2d[data[:,1].astype(np.int32),data[:,0].astype(np.int32)]=dat_b.astype(np.int32)
data_2d_full[data[:,1].astype(np.int32),data[:,0].astype(np.int32)]=data[:,2]
np.save("pixel_edep256.npy", np.int32(data_2d))

print(data)
print(data_2d)
print(data_2d_full)
plt.imshow(data_2d, cmap='jet')
#plt.imshow(data_2d_full, cmap=plt.cm.rainbow)
plt.colorbar()

plt.show()
