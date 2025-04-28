import numpy as np

x = np.linspace(0,1,201)
y = np.linspace(0,1,201)
Y,X = np.meshgrid(x,y)

# read in biodata
image = np.load(snakemake.input[0],allow_pickle=True)
mass = np.sum(image)/(201*201)
mean_x = np.sum(X[image==1])/np.sum(image==1)
mean_y = np.sum(Y[image==1])/np.sum(image==1)
vessel_length = np.max(X[image==1])
imstats = np.concatenate((np.array([mass]),np.array([mean_x]),np.array([mean_y]),np.array([vessel_length])))

np.savetxt(snakemake.output[0],np.reshape(imstats,(1,4)),delimiter=",",fmt='%1.9f')

