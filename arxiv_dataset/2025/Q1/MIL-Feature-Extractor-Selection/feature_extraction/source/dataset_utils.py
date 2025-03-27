import copy
import numpy as np
import torch

'''
From: A method for normalizing histology slides for quantitative analysis, 
M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley, G Xiaojun, 
C Schmitt, NE Thomas, IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250

https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

'''
def collate_patch_filepaths(batch):
    item = batch[0]
    idx = torch.LongTensor([item[0]])
    fp = item[1]
    return [idx, fp]

def stain_norm(img, Io = 240, alpha = 1, beta = 0.15):

    original_img = copy.deepcopy(img)
                               
    HERef = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(float)+1)/Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]

    # compute eigenvectors
    try:
        _ , eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    except:
        return original_img

    #eigvecs *= -1

    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues          
    That = ODhat.dot(eigvecs[:,1:3])

    phi = np.arctan2(That[:,1],That[:,0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)     

    return Inorm