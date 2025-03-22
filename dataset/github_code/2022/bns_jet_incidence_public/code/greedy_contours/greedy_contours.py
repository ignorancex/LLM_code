import numpy as np
from scipy.ndimage import gaussian_filter

# 2d greedy binning
def samples_to_mesh(x,y,bins=(30,31),smooth=0.7):
    h,bx,by = np.histogram2d(x,y,bins=bins,range=[[x.min()*0.9, x.max()*1.1], [y.min()*0.9, y.max()*1.1]])
    h = gaussian_filter(h,sigma=smooth)
    h_r = h.ravel()
    c_r = np.zeros_like(h_r)
    sort_idx = np.argsort(h_r)
    c_r[sort_idx] = np.cumsum(h_r[sort_idx])
    c = c_r.reshape(h.shape)
    c/=c.max()
    bxm,bym = np.meshgrid((bx[1:]+bx[:-1])/2.,(by[1:]+by[:-1])/2.)
    return (1.-c).T,bxm,bym

def mesh_to_mesh(z,smooth=0.7):
    h = gaussian_filter(z,sigma=smooth)
    h_r = h.ravel()
    
    c_r = np.zeros_like(h_r)
    sort_idx = np.argsort(h_r)
    c_r[sort_idx] = np.cumsum(h_r[sort_idx])
    c = c_r.reshape(z.shape)
    c/=c.max()
    return (1.-c)
    
