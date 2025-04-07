import numpy as np
import torch
from PIL import Image
import scipy.ndimage as ndi

# Computes a 2D histogram of vertical pixel pairs on an integer, grayscale image
def co_occur_unit(X,L=256):
    assert len(X.shape) == 2, 'Only accepts rectangular matrices'
    assert np.issubdtype(X.dtype,np.integer), 'Only accepts integer inputs'
    assert X.min() >= 0 and X.max() < L, f'Input must be non-negative and less than L ({L})'
    return np.bincount((L*X[:-1]+X[1:]).reshape(-1),minlength=L**2).reshape((L,L))


# Compte vertical and horizontal co-occurrence histograms for each RGB channel
# Output is of shape (256,256,6), with the channels ordered Rv, Gv, Bv, Rh, Gh, Bh
def co_occur(X,L=256):
    assert len(X.shape) == 3, f'Only accepts rectangular matrices, shape was {X.shape}'
    assert X.shape[2] == 3, f'Input must be 3 channel RGB, shape was {X.shape}'
    assert np.issubdtype(X.dtype,np.integer), f'Only accepts integer inputs, dtype was {X.dtype}'
    assert X.min() >= 0, 'Input must be non-negative'

    return np.dstack([co_occur_unit(img[:,:,c],L) for img in [X,X.transpose((1,0,2))] for c in range(3)])




class CoOccur:
    def __init__(self,L=256):
        self.L = L
    def __call__(self,X):
        return co_occur(X,self.L)



lap_filt = -1*np.ones((3,3,1))
lap_filt[1,1,0] = 8
    

class Laplace:
    def __call__(self,X):
        return ndi.filters.convolve(X,lap_filt)


class Normalize:
    def __call__(self,X):
        v_max = np.abs(X).max()
        if v_max == 0: return X
        else: return X/v_max


class ToTensor:
    def __call__(self,X):
        return torch.from_numpy(X.astype(np.float32).transpose((2,0,1)))



class REDDataset(torch.utils.data.Dataset):

    def __init__(self,fnames,labels,transform=None,debug=False):
        if debug:
            inds = np.random.choice(len(fnames),size=64)
            self.fnames = [fnames[i] for i in inds]
            self.labels = [labels[i] for i in inds]
        else:
            self.fnames = fnames
            self.labels = labels
        self.transform = transform
        self.n_classes = np.max(labels) + 1

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self,idx):
        img = np.array(Image.open(self.fnames[idx]))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]



