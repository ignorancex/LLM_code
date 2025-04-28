import torch, pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def load_ABM_data(file_path, plot=False):
    
    # load data
    file = np.load(file_path, allow_pickle=True).item()
    
    # extract data
    #ignore first spatial point because singularity there in spherical coords.
    x = file['x'][:,0].copy().astype(float)
    t = file['t'].copy()[0, :]
    X = file['X'].copy().astype(float)
    T = file['T'].copy()
    U = file['C'].copy()
    shape = U.shape
    
    # variable scales
    x_scale = 1.0 # 100.0 -> 100.0
    t_scale = 1.0 # 1000.0 -> 1000.0
    u_scale = 1.0

    # scale variables
    x *= x_scale
    t *= t_scale
    X *= x_scale
    T *= t_scale
    U *= u_scale
    
    # flatten for MLP
    inputs = np.concatenate([X.reshape(-1)[:, None],
                             T.reshape(-1)[:, None]], axis=1)
    outputs = U.reshape(-1)[:, None]


    if plot:
    
        # plot surface
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X, T, U, cmap=cm.coolwarm, alpha=1)
        #ax.scatter(X.reshape(-1), T.reshape(-1), U.reshape(-1), s=5, c='k')
        #plt.title(f'Cell line: {cell_line}, replicate: {replicate}')
        ax.set_xlabel('Position (millimeters)')
        ax.set_ylabel('Time (days)')
        ax.set_zlabel('Cell density (cells/mm^2)')
        #ax.set_zlim(0, 2.2e3)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.show()
        
    return inputs, outputs, shape

