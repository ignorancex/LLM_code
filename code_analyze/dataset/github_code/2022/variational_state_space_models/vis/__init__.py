import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import numpy as np

def plot_covariances(mean, cov, nstd=1, cmap=cm.autumn, **kwargs):
    T = cov.shape[0]
    for t in range(T):
        mu = mean[t,:]
        covs = cov[t,:,:]
        args = dict(kwargs)
        if 'color' not in args:
            args['color'] = cmap(t/T)
        plot_covariance(mu, covs, nstd, **args)
    x = mean[:,0]
    y = mean[:,1]
    plt.scatter(x, y, color='black', s=0.01)

def plot_covariance(mean, cov, nstd=1, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    width, height = 2 * nstd * np.sqrt(eigvals)
    ell = matplotlib.patches.Ellipse(xy=mean, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)
    plt.gca().add_artist(ell)


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)