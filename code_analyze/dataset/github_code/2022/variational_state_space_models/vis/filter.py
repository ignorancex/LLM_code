from vis import plot_covariances
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

model_colors = [
    [0.540, 0.168, 0.883, 0.5],
    [0.8, 0.8, 0, 0.5],
]

def plt_filter(qzxs, prior, post, samples, state, model_names=[], color=True,
                xlabel='', ylabel=''):
    handles = [
        mpatches.Patch(color=[0.5, 0.1, 0.1], label='filter prior'),
        mpatches.Patch(color=[0.1, 0.5, 0.1], label='filter posterior')
    ]
    # qzxs is [T, n_models] stacked
    # post is [T] stacked
    # samples is [n_samples, T] stacked
    qzxs_cov = np.linalg.inv(qzxs.conc[..., 0:2, 0:2] + 0.001*np.eye(2))
    qzxs_mean = (qzxs_cov @ np.expand_dims(qzxs.inf[..., 0:2], -1)).squeeze(-1)
    for m, n in zip(range(qzxs_mean.shape[1]), model_names): # for each model
        if n is None:
            continue
        handles.append(mpatches.Patch(color=model_colors[m], label=f'qzx/qz {n}'))
        qzx_cov = qzxs_cov[:, m]
        qzx_mean = qzxs_mean[:, m]
        plot_covariances(qzx_mean, qzx_cov, color=model_colors[m], zorder=0)

    prior_cov = np.linalg.inv(prior.conc[..., 0:2, 0:2])
    prior_mean = (prior_cov @ np.expand_dims(prior.inf[..., 0:2], -1)).squeeze(-1)
    plot_covariances(prior_mean, prior_cov, color=[0.5, 0.1, 0.1, 0.5], zorder=0)

    post_cov = np.linalg.inv(post.conc[..., 0:2, 0:2])
    post_mean = (post_cov @ np.expand_dims(post.inf[..., 0:2], -1)).squeeze(-1)
    plot_covariances(post_mean, post_cov, color=[0.1, 0.5, 0.1, 0.5], zorder=0)

    c = np.arange(samples.shape[1])
    for i in range(samples.shape[0]):
        z = samples[i]
        if color:
            scat = plt.scatter(z[...,0], z[...,1], s=10, c=c, cmap=cm.copper) #color=[0, 0, 0], s=10)
        else:
            scat = plt.scatter(z[...,0], z[...,1], s=10, color=[0, 0, 0]) #color=[0, 0, 0], s=10)
        if i == 0:
            scat.set_label('samples')
            handles.append(scat)
    if color:
        scat = plt.scatter(state[:, 0], state[:,1], c=c, cmap=cm.cool, s=30)#color=[0, 0, 1], s=50)
    else:
        scat = plt.scatter(state[:, 0], state[:,1], color=[0, 0, 1], s=30)
    scat.set_label('true state')
    handles.append(scat)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(handles=handles)
