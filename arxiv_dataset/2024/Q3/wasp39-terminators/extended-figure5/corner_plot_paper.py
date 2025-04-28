import numpy as np
import matplotlib.pyplot as plt
import pickle

import corner

import seaborn as sns
sns.set_style('ticks')

def get_quantiles(dist, alpha=0.68, method='median'):
    """
    get_quantiles function
    DESCRIPTION
        This function returns, in the default case, the parameter median and the error%
        credibility around it. This assumes you give a non-ordered
        distribution of parameters.
    OUTPUTS
        Median of the parameter,upper credibility bound, lower credibility bound
    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples * (alpha / 2.) + 1)
    if (method == 'median'):
        med_idx = 0
        if (nsamples % 2 == 0.0):  # Number of points is even
            med_idx_up = int(nsamples / 2.) + 1
            med_idx_down = med_idx_up - 1
            param = (ordered_dist[med_idx_up] + ordered_dist[med_idx_down]) / 2.
            return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                   ordered_dist[med_idx_down-nsamples_at_each_side]
        else:
            med_idx = int(nsamples / 2.)
            param = ordered_dist[med_idx]
            return param,ordered_dist[med_idx+nsamples_at_each_side],\
                   ordered_dist[med_idx-nsamples_at_each_side]

pic=pickle.load(open('./OUTPUT/w39-prism.pic','rb'), \
                encoding='latin1')
samples=pic[:,:-1]

# Samples has...all the samples. In our case, shape [nsamples, 16]. Let's 
# extract the common properties of the limbs and plot them in a corner plot:

# 0: Tirr, 2: log C/O, 3: logKzz, 4: fsed, 5: logPbase, 6:logCldVMR
samples1 = samples[:, [0,2,3,4,5,6]]

# 10: Tirr2, 11: log C/O_2, 12: logKzz2, 13: fsed2, 14: logPbase2, 15:logCldVMR2
samples2 = samples[:, [10, 11, 12, 13, 14, 15]]

T_evening = samples2[:,0]#samples2[:,-2]
T_morning = samples1[:,0]

logCO_evening = samples2[:, 1]
logCO_morning = samples1[:, 1]

logPbase_evening = samples2[:, -2]
logPbase_morning = samples1[:, -2]

# Set up the parameters of the problem.
ndim, nsamples = 6, len(T_evening)

# Generate some fake data.
data = np.zeros([nsamples, ndim])
data[:, 0] = T_evening
data[:, 1] = T_morning
data[:, 2] = logCO_evening
data[:, 3] = logCO_morning
data[:, 4] = logPbase_evening
data[:, 5] = logPbase_morning

# Plot it.
figure = corner.corner(
    data,
    labels=[
        r"$T_{\mathrm{evening}}$ (K)",
        r"$T_{\mathrm{morning}}$ (K)",
        r"$\log C/O_{\mathrm{evening}}$",
        r"$\log C/O_{\mathrm{morning}}$",
        r"$\log P_{\mathrm{base,evening}}$",
        r"$\log P_{\mathrm{base, morning}}$"
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=False,
    title_kwargs={"fontsize": 12},
)

# Extract the axes
axes = np.array(figure.axes).reshape((ndim, ndim))

ax = axes[1, 0]
ax.plot([700,1100], [700,1100], '--', color = 'darkviolet')


plt.savefig('paper_corner_plot.pdf',
            dpi=250,#, rasterize=True,
            #transparent=True,
            bbox_inches='tight')
