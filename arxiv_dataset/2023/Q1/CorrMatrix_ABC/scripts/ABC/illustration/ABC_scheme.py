#!/usr/bin/env python3


import pylab as plt
import matplotlib
import numpy as np
import scipy.stats as stats


def plot_sample_points(cols=['r', 'g']):
    """ Plot sample points.
    """

    ysample = [ymax-0.02, ymax]
    for i in range(nsim):
        if abs(xsim[i]-xobs) > eps:
            # Rejected samples
            col = cols[0]
        else:
            # Accepted samples
            col = cols[1]
        plt.plot([xsim[i], xsim[i]], ysample, color=col)
    plt.text((xmin+xmax)/2-0.2, ymax+0.02, '$y_{\\rm mod}$ = samples under $L$ (model)', color='black', size=fs)


def plot_accepted_range():
    """ Plot acceptence range
    """

    plt.plot([xobs-eps, xobs-eps], [0, gpdf_eps_m], 'g')
    plt.plot([xobs+eps, xobs+eps], [0, gpdf_eps_p], 'g')
    plt.plot([xobs-eps, xobs-eps], [gpdf_eps_m, ymax], 'g-.')
    plt.plot([xobs+eps, xobs+eps], [gpdf_eps_p, ymax], 'g-.')
    iacc = (x>xobs-eps) & (x<xobs+eps)
    ax.fill_between(x[iacc], 0, gpdf[iacc], color='g')




### Sampling from L in observable space ###

# Mixture of Gaussians
weights = [0.8, 0.2]
means   = [2, 3]
sigma   = [0.4, 0.3]


# Sample points
icomp   = range(2)
nsim = 100
xsim = np.zeros(nsim)
which = np.random.choice(icomp, p=weights, size=nsim)
for i in range(nsim):
    xsim[i]  = np.random.normal(loc=means[which[i]], scale=sigma[which[i]], size=1)

# Data
xobs = 3.2

# Tolerance
eps = 0.1

fig = plt.figure()
ax  = plt.gca()
ax.set_aspect('auto')

# Font size
fs = 14

xmin = 0
xmax = 4.5
ymin = 0
ymax = 0.9
x    = np.linspace(xmin, xmax, 1000)

# PDF
gpdf = 0
gpdf_obs = 0
gpdf_eps_m = 0
gpdf_eps_p = 0
for i in (0, 1):
    gpdf += weights[i] * stats.norm.pdf(x, means[i], sigma[i])
    gpdf_obs += weights[i] * stats.norm.pdf(xobs, means[i], sigma[i])
    gpdf_eps_m += weights[i] * stats.norm.pdf(xobs-eps, means[i], sigma[i])
    gpdf_eps_p += weights[i] * stats.norm.pdf(xobs+eps, means[i], sigma[i])

# Plot pdf
plt.plot(x, gpdf)

# Plot horizontal line for data and label
plt.plot([xobs, xobs], [0, gpdf_obs], color='black')
plt.text(xobs-0.1, -0.10, '$y_{\\rm obs}$ (observed data)', color='black', size=fs)

plt.xlabel('$y$')
plt.ylabel('$L(y)$')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

ax.yaxis.label.set_size(fs)
ax.xaxis.label.set_size(fs)
plt.tick_params(labelsize=fs)

plot_sample_points(['b', 'b'])
plt.savefig('ABC_scheme_x_L')

# With acceptance range
plot_sample_points(['r', 'g'])
plot_accepted_range()

plt.savefig('ABC_scheme_x_acc')


# Histogram
fig = plt.figure()
ax  = plt.gca()


plt.xlabel('$y$')
plt.ylabel('$L(y)$')

ax.yaxis.label.set_size(fs)
ax.xaxis.label.set_size(fs)
plt.tick_params(labelsize=fs)

nsim2 = 50000
xsim2 = np.zeros(nsim2)
which = np.random.choice(icomp, p=weights, size=nsim2)
for i in range(nsim2):
    xsim2[i]  = np.random.normal(loc=means[which[i]], scale=sigma[which[i]], size=1)

v = plt.hist(xsim2, 20, density=True, align='mid')

# Plot horizontal line for data and label
ind_acc = np.where(v[1]<xobs)[0][-1]
val_acc = v[0][ind_acc]
plt.plot([xobs, xobs], [0, val_acc], color='black')
plt.text(xobs-0.1, -0.10, '$y_{\\rm obs}$ (observed data)', color='black', size=fs)

# Plot accepted bin
db = v[1][1] - v[1][0]
rect = matplotlib.patches.Rectangle((xobs-db/2, 0), db, val_acc, fill=True, color='green')
ax.add_patch(rect)

plot_sample_points(['r', 'g'])
plt.ylim(ymin, ymax)
plt.savefig('ABC_scheme_x_hist')


### Sampling from P in parameter space
# TODO
