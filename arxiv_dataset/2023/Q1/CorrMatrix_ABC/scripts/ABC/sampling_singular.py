from scipy.stats import multivariate_normal
import numpy as np
import pylab as plt
from scipy.stats import kde

# 2x2 covariance matrix
#cov = np.array([[1, -1], [-1, 1]])
cov = np.array([[1, 0.5], [0.5, 1]])

mean = np.array([0, 0])
size = 10000

print('Drawing {} samples from multi-variate normal...'.format(size))
y = multivariate_normal.rvs(mean=mean, cov=cov, size=size)

print('Plotting points...')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y[:,0], y[:,1], s=0.5)

print('Drawing density contours...')
xmin = -4
xmax = 4
dx   = 0.05
x1, x2 = np.mgrid[xmin:xmax:dx, xmin:xmax:dx]
grid   = np.dstack((x1, x2))
nbins = 20
x1i, x2i = np.mgrid[x1.min():x1.max():nbins*1j, x2.min():x2.max():nbins*1j]
k = kde.gaussian_kde([y[:,0], y[:,1]])
zi = k(np.vstack([x1i.flatten(), x2i.flatten()]))

if zi.max() == 0:
    print('Singular matrix, KDE density estimate return 0 everywhere on grid')
else:
    zi = zi / zi.max()

    sig    = np.array([0.68, 0.95, 0.99]) 
    levels = 1.0 - sig[::-1]

    contours = ax.contour(x1i, x2i, zi.reshape(x1i.shape), levels, size=8, colors='b')


print('Plotting analytical Gaussian...')

try:
    cov_inv = np.linalg.inv(cov)

    g = np.zeros(x1.shape)
    xb = np.arange(xmin, xmax, dx)
    for i, xx1 in enumerate(xb):
        for j, xx2 in enumerate(xb):
            x = [xx1, xx2]
            chi2 = np.einsum('i,ij,j', x, cov_inv, x)
            g[i,j] = np.exp(-0.5 * chi2)
    g = g / g.max()
    ax.contour(x1, x2, g.reshape(x1.shape), levels, size=8, colors='r', linestyles='dashed')

except np.linalg.linalg.LinAlgError:
    print('Numerically singular matrix, cannot compute numerical inverse required to evaluate Gaussian')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

det = np.linalg.det(cov)
plt.title('C = {}, |C| = {}'.format(str(cov.flatten()), det))
print('|C| = {}'.format(det))
plt.axis('equal')
plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)

plt.savefig('y')

