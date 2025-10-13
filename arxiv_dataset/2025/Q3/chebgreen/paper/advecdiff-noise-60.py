from chebgreen.greenlearning.utils import DataProcessor
from chebgreen.greenlearning.model import *
from chebgreen.backend import plt
from chebgreen.utils import runCustomScript

from chebgreen.chebpy2 import Chebfun2, Chebpy2Preferences
from chebgreen.chebpy2.chebpy.core.settings import ChebPreferences
from chebgreen.chebpy2.chebpy import chebfun, Chebfun
from chebgreen import ChebGreen
import shutil, time, sys, fileinput
from pathlib import Path

def green(x,s):
    g = 0
    num1 = (np.exp(3*x/2) - np.exp(3/2)) * (np.exp(1.5*(1+s)) - 1) * np.heaviside(x-s, 0.5)
    num2 = (np.exp(3*s/2) - np.exp(3/2)) * (np.exp(1.5*(1+x)) - 1) * np.heaviside(s-x, 0.5)
    factor = (2 * np.exp(s/2 - 2*x)) / (3 * (np.exp(3) - 1))
    g = factor * (num1 + num2)
    return g

noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# noise_levels = [0, 0.1]


Theta = [1.0,2.0,3.0]
theta_ = 2.5
domain = [-1,1,-1,1]
generateData = False
script = "generate_example"
example = "advection_diffusion"
dirichletBC = True

Nsample = 100
lmbda = 0.01
Nf = 500
Nu = 500
saveSuffix = "validation"
seed = 42

noise_level = 0.6
t0 = time.perf_counter()

print("-------------------------------------------------------------------------------")
print(f"Learning a chebfun model for example \'{example}\' with {noise_level*100}% noise")
model = ChebGreen(Theta, domain, generateData, script, f"{example}-{int(noise_level*100)}/{example}", dirichletBC, datapath = f"datasets/{example}/", noise_level = noise_level)

print("-------------------------------------------------------------------------------")
print(f"Interpolating a chebfun model for example \'{example}\' with {noise_level*100}% noise")
Ginterp, Ninterp = model.generateNewModel(theta_)

savePath = f"plots-interp/{example}"
Path(savePath).mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize = (8,6))
model.interpG[theta_].plot(fig = fig)
fig.savefig(f'{savePath}/{example}-{int(noise_level*100)}.png', dpi = fig.dpi)

print("-------------------------------------------------------------------------------")
print(f"Computing error for example \'{example}\' with {noise_level*100}% noise")
# Compute the empirical error
runCustomScript(script, example, theta_, Nsample, lmbda, Nf, Nu, 0, seed, saveSuffix)
datapath = f"datasets/{example}/{theta_:.2f}-{saveSuffix}/data.mat"
data = DataProcessor(datapath)
data.generateDataset(trainRatio = 0)
error = model.computeEmpiricalError(theta_, data)

xx = np.linspace(domain[0],domain[1],2000)
yy = np.linspace(domain[2],domain[3],2000)
x, y = np.meshgrid(xx,yy)
E = np.abs(Ginterp[x,y] - green(x,y))

vmin = 0
vmax = 0.024

fig = plt.figure(figsize = (13,10), frameon=False)
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')
levels = np.linspace(vmin, vmax, 100, endpoint = True)
plt.contourf(x,y, E, levels = levels, cmap = 'jet', vmin = vmin, vmax = vmax)
# ticks = np.linspace(vmin, vmax, 10, endpoint=True)
# cbar = plt.colorbar(ticks = ticks, fraction = 0.046, pad = 0.04)

savePath = f"plots/"
Path(savePath).mkdir(parents=True, exist_ok=True)
fig.savefig(f'{savePath}/interp-60-error.png', dpi = fig.dpi, bbox_inches='tight', pad_inches=0)