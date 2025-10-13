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
generateData = True
script = "generate_example"
example = "advection_diffusion"
dirichletBC = True

Nsample = 100
lmbda = 0.01
Nf = 500
Nu = 500
saveSuffix = "validation"
seed = 42

Error = []
for noise_level in noise_levels:
    t0 = time.perf_counter()

    print("-------------------------------------------------------------------------------")
    print(f"Learning a chebfun model for example \'{example}\' with {noise_level*100}% noise")
    model = ChebGreen(Theta, domain, generateData, script, example, dirichletBC, noise_level = noise_level)

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
    Error.append(error)

    xx = np.linspace(domain[0], domain[1], 1000)
    ss = np.linspace(domain[2], domain[3], 1000)
    X, S = np.meshgrid(xx, ss)
    e = np.abs(Ginterp[X,S] - green(X,S))
    fig = plt.figure(figsize = (8,6))
    plt.contourf(X, S, e, levels = 100, cmap = 'turbo')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('s')
    plt.title('Error of the interpolated Green\'s function')
    fig.savefig(f'{savePath}/{example}-interp-error-{int(noise_level*100)}.png', dpi = fig.dpi)
    plt.close()
    
    print(f"Error for a model with {noise_level*100} % noise is {Error[-1]*100}%")
    print("-------------------------------------------------------------------------------")
    shutil.rmtree(f"datasets/{example}")

    new_path = Path(f"savedModels/{example}-{int(noise_level*100)}")
    if not new_path.exists():
        new_path.mkdir(parents=True, exist_ok=True)
        shutil.move(f"savedModels/{example}", str(new_path))

    t1 = time.perf_counter()
    print(f"Time taken for example \'{example}\' with {noise_level*100}% noise: {t1-t0:.2f} seconds")
for error in Error:
    print(f"{error*100:.2f}")