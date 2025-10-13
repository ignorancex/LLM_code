from chebgreen import ChebGreen
import numpy as np
import matplotlib.pyplot as plt

from chebgreen.chebpy2 import Quasimatrix
from chebgreen.chebpy2.chebpy import chebfun
from chebgreen.chebpy2 import Chebfun2

from pathlib import Path
import shutil, time, sys

def chooseGreensFunction(example, model, validation, Theta, theta_, label):
    if label == 'A':
        Gplot = model.G[Theta[0]]
    elif label == 'B':
        Gplot = model.G[Theta[1]]
    elif label == 'C':
        Gplot = model.G[Theta[2]]
    elif label == 'D':
        Gplot = model.interpG[theta_]
    elif label == 'E':
        Gplot = validation.G[theta_]
    return Gplot

def main(example):
    print(f"Generating plots for example {example}...")
    print("-------------------------------------------------------------------------------")

    if example == "advection_diffusion":
        # Example 1: Advection-Diffusion
        Theta = [1.0,2.0,3.0]
        theta_ = 2.5
        domain = [-1,1,-1,1]
        generateData = True
        script = "generate_example"
        example = "advection_diffusion"
        dirichletBC = True
        vmin, vmax = -0.68,0.13
    elif example == "airy_equation":
        # Example 2: Airy Equation
        Theta = [1,5,10]
        theta_ = 7
        domain = [0,1,0,1]
        generateData = True
        script = "generate_example"
        example = "airy_equation"
        dirichletBC = True
        vmin, vmax = -0.24,0.03
    elif example == "fractional_laplacian":
        # Example 3: Fractional Laplacian
        Theta = [0.8,0.9,0.95]
        theta_ = 0.85
        domain = [-np.pi/2,np.pi/2,-np.pi/2,np.pi/2]
        generateData = True
        script = "generate_fractional"
        example = "fractional_laplacian"
        dirichletBC = False
        vmin, vmax = -0.23, 1.68
    else:
        raise ValueError("Invalid example name.")

    model = ChebGreen(Theta, domain, generateData, script, example, dirichletBC)

    _, _ = model.generateNewModel(theta_)

    validation = ChebGreen([theta_], domain, generateData, script, example, dirichletBC)

    labels = ['A','B','C','D','E']

    for label in labels:
        Gplot = chooseGreensFunction(example, model, validation, Theta, theta_, label)
        xx = np.linspace(domain[0],domain[1],2000)
        yy = np.linspace(domain[2],domain[3],2000)
        x, y = np.meshgrid(xx,yy)
        
        Gplot = Gplot[x,y]

        fig = plt.figure(figsize = (13,10), frameon=False)
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        levels = np.linspace(vmin, vmax, 50, endpoint = True)
        plt.contourf(x,y,Gplot, levels = levels, cmap = 'jet', vmin = vmin, vmax = vmax)
        # ticks = np.linspace(vmin, vmax, 10, endpoint=True)
        # cbar = plt.colorbar(ticks = ticks, fraction = 0.046, pad = 0.04)

        savePath = f"plots/{example}"
        Path(savePath).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{savePath}/{example}-{label}.png', dpi = fig.dpi, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    t0 = time.perf_counter()
    main(sys.argv[1])
    t1 = time.perf_counter()
    print(f"Total time taken: {(t1-t0)//60} minutes {(t1-t0)%60:.2f} seconds")