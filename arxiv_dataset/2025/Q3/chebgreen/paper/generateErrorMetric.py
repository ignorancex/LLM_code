from chebgreen import ChebGreen
import numpy as np
import matplotlib.pyplot as plt

from chebgreen.chebpy2 import Quasimatrix
from chebgreen.chebpy2.chebpy import chebfun
from chebgreen.chebpy2 import Chebfun2

from chebgreen.utils import runCustomScript, computeEmpiricalError
from chebgreen.greenlearning.utils import DataProcessor
from pathlib import Path
import shutil, time, sys

def main(example):
    print(f"Generating error metric for example {example}...")
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
        vmin, vmax = 0,1
    else:
        raise ValueError("Invalid example name.")

    model = ChebGreen(Theta, domain, generateData, script, example, dirichletBC)

    Ginterp, Ninterp = model.generateNewModel(theta_)

    validation = ChebGreen([theta_], domain, generateData, script, example, dirichletBC)

    # Compute error for paper
    Nsample = 100
    lmbda = 0.01
    Nf = 500
    Nu = 500
    noise_level = 0
    seed = 42
    saveSuffix = "validation"

    Error = []
    for theta in Theta:
        datapath = f"datasets/{example}/{theta:.2f}-{saveSuffix}/data.mat"
        if Path(datapath).is_file():
            print(f"Test dataset already present for Theta = {theta:.2f}")
        else:
            print(f"Generating test dataset for Theta = {theta:.2f}")
            runCustomScript(script,example,theta,Nsample,lmbda,Nf,Nu,noise_level, seed, saveSuffix)
        data = DataProcessor(datapath)
        data.generateDataset(trainRatio = 0)
        error = model.computeEmpiricalError(theta, data)
        Error.append(error)
        print(f"Empirical error for model at Theta = {theta:.2f} is {error}")
        shutil.rmtree(f"datasets/{example}/{theta:.2f}-{saveSuffix}")
        
    datapath = f"datasets/{example}/{theta_:.2f}-{saveSuffix}/data.mat"
    if Path(datapath).is_file():
        print(f"Test dataset already present for Theta = {theta_:.2f}")
    else:
        print(f"Generating test dataset for Theta = {theta_:.2f}")
        runCustomScript(script,example,theta_,Nsample,lmbda,Nf,Nu,noise_level, seed, saveSuffix)
    data = DataProcessor(datapath)
    data.generateDataset(trainRatio = 0)
    error = model.computeEmpiricalError(theta_, data)
    Error.append(error)
    print(f"Empirical error for interpolated model at Theta = {theta_:.2f} is {error}")  
    error = validation.computeEmpiricalError(theta_, data)
    print(f"Empirical error for validation model at Theta = {theta_:.2f} is {error}")
    Error.append(error)
    shutil.rmtree(f"datasets/{example}/{theta_:.2f}-{saveSuffix}")

    return np.array(Error)

if __name__ == "__main__":
    t0 = time.perf_counter()
    Error = main(sys.argv[1])
    for error in Error:
        print(f"{error*100:.2f}")
    t1 = time.perf_counter()
    print(f"Total time taken: {(t1-t0)//60} minutes {(t1-t0)%60:.2f} seconds")