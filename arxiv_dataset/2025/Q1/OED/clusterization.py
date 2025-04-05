import numpy as np
from matplotlib import pyplot as plt

from src.probability import Prior, Posterior
from src.forward import Heat
from src.observations import PointObservation

FS = 24 ## Font size

## Avoid clusterization by increasing this to e.g. 4
MODEL_ERROR = 0



def main():

    sig = 5e-2 ## Observation error amplitude
    N = 200 ## Number of spatial discretizatio points
    L = 1 ## Length of computational domain
    time = 3e-2 ## Time for heat dissipation in arbitrary units.
    alpha = 1. ## Coefficient of Laplacian in heat equation
    gamma = -1. ## Exponent of Laplacian in prior. 

    ## Choose one, comment out the other
    transform = 'dct' ## dct - Discrete Cosine Transform. Corresponds
                      ## to homogenoeus Neumann boundary condition.
    transform = 'dst' ## dst - Discrete Sine Transform. Corresponds
                      ## homogenoeus Dirichlet boundary condition.

    ## We need to regularize the Laplacian so it is invertible and can
    ## function as a useful prior. Hence we add the constant delta
    ## when employing homogeneous Neumann boundary condition.
    delta = 0. if transform == 'dst' else 0.5
    

    ## Forward model - the heat equation with BC chosen according to transform
    fwd = Heat(N=N,
               L=L,
               transform=transform,
               alpha=alpha,
               time=time)

    ## Prior - with precision \delta
    pr = Prior(N=N,
               L=L,
               transform=transform,
               gamma=gamma,
               delta=delta)

    ## Posterior. We don't care about data, so no data is involved,
    ## only observation operators. This effectively implements
    ## calculating the D-optimality criterion design and searching for
    ## a D-optimal design.
    post = Posterior(fwd=fwd,
                     prior=pr,
                     sigSqr=sig**2,
                     L=L,
                     N=N,
                     transform=transform,
                     model_error=MODEL_ERROR)


    
    F_gamma_Fstar = fwd.multiplier**2 * pr.multiplier ## Prior covariance in H_o
    precision = 1 / F_gamma_Fstar ## Prior precision
    
    k = 2 ## rank O^*O
    m = 4 ## Number of measurements

    ## See last part of Theorem in paper. This is the value the pusforward 
    ## posterior precision takes for the first k eigenvalues
    sigSqr = 2e-4
    theta = ((np.sum(precision[:k]) + m /sigSqr ) / k) 

    ## Plotting
    n = 4 ## Truncate after this many eigevanlues, for visualization purposes
    y = precision[:n]
    x = np.arange(n, dtype=int) + 1
    xtra = np.zeros(n)
    xtra[:k] = theta - y[:k]
   
    plt.bar(x, y, label=r"prior")
    plt.bar(x, xtra, bottom=y, label="measurements")
    plt.yscale('log')
    plt.xlabel("Eigenvector", fontsize=FS)
    plt.ylabel("Precision", fontsize=FS)
    plt.tight_layout()
    plt.savefig(f"latex/figs/FgammaFstar_modelError{MODEL_ERROR}.png")
    
    
    ###################################
    ## find and plot optimal designs ##
    ###################################
    
    dic = {}

    ## Form  number of observations in this range
    for m in range(1, 7):

        ## Calculate an optimal design
        design = post.optimize(m=m, n_iterations=500)['x']

        ## And store it in dic
        dic[m] = design

        ## Print it so we see
        print(transform, m, design)
            
    
    fig, ax = plt.subplots(figsize=(10,5)) ## Create figure object

    ## For every pair in dic, plot a the locations of the measurements
    ## at ordinate m, with different colors.
    for m, array in dic.items():

        colors = iter(list("brgkmc"))
        vals = np.repeat(m, len(array))
        ax.scatter(array, vals, s=0)
        for i, val in enumerate(array):
            ax.annotate(str(i+1),
                        xy=(val, m),
                        ha='center', 
                        va='center',
                        fontsize=FS,
                        color=next(colors))


    ax.set_xlabel('Measurement Location', fontsize=FS)
    ax.set_ylabel('No. of Measurements', fontsize=FS)
    ax.set_yticks(list(dic.keys()))
    ax.set_xlim(0,1)
    plt.tight_layout()

    plt.savefig(f"latex/figs/{transform}_modelError{MODEL_ERROR}.png")
    #plt.show()

    ##############################
    ## Plot scaled eigenvectors ##
    ##############################
    
    ## Number of measurements
    m = 4
    design = dic[m]
    
   
    plt.close('all')
    fig = plt.figure(figsize=(8,4))
    fs = 18
    lss = ['solid', 'dotted', 'dashed', 'dashdot']
    vals = np.zeros(m)
    plt.scatter(design, vals)
    x = fwd.x
    for i,ls in enumerate(lss):
        print(i, ls)
        ev = fwd.eigenvector(i)
        lam = fwd.multiplier[i] * np.sqrt(pr.multiplier[i])
        plt.plot(x, lam * ev, label=i+1, ls=ls)
    plt.plot([0,1], [0,0], ls='-', color='k', alpha=0.5)    
    plt.xlabel(r"$x \in \Omega$", fontsize=fs)
    plt.legend(title='eigenvector, weighted')
    plt.tight_layout()
    plt.savefig(f"latex/eigenvectors_{transform}_scaled.png")


    
    plt.close('all')
    fig = plt.figure(figsize=(8,4))
    fs = 18
    lss = ['solid', 'dotted', 'dashed', 'dashdot']
    vals = np.zeros(m)
    plt.scatter(design, vals)
    x = fwd.x
    for i,ls in enumerate(lss):
        print(i, ls)
        ev = fwd.eigenvector(i)
        lam = fwd.multiplier[i] * np.sqrt(pr.multiplier[i])
        plt.plot(x, ev, label=i+1, ls=ls)
    plt.plot([0,1], [0,0], ls='-', color='k', alpha=0.5)    
    plt.xlabel(r"$x \in \Omega$", fontsize=fs)
    plt.legend(title='eigenvector, weighted')
    plt.tight_layout()
    plt.savefig(f"latex/figs/eigenvectors_{transform}_modelError{MODEL_ERROR}.png")
   

    

main()

