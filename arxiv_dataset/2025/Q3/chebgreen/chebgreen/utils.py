from .backend import os, sys, Path, MATLABPath, parser, np, tempfile, config, chebgreen_path, print_settings
from chebgreen.chebpy2.chebpy.core.algorithms import chebpts2, vals2coeffs2, standard_chop
from chebgreen.chebpy2.chebpy.api import chebfun, Chebfun
from chebgreen.chebpy2 import Chebfun2
from chebgreen.greenlearning.utils import DataProcessor
from chebgreen.chebpy2.chebpy.core.settings import ChebPreferences, _preferences
from typing import Optional, List
import matplotlib.pyplot as plt
from scipy.io import loadmat

def runCustomScript(script      : str,
                    example     : str   = "data",
                    theta       : Optional[float]  = None,
                    Nsample     : int  = parser['MATLAB'].getint('Nsample'),
                    lmbda       : int  = parser['MATLAB'].getfloat('lambda'),
                    Nf          : int  = parser['MATLAB'].getint('Nf'),
                    Nu          : int  = parser['MATLAB'].getint('Nu'),
                    noise_level : float = parser['MATLAB'].getfloat('noise'),
                    seed        : int  = 0,
                    saveSuffix  : Optional[str]   = None) -> None:
    """
    Arguments:
    script: Name of the matlab script to run
    example: Name of the example to run/Name for the save file
    Nsample: Number of sampled pairs f/u
    lambda: Lengthscale of kernel for sampling f
    Nf: Discretization for f, lambda > (size of domain)/Nf 
    Nu: Discretization for u
    noise_level: Noise level of the solutions u
    theta: Control parameter value for the problem
    saveSuffix: Suffix to add to the save file name

    Returns:
    Nothing. Saves the data in the datasets folder.
    """
    sys.stdout.flush() # Flush the stdout buffer

    # # Set the appropriate example name
    example_name = "\'"+example+"\'"
    scriptsPath = chebgreen_path / "scripts"
    examplesPath = scriptsPath / "examples"

    # Depending on the script type, define the appropriate MATLAB command
    if theta is None:
        if saveSuffix is None:
            matlabcmd = " ".join(f"{MATLABPath} -nodisplay -nosplash -nodesktop -r \
            \"addpath('{scriptsPath}'); addpath('{examplesPath}');\
            {script}({example_name},{int(Nsample)},{lmbda},{int(Nf)},{int(Nu)},{noise_level:.2f},{seed}); exit;\" | tail -n +11".split())
        else:
            raise ValueError("saveSuffix cannot be provided for this script type.")
    else:
        if saveSuffix is not None:
            matlabcmd = " ".join(f"{MATLABPath} -nodisplay -nosplash -nodesktop -r \
            \"addpath('{scriptsPath}'); addpath('{examplesPath}');\
            {script}({example_name},{int(Nsample)},{lmbda},{int(Nf)},{int(Nu)},{noise_level:.2f},{seed},{theta:.2f},\'{saveSuffix}\'); exit;\" | tail -n +11".split())
        else:
            matlabcmd = " ".join(f"{MATLABPath} -nodisplay -nosplash -nodesktop -r \
            \"addpath('{scriptsPath}'); addpath('{examplesPath}');\
            {script}({example_name},{int(Nsample)},{lmbda},{int(Nf)},{int(Nu)},{noise_level:.2f},{seed},{theta:.2f}); exit;\" | tail -n +11".split())

    # Write the MATLAB command to a temporary file and run it
    temp = next(tempfile._get_candidate_names()) + '.sh'
    with open(temp, 'w') as f:
        f.write(matlabcmd)
        f.close()

    os.system(f"bash {temp}") # Run the temporary file
    os.remove(temp) # Remove the temporary file
    sys.stdout.flush() # Flush the stdout buffer
    
    plot_random_sample(example, theta, saveSuffix) # Plot a random sample from the dataset

    if theta is None:
        if saveSuffix is not None:
            with open(f"datasets/{example}-{saveSuffix}/settings.ini", 'w') as f:
                print_settings(file = f)
        else:
            with open(f"datasets/{example}/settings.ini", 'w') as f:
                print_settings(file = f)
    else:
        if saveSuffix is not None:
            with open(f"datasets/{example}/{theta:.2f}-{saveSuffix}/settings.ini", 'w') as f:
                print_settings(file = f)
        else:
            with open(f"datasets/{example}/{theta:.2f}/settings.ini", 'w') as f:
                print_settings(file = f)

def generateMatlabData(script: str, example: str, Theta: Optional[List] = None, noise_level: Optional[float] = parser['MATLAB'].getfloat('noise')) -> str:
    if Theta is None:
        runCustomScript(script, example, noise_level = noise_level) # Run the custom script without theta
    else:
        for theta in Theta:
            # Check if the dataset already exists
            if Path(f"datasets/{example}/{theta:.2f}/data.mat").is_file():
                print(f"Dataset found for Theta = {theta:.2f}. Skipping dataset generation.")
                continue
            sys.stdout.flush() # Flush the stdout buffer
            runCustomScript(script, example, theta, noise_level = noise_level) # Run the custom script with theta

    sys.stdout.flush() # Flush the stdout buffer
    return os.path.abspath(f"datasets/{example}") # Return the path to the dataset

def vec2cheb(f: np.ndarray, x: np.ndarray, domain: Optional[List] = None) -> Chebfun:
    f = f.reshape((-1))
    x = x.reshape((-1))
    if domain is None:
        domain = [np.min(x), np.max(x)] # Compute the bounds of the domain
    
    # Function is resampled at twice as many sampled points to maintain accuracy in a different basis.
    N = 2 * f.shape[0] # Check if this is fine
    
    # Compute the Chebyshev nodes and scale them to the domain
    xc = (chebpts2(N) + 1) * ((domain[1] - domain[0])/2) + domain[0]

    # Compute the interpolated value of the function at the Chebyshev nodes
    fc = np.interp(xc, x, f).reshape((-1,1))

    prefs = ChebPreferences()
    prefs.eps = np.finfo(config(np)).eps

    coeffs_fc = vals2coeffs2(fc)

    # Find tolerance for standard chop
    a, b = domain
    h = max(np.linalg.norm(domain, np.inf),1)
    hF = b - a
    hscale = max(h / hF, 1)
    tol = prefs.eps * hscale

    # Determine where to chop the coefficients
    npts = standard_chop(coeffs_fc, tol)

    cbfun = chebfun(coeffs_fc[:npts].reshape(-1,1), domain, prefs = prefs, initcoeffs = True)
    return cbfun

def computeEmpiricalError(data: DataProcessor, G: Chebfun2, N: Optional[Chebfun] = None) -> float:
    RE = []
    if N is None:
        print('Assuming a zero homogeneous solution.')
    for i in range(data.valDataset[1].cpu().numpy().shape[0]):
        xF, xU = data.xF, data.xU
        f, u  = data.valDataset[0].cpu().numpy()[i,:], data.valDataset[1].cpu().numpy()[i,:]

        # Artifically fix the domain precision issue: Need to fix this
        domainF = np.array([np.min(xF), np.max(xF)])
        domainU = np.array([np.min(xU), np.max(xU)])
        if np.isclose(domainF,G.domain[2:], atol = np.finfo(config(np)).eps).all():
            domainF = G.domain[2:]
        if np.isclose(domainU,G.domain[:2], atol = np.finfo(config(np)).eps).all():
            domainU = G.domain[:2]

        f0, u0 = vec2cheb(f,xF,domainF), vec2cheb(u,xU,domainU)
        if N is None:
            uc = G.T.integralTransform(f0)
        else:
            uc = G.T.integralTransform(f0) + N

        # Ensure that the the error computation is done with the correct precision
        prefs = ChebPreferences()
        prefs.eps = np.finfo(config(np)).eps
        re = (uc - u0).abs()/u0.abs().sum()
        _preferences.reset()
        RE.append(re)
    
    error = np.mean([re.sum() for re in RE])
    return error

# Load and plot data from the generated .mat file
def plot_random_sample(example, theta=None, saveSuffix=None):

    # Construct the path to the data file
    if theta is None:
        data_path = f"datasets/{example}/data.mat"
    else:
        if saveSuffix:
            data_path = f"datasets/{example}/{theta:.2f}-{saveSuffix}/data.mat"
        else:
            data_path = f"datasets/{example}/{theta:.2f}/data.mat"

    # Load the .mat file
    data = loadmat(data_path)

    # Extract the required arrays
    U = data['U']
    F = data['F']
    X = data['X'].flatten()
    Y = data['Y'].flatten()

    # Choose a random column
    random_col = np.random.randint(0, U.shape[1])

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot F vs Y
    ax1.plot(Y, F[:, random_col])
    ax1.set_title('Forcing Function (F)')
    ax1.set_xlabel('s')
    ax1.set_ylabel('F')

    # Plot U vs X
    ax2.plot(X, U[:, random_col])
    ax2.set_title('System Response (U)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('U')

    # Set the main title
    if theta is None:
        fig.suptitle(f'Random Sample from {example}')
    else:
        fig.suptitle(f'Random Sample from {example} ($\\theta = {{{theta:.2f}}}$)')

    plt.tight_layout()

    # Save the plot in the same directory as the data
    save_dir = os.path.dirname(data_path)
    save_path = os.path.join(save_dir, 'random_sample.png')
    plt.savefig(save_path)
    plt.close(fig)