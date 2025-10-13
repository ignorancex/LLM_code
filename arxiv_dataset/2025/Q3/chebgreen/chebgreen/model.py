from .greenlearning.utils import DataProcessor
from .greenlearning.model import GreenNN
from .chebpy2 import Chebfun2, Chebpy2Preferences, Quasimatrix
from .chebpy2.chebpy import chebfun, Chebfun
from .chebpy2.chebpy.core.settings import ChebPreferences
from .backend import os, sys, Path, np, ABC, MATLABPath, parser, ast, config, print_settings, plt
from .utils import generateMatlabData, computeEmpiricalError
from typing import Optional, List, Dict, Tuple

class ChebGreen(ABC):
    def __init__(self,
                Theta           : np.ndarray,
                domain          : List = [0, 1, 0, 1],
                generateData    : bool = True,
                script          : str = "generate_example",
                example         : Optional[str] = None,
                dirichletBC     : Optional[bool] = True,
                datapath        : Optional[str] = None,
                noise_level     : Optional[float] = 0.0
                ):
        super().__init__()
        """
        Arguments:        
            Theta: A numpy array of shape (N_models, N_dimension) which specifies parameteric value at
            which the models are specified or need to be evaluated at.

            domain: A list of size 4 which specifies the domain for the Green's function.

            generateData: If set to True, "path" should be the location of the matlab script which is
            run at the parametric values in Theta to generate the dataset.

            script: A string which specifies the name matlab script in the current directory for
            generating the dataset.

            example: This specifies the name of the example which the user wants to run.

            dirichletBC: A boolean which specifies whether the boundary conditions are Dirichlet or not.

            data: If generateData is set to False, then user must provide the path to the dataset, which
            should be in the following format:

            The dataset should be a .mat file with the following fields:
                - X: The positions of points in the domain at which the solutions are sampled.
                    An array of size (Nx, 1) where Nx is number of points of the afforementioned points.
                - Y: The positions of points in the domain at which the forcing functions are sampled.
                    An array of size (Ny, 1) where Ny is number of points of the afforementioned points.
                - F: The forcing terms for the Green's function.
                    An array of size (Ny, N) where N is the number of samples.
                - U: The solutions for the system under corresponding forcing functions.
                    An array of size (Nx, N) where N is the number of samples.

            noise_level: A float which specifies the noise level of the data when generating the dataset.
        """
        
        self.Theta = Theta

        if type(domain) is not np.ndarray and type(domain) is list:
            self.domain = np.array(domain)
        else:
            self.domain = domain
        
        self.dirichletBC = dirichletBC

        # Set data path for the model:
        if generateData: 
            print(f"Generating dataset for example \'{example}\'")
            if noise_level is not None:
                self.datapath = generateMatlabData(script, example, self.Theta, noise_level)
            else:
                self.datapath = generateMatlabData(script, example, self.Theta)
        else:
            print(f"Loading dataset at {datapath}")
            assert datapath is not None, "No datapath specified!"
            self.datapath = datapath

        # Load or fit greenlearning models on the dataset and learn a chebfun2 on them:
        sys.stdout.flush()
        print("-------------------------------------------------------------------------------\n")
        print("Generating chebfun2 models:")
        self.generateChebfun2Models(example)
        self.interpG = {}
        self.interpN = {}

    def generateChebfun2Models(self, example: str) -> None:
        model = GreenNN()
        self.G = {}
        self.N = {}
        for theta in self.Theta:
            
            GreenNNPath = "savedModels/" + example + f"/{theta:.2f}" 
            
            if model.checkSavedModels(loadPath = GreenNNPath):          # Check for stored models
                print(f"Found saved model, Loading model for example \'{example}\' at Theta = {theta:.2f}")
                model.build(dimension = 1, domain = self.domain, dirichletBC = self.dirichletBC, loadPath = GreenNNPath)
            else:
                data = DataProcessor(self.datapath + f"/{theta:.2f}/data.mat")
                data.generateDataset(trainRatio = 0.95)
                model.build(dimension = 1, domain = self.domain, dirichletBC = self.dirichletBC)
                print(f"Training greenlearning model for example \'{example}\' at Theta = {theta:.2f}")
                lossHistory = model.train(data, savePath = GreenNNPath)
                model.build(dimension = 1, domain = self.domain, dirichletBC = self.dirichletBC, loadPath = GreenNNPath)
                with open(f"savedModels/{example}/{theta:.2f}/settings.ini", 'w') as f:
                    print_settings(file = f)

                plt.figure(figsize=(8, 6))
                plt.semilogy(range(1, len(lossHistory['training']) + 1), lossHistory['training'], label='Training')
                plt.semilogy(range(1, len(lossHistory['validation']) + 1), lossHistory['validation'], label='Validation')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title(f'Loss History for Theta = {theta:.2f}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save the plot in the same folder as settings.ini
                plot_path = f"savedModels/{example}/{theta:.2f}/loss_history.png"
                plt.savefig(plot_path)
                plt.close()
            
            print(f"Learning a chebfun model for example \'{example}\' at Theta = {theta:.2f}")
            prefs = Chebpy2Preferences()
            if parser.has_option('CHEBFUN', 'eps_x'):
                prefs.prefx.eps = parser['CHEBFUN'].getfloat('eps_x')
            if parser.has_option('CHEBFUN', 'eps_y'):
                prefs.prefy.eps = parser['CHEBFUN'].getfloat('eps_y')
            self.G[float(theta)] = (Chebfun2(model.evaluateG, domain = self.domain, prefs = prefs, simplify = True))
            print(f"Chebfun model added for example \'{example}\' at Theta = {theta:.2f}\n")

            # Store the homogeneous solution
            prefs = ChebPreferences()
            prefs.eps = np.finfo(config(np)).eps
            self.N[float(theta)] = chebfun(model.evaluateN, domain = self.domain[2:], prefs = prefs)
    
        maxRank = np.min(np.array([self.G[theta].rank for theta in self.Theta]))

        for theta in self.Theta:
            self.G[theta].truncate(maxRank)

        
    def generateNewModel(self, theta: float) -> Chebfun2:
        if theta not in list(self.interpG.keys()):
            self.interpG[theta], self.interpN[theta] = modelInterp(self.G, self.N, theta)
        return self.interpG[theta], self.interpN[theta]
    
    def computeEmpiricalError(self, theta: float, data: Optional[DataProcessor] = None) -> float:        
        if theta in list(self.interpG.keys()):
            G = self.interpG[theta]
            N = self.interpN[theta]
        elif theta in list(self.G.keys()):
            G = self.G[theta]
            N = self.N[theta]
        else:
            raise RuntimeError("No model found for the specified parameter!")
        
        if data is not None:
            return computeEmpiricalError(data, G, N)

        assert self.datapath is not None, "Cannot find the datapath for the model datasets!"
        data = DataProcessor(self.datapath + f"/{theta:.2f}/data.mat")
        data.generateDataset(trainRatio = 0.95)

        return computeEmpiricalError(data, G, N)
            

def computeInterpCoeffs(interpParams : List[float], targetParam: float) -> np.ndarray:
    """Computes the interpolation coefficients (based on fitting Lagrange polynomials) for performing interpolation,
    when the parameteric space is 1D. Note that the function takes in parameters of any dimnesions

    --------------------------------------------------------------------------------------------------------------------
    Arguments:
        interpSet: Dictionary of models (Chebfun2 objects) which are used to generate an interoplated model at the
            parameter targetParam.

        targetParam: A float which defines the parameter at which we want to find a new model.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        A dict of Interpolation coefficents index
    """
    
    assert len(interpParams) > 2, "Need at least two interpolant models"
    assert all([isinstance(theta,float) for theta in interpParams]), \
        "Lagrange Polynomial based interpolation requires the parameteric space to be 1D."

    interpCoeffs = dict([(theta, 1.0) for theta in interpParams])

    for i,t1 in enumerate(interpCoeffs):
        for j,t2 in enumerate(interpCoeffs):
            if i != j:
                interpCoeffs[t1] = interpCoeffs[t1]*((targetParam - t2)/(t1 - t2))

    return interpCoeffs

def computeOrderSigns(R0: Quasimatrix, R1: Quasimatrix) -> Tuple[np.ndarray, np.ndarray]:
    """ Given two orthonormal matrices R0 and R1, this function computes the "correct" ordering and signs of the columns
    (modes) of R1 using R0 as a reference. The assumption is that these are orthonormal matrices, the columns of which
    are eigenmodes of systems which are close to each other and hence the eigenmodes will close to each other as well.
    We thus find an order such that the modes of the second matrix have the maximum inner product (in magnitude) with
    the corresponding mode from the first matrix. If such an ordering doesn't exist the function raises a runtime error.

    Once such an ordering is found, one can flip the signs for the modes of R1, if the inner product is not positive.
    This is necessary when we want to interpolate.

    --------------------------------------------------------------------------------------------------------------------
    Args:
        R0: Orthonormal quasimatrix, the columns of which are used as the reference to re-order and find signs

        R1: Orthonormal quasimatrix for which the columns are supposed to be reordered.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        New ordering and the signs (sign flips) for matrix R1.
    """
    rank = R0.shape[1]
    order = -1*np.ones(rank).astype(int)
    signs = np.ones(rank)
    
    used = set()
    # For each mode in R1, Search over all modes of R0 for the best matching mode.
    products = np.abs(R0.T * R1) # Compute all the pairwise innerproducts
    for i in range(rank):
        maxidx, maxval = -1, -1
        for j in range(rank):
            current = products[i,j]
            if current >= maxval and (j not in used):
                maxidx = j
                maxval = current
        order[i] = maxidx
        used.add(maxidx)
    
    # Raise an error if the ordering of modes is not a permutation.
    check = set()
    for i in range(rank):
        check.add(order[i])

    if len(check) != rank:
        raise RuntimeError('No valid ordering of modes found')
    
    # Signs are determined according to the correct ordering of modes
    for i in range(rank):
        if (R0[:,i].T * R1[:,int(order[i])]).item() < 0:
            signs[i] = -1
    
    return order, signs

def modelInterp(interpSet: Dict[float,Chebfun2], interpSetHom: Dict[float,Chebfun], targetParam: float) -> Chebfun2:
    """
    Interpolation for the models. The left and right singular functions are interpolated in the tangent space of
    (L^2(domain))^K (K is the model rank) using a QR based retraction map. The singular values are interpolated
    directly (entry-by-entry) using a Lagrange polynomial based inteporlation. Note that currently the interpolation
    only supports 1D parameteric spaces for the model as the method for interpolation within the tangent space only
    supports 1D parameteric spaces but this can be easily extended to higher dimensions. The lifting and retraction to
    the tangent space at an "origin" has no dependendence on the dimensionality of the parameteric space.

    --------------------------------------------------------------------------------------------------------------------
    Arguments:
        interpSet: Dictionary of models (Chebfun2 objects) which are used to generate an interoplated model at the
            parameter targetParam.
            
        targetParam: A float which defines the parameter at which we want to find a new model.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        An Chebfun2 object at the target parameter.
    """
    interpParams = list(interpSet.keys())

    assert len(interpParams) > 2, "Need at least two interpolant models"

    # Find the model which is closest to target parameter. Note that the distance is calculated in terms of norm of the
    # normalized parameters.
    refIndex = None
    minDistance = np.inf
    for theta in interpParams:
        distance = np.linalg.norm((theta-targetParam)/targetParam)
        if distance < minDistance:
            minDistance = distance
            refIndex = theta
    
    interpCoeffs = computeInterpCoeffs(interpParams, targetParam)
    
    # Define the origin
    U0, _, Vt0 = interpSet[refIndex].cdr()
    V0 = Vt0.T
    K = interpSet[refIndex].rank
    
    # Initialize the interpolated singular functions and singular values
    U_ = Quasimatrix(data = chebfun(np.zeros((2,K)), domain = interpSet[refIndex].domain[2:]), transposed = False)
    S_ = np.zeros(K)
    V_ = Quasimatrix(data = chebfun(np.zeros((2,K)), domain = interpSet[refIndex].domain[:2]), transposed = False)
    N_ = chebfun(0, domain = interpSetHom[refIndex].domain)
    
    # Interpolate the singular functions and singular values
    for theta, model in interpSet.items():
        U, S, Vt = model.cdr()
        order, signs = computeOrderSigns(U0,U)
        
        Uc = U[:,order] * np.diag(signs)
        S = np.diag(S)[order]
        Vc = (Vt.T)[:,order] * np.diag(signs)
        
        # Project to tangent space of model at origin
        Up = Uc - U0 * ((U0.T * Uc + Uc.T * U0)*0.5)
        Vp = Vc - V0 * ((V0.T * Vc + Vc.T * V0)*0.5)
        
        
        # Interpolate the singular functions
        U_ += Up * np.diag(np.ones(K)*interpCoeffs[theta])
        V_ += Vp * np.diag(np.ones(K)*interpCoeffs[theta])
        
        # Interpolate the singular values directly
        S_ += S * interpCoeffs[theta]
        N_ += interpSetHom[theta] * interpCoeffs[theta]
        

    # Retract the interpolated singular functions and singular values from the tangent space at the origin
    Un, _ = (U0 + U_).qr()
    Vn, _ = (V0 + V_).qr()
    
    # Match the order and signs with the origin
    order, signs = computeOrderSigns(U0,Un)
    Un = Un[:,order] * np.diag(signs)
    Sn = S_[order]
    Vtn = (Vn[:,order] * np.diag(signs)).T
    
    
    return Chebfun2([Un, Sn, Vtn]), N_




            
            

            

            