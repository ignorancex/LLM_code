from chebgreen.backend import torch, np, ABC, config, device
from .utils import generateEvaluationGrid, approximateDistanceFunction
from .quadrature_weights import get_weights

class LossGreensFunction(ABC):
    def __init__(self, G, N, xF, xU, domain, addADF, device) -> None:
        super().__init__()
        """
        Class defining the loss function for the Green's function approximation.

        ----------------------------------------------------------------------------------------------------------------
        Arguments:        
            G: A neural network which represents the Green's function for the domain.

            N: A neural network which represents the homogeneous solution for the domain.

            xF: A numpy array of shape (N_f, 1) which specifies the evaluation points for the function.

            xU: A numpy array of shape (N_u, 1) which specifies the evaluation points for the solution.

            domain: A numpy array of size 4 which specifies the domain for the Green's function.
            
            addADF: A boolean which specifies whether to add the approximate distance function to the loss.
                Adding an approximate distance function improves the Green's function approximation in case
                of a Dirichlet boundary condition.

            device: torch device on which the computations are performed.
        """

        # Store the Green's function, the homogeneous solution, the evaluation points for the function and the solution
        self.G = G
        self.N = N
        self.xF, self.xU = torch.from_numpy(xF).to(device), torch.from_numpy(xU).to(device)

        # Compute and store the quadrature weights for the evaluation points
        self.wF = torch.from_numpy(get_weights('trapezoidal', xF).astype(dtype = config(np)).T).to(device)
        self.wU = torch.from_numpy(get_weights('trapezoidal', xU).astype(dtype = config(np)).T).to(device)
        self.X = generateEvaluationGrid(self.xU, self.xF).to(device)
        self.addADF = addADF
        
        # Compute and store the approximate distance function if addADF is True
        if addADF:
            self.ADF = approximateDistanceFunction(self.X[:,0], self.X[:,1],
                                               domain.astype(config(np)),
                                               device
                                               ).to(device).reshape((-1,1))

    def __call__(self, fTrain, uTrain, trainHomogeneous = True):
        """
        Method to compute the loss function for the Green's function approximation.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            fTrain: A numpy array of shape (N_f, 1) which specifies the function values at the training points.

            uTrain: A numpy array of shape (N_u, 1) which specifies the solution values at the training points.
            
            trainHomogeneous: A boolean which specifies whether the homogeneous solution is trained or not.
        ----------------------------------------------------------------------------------------------------------------
        Returns:
            loss: A scalar tensor which represents the value of the loss function.
        """
        nF, nU = self.xF.shape[0], self.xU.shape[0]
        if self.addADF:
            G = torch.reshape(self.ADF*self.G(self.X),(nF, nU)).T
        else:
            G = torch.reshape(self.G(self.X),(nF, nU)).T            
        uHomogeneous = self.N(self.xU)
        uPred = (torch.matmul(G, torch.mul(fTrain, self.wF).T) + uHomogeneous).T

        if trainHomogeneous:
            loss = torch.sum(torch.square(uTrain - uPred)*self.wU, dim = 1)/ \
                    torch.sum(torch.square(uTrain)*self.wU, dim = 1)
        else:
            loss = torch.sum(torch.square(uTrain - uPred)*self.wU, dim = 1)/ \
                    torch.sum(torch.square(uTrain - uHomogeneous.T)*self.wU, dim = 1)
        return torch.mean(loss)