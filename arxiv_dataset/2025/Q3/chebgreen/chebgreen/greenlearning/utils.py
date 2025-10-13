from chebgreen.backend import torch, np, scipy, ABC, config, device, parser

class Dataset(torch.utils.data.Dataset):
    """ Dataset class for the Green's function approximation."""
    def __init__(self, F, U):
        self.F = F
        self.U = U
    
    def __len__(self):
        return self.U.shape[0]
    
    def __getitem__(self, index):
        return self.F[index, :], self.U[index, :]
    
class DataProcessor(ABC):
    def __init__(self, filePath, seed = 42):
        """
        Constructor for the DataProcessor class which loads the data from the file and splits it into training and validation datasets.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            filePath: A string which specifies the path to the data file.

            seed: An integer which specifies the seed for the random number generator.
        ----------------------------------------------------------------------------------------------------------------
        """
        self.data = None
        self.filePath = filePath
        self.seed = seed
        
    
    def generateDataset(self, trainRatio = parser['GREENLEARNING'].getfloat('trainRatio'), device = device):
        """
        Method to generate the training and validation datasets.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            trainRatio: A float which specifies the ratio of training data to the total data.
            device: A torch.device object which specifies the device to store the data.
        ----------------------------------------------------------------------------------------------------------------
        """
        data = scipy.io.loadmat(self.filePath)
        np.random.seed(self.seed)

        self.xF = data['Y'].astype(dtype = config(np))
        self.xU = data['X'].astype(dtype = config(np))
        # self.xG = data['XG'].astype(dtype = config(np))
        # self.yG = data['YG'].astype(dtype = config(np))

        F = torch.from_numpy(data['F'].astype(dtype = config(np))).to(device)
        U = torch.from_numpy(data['U'].astype(dtype = config(np))).to(device)

        # Train-validation split
        TrainTestIndices = np.random.choice(F.shape[1], F.shape[1], replace = False)
        iSplit = int(trainRatio*data['F'].shape[1])
        Train_Indices, Test_Indices = TrainTestIndices[:iSplit], TrainTestIndices[iSplit:]
        
        self.trainDataset = (F[:,Train_Indices].T, U[:,Train_Indices].T)
        self.valDataset = (F[:,Test_Indices].T, U[:,Test_Indices].T)

def generateEvaluationGrid(xU, xF):
    """
    Function to generate an evaluation grid for the Green's function.
    ----------------------------------------------------------------------------------------------------------------
    Arguments:
        xU: A tensor of shape (nU, d) which specifies the evaluation points for the Green's function.

        xF: A tensor of shape (nF, d) which specifies the evaluation points for the Green's function.
    ----------------------------------------------------------------------------------------------------------------
    Returns:
    A tensor of shape (nF*nU, 1) which specifies the evaluation points for the Green's function.
    """
    nF, nU, d = xF.shape[0], xU.shape[0], xU.shape[1]
    x, y = [],[]
    for i in range(d):
        x.append(torch.reshape(torch.tile(xU[:,i].reshape((1,nU)), [nF,1]), (nF*nU,1)))
        y.append(torch.reshape(torch.tile(xF[:,i].reshape((nF,1)), [1,nU]), (nF*nU,1)))
    return torch.stack(x+y, dim = 1)[...,0]


def approximateDistanceFunction(x, y, dom, device):
    """
    Function to evaluate Approximate Distance Function for a specified domain at a set of evaluation points.
    ----------------------------------------------------------------------------------------------------------------
    Arguments:
        x: A tensor of shape (n, 1) which specifies the x-coordinates of the evaluation points.

        y: A tensor of shape (n, 1) which specifies the y-coordinates of the evaluation points.

        domain: A list of size 4 which specifies the domain for the Green's function.
    ----------------------------------------------------------------------------------------------------------------
    Returns:
    A tensor of shape (n, 1) which specifies the Approximate Distance Function for the domain.
    """
    # Define a distance metric
    def distance(x1, y1, x2, y2):
        return torch.sqrt(torch.square(x2-x1) + torch.square(y2-y1))

    # Define a distance function for a line segment
    def lineSegment(x, y, x1, y1, x2, y2):
        L = distance(x1, y1, x2, y2)
        xc, yc = (x1+x2)/2, (y1+y2)/2
        f = (1/L) * ((x-x1)*(y2-y1) - (y-y1) * (x2-x1))
        t = (1/L) * ((L/2)**2 - torch.square(distance(x,y,xc,yc)))
        phi = torch.sqrt(torch.square(t) + torch.pow(f,4))
        return torch.sqrt(torch.square(f) + 0.25 * torch.square(phi-t)) + torch.finfo(config(torch)).eps
    
    # Define the line segments which define the domain
    R = torch.zeros((x.shape[0],1))
    segments = torch.from_numpy(np.array([[dom[0], dom[2], dom[1], dom[2]],
                         [dom[1], dom[2], dom[1], dom[3]],
                         [dom[1], dom[3], dom[0], dom[3]],
                         [dom[0], dom[3], dom[0], dom[2]]
                        ]).astype(config(np))).to(device)
    
    # Combine the distance functions for each line segment to get the distance function for the domain
    phi = []
    for i in range(4):
        phi.append(lineSegment(x,y,segments[i,0], segments[i,1], segments[i,2], segments[i,3]))
    # Phi = phi[0]*phi[1]*phi[2]*phi[3]/(phi[0]+phi[1]+phi[2]+phi[3])
    Phi = phi[0]*phi[1]*phi[2]*phi[3]/(phi[1]*phi[2]*phi[3]+phi[0]*phi[2]*phi[3]+phi[0]*phi[1]*phi[3]+phi[0]*phi[1]*phi[2])
    return Phi