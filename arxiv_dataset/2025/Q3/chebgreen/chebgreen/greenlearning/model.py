from chebgreen.backend import torch, np, Path, ABC, config, device, parser, ast
from .activations import get_activation
from .loss import LossGreensFunction
from .utils import approximateDistanceFunction

MODEL_SAVE_FREQUENCY = 50

class NN(torch.nn.Module):
    def __init__(self,
                 numInputs = 2,
                 numOutputs = 1,
                 layerConfig = ast.literal_eval(parser['GREENLEARNING']['layerConfig']),
                 activation = parser['GREENLEARNING']['activation'],
                 dtype = config(torch),
                 device = device):
        super(NN, self).__init__()
        """
        Class to define a generic feedforward neural network.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            numInputs: An integer which specifies the number of inputs to the neural network.

            numOutputs: An integer which specifies the number of outputs from the neural network.

            layerConfig: A list of integers which specifies the number of neurons in each layer of the neural network.

            activation: A string which specifies the activation function to be used in the neural network.

            dtype: A torch.dtype which specifies the data type for the neural network.

            device: A torch.device which specifies the device for the neural network.
        ----------------------------------------------------------------------------------------------------------------
        """

        # Initialize the Layers. We hold all layers in a ModuleList.
        self.layers = torch.nn.ModuleList()
        self.activationFunctions = torch.nn.ModuleList()

        # Add the input layer
        self.layers.append(torch.nn.Linear(
                                in_features  = numInputs,
                                out_features = layerConfig[0],
                                bias         = True ).to(dtype = dtype, device = device))
        self.activationFunctions.append(get_activation(activation))

        # Add the hidden layers based on the layer configuration.
        for neuronsIn, neuronsOut in zip(layerConfig[:-1],layerConfig[1:]):
            self.layers.append(torch.nn.Linear(
                                in_features  = neuronsIn,
                                out_features = neuronsOut,
                                bias         = True ).to(dtype = dtype, device = device))
            self.activationFunctions.append(get_activation(activation))

        # Add the output layer
        self.layers.append(torch.nn.Linear(
                                in_features  = layerConfig[-1],
                                out_features = numOutputs,
                                bias         = True ).to(dtype = dtype, device = device))
        
        # Initialize layers
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # Loop over all the layers and activation functions.
        for layer, activation in zip(self.layers[:-1],self.activationFunctions):
            X = activation(layer(X))
        
        # Last layer does not take an activation function.
        return self.layers[-1](X)
    
class GreenNN(ABC):
    def __init__(self) -> None:
        super().__init__()
        """
            Class to define a neural network architecture and training loop for the Green's function approximation.
        """

    def build(self,
            dimension = 1,
            domain = np.array([0,1,0,1]),
            layerConfig = ast.literal_eval(parser['GREENLEARNING']['layerConfig']),
            activation = parser['GREENLEARNING']['activation'],
            dirichletBC = True,
            loadPath = None,
            device = device):
        """
        Method to build the neural network architecture.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            dimension: An integer which specifies the dimension of the problem.

            domain: A list or numpy array which specifies the domain for the Green's function.

            layerConfig: A list of integers which specifies the number of neurons in each hidden layer of the neural network.

            activation: A string which specifies the activation function to be used in the neural network.

            dirichletBC: A boolean which specifies whether the problem has Dirichlet boundary condition or not.

            loadPath: A string which specifies the path to the saved model. If a path is provided,
                the model is loaded from the path.

            device: A torch.device which specifies the device for the neural network.
        ----------------------------------------------------------------------------------------------------------------
        """

        self.dimension = dimension
        if type(domain) is not np.ndarray and type(domain) is list:
            self.domain = np.array(domain)
        else:
            self.domain = domain
        self.device = device
        self.layerConfig = layerConfig
        self.activation = activation
        self.addADF = dirichletBC

        if loadPath == None:
            self.G = NN(numInputs = dimension*2, numOutputs = dimension, layerConfig = layerConfig, activation = activation).to(self.device)
            self.N = NN(numInputs = dimension, numOutputs = dimension, layerConfig = layerConfig, activation = activation).to(self.device)
        else:
            assert self.checkSavedModels(loadPath), "Saved models not found" 
            self.loadModels(loadPath, device = device)

    def train(self,
              data,
              epochs = {'adam':parser['GREENLEARNING'].getint('epochs_adam') ,
                        'lbfgs':parser['GREENLEARNING'].getint('epochs_lbfgs')},
              trainHomogeneous = True,
              savePath = "temp"
              ):
        """
        Method to train the neural network.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            data: An instance of the DataLoader class which contains the training and validation datasets.

            epochs: A dictionary which specifies the number of epochs for Adam and LBFGS optimizers.

            trainHomogeneous: A boolean which specifies whether to learn the homogeneous solution.
                This can be set to False, if we know beforehand that the homogeneous solution is zero.
        ----------------------------------------------------------------------------------------------------------------
        """
        if trainHomogeneous:
            params = list(self.G.parameters()) + list(self.N.parameters())
        else:
            params = list(self.G.parameters())
        initial_learning_rate = parser['GREENLEARNING'].getfloat('initLearningRate')
        self.optimizerAdam = torch.optim.Adam(params, lr = initial_learning_rate)
        final_learning_rate = parser['GREENLEARNING'].getfloat('finalLearningRate')
        gamma = np.exp(np.log(final_learning_rate / initial_learning_rate) / epochs['adam'])
        self.schedulerAdam = torch.optim.lr_scheduler.ExponentialLR(self.optimizerAdam, gamma=gamma)
        # self.schedulerAdam = torch.optim.lr_scheduler.StepLR(self.optimizerAdam,
        #                                                     step_size = parser['GREENLEARNING'].getint('stepSize'),
        #                                                     gamma = parser['GREENLEARNING'].getfloat('decayRate'))
                                                            
        self.optimizerLBFGS = torch.optim.LBFGS(params, lr = parser['GREENLEARNING'].getfloat('initLearningRate'))

        # Check if the dimension of the evaluation points match with the model dimension
        assert data.xF.shape[1] == self.dimension and data.xU.shape[1] == self.dimension,\
                f"Dimension of evaluation points for forcing, {data.xF.shape[1]}, and response, \
                {data.xU.shape[1]}, should match with the model dimension, {self.dimension}"
        
        self.init_loss(data.xF, data.xU) # Initialize the loss function

        lossHistory = {'training': [], 'validation':[]} # Initialize a dictionary to store the loss history
        
        bestLoss = np.inf
        print("Training with Adam:")
        for epoch in range(int(epochs['adam'])):
            fTrain, uTrain = data.trainDataset
            fTrain, uTrain = fTrain.to(device), uTrain.to(device)
            self.G.train()
            self.N.train()

            self.optimizerAdam.zero_grad()
            lossValue = self.lossfn(fTrain, uTrain, trainHomogeneous)
            lossValue.backward()

            self.optimizerAdam.step()
            self.schedulerAdam.step()

            # Change this to be an average over batches. Currently assuming a single batch
            lossHistory['training'].append(lossValue.item())

            # Save the model if the validation loss is the better than the previous best loss
            if (epoch+1)%MODEL_SAVE_FREQUENCY == 0 and lossHistory['validation'][-1] < bestLoss:
                bestLoss = lossHistory['validation'][-1]
                self.saveModels(path = savePath)

            with torch.no_grad():
                fVal, uVal = data.valDataset
                lossValue = self.lossfn(fVal.to(device), uVal.to(device), trainHomogeneous)
            lossHistory['validation'].append(lossValue.item())
            if (epoch+1) % 100 == 0:
                print(f"Loss at epoch {epoch+1}: Training = {lossHistory['training'][-1]:.3E}, Validation = {lossHistory['validation'][-1]:.3E}")
        
        print("Training with LBFGS:")
        for epoch in range(int(epochs['lbfgs'])):
            fTrain, uTrain = data.trainDataset
            fTrain, uTrain = fTrain, uTrain
            self.G.train()
            self.N.train()

            # Closure function for LBFGS
            def closure():
                self.optimizerLBFGS.zero_grad()
                lossValue = self.lossfn(fTrain, uTrain, trainHomogeneous)
                lossValue.backward()
                return lossValue
            
            with torch.no_grad():
                lossValue = self.lossfn(fTrain, uTrain, trainHomogeneous)
                
            self.optimizerLBFGS.step(closure)
            
            # Change this to be an average over batches. Currently assuming a single batch
            lossHistory['training'].append(lossValue.item())

            # Save the model if the validation loss is the better than the previous best loss
            if (epoch+1)%MODEL_SAVE_FREQUENCY == 0 and lossHistory['validation'][-1] < bestLoss:
                bestLoss = lossHistory['validation'][-1]
                self.saveModels(path = savePath)

            with torch.no_grad():
                fVal, uVal = data.valDataset
                lossValue = self.lossfn(fVal.to(device), uVal.to(device), trainHomogeneous)
            lossHistory['validation'].append(lossValue.item())
            if (epoch+1) % 10 == 0:
                print(f"Loss at epoch {epoch+1}: Training = {lossHistory['training'][-1]:.3E}, Validation = {lossHistory['validation'][-1]:.3E}")

        # Fallback to save the last model if the MODEL_SAVE_FREQUENCY is not reached
            if lossHistory['validation'][-1] < bestLoss:
                bestLoss = lossHistory['validation'][-1]
                self.saveModels(path = savePath)

        return lossHistory
    
    # Not implemented for Green's function of dimension > 1
    def evaluateG(self, x, s):
        """
        Method to evaluate the Green's function at a set of evaluation points.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            x: A numpy array or torch tensor which specifies the evaluation points in the x-direction.
            
            s: A numpy array or torch tensor which specifies the evaluation points in the s-direction.
        ----------------------------------------------------------------------------------------------------------------
        Returns:
            A numpy array which specifies the value of the Green's function at the evaluation points.
        """

        # Check to ensure that the input is in the correct format.
        if isinstance(x,config(np)):
            x = np.array([x])
        if isinstance(s,config(np)):
            s = np.array([s])
        if isinstance(x.dtype, np.float64) or (s.dtype, np.float64):
            x, s = x.astype(config(np)), s.astype(config(np))
        assert(x.shape == s.shape), "Both input need to have the same shape"
        shape = x.shape
        if x.dtype == config(np):
            X = torch.tensor(np.vstack([x.ravel(), s.ravel()]).T, dtype = config(torch)).to(self.device)
        with torch.no_grad():
            # Evaluate the Green's function depending on whether the approximate distance function was added to the loss
            if self.addADF:
                G = (approximateDistanceFunction(X[:,0], X[:,1],
                                             self.domain.astype(config(np)),
                                             self.device).reshape((-1,1))*self.G(X)).cpu().numpy().reshape(shape)
            else:
                G = self.G(X).cpu().numpy().reshape(shape)
        return G 

    # Not implemented for Green's function of dimension > 1
    def evaluateN(self, x):
        """
        Method to evaluate the neural network at a set of evaluation points.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            x: A numpy array or torch tensor which specifies the evaluation points.
        ----------------------------------------------------------------------------------------------------------------
        Returns:
            A numpy array which specifies the value of the neural network at the evaluation points.
        """

        if isinstance(x,config(np)):
            x = np.array([x])
        if isinstance(x.dtype, np.float64):
            x = x.astype(config(np))
        if isinstance(x,np.ndarray):
            x = x.astype(config(np))
        if x.dtype == config(np):
            X = torch.tensor(x.reshape(-1,1), dtype = config(torch)).to(self.device)
        with torch.no_grad():
            N = self.N(X).cpu().numpy()
        return N

    def init_loss(self, xF, xU):
        self.lossfn = LossGreensFunction(self.G, self.N, xF, xU, self.domain, self.addADF ,self.device)
    
    def saveModels(self, path = "temp"):
        """
        Method to save the neural network parameters.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            path: A string which specifies the path to save the model.
        ----------------------------------------------------------------------------------------------------------------
        """

        savedict = {'dimension': self.dimension,
            'layerConfig': self.layerConfig,
            'activation': self.activation,
            'G_state_dict': self.G.state_dict(),
            'N_state_dict': self.N.state_dict(),
           }
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(savedict, path + "/model.pth")

    def checkSavedModels(self, loadPath):
        """
        Method to check if the saved models exist.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            loadPath: A string which specifies the path to the saved model.
        ----------------------------------------------------------------------------------------------------------------
        Returns:
            A boolean which specifies whether the saved models exist.
        """
        return Path(loadPath+"/model.pth").is_file()
        
    def loadModels(self, loadPath, device = device):
        """
        Method to load the neural network parameters.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            loadPath: A string which specifies the path to the saved model.

            device: A torch.device which specifies the device for the neural network.
        ----------------------------------------------------------------------------------------------------------------
        """
        model = torch.load(loadPath+'/model.pth', weights_only=False)
        self.G = NN(numInputs = model['dimension']*2, numOutputs = model['dimension'], layerConfig = model['layerConfig'], activation = model['activation']).to(device)
        self.N = NN(numInputs = model['dimension'], numOutputs = model['dimension'], layerConfig = model['layerConfig'], activation = model['activation']).to(device)
        self.G.load_state_dict(model['G_state_dict'])
        self.N.load_state_dict(model['N_state_dict'])
        
         
        


