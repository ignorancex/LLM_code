from .backend import tf, np, ABC, config, parser, ast
from .utils import generateEvaluationGrid, approximateDistanceFunction
from .activations import get_activation
from .loss import LossGreensFunction

def NN(numInputs = 2,
       numOutputs = 1,
       layerConfig = ast.literal_eval(parser['GREENLEARNING']['layerConfig']),
       activation = parser['GREENLEARNING']['activation'],
       dtype = config(tf)):
    layers = []
    layers.append(tf.keras.layers.InputLayer(input_shape = (numInputs,), dtype = config(tf)))
        
    for units in layerConfig:
        layers.append(tf.keras.layers.Dense(units = units, dtype = config(tf)))
        layers.append(get_activation(activation))
    
    layers.append(tf.keras.layers.Dense(units = numOutputs, dtype = config(tf), activation = None))
    return tf.keras.Sequential(layers)

class GreenNN(ABC):
    def __init__(self) -> None:
        super().__init__()
        print(f"Using tensorflow {tf.__version__}")

    def build(self,
              dimension = 1,
              domain = np.array([0,1,0,1]),
              layerConfig = ast.literal_eval(parser['GREENLEARNING']['layerConfig']),
              activation = parser['GREENLEARNING']['activation'],
              homogenousBC = True,
              loadPath = None):    
        """
        greenlearning models benefit from using a few steps of L-BFGS optimizer during the training
        but between tensorflow probability not being available for M1 Macbooks (which I am
        protyping this code on) and tf2 dropping the contrib module which had an implementation for 
        supporting external optimizers like ScipyOptimizerInterface, this code just uses an Adam
        optimizer to train the G and N networks. But in case there is some developement on the
        issue here, https://github.com/tensorflow/tensorflow/issues/48167, do add a few epochs of
        L-BFGS optimizer to have the model converge nicely :)
        """
        self.dimension = dimension
        if type(domain) is not np.ndarray and type(domain) is list:
            self.domain = np.array(domain)
        else:
            self.domain = domain
        self.addADF = homogenousBC

        if loadPath == None:
            self.G = NN(numInputs = dimension*2, numOutputs = dimension, layerConfig = np.array(layerConfig), activation = activation)
            self.N = NN(numInputs = dimension, numOutputs = dimension, layerConfig = np.array(layerConfig), activation = activation)
        else:
            assert self.checkSavedModels(loadPath), "Saved models not found" 
            self.loadModels(loadPath)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate = parser['GREENLEARNING'].getfloat('initLearningRate'),
                        decay_steps = parser['GREENLEARNING'].getint('stepSize'),
                        decay_rate = parser['GREENLEARNING'].getfloat('decayRate'))
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule,
                                                  beta_1 = 0.9,
                                                  beta_2 = 0.999,
                                                  epsilon = 1e-8)
    
    def train(self,
              data,
              epochs = {'adam':parser['GREENLEARNING'].getint('epochs_adam') ,
                        'lbfgs':parser['GREENLEARNING'].getint('epochs_lbfgs')},
              trainHomogeneous = True):
        
        assert data.xF.shape[1] == self.dimension and data.xU.shape[1] == self.dimension,\
                f"Dimension of evaluation points for forcing, {data.xF.shape[1]}, and response, \
                {data.xU.shape[1]}, should match with the model dimension, {self.dimension}"
        
        self.init_loss(data.xF, data.xU)

        lossHistory = {'training': [], 'validation':[]}
        for epoch in range(int(epochs['adam'])):
            for fTrain, uTrain in data.trainDataset:
                with tf.GradientTape() as tape:
                    lossValue = self.lossfn(fTrain,uTrain)
                
                if trainHomogeneous:
                    gradG, gradN = tape.gradient(lossValue, [self.G.trainable_weights, self.N.trainable_weights])
                    self.optimizer.apply_gradients(zip(gradG, self.G.trainable_weights))
                    self.optimizer.apply_gradients(zip(gradN, self.N.trainable_weights))
                else:
                    gradG = tape.gradient(lossValue, self.G.trainable_weights)
                    self.optimizer.apply_gradients(zip(gradG, self.G.trainable_weights))

            # Change this to be an average over batches. Currently assuming a single batch
            lossHistory['training'].append(lossValue.numpy())
            for fVal, uVal in data.valDataset:
                lossValue = self.lossfn(fVal, uVal)
            lossHistory['validation'].append(lossValue.numpy())
            if (epoch+1) % 100 == 0:
                print(f"Loss at epoch {epoch+1}: Training = {lossHistory['training'][-1]:.3E}, Validation = {lossHistory['validation'][-1]:.3E}")
        return lossHistory
    
    # Not implemented for Green's function of dimension > 1
    def evaluateG(self, x, s):
        if isinstance(x,config(np)):
            x = np.array([x])
        if isinstance(s,config(np)):
            s = np.array([s])
        if isinstance(x.dtype, np.float64) or (s.dtype, np.float64):
            x, s = x.astype(config(np)), s.astype(config(np))
        assert(x.shape == s.shape), "Both input need to have the same shape"
        shape = x.shape
        if x.dtype == config(np):
            X = tf.constant(np.vstack([x.ravel(), s.ravel()]).T, dtype = config(tf))
        
        if self.addADF:
            G = (tf.reshape(approximateDistanceFunction(X[:,0], X[:,1],
                                                self.domain.astype(config(np))),(-1,1))*self.G(X)).numpy().reshape(shape)
        else:
            G = self.G(X).numpy().reshape(shape)
        return G

    def evaluateN(self, x):
        if isinstance(x,config(np)):
            x = np.array([x])
        if isinstance(x.dtype, np.float64):
            x = x.astype(config(np))
        if isinstance(x,np.ndarray):
            x = x.astype(config(np))
        if x.dtype == config(np):
            X = tf.constant(x.reshape(-1,1), dtype = config(tf))
        
        N = self.N(X).numpy()
        return N       

    def init_loss(self, xF, xU):
        self.lossfn = LossGreensFunction(self.G, self.N, xF, xU, self.domain, self.addADF)
    
    def saveModels(self, path = "temp"):
        self.G.save(path + "/G")
        self.N.save(path + "/N")

    def checkSavedModels(self, loadPath):
        return tf.saved_model.contains_saved_model(loadPath+"/G") and tf.saved_model.contains_saved_model(loadPath+"/N")
        
    def loadModels(self, loadPath):
        self.G = tf.keras.models.load_model(loadPath+"/G", compile = False)
        self.N = tf.keras.models.load_model(loadPath+"/N", compile = False)
        assert (self.G.input.shape[1] == self.dimension*2 and \
                self.N.input.shape[1] == self.dimension), "Dimension mismatch for the loaded model"
         
        


