from .backend import tf, np, ABC, config
from .utils import generateEvaluationGrid, approximateDistanceFunction
from .quadrature_weights import get_weights
from .utils import approximateDistanceFunction

class LossGreensFunction(ABC):
    def __init__(self, G, N, xF, xU, domain, addADF) -> None:
        super().__init__()
        self.G = G
        self.N = N
        self.xF, self.xU = xF, xU
        self.wF = tf.constant(get_weights('uniform', xF).T, dtype = config(tf))
        self.wU = tf.constant(get_weights('uniform', xU).T, dtype = config(tf))
        self.X = generateEvaluationGrid(self.xU, self.xF)
        self.addADF = addADF
        if self.addADF:
            self.ADF = tf.reshape(approximateDistanceFunction(self.X[:,0], self.X[:,1],
                                                domain.astype(config(np))),(-1,1))

    @tf.function
    def __call__(self, fTrain, uTrain, trainHomogeneous = True):
        nF, nU = self.xF.shape[0], self.xU.shape[0]
        if self.addADF:
            G = tf.transpose(tf.reshape(tf.multiply(self.ADF,self.G(self.X)),(nF, nU)))
        else:
            G = tf.transpose(tf.reshape(self.G(self.X),(nF, nU)))
            
        uHomogeneous = self.N(self.xU)
        uPred = tf.transpose(tf.matmul(G, tf.multiply(fTrain, self.wF), transpose_b = True) + uHomogeneous)
        if trainHomogeneous:
            loss = tf.divide(tf.math.reduce_sum(tf.multiply(tf.square(uTrain - uPred),self.wU), axis = 1), \
                         tf.math.reduce_sum(tf.multiply(tf.square(uTrain),self.wU), axis = 1))
        else:
            loss = tf.divide(tf.math.reduce_sum(tf.multiply(tf.square(uTrain - uPred),self.wU), axis = 1), \
                            tf.math.reduce_sum(tf.multiply(tf.square(uTrain - uHomogeneous),self.wU), axis = 1))
        return tf.math.reduce_mean(loss)