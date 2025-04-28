import numpy as np
######### tfp
import tensorflow as tf
# do not use gpu, because it is even slower
tf.config.set_visible_devices([], 'GPU')
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from scipy.special import softmax

class Glass:
    """
    The Glass class implements a multinomial logistic regression model combined with variational inference 
    for efficient posterior estimation. 
    """
    def __init__(self, shrinkage_factor=0, dtype = tf.float32) -> None:
        """
        Initializes the GLASS model.

        Args:
            shrinkage_factor (float, optional): The soft-thresholding factor applied to impose sparsity in the 
                time-varying effects. Defaults to 0.
            dtype (tf.DType, optional): Data type for TensorFlow operations. Higher precision can be achieved with 
                tf.float64. Defaults to tf.float32.
        """
        self.shrinkage_factor, self.dtype = shrinkage_factor, dtype
    
    def process_data(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Processes and prepares the training data for the GLASS model.

        Args:
            X_train (np.ndarray): EEG response data, shaped as 
                [# of half sequences, # of stimuli per half-sequence, # of channels, # of time points].
            y_train (np.ndarray): Binary labels for stimulus types, with only one '1' per row indicating the 
                target stimulus. Shape: [# of half sequences, # of stimuli per half-sequence].
        """
        self.nchannel, self.nT = X_train.shape[2:]
        self.X_train = tf.constant(X_train, dtype = self.dtype)
        self.y_train = tf.constant(y_train, dtype = self.dtype)
        def jointmodel():
            sigma = yield tfd.Sample(tfd.HalfCauchy(0.0, tf.cast(1, self.dtype)), 2)
            # weights
            beta = yield tfb.Cumsum()(tfd.Sample(tfd.Normal(0.0, 1.0), (2, self.nT)))
            beta = sigma[:, None] * beta
            beta = tfp.math.soft_threshold(beta, self.shrinkage_factor)
            probs = yield tfd.RelaxedOneHotCategorical(1, logits=tf.zeros([self.nchannel, 3]))
            weights = yield tfd.Sample(tfd.Normal(0.0, 1.0), [self.nchannel, 2])
            weights = tf.linalg.normalize(weights, axis=0)[0]
            beta = tf.linalg.matmul(probs[:,1:]*weights, beta)
            logits = tf.linalg.tensordot(self.X_train, beta, axes = [[2, 3], [0, 1]])
            y = yield tfd.Multinomial(logits = logits, total_count = 1)
        self.joint = tfd.JointDistributionCoroutineAutoBatched(jointmodel)

    def mfvb(self, num_steps=2000, sample_size=10, importance_sample_size=10, learning_rate=0.05, seed=1):
        """
        Runs the mean-field variational Bayes (MFVB) algorithm to approximate the posterior distribution.

        Args:
            num_steps (int, optional): Number of optimization steps for fitting the surrogate posterior. 
                Defaults to 2000.
            sample_size (int, optional): Number of Monte Carlo samples to use in estimating the variational divergence. Defaults to 10.
            importance_sample_size (int, optional): Number of terms used to define an importance-weighted divergence. Defaults to 10.
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 0.05.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
        """
        self.posterior = tfd.JointDistributionSequentialAutoBatched([
            tfd.LogNormal(
                tf.Variable(tf.zeros(2, dtype = self.dtype) - 3),
                tfp.util.TransformedVariable(0.1 * tf.ones(2, dtype = self.dtype), bijector = tfb.Softplus())),
            tfd.Normal(
                tf.Variable(tf.random.normal((2, self.nT), stddev=3*self.shrinkage_factor, dtype = self.dtype), 
                            dtype = self.dtype),
                tfp.util.TransformedVariable(0.01 * tf.ones((2, self.nT), dtype = self.dtype), 
                                             bijector = tfb.Softplus())),
            tfd.RelaxedOneHotCategorical(0.5, logits=tf.Variable(tf.zeros((self.nchannel, 3)))),
            tfd.Normal(tf.Variable(tf.zeros([self.nchannel, 2], dtype = self.dtype), dtype = self.dtype),
                       tfp.util.TransformedVariable(tf.ones([self.nchannel, 2], dtype = self.dtype), 
                                                    dtype = self.dtype,
                                                    bijector = tfb.Softplus()))
        ])

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        tf.random.set_seed(seed)
        self.losses = list(tfp.vi.fit_surrogate_posterior(
            self.loglik, 
            self.posterior,
            optimizer = optimizer,
            num_steps = num_steps, 
            sample_size = sample_size,
            importance_sample_size=importance_sample_size))
        self.losses = [float(x) for x in self.losses]
        
        self.samples = self.posterior.sample(5000)
        self.process_samples()

    @property
    def median_sample(self):
        """
        Returns the median of sampled posterior values for each parameter.
        """
        return [np.median(x, axis = 0) for x in self.samples]
    
    def loglik(self, *args):
        """
        Returns the log-likelihood of a given parameter.
        """
        return self.joint.log_prob(*args, self.y_train)
        
    def process_samples(self):
        """
        Process the samples to construct some derived parameters. For internal use. 
        """
        sigmas, self.betagMats, self.probs, self.weights = [np.array(x) for x in self.samples]
        l2 = np.sqrt(np.sum(np.square(self.weights), 1))
        self.weights = self.weights/l2[:, None, :]
        self.effective_weights = self.probs[:,:,1:]*self.weights
        self.weight = self.weights.mean(axis = 0)
        self.effective_weight = self.effective_weights.mean(axis = 0)
        self.betagMats = sigmas[:, :, None] * self.betagMats
        self.betagMats = np.array(tfp.math.soft_threshold(self.betagMats, self.shrinkage_factor))
        self.betagMat = np.median(self.betagMats, axis=0)
        self.betaMats = np.matmul(self.effective_weights, self.betagMats)
        self.betaMat = np.median(self.betaMats, axis=0)
    
    def predict_prob(self, newX: np.ndarray, method = 'median'):
        """
        Predicts class probabilities for new EEG data based on the learned model.

        Args:
            newX : np.ndarray
                New EEG responses for prediction, structured similarly to the training data `X_train`, 
                with shape [# of sequences, # of stimuli, # of channels, # of time points].
            method : str, optional
                Method for prediction. Options are:
                    - 'median': Uses the median of posterior samples to predict.
                    - 'vote': Uses each posterior sample to make a prediction and averages the results.
                Defaults to 'median'.

        Returns:
            np.ndarray Predicted probabilities of being the target, with the shape [# of sequences, # of stimuli].
        """
        if method == 'median':
            logodds = np.tensordot(newX, self.betaMat, axes=[[2, 3], [0, 1]])
            probs = softmax(logodds, axis=1)
        elif method == 'vote':
            logodds = np.tensordot(newX, self.betaMats, axes=[[2, 3], [1, 2]])
            probs = softmax(logodds, axis=1)
            probs = probs.mean(axis=2)
        return probs
