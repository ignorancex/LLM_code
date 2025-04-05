###############################################################################
"""
This module implements the quantum noise detector. It has these classes:
    QuantumController      : This is an internal class for generating control pulses
    Distance               : This is an internal class for calculating average distance between a set of matrices
    detector               : This is the main class that defines the noise detector 
"""
################################################################################
# Preamble
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers,Model
import zipfile    
import os
import pickle
###############################################################################    
class QuantumController(layers.Layer):
    """
    This class defines a custom tensorflow layer that implemements a trainable pulse generator
    """
    
    def __init__(self, T, M, max_amp, n_max, **kwargs):
        """
        Class constructor.
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        max_amp       : Maximum amplitude of the control pulses
        n_max         : Maximum number of control pulses in the sequence
        """  
        # we must call thus function for any tensorflow custom layer
        super(QuantumController, self).__init__(**kwargs)
        
        # define and store time range
        self.T          = T
        self.M          = M
        self.max_amp    = max_amp
        self.n_max      = n_max
        self.time_range = tf.constant( np.reshape( [(0.5*T/M) + (j*T/M) for j in range(M)], (1,M,1,1) ) , dtype=tf.float32)
        
        # define the constant parmaters to shift the pulses correctly
        self.pulse_width = (0.5*self.T/self.n_max)
        
        self.a_matrix    = np.ones((self.n_max, self.n_max))
        self.a_matrix[np.triu_indices(self.n_max,1)] = 0
        self.a_matrix    = tf.constant(np.reshape(self.a_matrix,(1,self.n_max,self.n_max)), dtype=tf.float32)
        
        self.b_matrix    = np.reshape([idx + 0.5 for idx in range(self.n_max)], (1,self.n_max,1) ) * self.pulse_width
        self.b_matrix    = tf.constant(self.b_matrix, dtype=tf.float32)
        
        # define custom traninable weights
        self.mu    = self.add_weight(name = "mu",   shape=(1, n_max, 1), dtype=tf.float32, trainable=True)    
        self.A     = self.add_weight(name = "A",    shape=(1, n_max, 1), dtype=tf.float32, trainable=True)

    def call(self, inputs):
        """
        Tensorflow method where all the calculations are done
        
        """
        
        # construct the signal parameters in such a way to respect the amplutide and position constraints
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        a_matrix    = tf.tile(self.a_matrix, temp_shape)
        b_matrix    = tf.tile(self.b_matrix, temp_shape)
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([self.n_max,1],dtype=np.int32))],0 )     
        amplitude   = self.max_amp*tf.tanh(self.A)
        position    = 0.5*self.pulse_width + tf.sigmoid(self.mu)*( ( (self.T - self.n_max*self.pulse_width)/(self.n_max+1) ) - 0.5*self.pulse_width)
        position    = tf.matmul(a_matrix, position) + b_matrix
        std         = self.pulse_width * tf.ones(temp_shape, dtype=tf.float32)/6

        # construct the signal
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1,1],dtype=np.int32))],0 )     
        time_range = tf.tile(self.time_range, temp_shape)
        tau   = [tf.reshape( tf.matmul(position[:,idx,:],  tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]
        A     = [tf.reshape( tf.matmul(amplitude[:,idx,:], tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]
        sigma = [tf.reshape( tf.matmul(std[:,idx,:]      , tf.ones([1,self.M]) ), (tf.shape(time_range)) ) for idx in range(self.n_max)]
        signal = [tf.multiply(A[idx], tf.exp( -0.5*tf.square(tf.divide(time_range - tau[idx], sigma[idx])) ) ) for idx in range(self.n_max)] 
        signal = tf.add_n(signal)
        
        return signal
###############################################################################
class Distance(layers.Layer):
    """
    A custom tensorflow class to calculate the average Frobenius distance between a set of matrices
    
    """
    
    def __init__(self, **kwargs):
        """
        Class constructor.
        
        """
        super(Distance, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        Tensorflow method where all the calculations are done
        
        """
        # find the set of all unique pairs of matrices from the list of inputs
        N       = len(inputs)
        indices = sum([ [(i,j) for j in range(i+1, N)] for i in range(N)], [])
        
        # calculate the average distance [nomalized to 1]
        distance = tf.add_n( [tf.norm(inputs[i]-inputs[j], axis=[1,2]) for (i,j) in indices] )
        return tf.math.real(distance)/(N*np.sqrt(2))
###############################################################################
class detector():
    """
    A class that implements an ML-based detector given a set of quantum measurements
    
    """
    def __init__(self, max_amp, n_max, mlmodels):
        """
        Class constructor.
        
        mlmodels: A list of pretrained "qubitmlmodel" classes corresponding to each noise profile
        max_amp : The maximum allowed amplitude for the control pulses
        n_max   : The number of pulses per control sequence
        """
        # store parameters
        self.N = len(mlmodels)
        self.T = mlmodels[0].T
        self.M = mlmodels[0].M
        #######################################################################
        # define the controller TF model
        
        # define a dummy input for the quantum controller
        dummy_input = layers.Input(batch_shape=(1,1))    
        
        # add custom TF layers to generate the optimal detecting pulse
        optimal_pulse   = QuantumController(self.T, self.M, max_amp, n_max, trainable=True, name="optimal_pulse")(dummy_input)
        
        # define non-trainable copies of the pre-trained qubit models      
        qubit_models     = []
        for profile in range(self.N):
            # define a tensorflow model that captures the trainined part of the qubit model without the measurement layers
            qubit_models.append( Model(inputs=mlmodels[profile].model.input, outputs=mlmodels[profile].model.get_layer("V0").output) )
         
            # prevent training any layer of the already-trained model
            for layer in qubit_models[profile].layers:
                layer.trainable  = False
        
        self.qubit_models = mlmodels
        
        # connect a copy of the detector
        optimal_measurements = layers.Reshape((1,))( Distance()( [qubit_models[profile](optimal_pulse) for profile in range(self.N) ] ) )
        
        # define the training model of the detector    
        self.pulse_training_model = Model(dummy_input , optimal_measurements)
        self.pulse_training_model.compile(optimizer=optimizers.Adam(lr=0.5), loss='mse')
        self.pulse_training_model.summary()
        #######################################################################
        # define the detector TF model
        features              = layers.Input((3,)) 
        classification        = layers.Dense(self.N*1, activation = "tanh")(features)
        classification        = layers.Dense(self.N*3, activation = "tanh")(classification)
        classification        = layers.Dense(self.N   , activation = "softmax", name="classification_out")(classification)
        
        self.detector         = Model(features, classification)
        self.detector.compile(optimizer=optimizers.Adam(lr=0.01), loss="mse", metrics=["categorical_accuracy"])    
        self.detector.summary()
        
    def train_optimal_pulse(self, epochs):
        """
        A method to train the quantum controller to find the optimal disciminating pulse sequence
        
        epochs: Number of training iterations
        """
        self.controller_training_hostory = self.pulse_training_model.fit(np.ones((1,)), np.ones((1)), epochs=epochs, verbose=2)


    def train_detector(self, epochs, K=10000):
        """
        A method to train the detector
        
        epochs: number of training iterations
        K     : number of noisy repeations that are used to construct the training set
        """
     
        # 1) construct the optimal pulse waveform
        optimal_pulse = self.predict_optimal_pulse()
        
        # 2) predict the corresponding quantum measurements using the pre-trained model
        optimal_measurements = [self.qubit_models[profile].predict_measurements(optimal_pulse) for profile in range(self.N)]
        
        # 3) construct K copies of the measurements with added noise to simulate variability between trained qubit models and actual experiment
        dithered_measurements = np.concatenate( [np.tile(m, (K,1)) for m in optimal_measurements], axis=0 ) + np.random.randn(K*self.N,3)*0.01
        
        # 4) prepare the ideal classification outputs
        training_y = []
        for profile in range(self.N):
            y = np.zeros((K,self.N))
            y[0:K,profile] = 1.
            training_y.append(y)
        training_y = np.concatenate(training_y, axis=0)
        
        # 5) perform the training and strore the training history
        self.detector_training_history = self.detector.fit(dithered_measurements, training_y, epochs=epochs, verbose=2, validation_split=0.1)
        
    def predict_optimal_pulse(self):
        """
        A method to predict the optimal pulse for detection, usually called after training the mode;
        """
        
        pulse_model = Model(self.pulse_training_model.input, self.pulse_training_model.get_layer("optimal_pulse").output)
        return pulse_model.predict(np.ones((1,)))
    
    def detect(self, measurements):
        """
        A method to classify the noise based on the optimal pulse measurements
        
        measurements: an np (3,1) array storing the optimal pulse measurements
        
        returns the probability of each class, and the index of the class with maximum probability 
        """
        
        classification = self.detector.predict(measurements)
        return classification, np.argmax(classification, axis=1)
    
    def save_model(self, fname):
        """
        A method to save the trained model
        """
        
        self.detector.save_weights(fname+"_detector.mlmodel")
        self.pulse_training_model.save_weights(fname+"_pulses_mdl.mlmodel")
        np.save(fname+"_pulse", self.predict_optimal_pulse())
        
    def load_model(self, fname):
        """
        A method to load a trained model
        """
        
        self.detector.load_weights(fname+"_detector_mdl.mlmodel")
        self.pulse_training_model.load_weights(fname+"_pulses_mdl.mlmodel")
