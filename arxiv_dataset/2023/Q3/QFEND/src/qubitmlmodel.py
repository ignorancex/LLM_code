"""
This module implements the machine learning-based model for the qubit. It has three classes:
    HamiltonianConstruction: This is an internal class for constructing Hamiltonians
    QuantumCell            : This is an internal class required for implementing time-ordered evolution
    QuantumEvolution       : This is an internal class to implement time-ordered quantum evolution
    VoLayer                : This is an internal class to calculate Vo given the interaction unitary and the observable
    QuantumMeasurement     : This is an internal class to model coupling losses at the output.
    qubitMLmodel           : This is the main class that defines machine learning model for the qubit.  
"""
###############################################################################
# Preamble
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers,Model,initializers,callbacks
import zipfile    
import os
import pickle 
##############################################################################
class HamiltonianConstruction(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the Hamiltonian parameters as input, and generates the
    Hamiltonain matrix as an output at each time step for each example in the batch
    """
    
    def __init__(self, dynamic_operators, static_operators, **kwargs):
        """
        Class constructor 
        
        dynamic_operators: a list of all operators that have time-varying coefficients
        static_operators : a list of all operators that have constant coefficients
        """
        
        self.dynamic_operators = [tf.constant(op, dtype=tf.complex64) for op in dynamic_operators]
        self.static_operators  = [tf.constant(op, dtype=tf.complex64) for op in static_operators]
           
        # this has to be called for any tensorflow custom layer
        super(HamiltonianConstruction, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 

        H = []
        # loop over the strengths of all dynamic operators
        
        for idx_op, op in enumerate(self.dynamic_operators):

            # select the particular strength of the operator
            h = tf.cast(inputs[:,:,:,idx_op:idx_op+1] ,dtype=tf.complex64)

            # construct a tensor in the form of a row vector whose elements are [d1,d2,d3, 1,1], where d1, d2, and d3 correspond to the
            # number of examples, number of time steps of the input, and number of realizations
            temp_shape = tf.concat( [tf.shape(inputs)[0:3],tf.constant(np.array([1,1],dtype=np.int32))],0 )

            # add two extra dimensions for batch, time, and realization
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            operator = tf.expand_dims(operator,0)
            
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            
            # repeat the pulse waveform to as 2x2 matrix
            temp_shape = tf.constant(np.array([1,1,1,2,2],dtype=np.int32))
            h = tf.expand_dims(h,-1)
            h = tf.tile(h, temp_shape)
            
            # Now multiply each operator with its corresponding strength element-wise and add to the list of Hamiltonians
            H.append( tf.multiply(operator, h) )
       
        # loop over the strengths of all static operators
        for op in self.static_operators:          
            # construct a tensor in the form of a row vector whose elements are [d1,d2,d3,1,1], where d1, d2, and d2 correspond to the
            # number of examples, number of time steps of the input, and number of realizations
            temp_shape = tf.concat( [tf.shape(inputs)[0:3],tf.constant(np.array([1,1],dtype=np.int32))],0 )

            # add two extra dimensions for batch and time
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            operator = tf.expand_dims(operator,0)
            
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            
            # Now add to the list of Hamiltonians
            H.append( operator )
        
        # now add all componenents together
        H =  tf.add_n(H)
                            
        return H    
###############################################################################
class QuantumCell(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces one step forward propagator
    """
    
    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
        delta_T: time step for each propagator
        """  
        
        # here we define the time-step including the imaginary unit, so we can later use it directly with the expm function
        self.delta_T= tf.constant(delta_T*-1j, dtype=tf.complex64)

        # we must define this parameter for RNN cells
        self.state_size = [1]
        
        # we must call thus function for any tensorflow custom layer
        super(QuantumCell, self).__init__(**kwargs)

    def call(self, inputs, states):        
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        states: The tensor representing the state of the cell. This is passed automatically by tensorflow.
        """         
        
        previous_output = states[0] 
        
        # evaluate -i*H*delta_T
        Hamiltonian = inputs * self.delta_T
        
        #evaluate U = expm(-i*H*delta_T)
        U = tf.linalg.expm( Hamiltonian )
        
        # accuamalte U to to the rest of the propagators
        new_output  = tf.matmul(U, previous_output)    
        
        return new_output, [new_output]
###############################################################################
class QuantumEvolution(layers.RNN):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces the time-ordered evolution unitary as output
    """
    
    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
              
        delta_T: time step for each propagator
        """  
        
        # use the custom-defined QuantumCell as base class for the nodes
        cell = QuantumCell(delta_T)

        # we must call thus function for any tensorflow custom layer
        super(QuantumEvolution, self).__init__(cell,  **kwargs)
      
    def call(self, inputs):          
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        """
        
        # define identity matrix with correct dimensions to be used as initial propagtor 
        dimensions = tf.shape(inputs)
        I          = tf.eye( dimensions[-1], batch_shape=[dimensions[0], dimensions[2]], dtype=tf.complex64 )
        
        return super(QuantumEvolution, self).call(inputs, initial_state=[I])         
###############################################################################    
class QuantumMeasurement(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the unitary as input, 
    and generates the measurement outcome probability as output
    """
    
    def __init__(self, initial_state, measurement_operator, **kwargs):
        """
        Class constructor
        
        initial_state       : The inital density matrix of the state before evolution.
        Measurement_operator: The measurement operator
        """          
        self.initial_state        = tf.constant(initial_state, dtype=tf.complex64)
        self.measurement_operator = tf.constant(measurement_operator, dtype=tf.complex64)
    
        # we must call thus function for any tensorflow custom layer
        super(QuantumMeasurement, self).__init__(**kwargs)
            
    def call(self, x): 
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
    
        # extract the different inputs of this layer which are the Vo and Uc
        Vo, Uc = x
        
        Uc = Uc[:,0,:]
        # construct a tensor in the form of a row vector whose elements are [d1,1,1], where d1 correspond to the
        # number of examples of the input
        temp_shape = tf.concat( [tf.shape(Uc)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )

        # add an extra dimension for the initial state and measurement tensors to represent batch
        initial_state        = tf.expand_dims(self.initial_state,0)
        measurement_operator = tf.expand_dims(self.measurement_operator,0)   
        
        # repeat the initial state and measurment tensors along the batch dimensions
        initial_state        = tf.tile(initial_state, temp_shape )
        measurement_operator = tf.tile(measurement_operator, temp_shape)   
        
        # evolve the initial state using the propagator provided as input
        final_state = tf.matmul(tf.matmul(Uc, initial_state), Uc, adjoint_b=True )
        
        # calculate the probability of the outcome
        expectation = tf.linalg.trace( tf.matmul( tf.matmul( Vo, final_state), measurement_operator) ) 
        
        return tf.squeeze( tf.reshape( tf.math.real(expectation), temp_shape), axis=-1 )
###############################################################################    
class VoLayer(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes a vector of parameters represneting eigendecompostion and reconstructs a 2x2 Hermitian traceless matrix. 
    """
    
    def __init__(self, O, **kwargs):
        """
        Class constructor
        
        O: The observable to be measaured
        """
        # this has to be called for any tensorflow custom layer
        super(VoLayer, self).__init__(**kwargs)
    
        self.O = tf.constant(O, dtype=tf.complex64)         
        
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        
        # retrieve the two types of parameters from the input: 3 eigenvector parameters and 1 eigenvalue parameter
        if len(x)==2:
            UI,Uc = x
        else:
            UI,Uc,w = x
            temp_shape = tf.constant(np.array([1,1,2,2],dtype=np.int32))
            w  = tf.cast( tf.tile(w, temp_shape), dtype=tf.complex64)
        
        UI_tilde = tf.matmul(Uc, tf.matmul(UI,Uc, adjoint_b=True) )

        # expand the observable operator along batch and realizations axis
        O = tf.expand_dims(self.O, 0)
        O = tf.expand_dims(O, 0)
         
        temp_shape = tf.concat( [tf.shape(Uc)[0:2], tf.constant(np.array([1,1],dtype=np.int32))], 0 )
        O = tf.tile(O, temp_shape)

        # Construct Vo operator         
        VO = tf.matmul(O, tf.matmul( tf.matmul(UI_tilde,O, adjoint_a=True), UI_tilde) )
        
        if len(x)==2:
            VO = tf.reduce_mean(VO, axis=1, keepdims=False)
        else:
            VO = tf.reduce_sum( tf.multiply(VO, w), axis=1, keepdims=False) 
        
        return VO 
###############################################################################
class Noise(layers.Layer):
    """
    This class defines a custom tensorflow layer that generates realizations of a random process simulating noise.
    
    """
    def __init__(self, dim, K=1, **kwargs):
        """
        class constructor
        
        dim : how many axes to generate the noise 
        K   : number of realizations
        """
        # we must call thus function for any tensorflow custom layer
        super(Noise, self).__init__(**kwargs)
        
        # store the constructor parameters
        self.K = K
        self.dim = dim
        
    def build(self, input_shape):
        """
        This method must be defined for any custom layer, here you define the training parameters.
        
        input_shape: a tensor that automatically captures the dimensions of the input by tensorflow. 
        """ 
        
        M = input_shape.as_list()[1] # retreive the number of samples from the input
        
        # define custom weights to be generate the realizations
        
        # 1) time-domain samples
        self.beta = self.add_weight(name = "beta", shape=(1, M, self.K, self.dim), initializer=initializers.glorot_uniform(), dtype=tf.float32, trainable=True)        
        
        # 2) weight of each realizations to be used when calculating averages
        self.w = self.add_weight(name="w", shape=(1, self.K, 1, 1), dtype=tf.float32, initializer=initializers.glorot_uniform())   
        
         # we must call thus function for any tensorflow custom layer
        super(Noise, self).build(input_shape)
        
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done. 
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow.
        """
        # make sure the weights add up to 1
        w    =  tf.nn.softmax(self.w, axis=1)
        
        # repeat the weights and time-realization for each example in the batch [for fixing training issues]         
        temp_shape = tf.concat( [tf.shape(x)[0:1], tf.constant(np.array([1,1,1],dtype=np.int32))], 0 )
        beta       = tf.tile(self.beta, temp_shape)
        w          = tf.tile(w, temp_shape)
        
        # return the weights
        return [beta,w] 
###############################################################################
class qubitMLmodel():
    """
    This is the main class that defines machine learning model of the qubit.
    """    
      
    def __init__(self, T, M, dynamic_operators, static_operators, noise_operators, measurement_operators, initial_states, K=1):
        """
        Class constructor.

        T                : Evolution time
        M                : Number of time steps
        dynamic_operators: A list of arrays that represent the terms of the control Hamiltonian (that depend on pulses)
        static_operators : A list of arrays that represent the terms of the drifting Hamiltonian (that are constant)
        noise_operators  : A list of arrays that represent the terms of the classical noise Hamiltonians
        measurement_operators: A list of arrays representing the measuremetion operators
        initial_states   : A list of arrays representing the initial states of the quantum system
        K                : Number of noise realizations
        """
        
        # store the constructor arguments
        self.K       = K
        self.T       = T
        self.M       = M
        self.delta_T = T/M
 
       # define lists for stroring the training history
        self.training_history      = []
        self.val_history           = []
        
        # define tensorflow input layers for the pulse sequence and noise realization in time-domain
        pulse_time_domain = layers.Input(shape=(M,1,len(dynamic_operators)) , name="control")
        noise_time_domain,w = Noise(len(noise_operators), K=K, name="noise")(pulse_time_domain)
  
        # define the custom defined tensorflow layer that constructs the H0 part of the Hamiltonian from parameters at each time step
        H0 = HamiltonianConstruction(dynamic_operators=dynamic_operators, static_operators=static_operators, name="H0")(pulse_time_domain)  

        # define the custom defined tensorflow layer that constructs the H1 part of the Hamiltonian from parameters at each time step
        H1 = HamiltonianConstruction(dynamic_operators=noise_operators, static_operators=[], name="H1")(noise_time_domain)
    
        # define the custom defined tensorflow layer that constructs the time-ordered evolution of H0 
        U0 = QuantumEvolution(self.delta_T, return_sequences=True, name="U0")(H0)
    
        # define Uc which is U0(T)
        Uc = layers.Lambda(lambda u0: u0[:,-1,:,:,:], name="Uc")(U0)
        
        # define custom tensorflow layer to calculate HI
        U0_ext = layers.Lambda(lambda x: tf.tile(x, tf.constant([1,1,K,1,1], dtype=tf.int32) ) )(U0)
        HI = layers.Lambda(lambda x: tf.matmul( tf.matmul(x[0],x[1], adjoint_a=True), x[0] ), name="HI" )([U0_ext, H1])
    
        # define the custom defined tensorflow layer that constructs the time-ordered evolution of HI
        UI = QuantumEvolution(self.delta_T, return_sequences=False, name="UI")(HI)
        
        # construct the Vo operators
        Uc_ext = layers.Lambda(lambda x: tf.tile(x, tf.constant([1,K,1,1], dtype=tf.int32) ) )(Uc)
        
        Vo = [VoLayer(O, name="V%d"%idx_O)([UI,Uc_ext,w]) for idx_O, O in enumerate(measurement_operators)]
        
        # add the custom defined tensorflow layer that calculates the measurement outcomes
        expectations = [
                [QuantumMeasurement(rho,X, name="rho%dM%d"%(idx_rho,idx_X))([Vo[idx_X],Uc]) for idx_X, X in enumerate(measurement_operators)]
                for idx_rho,rho in enumerate(initial_states)]
       
        # concatenate all the measurement outcomes
        expectations = layers.Concatenate(axis=-1)(sum(expectations, [] ))

        # construct the model        
        self.model    = Model( inputs = pulse_time_domain,  outputs = expectations )          

        # specify the optimizer and loss function for training 
        self.model.compile(optimizer=optimizers.Adam(lr=0.01), loss='mse')
    
        # print a summary of the model showing the layers, their connections, and the number of training parameters
        self.model.summary()

     
    def train_model(self, training_x, training_y, epochs, batch_size):
        """
        This method is for training the model given the training set
        
        training_x: A numpy array that stores the time-domain represenation of control pulses of dimensions (number of examples, number of time steps, 1, number of axes)
        training_y: A numpy array that stores the measurement outcomes (number of examples, number of measurements).
        epochs    : The number of iterations to do the training     
        batch_size: The batch size
        """        
        
        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.training_history = self.model.fit(training_x, training_y, epochs=epochs, batch_size=batch_size,verbose=2).history["loss"] 
        
    def train_model_val(self, training_x, training_y, val_x, val_y, epochs, batch_size):
        """
        This method is for training the model given the training set and the validation set
        
        training_x: A numpy array that stores the time-domain represenation of control pulses of dimensions (number of examples, number of time steps, 1, number of axes)
        training_y: A numpy array that stores the measurement outcomes (number of examples, number of measurements).
        val_x     : The validation input array [similar to training_x]
        val_y     : The valiation desired output array [similar to training_y]  
        epochs    : The number of iterations to do the training     
        batch_size: The batch size    
        """        

        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        h  =  self.model.fit(training_x, training_y, epochs=epochs, batch_size=batch_size,verbose=1,validation_data = (val_x, val_y)) 
        self.training_history  = h.history["loss"]
        self.val_history       = h.history["val_loss"]
               
    def predict_measurements(self, testing_x):
        """
        This method is for predicting the measurement outcomes using the trained model. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """        
        return self.model.predict(testing_x)
    
    def predict_control_unitary(self,testing_x):
        """
        This method is for evaluating the control unitary. Usually called after training.
        
        testing_x: A numpy array of shape (number of examples, number of time steps, number of axes)
        """
        
        # define a new model that connects the input voltage and the "UC" later output
        unitary_model = Model(inputs=self.model.input, outputs=self.model.get_layer('Uc').output)
    
        # evaluate the output of this model
        return unitary_model.predict(testing_x)            
    
    def predict_noise(self, testing_x):
        """
        This method is for predicting the noise parameters
        """
        #beta = np.sin(self.model.get_weights()[0])
        beta = self.model.get_weights()[0]
        w    = np.exp(self.model.get_weights()[1])/np.sum(np.exp(self.model.get_weights()[1]))
        
        return beta,w
    
    def predict_Vo(self, testing_x):
        """
        This method is for predicting the Vo operators. Usally called after training.
       
        testing_x: A numpy array of shape (number of examples, number of time steps, number of axes)
        """
          
        # define a new model that connects the inputs to each of the Vo output layers
        Vo_model = Model(inputs=self.model.input, outputs=[self.model.get_layer(V).output for V in ["V%d"%idx for idx in range(self.m)] ] )
        
        # predict the output of the truncated model. This physically represents <U_I' O U_I>. We still need to multiply by O to get Vo = <O U_I' O U_I>
        Vo = Vo_model.predict(testing_x)
      
        return Vo
           

     
    def save_model(self, filename):
        """
        This method is to export the model to an external .mlmodel file
        
        filename: The name of the file (without any extensions) that stores the model.
        """
        
        # first save the ml model
        self.model.save_weights(filename+"_model.h5")
        
        # second, save all other variables
        data = {'training_history':self.training_history, 
                'val_history'     :self.val_history,
                }
        f = open(filename+"_class.h5", 'wb')
        pickle.dump(data, f, -1)
        f.close()
	
        # zip everything into one zip file
        f = zipfile.ZipFile(filename+".mlmodel", mode='w')
        f.write(filename+"_model.h5")
        f.write(filename+"_class.h5")
        f.close()
        
        # now delete all the tmp files
        os.remove(filename+"_model.h5")
        os.remove(filename+"_class.h5")

    def load_model(self, filename):
        """
        This method is to import the models from an external .mlmodel file
        
        filename: The name of the file (without any extensions) that stores the model.
        """       
        #unzip the zipfile
        f = zipfile.ZipFile(filename+".mlmodel", mode='r')
        f.extractall()
        f.close()
        
        # first load the ml model
        self.model.load_weights(filename+"_model.h5")

                
        # second, load all other variables
        f = open(filename+"_class.h5", 'rb')
        data = pickle.load(f)
        f.close()          
        self.training_history  = data['training_history']
        self.val_history       = data['val_history']

        # now delete all the tmp files
        os.remove(filename+"_model.h5")
        os.remove(filename+"_class.h5")
###############################################################################
