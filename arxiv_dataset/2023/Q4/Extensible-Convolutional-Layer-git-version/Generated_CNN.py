import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


#use multiple 1 filter CNN layer to comprise a multi filter CNN layer

#ideas for reducing resolution by factor of 2
#1. use 4x4 filter and stride 2
#2. use 2x2 filter and stride 2
#3. use a flatten layer for computation (?)
images_x_0_9 = [0]*10

def initialize_x_0_9(images_0_9):
    global images_x_0_9
    images_x_0_9 = images_0_9
#customized activation function
def bell_tanh_activation(x):
    #shapes like a bell
    #when x in 0 to 1, y approaches 1
    #and decreases on both sides and approaches -1
    #use 5+5x tanh
    x1 = 10-5*x
    x2 = 5*x
    return tf.minimum(tf.tanh(x1), tf.tanh(x2))
    
def gaussian_activation(x, tilt_level = 0.85): #last best -- 1.1
    x = tilt_level-tilt_level*x
    return 2*tf.exp(-(x**2)) - 1 #range -1 to 1


def gaussian_activation_01(x, tilt_level = 1):#0.85): #last best -- 1.1
    #x = tilt_level-tilt_level*x
    return tf.exp(-(x**2)) #range 0 to 1


#sigmoid
#1/(1+e^-x)

#======================================================================
# Notes:
# proves that sigmoid is better, for unknown reason
# probably because the gradient is non-zero for all values of x
# In contrast, bell has gradient approaching 0 not only on two ends, but in middle, and hence is harder to train
#======================================================================


class extensible_CNN_layer_multi_module_3D(tf.keras.Model):
    #growth model
    #input: nxnx1
    #output: 1*n
    #filter number can increase
    #last parameter:
    # kernel_size = (4,4), stride = 2, activation = 'gaussian_bell', padding = 'valid', optimizer = 'adam'
    #best: gaussian bell of 1 tilt level, with reg on weight and bias, and 3x3 filter with stride 1
    def __init__(self, kernel_size = (4,4,1), stride = 2, activation = 'sigmoid', padding = 'valid', optimizer = 'adam'): #best -- gaussian_bell, 4x4, stride 2
        super(extensible_CNN_layer_multi_module_3D, self).__init__()
        self.random_initilization = False
        self.filter_list = []
        self.bias_list = []
        self.kernel_size = kernel_size
        self.stride = stride
        self.decay_rate = 0 #other option -- 0.5
        self.visualising = False
        self.activation = activation
        self.channels = 1
        self.threshold = 0.5
        if self.activation == "gaussian_bell":
            self.threshold = 0.4 #0.34 final -- 0.4 one layer
        if self.activation == "sigmoid":
            self.threshold = 0.65 #from 0.5 to 0.7
        if self.activation == "relu":
            self.threshold = 0.5
        if self.activation == "gaussian_bell_01":
            self.threshold = 0.7 #last 0.5    
        
        if (activation == 'bell_tanh'):
            self.activation = bell_tanh_activation
        if (activation == 'gaussian_bell'):
            self.activation = gaussian_activation
        if (activation == 'gaussian_bell_01'):
            self.activation = gaussian_activation_01
         

        
        
        self.padding = padding
        self.optimizer = optimizer
        if optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        else:
            print("\noptimizer not implemented\n")
            return
        
        self.sample_space = {}
        self.filter_list = [] 
        #initialized with 1 filter
        self.aggregated_conv = None
    
    def call(self, input_x):
        #input
        feature_maps = []
        for filter_i in self.filter_list:
            feature_maps.append(filter_i(input_x))
        #feature map of a conv3d layer is: (batch, height, width, depth, channels)
        #ignore batch and channels
        #concatenate the 3D feature maps on the depth axis
        #print("feature_maps shape: ", feature_maps[0].shape)
        
        #print(len(feature_maps), feature_maps[0].shape)
        feature_maps = tf.concat(feature_maps, axis = 3)
        #print("feature_maps shape: ", feature_maps.shape)
        #Final shape -- (batch, height, width, depth, channels), where batch and channels are 1
        return feature_maps
    
    def update_depth(self, new_depth):
        #because with change of previous layer's units, the depth of inputs to this layer changes
        #So update the depth in all filters and sample space of this layer
        #filter_list -- list
        #sample_space -- dict -- key: filter_index, value: sample
        print("original depth: ", self.filter_list[0].get_weights()[0].shape)
        self.kernel_size = (self.kernel_size[0], self.kernel_size[1], new_depth) #update kernel size for new depth
        for filter_index in range(len(self.filter_list)):
            filter_depth = self.filter_list[filter_index].kernel_size[2] #filter depth
            if filter_depth > new_depth:
                #this might be caused by an error
                print("filter depth is larger than new depth\nOnly increase is allowed at this version")
                return
            if filter_depth == new_depth:
                #no need to update
                continue
            
            #create a new filter
            new_filter = tf.keras.layers.Conv3D(filters = 1, kernel_size = self.kernel_size, strides = self.stride, padding = self.padding, activation = self.activation)
            #weights
            new_filter.build(input_shape = (1, self.kernel_size[0], self.kernel_size[1], new_depth, 1))
            new_filter_weights = np.zeros((self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1, 1)).astype('float32') #initialize with 0
            #reduce the newly added weights by 1/sum(newly added weights)
            count_new_weights = self.kernel_size[0]*self.kernel_size[1]*(self.kernel_size[2]-filter_depth)
            reduce_map = np.zeros((self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1, 1)).astype('float32') #initialize with 0
            reduce_map[:, :, :, :, :] -= 1/(new_depth - filter_depth) #reduce by 1/sum(newly added weights) for the newly added weights
            #try to reject the new activation from the input
            #so that after expansion of the input size, the sub-features are considered as new classes
            
            #another way is to not use the reduce map
            #this assumes that the new features are classified as current classes
            
            #a different reduce map will result in different mode. Current mode --> inclusive, may recognize some new patterns as its own class
            #add the old weights
            old_weights = self.filter_list[filter_index].get_weights()[0]
            new_filter_weights[:, :, :filter_depth, 0, 0] += old_weights[:, :, :, 0, 0] #reduce_map[:, :, filter_depth:, 0, 0]
            #add the reduce map
            new_filter_weights[:, :, filter_depth:, 0, 0] += reduce_map[:, :, filter_depth:, 0, 0]
            #add the old weights to the new filter and reduced weights to the new filter
            new_bias = self.filter_list[filter_index].get_weights()[1]
            
            #set the new weights
            new_filter.set_weights([new_filter_weights, new_bias])
            #update the filter
            self.filter_list[filter_index] = new_filter
            
            #update the sample space
            sample_i = self.sample_space[filter_index]
            #add 0s to the end of the sample
            zero_sample = np.zeros((1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1)).astype('float32')
            zero_sample[0, :, :, :filter_depth, 0] += sample_i[0, :, :, :, 0]
            #update the sample space
            self.sample_space[filter_index] = zero_sample
        #at the end, print a message to show the update is done
        #take the first filter's shape as example
        first_filter_shape = self.filter_list[0].get_weights()[0].shape
        print("update of layer's depth done, new depth: ", first_filter_shape)
        print("new sample space shape: ", self.sample_space[0].shape)
        
        
            
            
            
    
    
    def add_filter(self, x, epochs = 10, refit = False, regularization = True, image_x = None):
        #use autoencoder to generate a new filter which accepts x
        #x: nxnxnx1
        #check if x is fit for filter size
        
        if x.shape != (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1):
            print("x is not fit for filter size")
            print("x shape: ", x.shape)
            print("expected shape: ", (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1))
            return
        
        #reshape x to (1, nxnxn, 1)
        x = x.reshape(1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1) #for 3D conv
        
        
        

        
        if image_x is not None:
            print("\nUpdating the sample space")
            #get the feature map for different filters
            for filter_i in range(max(len(self.filter_list) - 5,0), len(self.filter_list)):
                feature_map = self.filter_list[filter_i](image_x)
                #get one of the max value's location
                if np.max(feature_map) < self.threshold:
                    #when the max value is less than threshold, then the filter is not activated for this image
                    continue
                
                max_loc_fm = np.unravel_index(np.argmax(feature_map, axis=None), feature_map.shape)#[0]
                #print(max_loc_fm, feature_map.shape, image_x.shape)
                #map back to the image_x
                org_x = max_loc_fm[1]*self.stride
                org_y = max_loc_fm[2]*self.stride
                org_z = max_loc_fm[3]*self.stride
                #find the patch from image_x
                self.sample_space[filter_i] = image_x[:, 
                                                      (org_x):(org_x + self.kernel_size[0]), 
                                                      (org_y):(org_y + self.kernel_size[1]), 
                                                        (org_z):(org_z + self.kernel_size[2]),
                                                      :].reshape(1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1)

        #initialize a new filter with decoder

        #check if x is equivalent to any existing filter's sample space
        for sample_i in self.sample_space.keys():
            if np.sum(abs(self.sample_space[sample_i] - x)) < 0.2:
                print("x is already in sample space")
                return 0
            
            
        new_filter = tf.keras.layers.Conv3D(1, self.kernel_size, 
                                            padding=self.padding, activation=self.activation,strides=self.stride)
        #decoder = tf.keras.layers.Conv3DTranspose(1, self.kernel_size, 
        #                                          padding=self.padding, activation=self.activation)
        
        decoder = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape = (1,1,1,1,1)),
                tf.keras.layers.Dense(np.prod(self.kernel_size), activation = None, use_bias = False),
                tf.keras.layers.Reshape((self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1))
            ])
        
        #set weights of new filter to be the same as x
        print("new filter initialized, id = ", len(self.filter_list))
        
        #initialize new filter
        
        #get the 1 matrix of x
        #TODO: use a better way to get the 1 matrix of x
        #  by giving a suitable threshold -- example, 0.5
        zero_pixel_threshold = 0.001#self.threshold
        
        #matrix_1 = (x > zero_pixel_threshold).astype(np.float64)
        #make the matrix_1, when multiplied by x, the output is 1
        #assigning top n to 1, and rest to -1, to make the output of the dot product to be 1
        #find the threshold, where the values below and above sum to same value
        #because the range of x is 0 to 1, the sum of the values below and above the threshold is 0.5
        random_filter = np.random.rand(self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1)


        matrix_1 = (x - np.mean(x))
        
        blank_filter = False
        counter_std_enlarge = 0
        if (np.std(x) < 0.1 and np.std(x) != 0): #if it is all same values
            print("std is too small, increase std to a higher value")
            #matrix_1 = np.zeros(x.shape) - 5/(np.prod(x.shape) - 1) #if it is all the same values
            #max_idx = np.unravel_index(np.argmax(x, axis=None), x.shape)
            #matrix_1[:, self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2, :] = 5
            #blank_filter = True

            while (np.std(matrix_1) < 0.1 and counter_std_enlarge < 100 ):
                #matrix_1 = matrix_1/np.max(np.abs(matrix_1))
                #matrix_1 = np.multiply(matrix_1, np.abs(matrix_1))
                #matrix_1 = matrix_1 - np.mean(matrix_1)
                #shrink for greater than 0 or less than 0
                matrix_1_pos_loc = np.where(matrix_1 > 0)
                matrix_1_neg_loc = np.where(matrix_1 < 0)
                matrix_1[matrix_1_pos_loc] = matrix_1[matrix_1_pos_loc]/np.max(matrix_1[matrix_1_pos_loc])
                matrix_1[matrix_1_pos_loc] = np.multiply(matrix_1[matrix_1_pos_loc], matrix_1[matrix_1_pos_loc])

                matrix_1[matrix_1_neg_loc] = matrix_1[matrix_1_neg_loc]/np.abs(np.min(matrix_1[matrix_1_neg_loc]))
                matrix_1[matrix_1_neg_loc] = np.multiply(matrix_1[matrix_1_neg_loc], np.abs(matrix_1[matrix_1_neg_loc]))
                #print(matrix_1)
                matrix_1 = matrix_1 - np.mean(matrix_1)

                print(np.std(matrix_1), matrix_1)
                counter_std_enlarge += 1
            #matrix_1 = np.ones(x.shape)
        
        if np.std(x) == 0:
            #blank filter
            avg_val = np.mean(x)
            weight_count = np.prod(x.shape)
            
            #to jump out of the loop
            matrix_1 = np.ones(x.shape)*avg_val
            blank_filter = True
        else:
            matrix_1 = matrix_1/(np.sum(np.multiply(matrix_1, x)))
        #matrix_1 = x
        #matrix_1 = np.multiply(matrix_1, x)
        #matrix_1 = matrix_1/np.max(matrix_1)
        print("result 0", np.sum(np.multiply(matrix_1, x)))
        print("max abs", np.max(np.abs(matrix_1)))
        

        
        
        bias = np.array([-1]) #0
        
        if (np.max(x) > zero_pixel_threshold) and np.sum(x) != np.prod(x.shape): #if sum equas to prod, then all pixels are 1, which is used for initialization
            #characterization =(x/np.max(x))
            #equals to the 1 matrix of x
            #random_filter = np.random.rand(3,3,1)/5
            
            #avoid a random_filter value on pixel <= 0
            #multiply by dot product of matrix_1 and random_filter
            random_filter = np.multiply(matrix_1, random_filter)
            
            #devide by 5 so that the maximum output is 0.2*16 - 1 = 2.2
            #characterization = characterization/5  #/np.average(characterization)
            
            #or divide by the size of the filter, setting bias to 0, so the output is 1
            
            #bias = np.array([-np.sum(matrix_1)])
            if self.activation == gaussian_activation_01:
                bias = np.array([-(np.sum(np.multiply(matrix_1, x)))])
            else:
                print("sum of matrix_1 = ", np.sum(matrix_1))
                #print(matrix_1)
                bias = np.array([0])

            #V3
            
            characterization = (matrix_1)# - zero_pixel_threshold
            if self.random_initilization:
                print("Set to random filter")
                characterization = random_filter
            #print("characterization shape = ", characterization.shape, "bias = ", float(bias), characterization)
            #characterization -= random_filter
            #print("characterization = ", characterization)
            print("char sum:", np.sum(characterization))
            
            #characterization = characterization/np.sum(characterization) #so when product with x, the output is 1
            
            
        blank_filter = False
        if np.sum(matrix_1) == np.prod(matrix_1.shape) and np.max(x) > zero_pixel_threshold:
            print("all 1")
            
            characterization = np.ones(x.shape)/np.prod(x.shape)
            bias = np.array([-np.sum(characterization)/2])
            return 0
            #blank_filter = True
        if np.max(x) <= zero_pixel_threshold:
            bias = np.array([1])
            characterization = x - 2*(bias)    #/(np.max(x)+0.01)
            #print(characterization)
        #print("characterization = ", characterization, "matrix_1 = ", matrix_1)
        #characterization = np.asarray(characterization - 1 + matrix_1).reshape(1,self.kernel_size[0],self.kernel_size[1],1).astype(np.float64)
        characterization = np.asarray(characterization).reshape(1,self.kernel_size[0],self.kernel_size[1], self.kernel_size[2], 1).astype(np.float64)
        
        #normalize
        #characterization = characterization/np.max(characterization)
        new_filter.build(input_shape = (None,self.kernel_size[0],self.kernel_size[1],self.kernel_size[2],1))
        weight = characterization.reshape(self.kernel_size[0],self.kernel_size[1],self.kernel_size[2],1,1)
        new_filter.set_weights([weight, bias])
        decoder.build(input_shape = (None,1,1,1,1))
        
        adjust_value = 1
        if self.activation == "gelu":
            adjust_value = 0.84
        if self.activation == "sigmoid":
            adjust_value = 0.73
        weight2 = x.reshape(self.kernel_size[0],self.kernel_size[1],self.kernel_size[2],1,1)/(adjust_value) #so that the values will not shift
        weight2 = weight2.reshape(1,np.prod(weight2.shape))

        #decoder.set_weights([weight, bias])
        decoder.set_weights([weight2])#, np.array([0]*np.prod(weight2.shape))])
        #the weight of this filter is characterized by x

        #train the new filter
        #new filter must reject all other x in the sample space
        self.filter_list.append(new_filter)
        
        discard = False
            
        #==============================================
        #sub-functions that can be reused in this function
        def call_autoencoder(x):
            y = new_filter(x)
            y = decoder(y)
            #because decoder has activation, so y is in range [0,1]
            #hence magnify y by max in x
            #y = y * np.max(x)
            return y
        
        def combined_fit_v3(target, negative_samples):
            loss = np.array([0.0])
            global discard
            #the negative samples in this version is a list of all negative samples
            with tf.GradientTape(persistent=True) as tape:
                #y = call_autoencoder(target)
                #use sum of square error as loss function
                #loss = tf.reduce_sum(tf.square(y - target))
                
                #expect y to be close to 0
                #fetch the loss of each negative sample
                #if self.activation == gaussian_activation_01 or self.activation == "gelu" or self.activation == "relu":
                #loss += (tf.reduce_sum(tf.square(new_filter(target)- 0.8)))
                positive_target = 0.85
                
                if len(negative_samples) >= 4:
                    #Using 1 as positive target
                    if self.decay_rate == 0:
                        positive_target = 1
                    loss += (tf.reduce_sum(tf.square(new_filter(target)- positive_target)))
                else:
                    positive_target = 0.73
                    if self.decay_rate == 0:
                        positive_target = self.threshold 
                        #target to threshold, because this would find a filter that recognise target by giving output as the value of threshold
                    loss += (tf.reduce_sum(tf.square(new_filter(target)- positive_target))) #conservative mode
                    
                        
                if (new_filter(target).numpy()[0] < 0):
                    #characterization might be wrong
                    print("characterization = ", characterization.reshape(self.kernel_size[0],self.kernel_size[1],self.kernel_size[2]))
                    print("bias = ", bias)

                    
                if self.activation != gaussian_activation_01:
                    
                    #for neg_sample in negative_samples:
                    #    z = new_filter(neg_sample)
                    #    target_z = 0#-1, make the negative samples always exist in loss back propagation instead of ignoring it during training
                    #    if self.activation == gaussian_activation_01:
                    #        target_z = -1
                    #    loss += tf.reduce_sum(tf.square(z - target_z)) #the averaged loss
                    
                    if len(negative_samples) > 3:
                        #negative_samples = negative_samples[0:]
                        target_vec = np.zeros(len(negative_samples)).astype(np.float32)
                        #to tensor

                        #turn to tensor
                        pred_vec = new_filter(np.array(negative_samples))
                        target_vec = pred_vec * self.decay_rate
                        target_vec = tf.convert_to_tensor(target_vec)
                        target_vec = tf.reshape(target_vec, [-1])
                        pred_vec = tf.reshape(pred_vec, [-1])
                        
                        #if pred_vec > threshold, then it is a negative sample
                        #only take the negative samples' loss
                        threshold = self.threshold
                        #if len(negative_samples) > 0: #could be adjusted to remove the influence from null filters
                        #    bool_vec = tf.math.greater(pred_vec, threshold)
                        #    #make the corresponding target to be 1
                        #    target_vec = tf.where(bool_vec, tf.ones_like(target_vec), target_vec).numpy()
                            
                            #pred_vec = tf.where(bool_vec, pred_vec, tf.zeros_like(pred_vec)).numpy()
                        #print("\npred_vec[0:5] = ", pred_vec[0:5])
                        #checking loss
                        '''
                        if self.activation == "gelu" or self.activation == "relu":
                            #consider the predictions of negative samples can be greater than 1
                            #which means the negative samples might be out of the range
                            #which implies a distribution out of the range of the filter
                            #assumes a normal distribution with a mean of x=1
                            #hence, the loss should bue 0 for those samples
                            bool_vec = tf.math.less(pred_vec, 2) #larger than 1 would cause explosion in loss
                            pred_vec = tf.where(bool_vec, pred_vec, tf.zeros_like(pred_vec)).numpy()
                        '''
                        #print("\npred_vec = ", pred_vec)#, " sum: ", np.sum(pred_vec), "\nin loss:", tf.reduce_sum((pred_vec - target_vec))/len(negative_samples))
                        #print(new_filter.weights[0].numpy().reshape(4,4,3))
                        #print("target_vec = ", target_vec)

                        #print("before pred loss: ", loss)
                        #use MSE on pred and target vec
                        #print("loss0 = ", loss)
                        mse_loss = tf.reduce_sum(tf.square(pred_vec - target_vec))/len(negative_samples)
                        loss += mse_loss
                        #print("loss = ", loss)
                        #loss += tf.reduce_sum(tf.square(pred_vec - target_vec))/len(negative_samples)
                        #print("final loss:", loss)
                        #print min max of weight
                        #print("min weight = ", np.min(new_filter.weights[0]), " max weight = ", np.max(new_filter.weights[0]))
                        #print("sum pred_vec = ", np.sum(pred_vec), " len(negative_samples) = ", len(negative_samples))
                        #if np.sum(pred_vec) >= len(negative_samples)/2:
                            #overfitted filter
                            #too sensitive to the negative samples
                            #discard this filter
                        #    discard = True
                        
                        #plus max value in weight
                        #loss += tf.math.reduce_max(new_filter.weights[0]) - 1
                    #make bias close to 0
                    #loss += tf.reduce_sum(tf.square(new_filter.weights[1]))
                    #if the depth is greater than 3, then it is after another layer. Use regularization
                    if self.kernel_size[2] > 3 and self.activation != "gelu" and self.activation != "relu":
                        #regularization -- make the weights close to 0
                        loss += np.sum(np.square(new_filter.weights[0]))



                else:
                    #for gaussian activation, the negative samples are not needed
                    #but need to make the bias close to 0, and have larger variance on the weights
                    loss += np.sum(np.square(new_filter.weights[1]))
                    var_w = np.var(new_filter.weights[0])
                    loss -= var_w
                    
                    #take last 10 negative samples
                    
                    neg_sample_list = []
                    if len(negative_samples) > 0 and self.kernel_size[2] <= 3:
                        for i in range( len(negative_samples)):
                            neg_sample_list.append(negative_samples[-i])
                        neg_sample_list = np.array(neg_sample_list)
                        neg_pred = new_filter(neg_sample_list)
                        loss += np.sum(np.square(neg_pred))/len(negative_samples)
                    

                
                    
            grad_filter = tape.gradient(loss, new_filter.trainable_variables)
            #grad_decoder = tape.gradient(loss, decoder.trainable_variables)
            #give a learning rate according to the loss
            #learning_rate = tf.math.maximum(0.00000000000001, loss/10).numpy()[0]
            #learning_rate = tf.math.minimum(0.05, learning_rate)
            #print(learning_rate)
            #print("learning rate = ", learning_rate)
            #self.optimizer.learning_rate.assign(learning_rate)
            
            #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate.numpy())
            #optimizer.apply_gradients(zip(grad_filter, new_filter.trainable_variables))
            self.optimizer.apply_gradients(zip(grad_filter, new_filter.trainable_variables))
            #last best not update decoder
            #self.optimizer.apply_gradients(zip(grad_decoder, decoder.trainable_variables))
            #clear tape
            tf.keras.backend.clear_session()
            return loss
        
        #
        progress_bar = tf.keras.utils.Progbar(epochs)
        

        '''
        print("\ncombined training")
        for epoch in (range(epochs)):
            loss = 0
            for i in range(len(self.filter_list) - 1):
                loss = combined_fit(x, self.sample_space[i])
            progress_bar.update(epoch, values=[("loss", loss)])
        ''' 
        
        print("\ncombined training v3")
        if blank_filter:
            print("Blank filter")
            return 0
        #print("Not fitting mode (only initialize the filter)")
        #epochs = 0
        neg_samples = []
        for i in range(len(self.filter_list) - 1):
            neg_samples.append(self.sample_space[i])
        if len(neg_samples) > 3:
            for epoch in (range(epochs)):
                loss = 0
                loss = combined_fit_v3(x, neg_samples)
                progress_bar.update(epoch, values=[("loss", loss)])
                if discard and len(self.filter_list) > 1:
                    print("discard this filter")
                    self.filter_list.pop()
                    self.sample_space.pop(len(self.sample_space) - 1)
                    return 0
        
        if image_x is not None and self.visualising:
            results_init = new_filter(image_x)
            resulting_fm_shape = int(np.sqrt(np.prod(results_init.shape)))
            #print("\n",resulting_fm_shape)
            
            plt.imshow(results_init.numpy().reshape(resulting_fm_shape,resulting_fm_shape), cmap = "gray")
            plt.colorbar()
            plt.show()
        
        self.sample_space[len(self.filter_list) - 1] = x #.reshape(1, self.kernel_size[0], self.kernel_size[1], 1) #add to sample space

        
        self.filter_list[len(self.filter_list) - 1] = new_filter #replace the filter
        
        

        if refit:
            #fit all the filters
            self.refit_all(epochs = epochs)
        
        return new_filter
    
    
    def refit_all(self, epochs = 100, image_x = None):
        #refit all the filters with their own sample (1) and other filters' samples (-1)
        #update the sample space
                
        def refit_filter(idx, epochs):
            print("\nrefitting filter ", idx)
            progress_bar = tf.keras.utils.Progbar(epochs)
            for epoch in range(epochs):
                filter_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                loss = 0
                with tf.GradientTape(persistent=True) as tape:
                    
                    #fit the filter
                    for i in range(len(self.filter_list)):
                        if (i == idx):
                            loss += tf.reduce_sum(tf.square(self.filter_list[i](self.sample_space[i]) - 1))
                        else:
                            loss += tf.reduce_sum(tf.square(self.filter_list[i](self.sample_space[i]) - -1))
                grad_filter = tape.gradient(loss, self.filter_list[idx].trainable_variables)
                filter_optimizer.apply_gradients(zip(grad_filter, self.filter_list[idx].trainable_variables))
                progress_bar.update(epoch, values=[("loss", loss)])
            #clear tape
            
            tf.keras.backend.clear_session()
            
        
        for i in range(max(int(len(self.filter_list)/2)-1,0), len(self.filter_list)):
            refit_filter(i, epochs)
    
    #TODO: following functions are needed potentially when the learning covers more images
    def collapse_check(self, example_img):
        #check if the filters are overlapping
        return
    
    def collapse_overlapping(self, example_img):
        #collapse the overlapping filters
        return
    
    
    
    def get_index_map(self, x):
        feature_maps = self.call(x)
        return np.argmax(feature_maps, axis = 3).reshape(np.shape(feature_maps)[1:-1])
    
    
    def call_seperated_fm(self, x):
        feature_maps = []
        for i in range(len(self.filter_list)):
            feature_maps.append(self.filter_list[i](x))
        return feature_maps
            
    def get_aggregated_conv(self):
        #return the aggregated convolutional layer by building a conv layer with the weights in the filter list
        #to set layer weights
        #kernel_size[0], kernel_size[1], kernel_size[2], channel, len(self.filter_list)
        weights = np.zeros((self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1, len(self.filter_list)))
        biases = np.zeros((len(self.filter_list)))
        for i in tqdm.tqdm(range(len(self.filter_list))):
            weight_i = self.filter_list[i].get_weights()[0].reshape(self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1, 1)
            bias_i = self.filter_list[i].get_weights()[1].reshape(1)
            weights[:,:,:,0,i] = weight_i[:,:,:,0,0]
            biases[i] = bias_i[0]
        new_conv = tf.keras.layers.Conv3D(len(self.filter_list), self.kernel_size, strides = self.stride, padding = "valid", activation =self.activation)
        new_conv.build((None, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1))
        new_conv.set_weights([weights, biases])
        self.aggregated_conv = new_conv
        return new_conv
    
    def get_aggregated_conv_2D(self, name = None):
        #the depth is the channel of the input
        
        weights = np.zeros((self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], len(self.filter_list)))
        biases = np.zeros((len(self.filter_list)))
        for i in tqdm.tqdm(range(len(self.filter_list))):
            weight_i = self.filter_list[i].get_weights()[0].reshape(self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1)
            bias_i = self.filter_list[i].get_weights()[1].reshape(1)
            weights[:,:,:,i] = weight_i[:,:,:,0]
            biases[i] = bias_i[0]
        print(self.kernel_size)
        if name == None:
            new_conv = tf.keras.layers.Conv2D(len(self.filter_list), kernel_size=(self.kernel_size[0], self.kernel_size[1]), strides = self.stride, padding = "valid", activation =self.activation)
        else:
            new_conv = tf.keras.layers.Conv2D(len(self.filter_list), kernel_size=(self.kernel_size[0], self.kernel_size[1]), strides = self.stride, padding = "valid", activation =self.activation, name = name)
        new_conv.build((None, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        new_conv.set_weights([weights, biases])
        #self.aggregated_conv = new_conv
        return new_conv
        
            

def show_all_index_maps_3D(model, data_0_9):
    figure = plt.figure(figsize=(20, 20))
    global images_x_0_9
    image_shape = images_x_0_9[0].shape
    image_size = image_shape[-3]
    if images_x_0_9[0].shape[-2] == 1:
        for i in range(len(data_0_9)):

            feature_maps = tf.convert_to_tensor(model.call(data_0_9[i]).numpy().astype('float64'))
            plt.subplot((len(data_0_9) * 2) // 5, 5, i+1 + (i // 5) * 5)
            plt.imshow(images_x_0_9[i].reshape(image_size,image_size), cmap='gray')
            plt.title("label "+ str(i))
            plt.subplot((len(data_0_9) * 2) // 5, 5, i+6 + (i // 5) * 5)
            index_map = np.argmax(feature_maps, axis = 3).reshape(np.shape(feature_maps)[1:-2])
            plt.imshow(index_map, cmap='gist_ncar')
        plt.show()
    else:
        #3 channel rgb images
        for i in range(len(data_0_9)):
            feature_maps = tf.convert_to_tensor(model.call(data_0_9[i]).numpy().astype('float64'))
            plt.subplot((len(data_0_9) * 2) // 5, 5, i+1 + (i // 5) * 5)
            plt.imshow(images_x_0_9[i].reshape(image_size,image_size,3))
            plt.title("label "+ str(i))
            plt.subplot((len(data_0_9) * 2) // 5, 5, i+6 + (i // 5) * 5)
            
            index_map = np.argmax(feature_maps, axis = 3).reshape(np.shape(feature_maps)[1:-2])
            plt.imshow(index_map, cmap='gist_ncar')
        plt.show()

        
        
def generate_filter(image, model, image_shape, fm_0_9, show_mapping = False):
    if image_shape[2] > model.kernel_size[2]:
        model.update_depth(image_shape[2])
    print("image shape", image_shape)
    kernel_size = model.kernel_size
    stride = model.stride
    #image_shape ---> (x, y, z)
    image = image.reshape(1, image_shape[0], image_shape[1], image_shape[2], 1)
    feature_map_size_xy = (image_shape[0] - kernel_size[0]) // stride + 1
    feature_map_size_z = (image_shape[2] - kernel_size[2]) // stride + 1
    feature_map_size = (feature_map_size_xy, feature_map_size_xy, feature_map_size_z)

    if len(model.filter_list) == 0:
        null_input = np.zeros((kernel_size[0], kernel_size[1], kernel_size[2], 1))
        new_filter = model.add_filter(null_input, epochs = 1)
        
        #one_input = np.zeros((kernel_size[0], kernel_size[1], kernel_size[2], 1))+0.1
        ##make the center point to 1
        #one_input[kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2, 0] = 1
        #new_filter = model.add_filter(one_input, epochs = 100)
        #one_input = np.ones((kernel_size[0], kernel_size[1], kernel_size[2], 1))#*(1/np.prod(kernel_size)) #0.7 is around the 1 of the sigmoid function
        #new_filter = model.add_filter(one_input, epochs = 100)
        
        if new_filter is None:
            print("ERROR null filter")
    
    feature_maps = model.call(image.reshape(1, image_shape[0], image_shape[1], image_shape[2], 1))
    #feature map size --  1, x, y, z, 1
    #separate to multiple feature maps corresponding back to different filters
    feature_maps_of_filters = []
    
    for i in range(len(model.filter_list)):
        depth_start = i * feature_map_size[2]
        depth_end = (i+1) * feature_map_size[2]
        feature_maps_of_filters.append(feature_maps[:, :, :, depth_start:depth_end, :])
    #add together and divide by the number of filters
    
    #find the maximum values within the feature maps
    #print(model)
    max_values_on_maps = np.zeros(feature_map_size)
    for fm in feature_maps_of_filters:
        fm_greater = (fm>model.threshold).numpy().astype(np.float32)
        fm_smaller = (fm<model.threshold*3).numpy().astype(np.float32)
        fm = np.multiply(fm, fm_greater)
        fm = np.multiply(fm, fm_smaller)
        max_values_on_maps += fm.reshape(feature_map_size) #take the locations that are non-zero

    
    inactive_points = np.where(max_values_on_maps == 0)
    inactive_ratio_0 = len(inactive_points[0])/np.prod(np.shape(max_values_on_maps))
    #check how many points are inactive
    print ("inactive ratio ", inactive_ratio_0)
    
    if len(inactive_points[0]) == 0:
        return model, inactive_ratio_0
    
    point = np.random.choice(len(inactive_points[0]))
    selected = point# % 10 #first 10 points are usually the edge points
    
    
    x = inactive_points[0][selected]
    y = inactive_points[1][selected]
    z = inactive_points[2][selected]
    
    print("selected point", x, y, z)
    
    #get the patch of the image that corresponds to the selected point
    org_x_start = x * stride
    org_y_start = y * stride
    org_z_start = z * stride
    org_x_end = org_x_start + kernel_size[0]
    org_y_end = org_y_start + kernel_size[1]
    org_z_end = org_z_start + kernel_size[2]
    
    patch = image[:, org_x_start:org_x_end, org_y_start:org_y_end, org_z_start:org_z_end, :]
    patch = patch.reshape(kernel_size[0], kernel_size[1], kernel_size[2], 1)
    new_filter = model.add_filter(patch, epochs = 100, image_x = image)
    
    separated_new_fms = model.call_seperated_fm(image)
    max_map = np.zeros(separated_new_fms[0].shape)
    for fm in separated_new_fms:
        fm_greater = (fm>model.threshold).numpy().astype(np.float32)
        fm_smaller = (fm<model.threshold*3).numpy().astype(np.float32)
        fm = np.multiply(fm, fm_greater)
        fm = np.multiply(fm, fm_smaller)
        max_map += fm.reshape(separated_new_fms[0].shape)
        
    inactive_points_check = np.where(max_map == 0)
    inactive_ratio_1 = len(inactive_points_check[0])/np.prod(np.shape(max_map))
    print("inactive ratio after adding filter", inactive_ratio_1)
    
    if (inactive_ratio_1 >= inactive_ratio_0 and new_filter != 0):
        #remove the filter
        print("removing filter")
        #print(model.filter_list[-1].get_weights()[0])
        last_idx = len(model.filter_list) - 1
        model.filter_list.pop()
        model.sample_space.pop(last_idx)
    
    if show_mapping:
        show_all_index_maps_3D(model, data_0_9=fm_0_9)
    return model, inactive_ratio_1



def get_inactive_ratio(model, image):
    
    fm_i = model.call_seperated_fm(image)
    max_map = np.zeros(fm_i[0].shape)
    for fm in fm_i:
        fm_greater = (fm>model.threshold).numpy().astype(np.float32)
        fm_smaller = (fm<model.threshold*3).numpy().astype(np.float32)
        fm = np.multiply(fm, fm_greater)
        fm = np.multiply(fm, fm_smaller)

        max_map += fm.reshape(fm_i[0].shape)
    inactive_ratio = len(np.where(max_map == 0)[0])/np.prod(np.shape(max_map))
    return inactive_ratio
        
import tqdm
def get_inactive_ratio_list(model, images):
    inactive_ratios = []
    for img in tqdm.tqdm(images):
        inactive_ratios.append(get_inactive_ratio(model, img))
    return inactive_ratios

def generate_model_on_images(images, model, images_0_9, inactive_ratio_threshold = 0.1, n = 3):
    #initialize by generating 2 filters on the first image
    if model == None:
        model = extensible_CNN_layer_multi_module_3D()
    
    #model = generate_filter(images[0], model, threshold = model.threshold)
    
    #loop through the images, generate filters
    
    inactive_ratios = [0 for i in range(len(images))]
    '''
    for i in range(len(images)):
        print("\nimage ", i)
        model, ratio = generate_filter(images[i], model, images[i].shape[1:], fm_0_9=images_0_9)
        inactive_ratios[i] = ratio
    '''
    
    #If the model has no filter, first round, generate 1 filter on randomly selected n images
    if len(model.filter_list) == 0:
        for i in range(n):
            image_idx = np.random.randint(len(images))
            print("\nimage ", image_idx)
            model, ratio = generate_filter(images[image_idx], model, images[image_idx].shape[1:], fm_0_9=images_0_9)
            inactive_ratios[image_idx] = ratio
        
    #then generate 1 filter on the image with the highest inactive ratio
    
    inactive_ratios = get_inactive_ratio_list(model, images)
    
    mean_inactive_ratio = np.mean(inactive_ratios)
    max_inactive_ratio = np.max(inactive_ratios)
    print("inactive ratios mean", mean_inactive_ratio, inactive_ratios)
    
    if max_inactive_ratio < inactive_ratio_threshold:
        return model, max_inactive_ratio

    #get top n images with highest inactive ratio
    top_n_inactive_ratio_idx = np.argsort(inactive_ratios)[-n:]
    for i in top_n_inactive_ratio_idx:
        print("\nimage ", i)
        model, ratio = generate_filter(images[i], model, images[i].shape[1:], fm_0_9=images_0_9)
    
    
    print(model.activation, len(model.filter_list))
    return model, max_inactive_ratio


def generate_model_on_images_by_batch(images, model, images_0_9, inactive_ratio_threshold = 0.1, n = 3, batch = 32):
    #initialize by generating 2 filters on the first image
    if model == None:
        model = extensible_CNN_layer_multi_module_3D()
    
    #model = generate_filter(images[0], model, threshold = model.threshold)
    
    #loop through the images, generate filters
    
    inactive_ratios = [0]*len(images)

    
    #If the model has no filter, first round, generate 1 filter on randomly selected n images
    if len(model.filter_list) == 0:
        for i in range(n):
            image_idx = np.random.randint(len(images))
            print("\nimage ", image_idx)
            model, ratio = generate_filter(images[image_idx], model, images[image_idx].shape[1:], fm_0_9=images_0_9)
            inactive_ratios[image_idx] = ratio
        
    #then generate 1 filter on the image with the highest inactive ratio
    batches_count = len(images)//batch
    for i in range(batches_count):
        print("batch:", i)
        batch_image = images[i*batch:(i+1)*batch]
        inactive_ratios = get_inactive_ratio_list(model, batch_image)

        mean_inactive_ratio = np.mean(inactive_ratios)
        max_inactive_ratio = np.max(inactive_ratios)
        print("inactive ratios mean", mean_inactive_ratio, inactive_ratios)

        if max_inactive_ratio < inactive_ratio_threshold:
            continue #pass current batch

        #get top n images with highest inactive ratio
        top_n_inactive_ratio_idx = np.argsort(inactive_ratios)[-n:]
        for i in top_n_inactive_ratio_idx:
            print("\nimage ", i)
            model, ratio = generate_filter(batch_image[i], model, batch_image[i].shape[1:], fm_0_9=images_0_9)


        print(model.activation, len(model.filter_list))
    return model