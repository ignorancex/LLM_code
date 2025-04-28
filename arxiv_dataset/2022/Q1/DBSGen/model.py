from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_addons as tfa
import copy
import numpy as np

class networks(object):

    def __init__(
        self,
        is_training,
        n_input,
        n_hiddens,
        n_images,
        img_size, 
        fixed_image,
        filter_size = 7,
        channels_num = 32,
        X = None,
        indices=None
    ):
        tf.random.set_seed(1)
        self.is_training = is_training
        n_hiddens=copy.deepcopy(n_hiddens)
        self.loss= {}

        self.filter_size = filter_size
        self.channels_num = channels_num
        self.img_size = img_size
        self.n_images = n_images
        self.fixed_image = fixed_image 
        self.ResizeMethod = 'bicubic'
        n_hiddens.append(n_input)

        self.n_input =n_input 

        self.X = X

        self.indices = indices
        self.activation =  tf.nn.elu
        initializer = tf.keras.initializers.GlorotNormal()
        self.initializer =initializer
        l_reg = 0.005
        self.regularizer = tf.keras.regularizers.L2(l_reg)

        n_layers = len(n_hiddens)
        assert n_layers >= 2

        with tf.compat.v1.variable_scope("decoder"):
            self.W=[]
            self.b=[]
            self.t = tf.compat.v1.get_variable(name="t", shape=(n_images,n_hiddens[0]), dtype=tf.float32, initializer= initializer, regularizer=self.regularizer)

            for i in range(1,n_layers):
                layer = str(i)

                self.W.append( tf.compat.v1.get_variable(name="deep_Weight_"+layer, shape=(n_hiddens[i-1], n_hiddens[i]), dtype=tf.float32, initializer= initializer, regularizer=self.regularizer) )

                self.b.append(tf.compat.v1.get_variable(name="deep_bias_"+layer, shape=(n_hiddens[i]) , dtype=tf.float32, initializer= tf.zeros_initializer))

        with tf.compat.v1.variable_scope("inputs"):
            height,width,_= img_size
            self.z = tf.compat.v1.get_variable(name="z", shape=(n_images,height//8, width//8,1), dtype=tf.float32, initializer= tf.keras.initializers.VarianceScaling())

        with tf.compat.v1.variable_scope("params"): 
            self.illum = tf.compat.v1.get_variable(name="illumination", shape=(n_images,n_input), dtype=tf.float32, initializer= initializer, regularizer=self.regularizer)

    def net(self, t, W, b):
        
        hidden = t
            
        for i in range(0,len(W)-1):
            hidden = tf.matmul(hidden, W[i]) + b[i]
            hidden = tf.compat.v1.layers.batch_normalization(hidden, training=self.is_training)
            hidden = self.activation(hidden)
                       
        X_hat  = tf.nn.sigmoid(tf.matmul(hidden, W[-1]) + b[-1])
        return X_hat 

    def conv_decoder(self, hidden , scale):
        stride=2
        batch_size = tf.shape(hidden)[0]
        _,input_height,input_width,in_channels = hidden.get_shape()
        filter_size = self.filter_size
        channels_num = self.channels_num

        
        with tf.compat.v1.variable_scope(scale, reuse=tf.compat.v1.AUTO_REUSE):
            
            out_channels=channels_num
            shape =[filter_size, filter_size, in_channels, out_channels]
            w_conv1 = tf.compat.v1.get_variable(
                    name='W_conv1',
                    shape=shape,
                    initializer=self.initializer,
                    regularizer=self.regularizer
            )              
            conv1 = tf.nn.conv2d(
                hidden,
                w_conv1,
                strides=[1, 1, 1, 1],
                padding="SAME")  
            conv1 =self.activation(conv1)
            
            
            in_channels = channels_num
            out_channels = channels_num
            shape = [filter_size, filter_size, out_channels, in_channels]
            W_deconv1 = tf.compat.v1.get_variable(
                    name='W_deconv1',
                    shape=shape,
                    initializer=self.initializer,
                    regularizer=self.regularizer
            )
            output_shape = [batch_size,input_height*stride,input_width*stride, out_channels]
            deconv1 = tf.nn.conv2d_transpose(conv1, W_deconv1, output_shape, strides=[1, 2, 2, 1], padding="SAME")
            deconv1 =self.activation(deconv1)
            
            
            in_channels = channels_num
            out_channels = channels_num
            shape =[filter_size, filter_size, in_channels, out_channels]
            w_conv2 = tf.compat.v1.get_variable(
                    name='W_conv2',
                    shape=shape,
                    initializer=self.initializer,
                    regularizer=self.regularizer
            )               
            conv2 = tf.nn.conv2d(
                deconv1,
                w_conv2,
                strides=[1, 1, 1, 1],
                padding="SAME")  
            conv2 =self.activation(conv2)            
            
            out_channels = 2
            in_channels = channels_num
            shape = [filter_size, filter_size, out_channels, in_channels]
            W_deconv2 = tf.compat.v1.get_variable(
                    name='W_deconv2',
                    shape=shape,
                    initializer=self.initializer,
                    regularizer=self.regularizer
            )   
            output_shape = [batch_size,input_height*stride,input_width*stride, out_channels]
            deconv2 = tf.nn.conv2d_transpose(conv2, W_deconv2,output_shape, strides=[1, 1, 1, 1], padding="SAME")
            deconv2 =self.activation(deconv2)

            return deconv1,deconv2        

    def warp(self, X , z ):
        
        X = tf.reshape(X,(-1,*self.img_size))
        _,h,w,c = X.shape        
        scale1_4=[(h//8)*2,(w//8)*2]
        scale1_2=[(h//8)*4,(w//8)*4]        
        X1_4 = tf.image.resize(X,scale1_4,method=self.ResizeMethod)
        X1_2 = tf.image.resize(X,scale1_2,method=self.ResizeMethod)
                
        z = tf.gather(z,self.indices)
        z1,f1_4  = self.conv_decoder(z,"scale_1")
        z2,f1_2  = self.conv_decoder(z1,"scale_2")
        _,f1 = self.conv_decoder(z2,"scale_3")
        
        _,input_height,input_width,_ = f1_4.get_shape()
        T1_4=f1_4        
        T1_2 = f1_2 + tf.image.resize(T1_4,[input_height*2,input_width*2],method=self.ResizeMethod)
        T1 = f1 + tf.image.resize(T1_2,[input_height*4,input_width*4],method=self.ResizeMethod)

        warped_X = tfa.image.dense_image_warp(X,T1,name='dense_image_warp_1')
        warped_X =tf.reshape(warped_X,(-1,self.n_input))
        
        warped_X1_2 = tfa.image.dense_image_warp(X1_2,T1_2,name='dense_image_warp_1_2')
        size=scale1_2[0]*scale1_2[1]*c
        warped_X1_2 =tf.reshape(warped_X1_2,(-1,size))
        
        warped_X1_4 = tfa.image.dense_image_warp(X1_4,T1_4,name='dense_image_warp_1_4')        
        size=scale1_4[0]*scale1_4[1]*c
        warped_X1_4 =tf.reshape(warped_X1_4,(-1,size))        
        
        T1_loss = tf.reduce_mean(tf.reduce_sum(tf.square( tf.abs(T1)),axis=1))
        T1_2loss = tf.reduce_mean(tf.reduce_sum(tf.square( tf.abs(T1_2)),axis=1))
        T1_4loss = tf.reduce_mean(tf.reduce_sum(tf.square( tf.abs(T1_4)),axis=1))
        
        return T1, warped_X, warped_X1_2, warped_X1_4, T1_loss, T1_2loss, T1_4loss

    def get_scale_losses(self,  warped_X1_4, warped_X1_2, warped_X):

        I_fixed=tf.reshape(self.fixed_image,(1,*self.img_size))
        
        _,h,w,c = I_fixed.shape        
        scale1_4=[(h//8)*2,(w//8)*2]
        scale1_2=[(h//8)*4,(w//8)*4]          
        
        I_fixed1_4 = tf.image.resize(I_fixed,scale1_4,method=self.ResizeMethod)
        I_fixed1_2 = tf.image.resize(I_fixed,scale1_2,method=self.ResizeMethod)
        
        size=scale1_4[0]*scale1_4[1]*c        
        I_fixed1_4 = tf.reshape(I_fixed1_4 ,(1,size))
        I_fixed1_4 =tf.cast(I_fixed1_4,tf.float32)
        diff = tf.abs(I_fixed1_4 -warped_X1_4)
        loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(diff),axis=1))
        
        size=scale1_2[0]*scale1_2[1]*c      
        I_fixed1_2 = tf.reshape(I_fixed1_2 ,(1,size))
        I_fixed1_2 =tf.cast(I_fixed1_2,tf.float32)
        diff = tf.abs(I_fixed1_2 -warped_X1_2)
        loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(diff),axis=1))
        
        I_fixed = tf.reshape(I_fixed,(1,self.n_input))
        I_fixed =tf.cast(I_fixed,tf.float32)
        diff = tf.abs(I_fixed -warped_X)
        loss3 = tf.reduce_mean(tf.reduce_sum(tf.square(diff),axis=1))
        
        return loss1, loss2, loss3 

    def get_loss(self):
        reg_loss = tf.reduce_mean(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        self.flow, warped_X, warped_X1_2, warped_X1_4, T1_loss, T1_2loss, T1_4loss = self.warp( self.X , self.z )
        self.warped_img = warped_X
        
        beta = 1
        T_loss = 1*T1_loss + beta*T1_2loss + beta*beta*T1_4loss
        loss1, loss2, loss3 = self.get_scale_losses( warped_X1_4, warped_X1_2, warped_X)
        beta = 0.25
        loss_1_2_3 = 1*loss1 + beta*loss2 + beta*beta*loss3
        warping_loss = 1*loss_1_2_3 + 1*T_loss
        t = tf.gather(self.t,self.indices)
        X_hat= self.net(t,self.W,self.b)
        self.X_hat_0=X_hat
        diff = tf.abs(warped_X - X_hat)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(diff, axis=1))       
        
        foreground = warped_X - X_hat
        foreground = tf.reshape(foreground,(-1,*self.img_size))
        warped_foreground = tfa.image.dense_image_warp(foreground,-1*self.flow,name='foreground_image_warp')
        self.warped_foreground =tf.reshape(warped_foreground,(-1,self.n_input))
               
        total_loss =  reg_loss + 1*warping_loss + 0.1*reconstruction_loss
        
        self.loss =  {'total_loss':total_loss, 'warping_loss':warping_loss,'loss_1_2_3 ':loss_1_2_3 , 'loss1':loss1,'loss2':loss2,'loss3':loss3,'T_loss':T_loss,'T1_loss':T1_loss,'T1_2loss':T1_2loss, 'T1_4loss':T1_4loss,'reconstruction_loss':reconstruction_loss,'reg_loss':reg_loss   }

        return self.loss,self.warped_img
