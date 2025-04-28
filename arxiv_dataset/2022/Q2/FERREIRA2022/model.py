import tensorflow as tf
from tensorflow.keras.layers import (Activation, Input, Dense, 
                                    Conv2D, MaxPooling2D, 
                                    BatchNormalization, Dropout, 
                                    GlobalAveragePooling2D)

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation

class MonteCarloDropout(tf.keras.layers.Dropout):

    def call(self, inputs):
        return super().call(inputs, training=True)

class ConvBlock(tf.keras.layers.Layer):
        
    def __init__(self, filters, 
                 number_conv_per_block=2, 
                 l2_regularization=0.1, 
                 kernel_size=3, 
                 name='Conv2DBlock', 
                 **kwargs):

        super(ConvBlock, self).__init__(name=name, **kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.number_conv_per_block = number_conv_per_block
        self.l2_regularization = l2_regularization
        
        self.conv2d = [Conv2D(self.filters, self.kernel_size, kernel_regularizer=l2(l2_regularization), 
                              bias_regularizer=l2(l2_regularization), 
                              padding='valid') for n in range(self.number_conv_per_block)]
        
        self.act_fn = [Activation("relu") for n in range(self.number_conv_per_block)]
        self.batch_norm = [BatchNormalization() for n in range(self.number_conv_per_block)]
      
    def call(self, x):
        for i in range(self.number_conv_per_block):
            x = self.conv2d[i](x)
            x = self.act_fn[i](x)
            x = self.batch_norm[i](x)
        return x



def FERREIRA2020Net(num_classes=2,
                    input_shape=(128, 128, 1), 
                    conv_blocks=3, 
                    conv_per_block=2,
                    fn=32,
                    ks=11,
                    fc_layers=2,
                    fc_layers_size=128,
                    l2_reg=0.1,
                    dropout=0.5,
                    random_rotations=True):


    inputs = Input(shape=input_shape)

    x = RandomRotation((-0.1, 0.1))(inputs)

    for n in range(conv_blocks):
        x = ConvBlock(fn, kernel_size=ks, 
                      number_conv_per_block=conv_per_block, 
                      l2_regularization=l2_reg, 
                      name=f'ConvBlock-{n}')(x)

        if ks > 3: ks -= 2
        
        fn *= 2
        if n < conv_blocks-1:
            x = MaxPooling2D((2,2), padding='same')(x)
        
    x = GlobalAveragePooling2D()(x)
        
    for fcl in range(fc_layers):
        x = Dense(fc_layers_size)(x)
        x = MonteCarloDropout(dropout)(x)
        x = Activation('relu')(x)

    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='FERREIRA2020_class')

    return model