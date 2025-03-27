
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.keras.layers import Dense, Flatten, Softmax,Activation
from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.python.keras.layers import (BatchNormalization, Conv3D, Conv3DTranspose, Dropout,
                          Input, Layer, ReLU, Reshape, Softmax, concatenate,
                          merge)
from tensorflow.python.keras.losses import categorical_crossentropy
from .ac_layer import pyramidACBlock
import numpy as np
K = keras.backend


def CLS_loss(y_true, y_pred):
    cls_loss = categorical_crossentropy(y_true, y_pred)
    return cls_loss

def cross_entropy(y_true, y_pred):
    ce_loss = categorical_crossentropy(y_true, y_pred)
    return ce_loss

class T_SEG_module(object):
    def __init__(self, deploy=True):
        super().__init__()
        self.deploy = deploy
        self.bn_size = 4
        self.dense_conv = 3
        self.growth_rate = 16
        self.dropout_rate = 0.1
        self.compress_rate = 0.5
        self.num_init_features = 32
        self.conv_size = 3
        self.weight_decay = 0.0001
        self.images = Input(shape=[32, 32, 32, 1], name='input')
        self.num_classes = 4  # BK,CSF,GM,WM

    def switch_to_deploy(self):
        self.deploy = True

    def encoder(self, input_img):
        img = Conv3D(filters=self.num_init_features, kernel_size=3, strides=1, padding='same',
                     kernel_initializer=keras.initializers.he_normal(),
                     kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name='feature_conv')(input_img)
        img = BatchNormalization(name='feature_bn')(img)
        img = ReLU(name='feature_relu')(img)

        dense1 = self.dense_block(img, 'dense1')
        trans1 = self.transition_down(dense1, 'trans1')

        dense2 = self.dense_block(trans1, 'dense2')
        trans2 = self.transition_down(dense2, 'trans2')

        dense3 = self.dense_block(trans2, 'dense3')

        trans3 = self.transition_down(dense3, 'trans3')
        dense4 = self.dense_block(trans3, 'dense4')

        return dense1, dense2, dense3, dense4

    def decoder(self, dense1, dense2, dense3, dense4):
        trans4 = self.transition_up(dense4, 'trans4')

        # decode path
        concat1 = self.SkipConn(dense3, trans4, 'concat1')
        dense5 = self.dense_block(concat1, 'dense5')
        trans5 = self.transition_up(dense5, 'trans5')

        concat2 = self.SkipConn(dense2, trans5, 'concat2')
        dense6 = self.dense_block(concat2, 'dense6')
        trans6 = self.transition_up(dense6, 'trans6')

        concat3 = self.SkipConn(dense1, trans6, 'concat3')
        dense7 = self.dense_block(concat3, 'dense7')

        return dense7

    def build_model(self):
        enc = self.encoder(self.images)
        dec = self.decoder(*enc)

        seg = Conv3D(filters=self.num_classes, kernel_size=1, strides=1,
                     padding='same', dilation_rate=1, use_bias=False,
                     name='seg_conv')(dec)
        seg = Softmax(name='output')(seg)

        model = keras.Model(inputs=self.images, outputs=seg, name='T_SEG')
        return model

    def bottle_layer(self, x_in, out_channel, padding='same', use_bias=True, name='bottle'):
        x = Conv3D(filters=out_channel, kernel_size=1, strides=1,
                   padding=padding, use_bias=use_bias,
                   kernel_initializer=keras.initializers.he_normal(),
                   kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.conv0')(x_in)
        x = BatchNormalization(name=name + '.bn0')(x)
        x = ReLU(name=name + '.relu0')(x)
        return x

    def dense_layer(self, f_in, name):
        x = self.bottle_layer(f_in, self.bn_size * self.growth_rate, name=name + '.bottle')
        x = self._aconv(x, self.growth_rate, 1, name=name + '.aconv')
        x = Dropout(rate=self.dropout_rate, name=name + '.drop')(x)
        x = concatenate([f_in, x], name=name + '.cat')
        return x

    def dense_block(self, f_in, name='dense_block0'):
        x = f_in
        for i in range(self.dense_conv):
            x = self.dense_layer(x, name=name + '.denselayer{}'.format(i))
        return x

    def _aconv(self, x_in, out_channel, stride=1, name=None):
        x = pyramidACBlock(x_in, name + '.pyacb', out_channels=out_channel, strides=stride,
                           kernel_initializer=keras.initializers.he_normal(),
                           kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                           deploy=self.deploy)
        x = ReLU(name=name + '.relu')(x)
        return x

    def _deconv(self, x_in, out_channel, padding='same', name=None):
        x = Conv3DTranspose(filters=out_channel, kernel_size=self.conv_size, strides=2, padding=padding,
                            kernel_initializer=keras.initializers.he_normal(),
                            kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.convtrans')(
            x_in)
        x = BatchNormalization(name=name + '.bn')(x)
        x = ReLU(name=name + '.relu')(x)
        return x

    def transition_down(self, f_in, name='trans_down'):
        channels = f_in.get_shape()[-1]
        x = self._aconv(f_in, int(channels * self.compress_rate), 1, name=name + '.conv0')
        x = self._aconv(x, x.get_shape()[-1], 2, name=name + '.conv1')
        return x

    def transition_up(self, f_in, name='trans_up'):
        channels = f_in.get_shape()[-1]
        x = self._deconv(f_in, int(channels * self.compress_rate), name=name + '.deconv')
        return x

    def SkipConn(self, enc_f, dec_f, name='skip'):
        """
        f_in_1: the feature map from the encoder path
        f_in_2: the feature map from the decoder path
        """
        return concatenate([enc_f, dec_f], name=name)

class CLS_module(object):
    def __init__(self, deploy=True):  # deploy=False
        super().__init__()
        self.deploy = deploy
        self.bn_size = 4
        self.dense_conv = 3
        self.growth_rate = 16
        self.dropout_rate = 0.1
        self.compress_rate = 0.5
        self.num_init_features = 32
        self.conv_size = 3
        self.weight_decay = 0.0001
        self.images = Input(shape=[32, 32, 32, 1], name='input')
        self.num_classes = 1

    def switch_to_deploy(self):
        self.deploy = True

    def encoder(self, input_img):
        img = Conv3D(filters=self.num_init_features, kernel_size=3, strides=1, padding='same',
                     kernel_initializer=keras.initializers.he_normal(),
                     kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name='feature_conv')(input_img)
        img = BatchNormalization(name='feature_bn')(img)
        img = ReLU(name='feature_relu')(img)

        # encode path
        dense1 = self.dense_block(img, 'dense1')  # 80 32*32*32
        trans1 = self.transition_down(dense1, 'trans1')  # 40

        dense2 = self.dense_block(trans1, 'dense2')  # 88 16*16*16
        trans2 = self.transition_down(dense2, 'trans2')  # 44

        dense3 = self.dense_block(trans2, 'dense3')  # 92 8*8*8

        trans3 = self.transition_down(dense3, 'trans3')  # 46
        dense4 = self.dense_block(trans3, 'dense4')  # 94 4*4*4

        gap_tensor = self.gap_block(dense4)# 94 1*1*1
        tensor_flatten = self.Flat(gap_tensor)#94 1(94,1)
        x = self.d1(tensor_flatten)#(10,1)
        x = self.d2(x)#(2,1)
        x = self.softmax(x)
        return x

    def build_model(self):
        enc = self.encoder(self.images)
        model = keras.Model(inputs=self.images, outputs=enc, name='CLS')
        return model

    def bottle_layer(self, x_in, out_channel, padding='same', use_bias=True, name='bottle'):
        x = Conv3D(filters=out_channel, kernel_size=1, strides=1,
                   padding=padding, use_bias=use_bias,
                   kernel_initializer=keras.initializers.he_normal(),
                   kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.conv0')(x_in)
        x = BatchNormalization(name=name + '.bn0')(x)
        x = ReLU(name=name + '.relu0')(x)
        return x

    def dense_layer(self, f_in, name):
        x = self.bottle_layer(f_in, self.bn_size * self.growth_rate, name=name + '.bottle')
        x = self._aconv(x, self.growth_rate, 1, name=name + '.aconv')
        x = Dropout(rate=self.dropout_rate, name=name + '.drop')(x)
        x = concatenate([f_in, x], name=name + '.cat')
        return x

    def dense_block(self, f_in, name='dense_block0'):
        x = f_in
        for i in range(self.dense_conv):
            x = self.dense_layer(x, name=name + '.denselayer{}'.format(i))
        return x

    def gap_block(self, f_in, name="GAP_lol"):
        x = f_in
        layer = tf.keras.layers.AveragePooling3D(pool_size=4, name=name)
        x = layer(x)
        return x

    def Flat(self, f_in):
        x = f_in
        layer = Flatten()
        x = layer(x)
        return x

    def d1(self, f_in):
        x = f_in
        layer = Dense(10, activation='relu')
        x = layer(x)
        return x

    def d2(self, f_in):
        x = f_in
        # layer = Dense(2, activation='softmax')
        layer = Dense(2)
        x = layer(x)
        return x

    def softmax(self, f_in):
        x = f_in
        layer = Softmax(name='output')
        x = layer(x)
        return x

    def _aconv(self, x_in, out_channel, stride=1, name=None):
        x = pyramidACBlock(x_in, name + '.pyacb', out_channels=out_channel, strides=stride,
                           kernel_initializer=keras.initializers.he_normal(),
                           kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                           deploy=self.deploy)
        x = ReLU(name=name + '.relu')(x)
        return x

    def _deconv(self, x_in, out_channel, padding='same', name=None):
        x = Conv3DTranspose(filters=out_channel, kernel_size=self.conv_size, strides=2, padding=padding,
                            kernel_initializer=keras.initializers.he_normal(),
                            kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.convtrans')(
            x_in)
        x = BatchNormalization(name=name + '.bn')(x)
        x = ReLU(name=name + '.relu')(x)
        return x

    def transition_down(self, f_in, name='trans_down'):
        channels = f_in.get_shape()[-1]  # .value
        x = self._aconv(f_in, int(channels * self.compress_rate), 1, name=name + '.conv0')
        x = self._aconv(x, x.get_shape()[-1], 2, name=name + '.conv1')  # .value
        # x = self._aconv(x, x.get_shape()[-1], 2, name=name+'.conv1')
        return x

class CMG_module(object):
    def __init__(self, deploy=True):
        super().__init__()
        self.deploy = deploy
        self.bn_size = 4
        self.dense_conv = 3
        self.growth_rate = 16
        self.dropout_rate = 0.1
        self.compress_rate = 0.5
        self.num_init_features = 32#32
        self.conv_size = 3
        self.weight_decay = 0.0001
        self.images = Input(shape=[32, 32, 32, 1], name='input')
        self.mask = Input(shape=[32, 32, 32, 1], name='input_mask')#Focus the counterfactual map on the foreground of the image
        self.num_classes = 1

        #The number of channels is 2 because there are two "swich" states
        self.c1 = Input(shape=[32, 32, 32, 2], name='c1')
        self.c2 = Input(shape=[16, 16, 16, 2], name='c2')
        self.c3 = Input(shape=[8, 8, 8, 2], name='c3')

    def concat(self, x, y, n):
        return concatenate([x, y], name=n + "_concat_target_c")
    def conv_bn_relu(self, input_img,filters,name):
        img = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same',
                     kernel_initializer=keras.initializers.he_normal(),
                     kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name+'_conv')(input_img)
        img = BatchNormalization(name=name+'_bn')(img)
        img = ReLU(name=name+'_relu')(img)

        return img
    def trans_up(self, input_img,name):
        channels = input_img.get_shape()[-1]
        img = Conv3DTranspose(filters=channels, kernel_size=self.conv_size, strides=2, padding='same',
                        kernel_initializer=keras.initializers.he_normal(),
                        kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.trans')(input_img)

        return img

    def max_pool_block(self, f_in,name):
        x = f_in
        layer = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2),name=name+'_max_pool')
        x = layer(x)
        return x
    def up(self, x_in, out_channel, padding='same', name=None):
        x = Conv3DTranspose(filters=out_channel, kernel_size=self.conv_size, strides=2, padding=padding,
                            kernel_initializer=keras.initializers.he_normal(),
                            kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.convtrans')(
            x_in)


    def encoder(self, input_img):
        img32 = self.conv_bn_relu(input_img, self.num_init_features, 'enc32_1')
        img32 = self.conv_bn_relu(img32, 16, 'enc32_2')

        img16 = self.max_pool_block(img32, 'enc32_to_16')
        img16 = self.conv_bn_relu(img16, 32, 'enc16_1')
        img16 = self.conv_bn_relu(img16, 64, 'enc16_2')

        img8 = self.max_pool_block(img16, 'enc16_to_8')
        img8 = self.conv_bn_relu(img8, 64, 'enc8_1')
        img8 = self.conv_bn_relu(img8, 128, 'enc8_2')

        img4 = self.max_pool_block(img8, 'enc8_to_4')
        img4 = self.conv_bn_relu(img4, 128, 'enc4_1')
        img4 = self.conv_bn_relu(img4, 256, 'enc4_2')

        return img32, img16, img8, img4

    def decoder(self, enc32, enc16, enc8, enc4):
        dec8 = self.trans_up(enc4, 'enc4')
        # decode path
        concat8 = self.SkipConn(enc8, dec8, 'concat8')
        concat8 = self.concat(concat8, self.c3, 'c3')
        dec8 = self.conv_bn_relu(concat8, 128, 'dec8_1')
        dec8 = self.conv_bn_relu(dec8, 64, 'dec8_2')

        dec16 = self.trans_up(dec8, 'dec8')
        concat16 = self.SkipConn(enc16, dec16, 'concat16')
        concat16 = self.concat(concat16, self.c2, 'c2')
        dec16 = self.conv_bn_relu(concat16, 32, 'dec16_1')
        dec16 = self.conv_bn_relu(dec16, 16, 'dec16_2')

        dec32 = self.trans_up(dec16, 'dec16')
        concat32 = self.SkipConn(enc32, dec32, 'concat32')
        concat32 = self.concat(concat32, self.c1, 'c1')
        dec32 = self.conv_bn_relu(concat32, 8, 'dec32_1')
        dec32 = self.conv_bn_relu(dec32, 4, 'dec32_2')

        return dec32

    def build_model(self):
        enc = self.encoder(self.images)
        dec = self.decoder(*enc)
        seg = Conv3D(filters=self.num_classes, kernel_size=1, strides=1,
                     padding='same', dilation_rate=1, use_bias=False,
                     name='seg_conv')(dec)
        seg = ReLU(name='end_relu')(seg)
        seg = seg * self.mask

        model = keras.Model({"input": self.images, "mask": self.mask, "c1": self.c1, "c2": self.c2, "c3": self.c3},
                                     {"CF_output": seg}, name="CMG")
        return model

    def gap_block(self, f_in,name="GAP_lol"):
        x = f_in
        layer = tf.keras.layers.AveragePooling3D(pool_size=4,name=name)
        x = layer(x)
        return x
    def Flat(self, f_in):
        x = f_in
        layer = Flatten()
        x = layer(x)
        return x
    def d1(self, f_in):
        x = f_in
        layer = Dense(10, activation='relu')
        x = layer(x)
        return x

    def d2(self, f_in):
        x = f_in
        layer = Dense(2)
        x = layer(x)
        return x

    def softmax(self, f_in):
        x = f_in
        layer = Softmax(name='output')
        x = layer(x)
        return x

    def bottle_layer(self, x_in, out_channel, padding='same', use_bias=True, name='bottle'):
        x = Conv3D(filters=out_channel, kernel_size=1, strides=1,
                   padding=padding, use_bias=use_bias,
                   kernel_initializer=keras.initializers.he_normal(),
                   kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.conv0')(x_in)
        x = BatchNormalization(name=name + '.bn0')(x)
        x = ReLU(name=name + '.relu0')(x)
        return x

    def dense_layer(self, f_in, name):
        x = self.bottle_layer(f_in, self.bn_size * self.growth_rate, name=name + '.bottle')
        x = self._aconv(x, self.growth_rate, 1, name=name + '.aconv')
        x = Dropout(rate=self.dropout_rate, name=name + '.drop')(x)
        x = concatenate([f_in, x], name=name + '.cat')
        return x

    def dense_block(self, f_in, name='dense_block0'):
        x = f_in
        for i in range(self.dense_conv):
            x = self.dense_layer(x, name=name + '.denselayer{}'.format(i))
        return x

    def _aconv(self, x_in, out_channel, stride=1, name=None):
        x = pyramidACBlock(x_in, name + '.pyacb', out_channels=out_channel, strides=stride,
                           kernel_initializer=keras.initializers.he_normal(),
                           kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                           deploy=self.deploy)
        x = ReLU(name=name + '.relu')(x)
        return x

    def _deconv(self, x_in, out_channel, padding='same', name=None):
        x = Conv3DTranspose(filters=out_channel, kernel_size=self.conv_size, strides=2, padding=padding,
                            kernel_initializer=keras.initializers.he_normal(),
                            kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.convtrans')(
            x_in)
        x = BatchNormalization(name=name + '.bn')(x)
        x = ReLU(name=name + '.relu')(x)
        return x

    def transition_down(self, f_in, name='trans_down'):
        channels = f_in.get_shape()[-1]  # .value
        x = self._aconv(f_in, int(channels * self.compress_rate), 1, name=name + '.conv0')
        x = self._aconv(x, x.get_shape()[-1], 2, name=name + '.conv1')
        return x

    def transition_up(self, f_in, name='trans_up'):
        channels = f_in.get_shape()[-1]
        x = self._deconv(f_in, int(channels * self.compress_rate), name=name + '.deconv')
        return x

    def SkipConn(self, enc_f, dec_f, name='skip'):
        """
        f_in_1: the feature map from the encoder path
        f_in_2: the feature map from the decoder path
        """
        return concatenate([enc_f, dec_f], name=name)

class P_SEG_module(object):
    """input T1 + P_seg + CMG"""
    def __init__(self, deploy=True):
        super().__init__()
        self.deploy = deploy
        self.bn_size = 4
        self.dense_conv = 3
        self.growth_rate = 16
        self.dropout_rate = 0.1
        self.compress_rate = 0.5
        self.num_init_features = 32
        self.conv_size = 3
        self.weight_decay = 0.0001
        self.images = Input(shape=[32, 32, 32, 6], name='input')
        self.num_classes = 1

    def switch_to_deploy(self):
        self.deploy = True

    def encoder(self, input_img):
        img = Conv3D(filters=self.num_init_features, kernel_size=3, strides=1, padding='same',
                     kernel_initializer=keras.initializers.he_normal(),
                     kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name='feature_conv')(input_img)
        img = BatchNormalization(name='feature_bn')(img)
        img = ReLU(name='feature_relu')(img)

        # encode path
        dense1 = self.dense_block(img, 'dense1')
        trans1 = self.transition_down(dense1, 'trans1')

        dense2 = self.dense_block(trans1, 'dense2')
        trans2 = self.transition_down(dense2, 'trans2')

        dense3 = self.dense_block(trans2, 'dense3')

        trans3 = self.transition_down(dense3, 'trans3')
        dense4 = self.dense_block(trans3, 'dense4')

        return dense1, dense2, dense3, dense4

    def decoder(self, dense1, dense2, dense3, dense4):
        trans4 = self.transition_up(dense4, 'trans4')

        # decode path
        concat1 = self.SkipConn(dense3, trans4, 'concat1')
        dense5 = self.dense_block(concat1, 'dense5')
        trans5 = self.transition_up(dense5, 'trans5')

        concat2 = self.SkipConn(dense2, trans5, 'concat2')
        dense6 = self.dense_block(concat2, 'dense6')
        trans6 = self.transition_up(dense6, 'trans6')

        concat3 = self.SkipConn(dense1, trans6, 'concat3')
        dense7 = self.dense_block(concat3, 'dense7')

        return dense7

    def output(self, f_in, name="output"):
        x = f_in

        return x

    def build_model(self):
        enc = self.encoder(self.images)
        dec = self.decoder(*enc)

        seg = Conv3D(filters=self.num_classes, kernel_size=1, strides=1,
                     padding='same', dilation_rate=1, use_bias=False,
                     name='seg_conv')(dec)

        seg = Activation(tf.nn.sigmoid, name='output')(seg)
        model = keras.Model(inputs=self.images, outputs=seg, name='P_SEG')
        return model

    def gap_block(self, f_in, name="GAP_lol"):
        x = f_in
        layer = tf.keras.layers.AveragePooling3D(pool_size=4, name=name)
        x = layer(x)
        return x

    def Flat(self, f_in):
        x = f_in
        layer = Flatten()
        x = layer(x)
        return x

    def d1(self, f_in):
        x = f_in
        layer = Dense(10, activation='relu')
        x = layer(x)
        return x

    def d2(self, f_in):
        x = f_in
        # layer = Dense(2, activation='softmax')
        layer = Dense(2)
        x = layer(x)
        return x

    def softmax(self, f_in):
        x = f_in
        layer = Softmax(name='output')
        x = layer(x)
        return x

    def bottle_layer(self, x_in, out_channel, padding='same', use_bias=True, name='bottle'):
        x = Conv3D(filters=out_channel, kernel_size=1, strides=1,
                   padding=padding, use_bias=use_bias,
                   kernel_initializer=keras.initializers.he_normal(),
                   kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name + '.conv0')(x_in)
        x = BatchNormalization(name=name + '.bn0')(x)
        x = ReLU(name=name + '.relu0')(x)
        return x

    def dense_layer(self, f_in, name):
        x = self.bottle_layer(f_in, self.bn_size * self.growth_rate, name=name + '.bottle')
        x = self._aconv(x, self.growth_rate, 1, name=name + '.aconv')
        x = Dropout(rate=self.dropout_rate, name=name + '.drop')(x)
        x = concatenate([f_in, x], name=name + '.cat')
        return x

    def dense_block(self, f_in, name='dense_block0'):
        x = f_in
        for i in range(self.dense_conv):
            x = self.dense_layer(x, name=name + '.denselayer{}'.format(i))
        return x

    def _aconv(self, x_in, out_channel, stride=1, name=None):
        x = pyramidACBlock(x_in, name + '.pyacb', out_channels=out_channel, strides=stride,
                           kernel_initializer=keras.initializers.he_normal(),
                           kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                           deploy=self.deploy)
        x = ReLU(name=name + '.relu')(x)
        return x

    def _deconv(self, x_in, out_channel, padding='same', name=None):
        x = Conv3DTranspose(filters=out_channel, kernel_size=self.conv_size, strides=2, padding=padding,
                            kernel_initializer=keras.initializers.he_normal(),
                            kernel_regularizer=keras.regularizers.l2(l=self.weight_decay),
                            name=name + '.convtrans')(
            x_in)
        x = BatchNormalization(name=name + '.bn')(x)
        x = ReLU(name=name + '.relu')(x)
        return x

    def transition_down(self, f_in, name='trans_down'):
        channels = f_in.get_shape()[-1]  # .value
        x = self._aconv(f_in, int(channels * self.compress_rate), 1, name=name + '.conv0')
        x = self._aconv(x, x.get_shape()[-1], 2, name=name + '.conv1')  # .value
        # x = self._aconv(x, x.get_shape()[-1], 2, name=name+'.conv1')
        return x

    def transition_up(self, f_in, name='trans_up'):
        channels = f_in.get_shape()[-1]  # .value
        x = self._deconv(f_in, int(channels * self.compress_rate), name=name + '.deconv')
        return x

    def SkipConn(self, enc_f, dec_f, name='skip'):
        """
        f_in_1: the feature map from the encoder path
        f_in_2: the feature map from the decoder path
        """
        return concatenate([enc_f, dec_f], name=name)




