from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import keras
import keras.backend as K
import keras.layers as KL
from keras.layers import Input, Flatten

import keras.engine as KE
import keras.models as KM
import tensorflow.keras.layers as E
import tensorflow.contrib.slim as slim

from keras.models import Model
from keras.layers import Concatenate , Lambda, Activation, Conv2D, MaxPooling2D, Add, Input, BatchNormalization, UpSampling2D, Concatenate , Dense 
from keras.layers.merge import concatenate, add
from keras.regularizers import l2
from PIL import Image


Backbone = "resnet101"
TRAIN_BN = "True"
TOP_DOWN_PYRAMID_SIZE = 256


def Blender_discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(Blender_conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(Blender_conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(Blender_conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(Blender_conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = Blender_conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)


        return h4

def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, ks=4, s=2 ,name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, ks=4, s=2 ,name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, ks=4, s=2 ,name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, ks=4, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, ks=4, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)

        return h4


def generator_unet(image, options, reuse=False, name="generator"):

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        # # ////////////////////////////////////////////////
        # e9 = instance_norm(conv2d(lrelu(e8), options.gf_dim*8, name='g_e9_conv'), 'g_bn_e9')
        # # e7 is (2 x 2 x self.gf_dim*8)
        # e10 = instance_norm(conv2d(lrelu(e9), options.gf_dim*8, name='g_e10_conv'), 'g_bn_e10')
        # # e8 is (1 x 1 x self.gf_dim*8)


        # d10 = deconv2d(tf.nn.relu(e10), options.gf_dim*8, name='g_d10')
        # d10 = tf.nn.dropout(d10, dropout_rate)
        # d10 = tf.concat([instance_norm(d10, 'g_bn_d10'), e9], 3)
        # # d1 is (2 x 2 x self.gf_dim*8*2)

        # d9 = deconv2d(tf.nn.relu(d10), options.gf_dim*8, name='g_d9')
        # d9 = tf.nn.dropout(d9, dropout_rate)
        # d9 = tf.concat([instance_norm(d9, 'g_bn_d9'), e8], 3)
        # # d2 is (4 x 4 x self.gf_dim*8*2)

        # # ///////////////////////////////////////////

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred

def Blender_generator_resnet(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            # p = int((ks - 1) / 2)
            p = 0
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(Blender_conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(Blender_conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # c00 = Conv2D(64 , (7, 7), strides=(1, 1), padding='valid', name='FC1')(c0)
        c00 = tf.nn.relu(instance_norm(Blender_conv2d(c0, options.gf_dim, 3, 2, name='g_e1_c'), 'g_e1_bn'))
        c1 = tf.nn.relu(instance_norm(c00))
        c2 = tf.nn.relu(instance_norm(Blender_conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(Blender_conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))  
        #    Output of C3 is 262*262  

        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')
        #    Output of C3 is 262*262  

        d1 = Blender_deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        #    Output of D1 is 262*262  

        d2 = Blender_deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        #    Output of D2 is 262*262  

        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        #    Output of D2 is 268*268  

        pred = tf.nn.tanh(Blender_conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))
        #    Output of pred is 268*268  
        output = tf.image.crop_to_bounding_box(pred , 3,3,256,256)

        return output


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def blender(image, Object ,  options, reuse=False, name="blender"):
    # /////// this model returns the feature maps of object and the target image ///////////
    Image_Features , Object_Features  = resnext_fpn(image , Object , 64)
    x = FPN_Header(Image_Features , Object_Features )
    return x

def Transparent(img):
    # ////// img size is    ?,256,256,3
    im = img
    # im = tf.dtypes.cast(im, tf.int32)
    image = tf.add(im[0,:,:,0],im[0,:,:,1])
    image = tf.add(image,im[0, :,:,2])
    #  Shape is  ?*256*256 !!!!

    #///////////////////  Convert black points to 0 and white points to 1 ////////////////
    image_mask = tf.equal(image, 3.)
    NonInv_image_mask = tf.dtypes.cast(image_mask , tf.int32)
    NonInv_image_mask = tf.dtypes.cast(NonInv_image_mask , tf.float32)
    #///////// Size of Image_mask is   ?,256,256

    #////////////////// This part convert the mask to an image ////////////
    image_mask = tf.logical_not(image_mask)
    Inv_image_mask = tf.dtypes.cast(image_mask , tf.int32)
    Inv_image_mask = tf.dtypes.cast(Inv_image_mask , tf.float32)


    # a = tf.dtypes.cast(tf.constant(0, shape=(256, 256)), tf.float32)
    # b = tf.dtypes.cast(tf.constant(1, shape=(256, 256)), tf.float32)
    # c = tf.dtypes.cast(tf.constant(1, shape=(256, 256)), tf.float32)
    # d = tf.stack([image_mask ,image_mask , a])
    # image_mask = tf.expand_dims(d, axis = 0)
    # image_mask = tf.transpose(image_mask , (0, 2,3,1))


    #     ///////////////////// Coverting white background to black /////////////////
    im = tf.transpose(im , (0,3,1,2))
    image = tf.multiply(Inv_image_mask,im[:,:])
    image = tf.transpose(image , (0,2,3,1))

    return (image , NonInv_image_mask )


def euclidean_distance(x , y):
    # x , y = val
    return (x,y)

def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1, shape2)


def Embed_Images(im1 , mask , im2 , Parameters, options,):
    # # /////////////////////////////////////////////////////////////////////////////////////////////////////////
    # # This function get the object , it's mask and thetarget images
    # # it place the object on the target image respect to the cordinate data that has been sent 
    # # ////////// This function get first images and it's binary mask and the seconf image to merge them togetehr ////////
    # # //////////////////////////////////////////////////////////////////////////////////////////////////////    
    # ///////////////////////////////////

    ratio= Parameters[0]
    x= tf.cast(Parameters[1], tf.float32)
    y= tf.cast(Parameters[2], tf.float32)

    Target_V = int(im2.get_shape()[1])
    Target_H = int(im2.get_shape()[2])
        
    V = int(im1.get_shape()[1])
    H = int(im1.get_shape()[2])
    Aspect_ratio = H // V

    V = tf.cast(ratio * V, tf.int32)
    H = tf.cast( V * Aspect_ratio, tf.int32)

    newsize = [V, H]

    mask = tf.expand_dims(mask, axis = 0)
    mask = tf.expand_dims(mask, axis = 3)

    padded_mask = tf.ones_like(mask)

    mask = tf.image.resize( mask , newsize) 
    mask = tf.transpose (mask , (0,3,1,2))

    im1 = tf.image.resize( im1, newsize) 

    im1 = tf.transpose (im1 , (0,3,1,2))

    padded_im1 = tf.zeros_like(im2)
    padded_im1 = tf.transpose (padded_im1 , (0,3,1,2))

    top_pad = tf.cast(tf.multiply(tf.cast((Target_V - V),  tf.float32)  , y) ,  tf.int32)
    button_pad = (Target_V - V) - top_pad

    lef_pad = tf.cast(tf.multiply(tf.cast((Target_H - H) ,  tf.float32) ,  x ) ,  tf.int32)
    right_pad = (Target_H - H - lef_pad)

    Padding = [[0 , 0],[0 , 0],[top_pad , button_pad ],[lef_pad  , right_pad]]
    padded_im1 = tf.pad(im1[:,:], Padding, 'constant')
    padded_im1 = tf.transpose (padded_im1 , (0,2,3,1))
    padded_im1 = tf.image.resize( padded_im1 , [256,256]) 
    padded_im1 = tf.transpose (padded_im1 , (0,3,1,2))

    padded_mask = tf.transpose (padded_mask , (0,3,1,2))
    padded_mask = tf.pad(mask[:,:], Padding , 'constant' , constant_values=1 )
    padded_mask = tf.transpose (padded_mask , (0,2,3,1))
    padded_mask = tf.image.resize( padded_mask[:,:] , [256,256]) 
    padded_mask = tf.transpose (padded_mask , (0,3,1,2))
    padded_mask = tf.squeeze(padded_mask, axis=1)
    padded_mask = tf.squeeze(padded_mask, axis=0)
    padded_mask  = tf.cast(padded_mask, tf.float32)


    im2 = tf.transpose(im2 , (0,3,1,2))
    image = tf.multiply(padded_mask,im2[:,:])
    image = tf.add(padded_im1,image)
    image = tf.transpose(image , (0,2,3,1))

    # image = Mask_To_Image(padded_mask)

    return (image)


def Mask_To_Image(Mask):
    a = tf.dtypes.cast(tf.constant(0, shape=(256, 256)), tf.float32)
    b = tf.dtypes.cast(tf.constant(0, shape=(256, 256)), tf.float32)
    # c = tf.dtypes.cast(tf.constant(1, shape=(256, 256)), tf.float32)
    d = tf.stack([Mask ,a , b])
    image = tf.expand_dims(d, axis = 0)
    image = tf.transpose(image , (0, 2,3,1))
    return image

def Object_blender(Parameters, real, Object, options):
    image , mask = Transparent(Object)
    image = Embed_Images(image , mask , real , Parameters , options)
    # image = tf.nn.relu(instance_norm(conv2d(image, options.gf_dim, 1, 1, padding='SAME', name='g_e1_c'), 'g_e1_bn'))
    return  tf.cast(image, tf.float32)

def FPN_Header(Image_Feature_maps , Object_Feature_maps):
    batch_momentum=0.9
    bn_axis = 3

    print("Feature map is equal yo ", Image_Feature_maps)
    print("Object Feature map is equal yo ", Object_Feature_maps)

    x = concatenate([Image_Feature_maps, Object_Feature_maps],axis=3)

    # # /////////////////////////////    Image featuress /////////////////////////////////////
    # x11 = Conv2D(12 , (256, 256), strides=(1, 1), padding='valid', name='FCcccc12')(Object_Feature_maps)
    # x11 = BatchNormalization(axis=bn_axis, name='bn_conv12', momentum=batch_momentum)(x11)
    # x11 = Activation('relu')(x11)

    # x11 = Conv2D(12 , (1, 1), padding='valid', name='FCffff22')(x11)
    # x11 = BatchNormalization(axis=bn_axis, name='bn_conv12', momentum=batch_momentum)(x11)
    # x11 = Activation('relu')(x11)

    # shared1 = KL.Lambda(lambda x: K.squeeze(K.squeeze(x11, 2), 1),
    #                    name="pool_squeeze2")(x11)

    # /////////////////////////////    Image featuress /////////////////////////////////////
    x = Conv2D(12 , (256, 256), strides=(1, 1), padding='valid', name='FCcccc1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(12 , (1, 1), padding='valid', name='FCffff2')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)


    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 2), 1),
                       name="pool_squeeze")(x)


    # First_Layer = shared1
    # Second_Layer = shared
    # Result = Concatenate([x11, x])

    # Regressor
    # Blender_Parameters = Dense( 3 , activation='linear')(shared)
    Blender_Parameters = Dense( 3 , activation='sigmoid')(shared)

    print("4444444444444CCCCCCCCCCXXXXXXXXXXXXXXXX s s " , Blender_Parameters)

    return Blender_Parameters


def resnext_fpn(input_tensor , Object , nb_labels, depth=(3, 4, 6, 3), cardinality=32, width=4, weight_decay=5e-4, batch_norm=True,
                batch_momentum=0.9):
    """
    TODO: add dilated convolutions as well
    Resnext-50 is defined by (3, 4, 6, 3) [default]
    Resnext-101 is defined by (3, 4, 23, 3)
    Resnext-152 is defined by (3, 8, 23, 3)
    :param input_shape:
    :param nb_labels:
    :param depth:
    :param cardinality:
    :param width:
    :param weight_decay:
    :param batch_norm:
    :param batch_momentum:
    :return:
    """
# //////////////////////////////////////     Image feature map    ////////////////////////////////////////////////////
    # nb_rows, nb_cols, _ = input_shape
    nb_rows = 256
    nb_cols = 256

    # input_tensor = Input(shape=input_shape)

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input_tensor)
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    stage_1 = x

    # filters are cardinality * width * 2 for each depth level
    for i in range(depth[0]):
        x = bottleneck_block(x, 128, cardinality, strides=1, weight_decay=weight_decay)
    stage_2 = x

    # this can be done with a for loop but is more explicit this way
    x = bottleneck_block(x, 256, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[1]):
        x = bottleneck_block(x, 256, cardinality, strides=1, weight_decay=weight_decay)
    stage_3 = x

    x = bottleneck_block(x, 512, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[2]):
        x = bottleneck_block(x, 512, cardinality, strides=1, weight_decay=weight_decay)
    stage_4 = x

    x = bottleneck_block(x, 1024, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[3]):
        x = bottleneck_block(x, 1024, cardinality, strides=1, weight_decay=weight_decay)
    stage_5 = x

    P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(stage_5)
    P4 = Add(name="fpn_p4add")([UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4', padding='same')(stage_4)])
    P3 = Add(name="fpn_p3add")([UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(stage_3)])
    P2 = Add(name="fpn_p2add")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2', padding='same')(stage_2)])
    # Attach 3x3 conv to all P layers to get the final feature maps. --> Reduce aliasing effect of upsampling
    P2 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)

    head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv")(P2)
    head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv_2")(head1)

    head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv")(P3)
    head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv_2")(head2)

    head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv")(P4)
    head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv_2")(head3)

    head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv")(P5)
    head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv_2")(head4)

    f_p2 = UpSampling2D(size=(8, 8), name="pre_cat_2")(head4)
    f_p3 = UpSampling2D(size=(4, 4), name="pre_cat_3")(head3)
    f_p4 = UpSampling2D(size=(2, 2), name="pre_cat_4")(head2)
    f_p5 = head1

    x = Concatenate(axis=-1)([f_p2, f_p3, f_p4, f_p5])
    x = Conv2D(nb_labels, (3, 3), padding="SAME", name="final_conv", kernel_initializer='he_normal',
               activation='linear')(x)
    x = UpSampling2D(size=(4, 4), name="final_upsample")(x)
    x = Activation('sigmoid')(x)


# //////////////////////////////////////     Object feature map    ////////////////////////////////////////////////////

    x1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv12', kernel_regularizer=l2(weight_decay))(Object)
    if batch_norm:
        x1 = BatchNormalization(axis=bn_axis, name='bn_conv122', momentum=batch_momentum)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x1)
    stage_1 = x1

    # filters are cardinality * width * 2 for each depth level
    for i in range(depth[0]):
        x1 = bottleneck_block(x1, 128, cardinality, strides=1, weight_decay=weight_decay)
    stage_2 = x1

    # this can be done with a for loop but is more explicit this way
    x1 = bottleneck_block(x1, 256, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[1]):
        x1 = bottleneck_block(x1, 256, cardinality, strides=1, weight_decay=weight_decay)
    stage_3 = x1

    x1 = bottleneck_block(x1, 512, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[2]):
        x1 = bottleneck_block(x1, 512, cardinality, strides=1, weight_decay=weight_decay)
    stage_4 = x1

    x1 = bottleneck_block(x1, 1024, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[3]):
        x1 = bottleneck_block(x1, 1024, cardinality, strides=1, weight_decay=weight_decay)
    stage_5 = x1

    P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p52')(stage_5)
    P4 = Add(name="fpn_p4add2")([UpSampling2D(size=(2, 2), name="fpn_p5upsampled2")(P5),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p42', padding='same')(stage_4)])
    P3 = Add(name="fpn_p3add2")([UpSampling2D(size=(2, 2), name="fpn_p4upsampled2")(P4),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p32')(stage_3)])
    P2 = Add(name="fpn_p2add2")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled2")(P3),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p22', padding='same')(stage_2)])
    # Attach 3x3 conv to all P layers to get the final feature maps. --> Reduce aliasing effect of upsampling
    P2 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p22")(P2)
    P3 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p32")(P3)
    P4 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p42")(P4)
    P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p52")(P5)

    head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv2")(P2)
    head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv_22")(head1)

    head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv2")(P3)
    head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv_22")(head2)

    head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv2")(P4)
    head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv_22")(head3)

    head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv2")(P5)
    head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv_22")(head4)

    f_p2 = UpSampling2D(size=(8, 8), name="pre_cat_22")(head4)
    f_p3 = UpSampling2D(size=(4, 4), name="pre_cat_32")(head3)
    f_p4 = UpSampling2D(size=(2, 2), name="pre_cat_42")(head2)
    f_p5 = head1

    x1 = Concatenate(axis=-1)([f_p2, f_p3, f_p4, f_p5])
    x1 = Conv2D(nb_labels, (3, 3), padding="SAME", name="final_conv2", kernel_initializer='he_normal',
               activation='linear')(x1)
    x1 = UpSampling2D(size=(4, 4), name="final_upsample2")(x1)
    x1 = Activation('sigmoid')(x1)



    # output = [x , x1]

    output = x
    # model = Model(input_tensor, x)

    return x , x1


def grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = input
    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        group_list.append(x)

    group_merge = concatenate(group_list, axis=3)
    x = BatchNormalization(axis=3)(group_merge)
    x = Activation('relu')(x)
    return x


def bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    init = input
    grouped_channels = int(filters / cardinality)

    if init._keras_shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        init = BatchNormalization(axis=3)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=3)(x)

    x = add([init, x])
    x = Activation('relu')(x)
    return x