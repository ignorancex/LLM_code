import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model, Input
from resnet import *

class MyResnet(Model):
    def __init__(self, class_count, weight_decay=None):
        super().__init__()
        print("using my resnet.")
        if not isinstance(weight_decay, type(None)):
            regularizer = tf.keras.regularizers.L2(weight_decay)
        else:
            regularizer = None
        self.base_model = ResNet18()
        self.base_model.trainable = True
        for layer in self.base_model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)
        self.out_dense = Dense(class_count, use_bias=False)#shift bias to last
        self.last_mul = tf.Variable(initial_value=0.0, trainable=True)#make sure FUKL won't overflow
        self.last_bias = tf.Variable(initial_value=tf.zeros((1,class_count)), trainable=True)
        self.config = {
            "class_count":class_count,
            "weight_decay":weight_decay,
        }

    def get_config(self):
        return self.config

    def call(self, x):
        x = tf.cast(x, tf.float32)
        base_feature = self.base_model(x)
        base_feature = tf.reduce_mean(base_feature, axis=[1,2])
        dense_y = self.out_dense(base_feature)
        raw_y = dense_y*self.last_mul+self.last_bias
        out_y = tf.nn.softmax(raw_y)
        final_result = tf.argmax(out_y, axis=-1)
        return out_y, final_result, raw_y, base_feature

class DataAugmentGenerator(Model):
    def __init__(self, adjust_brightness=False, adjust_contrast=False, add_noise=False, 
        adjust_size=True, random_flip=True):
        super().__init__()
        # self.brightness_delta = brightness_delta#brightness_delta=0.2
        # self.contrast_bound = contrast_bound#contrast_bound=(0.7, 1.2)
        self.adjust_brightness = adjust_brightness
        self.adjust_contrast = adjust_contrast
        self.add_noise = add_noise
        self.adjust_size = adjust_size
        self.random_flip = random_flip

    def call(self, inputs, seed=None,\
        brightness_delta=20, contrast_bound=(0.7, 1.2), noise_std=5, min_crop_ratio=0.7,
        pad_para=[[0,0],[4,4],[4,4],[0,0]]):
        if not isinstance(seed, type(None)):
            apply_seed = True
            seed = (seed,0)#input seed must be an integer
        else:
            apply_seed = False

        if self.random_flip:
            if apply_seed:
                inputs = tf.image.stateless_random_flip_left_right(inputs, seed)
            else:
                inputs = tf.image.random_flip_left_right(inputs)

        if self.adjust_brightness:
            if apply_seed:
                inputs = tf.image.stateless_random_brightness(inputs, brightness_delta, seed)
            else:
                inputs = tf.image.random_brightness(inputs, brightness_delta)

        if self.adjust_contrast:
            if apply_seed:
                inputs = tf.image.stateless_random_contrast(inputs, *contrast_bound, seed)
            else:
                inputs = tf.image.random_contrast(inputs, *contrast_bound)

        if self.adjust_size:
            temp_inputs = tf.pad(inputs, pad_para, mode="REFLECT")
            if apply_seed:
                crop_ratio = tf.random.stateless_uniform((), seed, min_crop_ratio, 1)
            else:
                crop_ratio = tf.random.uniform((), min_crop_ratio, 1)
            crop_height = tf.cast(inputs.shape[1]*crop_ratio, tf.int32)
            crop_width = tf.cast(inputs.shape[2]*crop_ratio, tf.int32)

            if apply_seed:
                temp_inputs = tf.image.stateless_random_crop(temp_inputs, (temp_inputs.shape[0],crop_height,crop_width,3), seed)
            else:
                temp_inputs = tf.image.random_crop(temp_inputs, (temp_inputs.shape[0],crop_height,crop_width,3))
            inputs = tf.image.resize(temp_inputs, (inputs.shape[1], inputs.shape[2]))
        
        if self.add_noise:
            if apply_seed:
                noise = tf.random.stateless_normal((1,inputs.shape[1], inputs.shape[2],3), seed, stddev=noise_std)
            else:
                noise = tf.random.normal((1,inputs.shape[1], inputs.shape[2],3), stddev=noise_std)
            inputs = inputs+noise

        inputs = tf.clip_by_value(inputs, 0, 255)
        return inputs

if __name__ == "__main__":
    model = MyResnet50(1)
    model.build((None, 32,32,3))
    for var in model.non_trainable_variables:
        print(var.name, end=", ")