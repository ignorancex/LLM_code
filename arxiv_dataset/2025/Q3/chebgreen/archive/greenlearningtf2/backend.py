import numpy as np
import tensorflow as tf
import scipy
from abc import ABC
import configparser, ast

parser = configparser.ConfigParser(inline_comment_prefixes="#")
parser.read('settings.ini')

# tf.float64 doesn't work for Apple tf2
class Config(ABC):
    def __init__(self, precision):
        self.precision = precision
    
    def __call__(self, package):
        config = {
            32: {np: np.float32, tf: tf.float32},
            64: {np: np.float64, tf: tf.float64},
        }[self.precision]
        return config[package]
 
config = Config(parser['GENERAL'].getint('precision'))

# Need to explicitly set the config for float to float64 since tf.keras.Model.__call__ casts to tf.backend.floatx
if config(tf) == tf.float64:
    tf.keras.backend.set_floatx('float64')
