""" Simultaneous quantile regression:
    http://stefanmeili.ca/blog/quantile-regression
"""
import numpy as np
import tensorflow as tf

class SQ_Dense(tf.keras.layers.Dense):
    '''
    A Reshaped Dense layer to allow multiple quantiles to be predicted simultaneously
    The base Dense layer has units x n_quant units.
    The output is reshaped to (?, n_quant, units) 
    '''
    def __init__(self, units, n_quant, **kwargs):
        super(SQ_Dense, self).__init__(units * n_quant, **kwargs)
        self.units_ = units
        self.n_quant = n_quant

    def get_config(self):
        config = {
            'units': self.units,
            'n_quant': self.n_quant}
        base_config = super(SQ_Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input):
        super().build(input)

    def call(self, input, training = False):
        return tf.reshape(super(SQ_Dense, self).call(input), (-1, self.n_quant, self.units_))
    
class SQ_Loss:
    '''
    A general purpose quantile loss function. tau, y_pred and y_true are all reashaped appropriately for broadcasting.
    if tau is a float, this loss function can be used to model a single quantile.
    if tau is a list of floats, it can be used to model multiple quantiles simultaneously.
    '''
    def __init__(self, tau):
        self.__name__ = 'SQ_Loss'
        self.multiple = True if isinstance(tau, list) else False
        self.tau = np.array(tau).reshape(1,-1,1) if isinstance(tau, list) else tau

    def __call__(self, y_true, y_pred, **kwargs):
        y_true_ = tf.expand_dims(y_true, -2) if self.multiple else y_true
        #tensorflow seems to drop single element last dimensions. This fixes that.  
        y_pred_ = tf.expand_dims(y_pred, -1) if self.multiple and len(y_pred.shape) == 2 else y_pred
        err = y_true_ - y_pred_
        pinball = tf.maximum(self.tau * err, (self.tau - 1) * err)
        return tf.math.reduce_mean(pinball)
